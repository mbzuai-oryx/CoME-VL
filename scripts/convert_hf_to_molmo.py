import argparse
import logging
import math
import os
from pathlib import Path
from typing import Dict, Any

import torch
import einops
from flax.traverse_util import flatten_dict, unflatten_dict
from transformers import AutoModel, AutoModelForCausalLM, CLIPModel, SiglipModel, AutoModelForImageTextToText

from launch_scripts.utils import VISION_BACKBONES, LLMS, DEFAULT_LOAD_PATHS
from olmo import VisionBackboneConfig, ModelConfig, Molmo, BlockType
from olmo.util import prepare_cli_environment


def interpolate_position_embeddings(
    position_embeddings: torch.Tensor,
    num_patches: int,
    dim: int,
    patch_size: int,
    height: int,
    width: int,
    num_prefix_tokens: int = 1,
) -> torch.Tensor:
    from torch import nn as torch_nn

    num_positions = position_embeddings.shape[1] - num_prefix_tokens
    if num_patches == num_positions and height == width:
        return position_embeddings
    class_pos_embed = position_embeddings[:, :num_prefix_tokens]
    patch_pos_embed = position_embeddings[:, num_prefix_tokens:]
    height = height // patch_size
    width = width // patch_size
    # we add a small number to avoid floating point error in the interpolation
    # see discussion at https://github.com/facebookresearch/dino/issues/8
    height, width = height + 0.1, width + 0.1
    patch_pos_embed = patch_pos_embed.reshape(1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim)
    patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
    target_dtype = patch_pos_embed.dtype
    patch_pos_embed = torch_nn.functional.interpolate(
        patch_pos_embed.to(dtype=torch.float32),
        scale_factor=(float(height / math.sqrt(num_positions)), float(width / math.sqrt(num_positions))),
        mode="bicubic",
        align_corners=False,
    ).to(dtype=target_dtype)
    if int(height) != patch_pos_embed.shape[-2] or int(width) != patch_pos_embed.shape[-1]:
        raise ValueError("Width or height does not match with the interpolated position embeddings")
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    return torch.cat((class_pos_embed, patch_pos_embed), dim=1)


def convert_state_dict_clip(state_dict, vision_config: VisionBackboneConfig) -> Dict[str, Any]:
    state_dict = unflatten_dict(state_dict, sep=".")

    resblocks = {}
    for layer in range(vision_config.image_num_layers):
        layer_dict = state_dict["encoder"]["layers"][str(layer)]
        q, k, v, o = [
            layer_dict["self_attn"][f"{x}_proj"].pop("weight")
            for x in ["q", "k", "v", "out"]
        ]
        q_b, k_b, v_b, o_b = [
            layer_dict["self_attn"][f"{x}_proj"].pop("bias")
            for x in ["q", "k", "v", "out"]
        ]

        w1, w2 = [layer_dict["mlp"][f"{x}"].pop("weight") for x in ["fc1", "fc2"]]
        w1_b, w2_b = [layer_dict["mlp"][f"{x}"].pop("bias") for x in ["fc1", "fc2"]]

        mapped_layer_dict = {
            "attention": {
                "wq": dict(weight=q, bias=q_b),
                "wk": dict(weight=k, bias=k_b),
                "wv": dict(weight=v, bias=v_b),
                "wo": dict(weight=o, bias=o_b),
            },
            "feed_forward": {
                "w1": dict(weight=w1, bias=w1_b),
                "w2": dict(weight=w2, bias=w2_b), 
            },
            "attention_norm": {
                "weight": layer_dict["layer_norm1"].pop("weight"),
                "bias": layer_dict["layer_norm1"].pop("bias"),
            },
            "ffn_norm": {
                "weight": layer_dict["layer_norm2"].pop("weight"),
                "bias": layer_dict["layer_norm2"].pop("bias"),
            }
        }

        resblocks[str(layer)] = mapped_layer_dict
    
    # We accidentally set the number of layers for OpenAI CLIP ViT to 23 in experiments
    if str(vision_config.image_num_layers) in state_dict["encoder"]["layers"]:
        del state_dict["encoder"]["layers"][str(vision_config.image_num_layers)]

    # Interpolate position embeddings
    height, width = vision_config.image_default_input_size
    num_patches = vision_config.image_num_pos - 1
    position_embedding = state_dict["embeddings"]["position_embedding"].pop("weight")
    position_embedding = interpolate_position_embeddings(
        position_embedding.unsqueeze(0),
        num_patches,
        position_embedding.shape[-1],
        vision_config.image_patch_size,
        height,
        width,
    )

    patch_embedding = state_dict["embeddings"]["patch_embedding"].pop(
        "weight"
    ).permute(0, 2, 3, 1).reshape(vision_config.image_emb_dim, -1)

    pre_ln = {
        "weight": state_dict["pre_layrnorm"].pop("weight"),
        "bias": state_dict["pre_layrnorm"].pop("bias"),
    }

    out = {
        "class_embedding": state_dict["embeddings"].pop("class_embedding"),
        "positional_embedding": position_embedding[0],
        "patch_embedding": dict(weight=patch_embedding),
        "pre_ln": pre_ln,
        "transformer": dict(resblocks=resblocks),
    }
    out = flatten_dict(out, sep=".")
    del state_dict["post_layernorm"]
    for k in flatten_dict(state_dict):
        raise ValueError("Unused parameter:", k)
    return out


def convert_state_dict_siglip(state_dict, vision_config: VisionBackboneConfig) -> Dict[str, Any]:
    state_dict = unflatten_dict(state_dict, sep=".")

    resblocks = {}
    for layer in range(vision_config.image_num_layers):
        layer_dict = state_dict["encoder"]["layers"][str(layer)]
        q, k, v, o = [
            layer_dict["self_attn"][f"{x}_proj"].pop("weight")
            for x in ["q", "k", "v", "out"]
        ]
        q_b, k_b, v_b, o_b = [
            layer_dict["self_attn"][f"{x}_proj"].pop("bias")
            for x in ["q", "k", "v", "out"]
        ]

        w1, w2 = [layer_dict["mlp"][f"{x}"].pop("weight") for x in ["fc1", "fc2"]]
        w1_b, w2_b = [layer_dict["mlp"][f"{x}"].pop("bias") for x in ["fc1", "fc2"]]

        mapped_layer_dict = {
            "attention": {
                "wq": dict(weight=q, bias=q_b),
                "wk": dict(weight=k, bias=k_b),
                "wv": dict(weight=v, bias=v_b),
                "wo": dict(weight=o, bias=o_b),
            },
            "feed_forward": {
                "w1": dict(weight=w1, bias=w1_b),
                "w2": dict(weight=w2, bias=w2_b), 
            },
            "attention_norm": {
                "weight": layer_dict["layer_norm1"].pop("weight"),
                "bias": layer_dict["layer_norm1"].pop("bias"),
            },
            "ffn_norm": {
                "weight": layer_dict["layer_norm2"].pop("weight"),
                "bias": layer_dict["layer_norm2"].pop("bias"),
            }
        }

        resblocks[str(layer)] = mapped_layer_dict

    # Interpolate position embeddings
    height, width = vision_config.image_default_input_size
    num_patches = vision_config.image_num_pos
    position_embedding = state_dict["embeddings"]["position_embedding"].pop("weight")
    position_embedding = interpolate_position_embeddings(
        position_embedding.unsqueeze(0),
        num_patches,
        position_embedding.shape[-1],
        vision_config.image_patch_size,
        height,
        width,
        num_prefix_tokens=0,
    )

    patch_embedding = state_dict["embeddings"]["patch_embedding"].pop(
        "weight"
    ).permute(0, 2, 3, 1).reshape(vision_config.image_emb_dim, -1)
    patch_embedding_b = state_dict["embeddings"]["patch_embedding"].pop("bias")

    out = {
        "positional_embedding": position_embedding[0],
        "patch_embedding": dict(weight=patch_embedding, bias=patch_embedding_b),
        "transformer": dict(resblocks=resblocks),
    }
    out = flatten_dict(out, sep=".")
    del state_dict["post_layernorm"]
    del state_dict["head"]
    for k in flatten_dict(state_dict):
        raise ValueError("Unused parameter:", k)
    return out


def convert_state_dict_dino(state_dict, vision_config: VisionBackboneConfig) -> Dict[str, Any]:
    state_dict = unflatten_dict(state_dict, sep=".")

    resblocks = {}
    for layer in range(vision_config.image_num_layers):
        layer_dict = state_dict["encoder"]["layer"][str(layer)]
        q, k, v = [
            layer_dict["attention"]["attention"][f"{x}"].pop("weight")
            for x in ["query", "key", "value"]
        ]
        q_b, k_b, v_b = [
            layer_dict["attention"]["attention"][f"{x}"].pop("bias")
            for x in ["query", "key", "value"]
        ]
        o = layer_dict["attention"]["output"]["dense"].pop("weight")
        o_b = layer_dict["attention"]["output"]["dense"].pop("bias")

        w1, w2 = [layer_dict["mlp"][f"{x}"].pop("weight") for x in ["fc1", "fc2"]]
        w1_b, w2_b = [layer_dict["mlp"][f"{x}"].pop("bias") for x in ["fc1", "fc2"]]

        mapped_layer_dict = {
            "attention": {
                "wq": dict(weight=q, bias=q_b),
                "wk": dict(weight=k, bias=k_b),
                "wv": dict(weight=v, bias=v_b),
                "wo": dict(weight=o, bias=o_b),
            },
            "feed_forward": {
                "w1": dict(weight=w1, bias=w1_b),
                "w2": dict(weight=w2, bias=w2_b), 
            },
            "attention_norm": {
                "weight": layer_dict["norm1"].pop("weight"),
                "bias": layer_dict["norm1"].pop("bias"),
            },
            "ffn_norm": {
                "weight": layer_dict["norm2"].pop("weight"),
                "bias": layer_dict["norm2"].pop("bias"),
            },
            "lambda1": layer_dict["layer_scale1"].pop("lambda1"),
            "lambda2": layer_dict["layer_scale2"].pop("lambda1"),
        }

        resblocks[str(layer)] = mapped_layer_dict

    # Interpolate position embeddings
    height, width = vision_config.image_default_input_size
    num_patches = vision_config.image_num_pos - 1
    position_embedding = state_dict["embeddings"].pop("position_embeddings")
    position_embedding = interpolate_position_embeddings(
        position_embedding,
        num_patches,
        position_embedding.shape[-1],
        vision_config.image_patch_size,
        height,
        width,
    )

    patch_embedding = state_dict["embeddings"]["patch_embeddings"]["projection"].pop(
        "weight"
    ).permute(0, 2, 3, 1).reshape(vision_config.image_emb_dim, -1)
    patch_embedding_b = state_dict["embeddings"]["patch_embeddings"]["projection"].pop("bias")

    out = {
        "class_embedding": state_dict["embeddings"].pop("cls_token").reshape(-1),
        "positional_embedding": position_embedding[0],
        "patch_embedding": dict(weight=patch_embedding, bias=patch_embedding_b),
        "transformer": dict(resblocks=resblocks),
    }
    out = flatten_dict(out, sep=".")
    del state_dict["layernorm"]
    del state_dict["embeddings"]["mask_token"]
    for k in flatten_dict(state_dict):
        raise ValueError("Unused parameter:", k)
    return out



def convert_state_dict_dino1(state_dict, vision_config: VisionBackboneConfig) -> Dict[str, Any]:

    state_dict = unflatten_dict(state_dict, sep=".")
    
    # print(12, state_dict['embeddings'].keys())
    resblocks = {}
    for layer_idx in range(vision_config.image_num_layers):
        layer = state_dict["layer"][str(layer_idx)]
        # Attention projections
        q = layer["attention"]["q_proj"].pop("weight")
        q_b = layer["attention"]["q_proj"].pop("bias", None)
        k = layer["attention"]["k_proj"].pop("weight")
        k_b = layer["attention"]["k_proj"].pop("bias", None)
        v = layer["attention"]["v_proj"].pop("weight")
        v_b = layer["attention"]["v_proj"].pop("bias", None)
        o = layer["attention"]["o_proj"].pop("weight")
        o_b = layer["attention"]["o_proj"].pop("bias", None)

        # MLP projections
        w1 = layer["mlp"]["up_proj"].pop("weight")
        w1_b = layer["mlp"]["up_proj"].pop("bias", None)
        w2 = layer["mlp"]["down_proj"].pop("weight")
        w2_b = layer["mlp"]["down_proj"].pop("bias", None)

        # Norm layers
        attn_norm_w = layer["norm1"].pop("weight")
        attn_norm_b = layer["norm1"].pop("bias")
        ffn_norm_w = layer["norm2"].pop("weight")
        ffn_norm_b = layer["norm2"].pop("bias")

        # Layer scales (if any)
        lambda1 = layer.get("layer_scale1", {}).pop("lambda1", None)
        lambda2 = layer.get("layer_scale2", {}).pop("lambda1", None)

        resblocks[str(layer_idx)] = {
            "attention": {
                "wq": dict(weight=q, bias=q_b),
                "wk": dict(weight=k, bias=k_b),
                "wv": dict(weight=v, bias=v_b),
                "wo": dict(weight=o, bias=o_b),
            },
            "feed_forward": {
                "w1": dict(weight=w1, bias=w1_b),
                "w2": dict(weight=w2, bias=w2_b),
            },
            "attention_norm": {"weight": attn_norm_w, "bias": attn_norm_b},
            "ffn_norm": {"weight": ffn_norm_w, "bias": ffn_norm_b},
            "lambda1": lambda1,
            "lambda2": lambda2,
        }

    # Patch embeddings
        
    # If patch_embeddings also contains positional embeddings
    # pos_embd = state_dict["embeddings"].pop("position_embeddings", None)

    patch_w = state_dict["embeddings"]["patch_embeddings"].pop("weight")
    patch_b = state_dict["embeddings"]["patch_embeddings"].pop("bias", None)

    

    # CLS token
    cls_token = state_dict["embeddings"].pop("cls_token").reshape(-1)

    # Register tokens (if present)
    reg_tokens = state_dict["embeddings"].pop("register_tokens", None)
    if reg_tokens is not None:
        reg_tokens = reg_tokens.reshape(reg_tokens.shape[0], -1)

    # Final norm (after Transformer blocks)
    final_norm_w = state_dict.pop("norm", {}).get("weight", None)
    final_norm_b = state_dict.pop("norm", {}).get("bias", None)
    
    # # Interpolate position embeddings
    # height, width = vision_config.image_default_input_size
    # num_patches = vision_config.image_num_pos - 1
    # breakpoint()
    # position_embedding = state_dict["embeddings"].pop("position_embeddings")
    # position_embedding = interpolate_position_embeddings(
    #     position_embedding,
    #     num_patches,
    #     position_embedding.shape[-1],
    #     vision_config.image_patch_size,
    #     height,
    #     width,
    # )
    
    output = {
        "class_embedding": cls_token,
        # "register_tokens": reg_tokens,
        "patch_embedding": dict(weight=patch_w, bias=patch_b),
        "positional_embedding": dict(weight=patch_w, bias=patch_b),
        "transformer": dict(resblocks=resblocks),
        # "final_norm": dict(weight=final_norm_w, bias=final_norm_b),
    }

    output = flatten_dict(output, sep=".")

    # Clean up optional keys
    state_dict.get("embeddings", {}).pop("mask_token", None)

    remaining = flatten_dict(state_dict)
    if remaining:
        raise ValueError(f"Still unused parameters: {remaining}")

    return output



def convert_state_dict_olmoe(state_dict, config: ModelConfig, block_type: BlockType) -> Dict[str, Any]:
    state_dict = unflatten_dict(state_dict, sep=".")
    assert len(state_dict) == 2
    lmhead = state_dict["lm_head"]
    state_dict = state_dict["model"]

    blocks = {}
    for layer in range(config.n_layers):
        layer_dict = state_dict["layers"][str(layer)]
        q, k, v, o = [layer_dict["self_attn"][f"{k}_proj"].pop("weight") for k in ["q", "k", "v", "o"]]

        assert block_type == BlockType.moe

        router = layer_dict["mlp"]["gate"].pop("weight")
        mlp_gates = [layer_dict["mlp"]["experts"][str(i)]["gate_proj"].pop("weight") for i in range(config.moe_num_experts)]
        mlp_ups = [layer_dict["mlp"]["experts"][str(i)]["up_proj"].pop("weight") for i in range(config.moe_num_experts)]
        mlp_downs = [layer_dict["mlp"]["experts"][str(i)]["down_proj"].pop("weight").t() for i in range(config.moe_num_experts)]

        ffn = {
            "router": {
                "layer": dict(weight=router),
            },
            "experts": {
                "mlp": dict(
                    w1=torch.cat(mlp_gates, 0),
                    v1=torch.cat(mlp_ups, 0),
                    w2=torch.cat(mlp_downs, 0),
                ),
            }
        }

        mapped_layer_dict = {
            "ffn": ffn,
            "ff_norm": {
                "weight": layer_dict[f"post_attention_layernorm"].pop("weight")
            },
            "attn_norm": {
                "weight": layer_dict[f"input_layernorm"].pop("weight")
            },
            "att_proj": dict(
                weight=torch.cat((q, k, v), dim=0),
            ),
            "attn_out": dict(weight=o),
            "q_norm": {
                "weight": layer_dict["self_attn"]["q_norm"].pop("weight"),
            },
            "k_norm": {
                "weight": layer_dict["self_attn"]["k_norm"].pop("weight"),
            },
        }

        blocks[str(layer)] = mapped_layer_dict

    out = flatten_dict(dict(transformer=dict(blocks=blocks)), sep=".")
    assert list(lmhead) == ["weight"]
    out.update({
        "transformer.wte.embedding": state_dict["embed_tokens"].pop("weight"),
        "transformer.ln_f.weight": state_dict["norm"].pop("weight"),
        "transformer.ff_out.weight": lmhead.pop("weight"),
    })
    for k in flatten_dict(state_dict):
        raise ValueError("Unused parameter:", k)
    return out


def convert_state_dict_olmo_1024_preview(state_dict, config: ModelConfig, block_type: BlockType) -> Dict[str, Any]:
    state_dict = unflatten_dict(state_dict, sep=".")
    assert len(state_dict) == 2
    lmhead = state_dict["lm_head"]
    state_dict = state_dict["model"]

    blocks = {}
    for layer in range(config.n_layers):
        layer_dict = state_dict["layers"][str(layer)]
        q, k, v, o = [layer_dict["self_attn"][f"{k}_proj"].pop("weight") for k in ["q", "k", "v", "o"]]
        mlp_gate = layer_dict["mlp"]["gate_proj"].pop("weight")
        mlp_up = layer_dict["mlp"]["up_proj"].pop("weight")
        mlp_down = layer_dict["mlp"]["down_proj"].pop("weight")

        assert block_type == BlockType.sequential

        mapped_layer_dict = {
            "ff_proj": {
                "weight": torch.cat([mlp_up, mlp_gate], 0)
            },
            "ff_out": {
                "weight": mlp_down
            },
            "ff_norm": {
                "weight": layer_dict[f"post_feedforward_layernorm"].pop("weight")
            },
            "attn_norm": {
                "weight": layer_dict[f"post_attention_layernorm"].pop("weight")
            },
            "att_proj": dict(
                weight=torch.cat((q, k, v), dim=0),
            ),
            "attn_out": dict(weight=o),
            "q_norm": {
                "weight": layer_dict["self_attn"]["q_norm"].pop("weight"),
            },
            "k_norm": {
                "weight": layer_dict["self_attn"]["k_norm"].pop("weight"),
            },
        }

        blocks[str(layer)] = mapped_layer_dict

    out = flatten_dict(dict(transformer=dict(blocks=blocks)), sep=".")
    assert list(lmhead) == ["weight"]
    out.update({
        "transformer.wte.embedding": state_dict["embed_tokens"].pop("weight"),
        "transformer.ln_f.weight": state_dict["norm"].pop("weight"),
        "transformer.ff_out.weight": lmhead.pop("weight"),
    })
    for k in flatten_dict(state_dict):
        raise ValueError("Unused parameter:", k)
    return out


def convert_state_dict_qwen2(state_dict, config: ModelConfig, block_type: BlockType) -> Dict[str, Any]:
    state_dict = unflatten_dict(state_dict, sep=".")
    assert len(state_dict) == 2
    lmhead = state_dict["lm_head"]
    state_dict = state_dict["model"]

    blocks = {}
    for layer in range(config.n_layers):
        layer_dict = state_dict["layers"][str(layer)]
        q, k, v, o = [layer_dict["self_attn"][f"{k}_proj"].pop("weight") for k in ["q", "k", "v", "o"]]
        if config.qkv_bias:
            q_b, k_b, v_b = [layer_dict["self_attn"][f"{k}_proj"].pop("bias") for k in ["q", "k", "v"]]
        else:
            q_b, k_b, v_b = None, None, None

        mlp_gate = layer_dict["mlp"]["gate_proj"].pop("weight")
        mlp_up = layer_dict["mlp"]["up_proj"].pop("weight")
        mlp_down = layer_dict["mlp"]["down_proj"].pop("weight")

        if block_type == BlockType.llama:
            mapped_layer_dict = {
                "q_proj": dict(weight=q, bias=q_b),
                "k_proj": dict(weight=k, bias=k_b),
                "v_proj": dict(weight=v, bias=v_b),
                "attn_out": dict(weight=o),
                "ff_norm": {
                    "weight": layer_dict[f"post_attention_layernorm"].pop("weight")
                },
                "attn_norm": {
                    "weight": layer_dict[f"input_layernorm"].pop("weight")
                },
                "ff_proj1": dict(weight=mlp_gate),
                "ff_proj2": dict(weight=mlp_up),
                "ff_out": dict(weight=mlp_down),
            }
        elif block_type == BlockType.sequential:
            mapped_layer_dict = {
                "ff_proj": {
                    "weight": torch.cat([mlp_up, mlp_gate], 0)
                },
                "ff_out": {
                    "weight": mlp_down
                },
                "ff_norm": {
                    "weight": layer_dict[f"post_attention_layernorm"].pop("weight")
                },
                "attn_norm": {
                    "weight": layer_dict[f"input_layernorm"].pop("weight")
                },
                "att_proj": dict(
                    weight=torch.cat((q, k, v), dim=0),
                    bias=None if q_b is None else torch.cat((q_b, k_b, v_b), dim=0)
                ),
                "attn_out": dict(weight=o),
            }
        else:
            raise NotImplementedError(block_type)
        blocks[str(layer)] = mapped_layer_dict

    out = flatten_dict(dict(transformer=dict(blocks=blocks)), sep=".")
    assert list(lmhead) == ["weight"]
    out.update({
        "transformer.wte.embedding": state_dict["embed_tokens"].pop("weight"),
        "transformer.ln_f.weight": state_dict["norm"].pop("weight"),
        "transformer.ff_out.weight": lmhead.pop("weight"),
    })
    for k in flatten_dict(state_dict):
        raise ValueError("Unused parameter:", k)
    return out



def convert_state_dict_qwen3(state_dict, config: ModelConfig, block_type: BlockType) -> Dict[str, Any]:
    state_dict = unflatten_dict(state_dict, sep=".")
    lmhead = state_dict["lm_head"]
    # breakpoint()
    state_dict = state_dict["model"]
    # state_dict = state_dict["model"]["language_model"]

    blocks = {}
    for layer in range(config.n_layers):
        layer_dict = state_dict["layers"][str(layer)]
        q, k, v, o = [layer_dict["self_attn"][f"{k}_proj"].pop("weight") for k in ["q", "k", "v", "o"]]
        q_b = k_b = v_b = None
        if config.qkv_bias:
            q_b, k_b, v_b = [layer_dict["self_attn"][f"{k}_proj"].pop("bias") for k in ["q", "k", "v"]]

        mlp_gate = layer_dict["mlp"]["gate_proj"].pop("weight")
        mlp_up = layer_dict["mlp"]["up_proj"].pop("weight")
        mlp_down = layer_dict["mlp"]["down_proj"].pop("weight")

        # NEW: Pop q_norm and k_norm
        qn = layer_dict["self_attn"].get("q_norm", {}).pop("weight", None)
        kn = layer_dict["self_attn"].get("k_norm", {}).pop("weight", None)

        if block_type == BlockType.llama:
            mapped_layer_dict = {
                "q_proj": dict(weight=q, bias=q_b),
                "k_proj": dict(weight=k, bias=k_b),
                "v_proj": dict(weight=v, bias=v_b),
                "attn_out": dict(weight=o),
                "ff_norm": {"weight": layer_dict["post_attention_layernorm"].pop("weight")},
                "attn_norm": {"weight": layer_dict["input_layernorm"].pop("weight")},
                "ff_proj1": dict(weight=mlp_gate),
                "ff_proj2": dict(weight=mlp_up),
                "ff_out": dict(weight=mlp_down),
            }
        elif block_type == BlockType.sequential:
            mapped_layer_dict = {
                "ff_proj": {"weight": torch.cat([mlp_up, mlp_gate], 0)},
                "ff_out": {"weight": mlp_down},
                "ff_norm": {"weight": layer_dict["post_attention_layernorm"].pop("weight")},
                "attn_norm": {"weight": layer_dict["input_layernorm"].pop("weight")},
                "att_proj": dict(
                    weight=torch.cat((q, k, v), dim=0),
                    bias=None if q_b is None else torch.cat((q_b, k_b, v_b), dim=0)
                ),
                "attn_out": dict(weight=o),
            }
        else:
            raise NotImplementedError(block_type)

        # Add q_norm/k_norm if available
        if qn is not None:
            mapped_layer_dict["q_norm"] = {"weight": qn}
        if kn is not None:
            mapped_layer_dict["k_norm"] = {"weight": kn}

        blocks[str(layer)] = mapped_layer_dict

    out = flatten_dict(dict(transformer=dict(blocks=blocks)), sep=".")
    out.update({
        "transformer.wte.embedding": state_dict["embed_tokens"].pop("weight"),
        "transformer.ln_f.weight": state_dict["norm"].pop("weight"),
        "transformer.ff_out.weight": lmhead.pop("weight"),
    })

    for k in flatten_dict(state_dict):
        raise ValueError("Unused parameter:", k)
    return out




def convert_state_dict_gptoss(
    state_dict: Dict[str, Any],
    config: ModelConfig,
    block_type: BlockType
) -> Dict[str, Any]:
    """
    Convert a HuggingFace GptOssForCausalLM state_dict into Molmo's transformer format.

    This handles:
      - Unflattening and flattening nested dicts
      - Removing unneeded modules (rotary_emb, sinks, router)
      - Extracting q/k/v/o projections and biases (dropping o_proj bias)
      - Aggregating MoE expert weights or falling back to single-expert keys
      - Mapping to either LLaMA-style or sequential block types
      - Final embedding, layer norm, and LM head extraction

    Args:
        state_dict: Raw HF checkpoint state dict.
        config:   Molmo ModelConfig with n_layers, qkv_bias, etc.
        block_type: BlockType.llama or BlockType.sequential mapping style.

    Returns:
        A flattened dict suitable for Molmo transformer initialization.
    """
    # 1) Unflatten HF dict and pop off LM head
    state_dict = unflatten_dict(state_dict, sep=".")
    lmhead = state_dict.pop("lm_head")
    core = state_dict["model"]

    # 2) Drop unneeded modules
    core.pop("rotary_emb", None)

    blocks = {}
    for layer_idx in range(config.n_layers):
        layer = core["layers"][str(layer_idx)]

        # --- Attention projections ---
        q, k, v, o = [
            layer["self_attn"][f"{p}_proj"].pop("weight")
            for p in ("q", "k", "v", "o")
        ]
        q_b = k_b = v_b = None
        if config.qkv_bias:
            q_b, k_b, v_b = [
                layer["self_attn"][f"{p}_proj"].pop("bias")
                for p in ("q", "k", "v")
            ]
        # drop o_proj bias
        layer["self_attn"]["o_proj"].pop("bias", None)

        # drop sinks buffer
        layer["self_attn"].pop("sinks", None)

        # optional norms
        qn = layer["self_attn"].get("q_norm", {}).pop("weight", None)
        kn = layer["self_attn"].get("k_norm", {}).pop("weight", None)

        # --- MLP experts ---
        exp_dict = layer["mlp"]["experts"]
        numeric = [k for k in exp_dict.keys() if k.isdigit()]
        if numeric:
            gate_ls, up_ls, down_ls = [], [], []
            for idx in sorted(numeric, key=int):
                e = exp_dict[idx]
                gate_ls.append(e.pop("gate_proj")["weight"])
                up_ls.append(e.pop("up_proj")["weight"])
                down_ls.append(e.pop("down_proj")["weight"])
            mlp_gate = torch.cat(gate_ls, dim=0)
            mlp_up   = torch.cat(up_ls,   dim=0)
            mlp_down = torch.cat(down_ls, dim=0)
        else:
            # fallback: pop single-expert weights & biases
            gate_ent = exp_dict.pop("gate_proj", exp_dict.pop("gate_up_proj", None))
            up_ent   = exp_dict.pop("up_proj",   exp_dict.pop("up_proj_proj", None))
            down_ent = exp_dict.pop("down_proj", exp_dict.pop("down_proj_proj", None))
            # also pop bias versions
            exp_dict.pop("gate_proj_bias", None)
            exp_dict.pop("gate_up_proj_bias", None)
            exp_dict.pop("up_proj_bias", None)
            exp_dict.pop("up_proj_proj_bias", None)
            exp_dict.pop("down_proj_bias", None)
            exp_dict.pop("down_proj_proj_bias", None)

            mlp_gate = gate_ent["weight"] if isinstance(gate_ent, dict) else gate_ent
            mlp_up   = up_ent["weight"]   if isinstance(up_ent, dict)   else up_ent
            mlp_down = down_ent["weight"] if isinstance(down_ent, dict) else down_ent

        # drop router
        layer["mlp"].pop("router", None)

        # --- Norms ---
        ln1 = layer["input_layernorm"].pop("weight")
        ln2 = layer["post_attention_layernorm"].pop("weight")

        # --- Map blocks ---
        if block_type == BlockType.llama:
            mapped = {
                "q_proj":    {"weight": q,   "bias": q_b},
                "k_proj":    {"weight": k,   "bias": k_b},
                "v_proj":    {"weight": v,   "bias": v_b},
                "attn_out":  {"weight": o},
                "attn_norm": {"weight": ln1},
                "ff_norm":   {"weight": ln2},
                "ff_proj1":  {"weight": mlp_gate},
                "ff_proj2":  {"weight": mlp_up},
                "ff_out":    {"weight": mlp_down},
            }
        else:
            mapped = {
                "att_proj":  {
                    "weight": torch.cat((q, k, v), dim=0),
                    "bias": None if q_b is None else torch.cat((q_b, k_b, v_b), dim=0)
                },
                "attn_out":  {"weight": o},
                "attn_norm": {"weight": ln1},
                "ff_proj":   {"weight": torch.cat([mlp_up, mlp_gate], dim=0)},
                "ff_out":    {"weight": mlp_down},
                "ff_norm":   {"weight": ln2},
            }

        # re-add optional
        if qn is not None:
            mapped["q_norm"] = {"weight": qn}
        if kn is not None:
            mapped["k_norm"] = {"weight": kn}

        blocks[str(layer_idx)] = mapped

    # flatten + embeddings
    out = flatten_dict({"transformer": {"blocks": blocks}}, sep=".")
    out["transformer.wte.embedding"] = core["embed_tokens"].pop("weight")
    out["transformer.ln_f.weight"]   = core["norm"].pop("weight")
    out["transformer.ff_out.weight"] = lmhead.pop("weight")

    # validate no leftovers
    leftovers = list(flatten_dict(core).keys())
    if leftovers:
        raise ValueError(f"Unused parameter in conversion: {leftovers[0]}")

    return out





def get_default_load_path(model_name: str) -> str:
    default_load_path = DEFAULT_LOAD_PATHS[model_name]
    return "/".join(default_load_path.split("/")[1:])


CONVERT_FNS = {
    "openai": convert_state_dict_clip,
    "siglip": convert_state_dict_siglip,
    "dinov2_large_336": convert_state_dict_dino,
    "dinov3_large_224": convert_state_dict_dino1,
    "metaclip_l14_336": convert_state_dict_clip,
    "olmoe": convert_state_dict_olmoe,
    "olmo_1024_preview": convert_state_dict_olmo_1024_preview,
    "qwen2.5_3b": convert_state_dict_qwen2,
    "qwen2_3b": convert_state_dict_qwen2,
    "qwen2_7b": convert_state_dict_qwen2,
    "qwen2_72b": convert_state_dict_qwen2,

    "qwen3_1b": convert_state_dict_qwen3,
    "qwen3_4b": convert_state_dict_qwen3,

    "gptoss_20b": convert_state_dict_gptoss,
}


VIT_HF_SOURCES  = {
    "openai": "openai/clip-vit-large-patch14-336",
    # "siglip": "google/siglip-so400m-patch14-384",
    "siglip": "google/siglip2-so400m-patch16-384",
    "dinov2_large_336": "facebook/dinov2-large",
    "dinov3_large_224": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "metaclip_l14_336": "facebook/metaclip-l14-fullcc2.5b",
}


LLM_HF_SOURCES = {
    "olmoe": "allenai/OLMoE-1B-7B-0924",
    "olmo_1024_preview": "allenai/OLMo-7B-1024-preview",
    "qwen2_3b": "Qwen/Qwen2-3B",
    # "qwen2_7b": "Qwen/Qwen2-7B",
    "qwen2_7b": "Qwen/Qwen2.5-7B",
    "qwen2.5_3b": "Qwen/Qwen2.5-3B",
    "qwen2_72b": "Qwen/Qwen2-72B",

    "qwen3_1b": "Qwen/Qwen3-1.7B",
    "qwen3_4b": "Qwen/Qwen3-4B",
    # "qwen3_4b": "OpenGVLab/InternVL3_5-4B-HF",

    "gptoss_20b": "openai/gpt-oss-20b",
    
}


def main_vit(args: argparse.Namespace) -> None:
    hf_source = VIT_HF_SOURCES[args.model]
    cfg = ModelConfig(vision_backbone=VISION_BACKBONES[args.model])
    cfg.init_device = 'cpu'
    v_cfg = cfg.vision_backbone
    convert_fn = CONVERT_FNS[args.model]

    output_path = str(Path(args.data_dir).joinpath(get_default_load_path(args.model)))
    Path(output_path).parent.mkdir(exist_ok=True, parents=True)

    logging.info(f"Convert model {args.model} to olmo format and save to {output_path}...")

    logging.info(f"Loading model from {hf_source}...")

    model = AutoModel.from_pretrained(
        hf_source,
        torch_dtype=torch.float32,
        cache_dir=args.cache_dir,
    )
    if isinstance(model, (CLIPModel, SiglipModel)):
        model = model.vision_model

    state_dict = model.state_dict()

    logging.info("Converting...")

    vit_state_dict = convert_fn(state_dict, v_cfg)
    
    logging.info("Saving...")
    torch.save(vit_state_dict, output_path)


def main_llm(args: argparse.Namespace) -> None:
    hf_source = LLM_HF_SOURCES[args.model]
    cfg = LLMS[args.model]
    # print(cfg)
    cfg.init_device = 'cpu'
    convert_fn = CONVERT_FNS[args.model]

    output_path = str(Path(args.data_dir).joinpath(get_default_load_path(args.model)))
    Path(output_path).parent.mkdir(exist_ok=True, parents=True)

    logging.info(f"Convert model {args.model} to olmo format and save to {output_path}...")

    logging.info(f"Loading model from {hf_source}...")

    # model = AutoModelForImageTextToText.from_pretrained(
    #     hf_source,
    #     torch_dtype=torch.float32,
    #     trust_remote_code=True,
    #     cache_dir=args.cache_dir,
    #     revision="fp32" if args.model == "olmoe" else "main",
    # )

    model = AutoModelForCausalLM.from_pretrained(
        hf_source,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        cache_dir=args.cache_dir,
        revision="fp32" if args.model == "olmoe" else "main",
    )

    state_dict = model.state_dict()

    logging.info("Converting...")
    
    olmo_state_dict = convert_fn(state_dict, cfg, cfg.block_type)

    logging.info("Saving...")
    torch.save(olmo_state_dict, output_path)


def main(args: argparse.Namespace) -> None:
    prepare_cli_environment()

    if args.data_dir is None:
        if "MOLMO_DATA_DIR" not in os.environ:
            raise ValueError("Either `data_dir` or env variable MOLMO_DATA_DIR must be set")
        args.data_dir = os.environ["MOLMO_DATA_DIR"]
        logging.info(f"Defaulting to data dir {args.data_dir}.")

    print('model ', args.model, VISION_BACKBONES)
    if args.model in VISION_BACKBONES:
        main_vit(args)
    elif args.model in LLMS:
        main_llm(args)
    else:
        raise ValueError(f"Unknown model {args.model}")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="convert hf vit/llm to molmo format script")
    parser.add_argument(
        "model",
        type=str,
        help="Model to be converted",
        choices=list(CONVERT_FNS.keys()),
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Needed to save converted model weights. It is a directory",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        help="Directory to save HF parameters",
    )

    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    main(args)
