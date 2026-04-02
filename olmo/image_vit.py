import math
from copy import deepcopy
from functools import partial
from typing import Callable, List, Optional

import torch
import torch.backends.cuda
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import get_activation

from olmo.config import ModelConfig, AttentionType
import torch
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image

import argparse
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoImageProcessor
import pdb
from torch.distributed import get_rank
from olmo.dinov3 import *
from olmo.config import ModelConfig, AttentionType, VisionBackboneConfig
from einops import rearrange

def _expand_token(token, batch_size: int):
    return token.view(1, 1, -1).expand(batch_size, -1, -1)


def vit_activation_checkpoint_function(cfg: ModelConfig):
    v_cfg = cfg.vision_backbone
    preserve_rng_state = (
        (v_cfg.attention_dropout == 0.0) and (v_cfg.residual_dropout == 0.0)
    )
    from torch.utils.checkpoint import checkpoint

    return partial(
        checkpoint,
        preserve_rng_state=preserve_rng_state,
        use_reentrant=False,
    )


class ViTMultiHeadDotProductAttention(nn.Module):
    """MDPA for the image ViT"""

    def __init__(self, config: ModelConfig, use_bias: bool = True, is_vit_layer: Optional[bool] = True, encoder2: bool = False):
        super().__init__()
        self.config = config
        self.use_bias = use_bias
        if encoder2:
            v_cfg = config.vision_backbone2
        else:
            v_cfg = config.vision_backbone
        self.embed_dim = v_cfg.image_emb_dim
        self.num_heads = v_cfg.image_num_heads
        self.head_dim = v_cfg.image_head_dim
        self.num_key_value_heads = v_cfg.image_num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.initializer_range = v_cfg.initializer_range
        self.is_vit_layer = is_vit_layer

        nlayers = 1 if (is_vit_layer or config.vit_layers is None) else len(config.vit_layers)

        self.wq = nn.Linear(
            nlayers * self.embed_dim,
            self.num_heads * self.head_dim,
            bias=use_bias,
            device=config.init_device,
            )
        self.wk = nn.Linear(
            nlayers * self.embed_dim,
            self.num_key_value_heads * self.head_dim,
            bias=use_bias,
            device=config.init_device,
            )
        self.wv = nn.Linear(
            nlayers * self.embed_dim,
            self.num_key_value_heads * self.head_dim,
            bias=use_bias,
            device=config.init_device,
            )
        self.wo = nn.Linear(
            self.num_heads * self.head_dim,
            self.embed_dim,
            bias=use_bias,
            device=config.init_device,
            )
        self.attention_dropout: Optional[nn.Dropout] = None
        if v_cfg.attention_dropout > 0:
            self.attention_dropout = nn.Dropout(v_cfg.attention_dropout)
        self.residual_dropout = nn.Dropout(v_cfg.residual_dropout)

    def reset_parameters(self):
        nn.init.normal_(self.wq.weight, std=self.initializer_range)
        nn.init.normal_(self.wk.weight, std=self.initializer_range)
        nn.init.normal_(self.wv.weight, std=self.initializer_range)
        nn.init.normal_(self.wo.weight, std=self.initializer_range)
        if self.use_bias:
            nn.init.constant_(self.wq.bias, 0)
            nn.init.constant_(self.wk.bias, 0)
            nn.init.constant_(self.wv.bias, 0)
            nn.init.constant_(self.wo.bias, 0)

    def _split_heads(self, hidden_states, num_heads) -> torch.Tensor:
        return hidden_states.reshape(hidden_states.shape[:2] + (num_heads, self.head_dim))

    def _merge_heads(self, hidden_states) -> torch.Tensor:
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

    def forward(self, inputs_q: torch.Tensor, inputs_kv: Optional[torch.Tensor] = None) -> torch.Tensor:

        if inputs_kv is not None:
            inputs_k = inputs_kv
            inputs_v = inputs_kv
        else:
            inputs_k = inputs_q
            inputs_v = inputs_q
        
        xq, xk, xv = self.wq(inputs_q), self.wk(inputs_k), self.wv(inputs_v)

        xq = self._split_heads(xq, self.num_heads)
        xk = self._split_heads(xk, self.num_key_value_heads)
        xv = self._split_heads(xv, self.num_key_value_heads)

        if self.num_heads != self.num_key_value_heads:
            xk = xk.repeat_interleave(self.num_key_value_groups, dim=2, output_size=self.num_heads)
            xv = xv.repeat_interleave(self.num_key_value_groups, dim=2, output_size=self.num_heads)

        og_dtype = xq.dtype

        if self.config.float32_attention:
            xq = xq.to(torch.float)
            xk = xk.to(torch.float)

        if self.config.attention_type == AttentionType.direct:
            attn_weights = torch.einsum("...qhd,...khd->...hqk", xq / math.sqrt(xq.size(-1)), xk)
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(xq.dtype)
            if self.attention_dropout is not None:
                attn_weights = self.attention_dropout(attn_weights)
            attn_output = torch.einsum("...hqk,...khd->...qhd", attn_weights.to(xv.dtype), xv)

        elif self.config.attention_type == AttentionType.sdpa:
            attn_output = F.scaled_dot_product_attention(
                xq.transpose(1, 2).contiguous(),
                xk.transpose(1, 2).contiguous(),
                xv.transpose(1, 2).contiguous(),
                is_causal=False,
                dropout_p=self.config.vision_backbone.attention_dropout
            ).transpose(1, 2)
        else:
            raise NotImplementedError(self.config.attention_type)
        attn_output = attn_output.to(og_dtype)
        attn_output = self._merge_heads(attn_output)
        attn_output = self.wo(attn_output)
        attn_output = self.residual_dropout(attn_output)

        return attn_output


class ViTMLP(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        v_cfg = config.vision_backbone

        self.w1 = nn.Linear(
            v_cfg.image_emb_dim,
            v_cfg.image_mlp_dim,
            bias=True,
            device=config.init_device,
        )
        # Activation function.
        cfg = deepcopy(config)
        cfg.activation_type = v_cfg.image_mlp_activations
        self.act = get_activation(v_cfg.image_mlp_activations)
        self.w2 = nn.Linear(
            v_cfg.image_mlp_dim,
            v_cfg.image_emb_dim,
            bias=True,
            device=config.init_device,
        )

    def reset_parameters(self):
        v_cfg = self.config.vision_backbone
        nn.init.trunc_normal_(self.w1.weight, std=math.sqrt(1 / v_cfg.image_emb_dim), a=-2.0, b=2.0)
        nn.init.trunc_normal_(self.w2.weight, std=math.sqrt(1 / v_cfg.image_mlp_dim), a=-2.0, b=2.0)
        nn.init.zeros_(self.w1.bias)
        nn.init.zeros_(self.w2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.w1(x)
        x = self.act(x)
        x = self.w2(x)
        return x


class BlockCollection(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.grad_checkpointing: bool = False
        self._activation_checkpoint_fn: Callable = vit_activation_checkpoint_function(self.config)

        v_cfg = config.vision_backbone
        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(config) for _ in range(v_cfg.image_num_layers)
        ])

    def reset_parameters(self):
        for r in self.resblocks:
            r.reset_parameters()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        hidden_states = []
        count = 0
        for r in self.resblocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                # print("++++++++++++++++ ", count, x.shape)
                x = self._activation_checkpoint_fn(r, x)
            else:
                # print("+++------++++++ ", count, x.shapecs)
                x = r(x)
            hidden_states.append(x)
            count +=1 
        return hidden_states


class DinoBlockCollection(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.grad_checkpointing: bool = False
        self._activation_checkpoint_fn: Callable = vit_activation_checkpoint_function(self.config)

        v_cfg = config.vision_backbone
        self.resblocks = nn.ModuleList([
            DinoResidualAttentionBlock(config) for _ in range(v_cfg.image_num_layers)
        ])
    
    def reset_parameters(self):
        for r in self.resblocks:
            r.reset_parameters()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        hidden_states = []
        for r in self.resblocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = self._activation_checkpoint_fn(r, x)
            else:
                x = r(x)
            hidden_states.append(x)
        return hidden_states


class ResidualAttentionBlock(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        v_cfg = config.vision_backbone
        self.attention = ViTMultiHeadDotProductAttention(config)
        self.feed_forward = ViTMLP(config)
        self.attention_norm = nn.LayerNorm(
            v_cfg.image_emb_dim,
            eps=v_cfg.image_norm_eps,
            device=config.init_device,
        )
        self.ffn_norm = nn.LayerNorm(
            v_cfg.image_emb_dim,
            eps=v_cfg.image_norm_eps,
            device=config.init_device,
        )

    def reset_parameters(self):
        self.attention.reset_parameters()
        self.feed_forward.reset_parameters()
        self.attention_norm.reset_parameters()
        self.ffn_norm.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x))
        x = x + self.feed_forward(self.ffn_norm(x))
        return x

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     print(11, x.shape)
    #     x1 = self.attention_norm(x)
    #     print(12, x1.shape)
    #     x2 = self.attention(x1)
    #     print(13, x2.shape)
    #     x = x + x2
    #     print(14, x.shape)
    #     x3 = self.ffn_norm(x)
    #     print(15, x3.shape)
    #     x4 = self.feed_forward(x3)
    #     print(16, x4.shape)
    #     x = x + x4
    #     print(17, x.shape)
    #     return x


class DinoResidualAttentionBlock(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        v_cfg = config.vision_backbone
        self.attention = ViTMultiHeadDotProductAttention(config)
        self.feed_forward = ViTMLP(config)
        self.attention_norm = nn.LayerNorm(
            v_cfg.image_emb_dim,
            eps=v_cfg.image_norm_eps,
            device=config.init_device,
        )
        self.ffn_norm = nn.LayerNorm(
            v_cfg.image_emb_dim,
            eps=v_cfg.image_norm_eps,
            device=config.init_device,
        )
        self.lambda1 = nn.Parameter(
            torch.ones(v_cfg.image_emb_dim, device=config.init_device),
        )
        self.lambda2 = nn.Parameter(
            torch.ones(v_cfg.image_emb_dim, device=config.init_device),
        )

    def reset_parameters(self):
        self.attention.reset_parameters()
        self.feed_forward.reset_parameters()
        self.attention_norm.reset_parameters()
        self.ffn_norm.reset_parameters()
        nn.init.ones_(self.lambda1)
        nn.init.ones_(self.lambda2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x)) * self.lambda1
        x = x + self.feed_forward(self.ffn_norm(x)) * self.lambda2
        return x


class VisionTransformer(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        v_cfg = config.vision_backbone
        # class embeddings and positional embeddings
        self.scale = v_cfg.image_emb_dim ** -0.5
        self.class_embedding = nn.Parameter(
            torch.zeros(v_cfg.image_emb_dim, device=config.init_device),
        )
        self.num_prefix_tokens: int = 1
        self.positional_embedding = nn.Parameter(
            torch.zeros(v_cfg.image_num_pos, v_cfg.image_emb_dim, device=config.init_device),
        )

        image_patch_size = v_cfg.image_patch_size
        self.patch_embedding = nn.Linear(
            image_patch_size * image_patch_size * 3,
            v_cfg.image_emb_dim,
            bias=False,
            device=config.init_device,
            )

        self.pre_ln = nn.LayerNorm(
            v_cfg.image_emb_dim,
            eps=v_cfg.image_norm_eps,
            device=config.init_device,
        )

        self.transformer = BlockCollection(config)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    def reset_parameters(self):
        nn.init.normal_(self.class_embedding, std=self.scale)
        nn.init.normal_(self.positional_embedding, std=self.scale)
        nn.init.normal_(self.patch_embedding.weight, std=0.02)
        self.pre_ln.reset_parameters()
        self.transformer.reset_parameters()

    def add_pos_emb(self, x: torch.Tensor, patch_num: int) -> torch.Tensor:
        cls_emb = self.positional_embedding[0:1]
        pos_emb = self.positional_embedding[1:]

        pos_emb = pos_emb.reshape(
            (int(math.sqrt(pos_emb.shape[0])), int(math.sqrt(pos_emb.shape[0])), pos_emb.shape[1])
        )

        (patch_num_0, patch_num_1) = patch_num

        if pos_emb.shape[0] != patch_num_0 or pos_emb.shape[1] != patch_num_1:
            # Dervied from https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
            # antialias: default True in jax.image.resize
            pos_emb = pos_emb.unsqueeze(0).permute(0, 3, 1, 2)
            pos_emb = F.interpolate(
                pos_emb, size=(patch_num_0, patch_num_1), mode="bicubic", align_corners=False, antialias=True,
            )
            pos_emb = pos_emb.permute(0, 2, 3, 1).squeeze(0)

        pos_emb = pos_emb.reshape(-1, pos_emb.shape[-1])
        x = x + torch.cat([cls_emb[None, :, :], pos_emb[None, :, :]], dim=1).to(x.dtype)
        return x

    def forward(self, x: torch.Tensor, patch_num: int = None) -> List[torch.Tensor]:
        """
        : param x: (batch_size, num_patch, n_pixels)
        """
        if patch_num is None:
            patch_num = self.config.vision_backbone.image_num_patch
        B, N, D = x.shape

        x = self.patch_embedding(x)

        # class embeddings and positional embeddings
        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        x = self.add_pos_emb(x, patch_num)

        x = self.pre_ln(x)

        hidden_states = self.transformer(x)
        return hidden_states


class SiglipVisionTransformer(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        v_cfg = config.vision_backbone
        # positional embeddings
        self.scale = v_cfg.image_emb_dim ** -0.5
        self.num_prefix_tokens: int = 0 # no class embeddings
        self.positional_embedding = nn.Parameter(
            torch.zeros(v_cfg.image_num_pos, v_cfg.image_emb_dim, device=config.init_device),
        )

        image_patch_size = v_cfg.image_patch_size
        self.patch_embedding = nn.Linear(
            image_patch_size * image_patch_size * 3,
            v_cfg.image_emb_dim,
            bias=True,
            device=config.init_device,
        )

        self.transformer = BlockCollection(config)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable
    
    def reset_parameters(self):
        nn.init.normal_(self.positional_embedding, std=self.scale)
        nn.init.normal_(self.patch_embedding.weight, std=0.02)
        nn.init.zeros_(self.patch_embedding.bias)
        self.transformer.reset_parameters()
    
    def add_pos_emb(self, x: torch.Tensor, patch_num: int) -> torch.Tensor:
        pos_emb = self.positional_embedding

        pos_emb = pos_emb.reshape(
            (int(math.sqrt(pos_emb.shape[0])), int(math.sqrt(pos_emb.shape[0])), pos_emb.shape[1])
        )
    
        (patch_num_0, patch_num_1) = patch_num

        if pos_emb.shape[0] != patch_num_0 or pos_emb.shape[1] != patch_num_1:
            # Dervied from https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
            # antialias: default True in jax.image.resize
            pos_emb = pos_emb.unsqueeze(0).permute(0, 3, 1, 2)
            pos_emb = F.interpolate(
                pos_emb, size=(patch_num_0, patch_num_1), mode="bicubic", align_corners=False, antialias=True,
            )
            pos_emb = pos_emb.permute(0, 2, 3, 1).squeeze(0)

        pos_emb = pos_emb.reshape(-1, pos_emb.shape[-1])
        x = x + pos_emb[None, :, :].to(x.dtype)
        return x

    def forward(self, x: torch.Tensor, patch_num: int = None) -> List[torch.Tensor]:
        """
        : param x: (batch_size, num_patch, n_pixels)
        """
        if patch_num is None:
            patch_num = self.config.vision_backbone.image_num_patch
        B, N, D = x.shape
        # if get_rank() == 0:
        #     pdb.set_trace()
 
        x = self.patch_embedding(x)

        # class embeddings and positional embeddings
        x = self.add_pos_emb(x, patch_num)
        # print("---------------- ", x.shape)
        hidden_states = self.transformer(x)
        return hidden_states




# from .layers.rope_position_encoding import RopePositionEmbedding


# class DinoVisionTransformer(nn.Module):

#     def __init__(self, config: ModelConfig):
#         super().__init__()
#         self.config = config

#         v_cfg = config.vision_backbone
#         # class embeddings and positional embeddings
#         self.scale = v_cfg.image_emb_dim ** -0.5
#         self.class_embedding = nn.Parameter(
#             torch.zeros(v_cfg.image_emb_dim, device=config.init_device),
#         )
#         self.num_prefix_tokens: int = 1
#         self.positional_embedding = nn.Parameter(
#             torch.zeros(v_cfg.image_num_pos, v_cfg.image_emb_dim, device=config.init_device),
#         )
#         # RopePositionEmbedding
#         image_patch_size = v_cfg.image_patch_size
#         self.patch_embedding = nn.Linear(
#             image_patch_size * image_patch_size * 3,
#             v_cfg.image_emb_dim,
#             bias=True,
#             device=config.init_device,
#         )

#         # replace the Linear with Conv2d
#         # self.patch_embedding = nn.Conv2d(
#         #     in_channels=3,
#         #     out_channels=v_cfg.image_emb_dim,  # 1024
#         #     kernel_size=image_patch_size,      # 16
#         #     stride=image_patch_size,           # 16
#         #     bias=True,
#         #     device=config.init_device,
#         # )


#         self.transformer = DinoBlockCollection(config)

#     @torch.jit.ignore
#     def set_grad_checkpointing(self, enable=True):
#         self.transformer.grad_checkpointing = enable
    
#     def reset_parameters(self):
#         nn.init.normal_(self.class_embedding, std=self.scale)
#         nn.init.normal_(self.positional_embedding, std=self.scale)
#         nn.init.normal_(self.patch_embedding.weight, std=0.02)
#         self.transformer.reset_parameters()
    
#     def add_pos_emb(self, x: torch.Tensor, patch_num: int) -> torch.Tensor:
#         cls_emb = self.positional_embedding[0:1]
#         pos_emb = self.positional_embedding[1:]

#         pos_emb = pos_emb.reshape(
#             (int(math.sqrt(pos_emb.shape[0])), int(math.sqrt(pos_emb.shape[0])), pos_emb.shape[1])
#         )
    
#         (patch_num_0, patch_num_1) = patch_num

#         if pos_emb.shape[0] != patch_num_0 or pos_emb.shape[1] != patch_num_1:
#             # Dervied from https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
#             # antialias: default True in jax.image.resize
#             pos_emb = pos_emb.unsqueeze(0).permute(0, 3, 1, 2)
#             pos_emb = F.interpolate(
#                 pos_emb, size=(patch_num_0, patch_num_1), mode="bicubic", align_corners=False, antialias=True,
#             )
#             pos_emb = pos_emb.permute(0, 2, 3, 1).squeeze(0)

#         pos_emb = pos_emb.reshape(-1, pos_emb.shape[-1])
#         x = x + torch.cat([cls_emb[None, :, :], pos_emb[None, :, :]], dim=1).to(x.dtype)
#         return x

#     def forward(self, x: torch.Tensor, patch_num: int = None) -> List[torch.Tensor]:
#         """
#         : param x: (batch_size, num_patch, n_pixels)
#         """
#         if patch_num is None:
#             patch_num = self.config.vision_backbone.image_num_patch
#         B, N, D = x.shape

#         x = self.patch_embedding(x)

#         # class embeddings and positional embeddings
#         x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
#         x = self.add_pos_emb(x, patch_num)

#         hidden_states = self.transformer(x)
#         return hidden_states


class DinoVisionTransformer(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        v_cfg = config.vision_backbone
        # # class embeddings and positional embeddings
        # self.scale = v_cfg.image_emb_dim ** -0.5
        # self.class_embedding = nn.Parameter(
        #     torch.zeros(v_cfg.image_emb_dim, device=config.init_device),
        # )
        self.num_prefix_tokens1: int = 1
        # self.positional_embedding = nn.Parameter(
        #     torch.zeros(v_cfg.image_num_pos, v_cfg.image_emb_dim, device=config.init_device),
        # )
        # # RopePositionEmbedding
        # image_patch_size = v_cfg.image_patch_size
        # self.patch_embedding = nn.Linear(
        #     image_patch_size * image_patch_size * 3,
        #     v_cfg.image_emb_dim,
        #     bias=True,
        #     device=config.init_device,
        # )

        


        # self.transformer = DinoBlockCollection(config)

        
        name = "facebook/dinov3-vitl16-pretrain-lvd1689m"
        # processor = AutoImageProcessor.from_pretrained(name, trust_remote_code=True)
        self.transformer = AutoModel.from_pretrained(name, trust_remote_code=True)
        # self.transformer = vit_large()

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable
    
    # def reset_parameters(self):
    #     nn.init.normal_(self.class_embedding, std=self.scale)
    #     nn.init.normal_(self.positional_embedding, std=self.scale)
    #     nn.init.normal_(self.patch_embedding.weight, std=0.02)
    #     self.transformer.reset_parameters()
    
    # def add_pos_emb(self, x: torch.Tensor, patch_num: int) -> torch.Tensor:
    #     cls_emb = self.positional_embedding[0:1]
    #     pos_emb = self.positional_embedding[1:]

    #     pos_emb = pos_emb.reshape(
    #         (int(math.sqrt(pos_emb.shape[0])), int(math.sqrt(pos_emb.shape[0])), pos_emb.shape[1])
    #     )
    
    #     (patch_num_0, patch_num_1) = patch_num

    #     if pos_emb.shape[0] != patch_num_0 or pos_emb.shape[1] != patch_num_1:
    #         # Dervied from https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
    #         # antialias: default True in jax.image.resize
    #         pos_emb = pos_emb.unsqueeze(0).permute(0, 3, 1, 2)
    #         pos_emb = F.interpolate(
    #             pos_emb, size=(patch_num_0, patch_num_1), mode="bicubic", align_corners=False, antialias=True,
    #         )
    #         pos_emb = pos_emb.permute(0, 2, 3, 1).squeeze(0)

    #     pos_emb = pos_emb.reshape(-1, pos_emb.shape[-1])
    #     x = x + torch.cat([cls_emb[None, :, :], pos_emb[None, :, :]], dim=1).to(x.dtype)
    #     return x

    def forward(self, x: torch.Tensor, patch_num: int = None):
        """
        : param x: (batch_size, num_patch, n_pixels)
        """
        # if patch_num is None:
        #     patch_num = self.config.vision_backbone.image_num_patch
        # B, N, D = x.shape

        # x = self.patch_embedding(x)

        # # class embeddings and positional embeddings
        # x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        # x = self.add_pos_emb(x, patch_num)

        # imgs = rearrange(x, 'g (h w) (ph pw c) -> g c (h ph) (w pw)', h=32, w=32, ph=14, pw=14, c=3)

        imgs = rearrange(x, 'g (h w) (ph pw c) -> g c (h ph) (w pw)', h=24, w=24, ph=16, pw=16, c=3)
        resized = F.interpolate(imgs, size=(224, 224), mode="bilinear", align_corners=False, antialias=True).to(imgs.dtype)
        hidden_states = self.transformer(resized, output_hidden_states=True)
        
        return hidden_states




if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from PIL import Image
    import torch
    from transformers import AutoImageProcessor

    from olmo.dinov3 import *


    parser = argparse.ArgumentParser(description="Run DinoVisionTransformer inference")
    # parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth/.bin)")
    # parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # --- 1. Build config ---
    from olmo.config import ModelConfig, AttentionType, VisionBackboneConfig

    vision_backbone = vit_large()
    # --- 2. Init model ---

    # --- 3. Load checkpoint ---
    # ckpt = torch.load(Path(args.checkpoint), map_location=args.device)
    # state_dict = ckpt["model"] if "model" in ckpt else ckpt
    # breakpoint()
    # missing, unexpected = model.load_state_dict(state_dict, strict=False)
    
    # print("Missing keys:", missing)
    # print("Unexpected keys:", unexpected)
    # model.eval()
    # brea
    breakpoint()

    # --- 4. Preprocess image ---
    # processor = AutoImageProcessor.from_pretrained("facebook/dinov3-vitl16-pretrain")
    # image = Image.open(args.image).convert("RGB")
    # inputs = processor(images=image, return_tensors="pt")
    # pixel_values = inputs["pixel_values"].to(args.device)

    # # --- 5. Forward pass ---
    # with torch.no_grad():
    #     outputs = model(pixel_values, patch_num=(14, 14))

    # print("Number of hidden states:", len(outputs))
    # for i, out in enumerate(outputs):
    #     print(f"Layer {i} output shape: {out.shape}")
