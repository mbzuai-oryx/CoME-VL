import logging
import re
from pathlib import Path
from typing import Dict

import numpy as np

from olmo import DataConfig, DatasetEvaluatorConfig
from olmo.config import EvaluatorConfig, ModelConfig, VisionBackboneConfig, \
    TokenizerConfig, LayerNormType, AttentionType


DEBUG_MODEL = ModelConfig(
    d_model=128,
    n_heads=2,
    n_layers=1,
    max_sequence_length=4096,
    additional_vocab_size=128,
    vocab_size=152064,
    rope=True,
    embedding_size=None,
    weight_tying=False,
    vision_backbone=VisionBackboneConfig(
        image_num_layers=1,
    ),
    crop_mode="resize",
    tokenizer=TokenizerConfig(
        identifier="Qwen/Qwen3-4B",
        # identifier="/Med_Repo_Gen/molmo/Qwen-tokenizer"
    ),
)


def get_evaluator(name) -> EvaluatorConfig:
    """Gets the default evaluator for task `name`"""
    med_data = [
        # Report/Caption
        "medtrinity_report",
        "iuxray_report",
        "medpix_cliqa_report",
        "roco",
        "roco_v2",
        "chexpertplus_report",

        # VQA
        "vqa_med_2019",
        "pubmed_vision",
        "nih_classification",

        # BBox
        "nih_bbox",
        "deeplesion_bbox",
        "grazpedwri_dx_bbox",
        "cell_bacteria_bbox",
        "cell_ctc_bbox",
        "cell_deepcell_bbox",

        # # QNA/MCQ/Reasoning
        "pmc_instruct_qa",
        "medquad_qa",
        "medqa",
        "medical_meadow_medqa",
        "alphacare_qa",
        "chatdoctor_healthcaremagic",
        "chatdoctor_icliniq",
        "chatdoc_medqa_4option",
        "chatdoc_medqa_5option",
        "medical_meadow_flashcard",
        "medical_meadow_mediqa",
        "medical_meadow_pubmed_causal",
        "medical_meadow_wikidoc",
        "medical_meadow_wikidoc_patient_information",
        "medical_meadow_mmmlu",
        # "chatdoc_4option",
        # "chatdoc_5option",
        "mmmlu_anatomy",
        "mmmlu_clinical_knowledge",
        "mmmlu_college_biology",
        "mmmlu_college_medicine", 
        "mmmlu_medical_genetics",
        "mmmlu_professional_medicine",

        ## Summarization
        "medical_meadow_cord19",
        "mimic_ext_bhc",

        ## Reasoning
        "medical_o1_sft_mix",
        "medical_o1_verifiable_problem",
        "medical_r1_distill",
        "medreason",


    ]
    if name in med_data:
        return EvaluatorConfig(vqa_eval="vqa_score")
    if name in ["text_vqa", "okvqa", "coco_2014_vqa", "coco_2014_vqa_multi", "coco_bbox"]:
        return EvaluatorConfig(vqa_eval="vqa_score")
    elif name.startswith("math_vista"):
        return EvaluatorConfig(math_vista_eval=True)
    elif name == "a_okvqa_da":
        return EvaluatorConfig(vqa_eval="a_okvqa_score")
    elif name.startswith("android_control"):
        return EvaluatorConfig(android_eval=True)
    elif name == "vqa_v2_test":
        return EvaluatorConfig()
    elif name.startswith("chart_qa"):
        return EvaluatorConfig(vqa_eval="relaxed_correctness,em")
    elif name.startswith("refcoco"):
        return EvaluatorConfig(vqa_eval="bbox")
    elif name in [
        "doc_qa",
        "info_qa",
        "st_qa",
        "pixmo_docs_other",
        "pixmo_docs_tables",
        "pixmo_docs_diagrams",
        "pixmo_docs_charts",
        "scifi_document_qa",
        "scifi_table_qa",
        "scifi_diagram_qa",
        "scifi_charts_qa",
    ]:
        return EvaluatorConfig(vqa_eval="ansl,em")
    elif name in ["gqa", "tally_qa"]:
        return EvaluatorConfig(vqa_eval="em")
    elif name in ["science_qa", "a_okvqa_mc", "science_qa_img", "ai2_diagram", "ai2_diagram_v2", "ai2_diagram_v2_transparent"]:
        return EvaluatorConfig(vqa_eval="mc")
    elif name in ["ai2_diagram_v2_mix_transparent", "ai2_diagram_v2_mix_transparent_one_style"]:
        return EvaluatorConfig(vqa_eval="mc_ai2d_opaque,mc_ai2d_transparent")
    elif name.startswith("mmmu"):
        return EvaluatorConfig(vqa_eval="mmmu_score")
    elif name.startswith("countbench_qa") or name.startswith("pixmo_count"):
        return EvaluatorConfig(point_count_eval=True)
    elif name.startswith("real_world_qa"):
        return EvaluatorConfig(vqa_eval="real_world_qa_score")
    elif name == "clocks":
        return EvaluatorConfig(clock_eval=True)
    elif name == "pointing_eval":
        return EvaluatorConfig(pointing_eval=True)
    elif name == "clock_bench":
        return EvaluatorConfig(clock_bench_eval=True)
    elif name in ["countbench_qa"]:
        return EvaluatorConfig(count_eval=True)
    elif name in ["dense_caption_eval", "user_qa", "vqa_v2_test"]:
        # No metrics, but still save prediction file
        return EvaluatorConfig()
    else:
        raise NotImplementedError(name)


def get_evaluation(name, seq_len, batch_size, max_examples, num_workers=2) -> DatasetEvaluatorConfig:
    """Gets the default evaluation config for task (or task:split string) `name`"""
    if ":" in name:
        name, split = name.split(":")
    else:
        split = None

    if name == "chart_qa_weighted":
        name = "chart_qa"
    if name == "coco_2014_vqa_multi":
        name = "coco_2014_vqa"

    evaluator = get_evaluator(name)
    evaluator.num_wandb_examples = 64

    eval_only_tasks = [ "medtrinity_report", "iuxray_report", "roco", "chexpertplus_report",
        "vqa_med_2019", "pubmed_vision", "nih_classification", "nih_bbox", "deeplesion_bbox",
        "grazpedwri_dx_bbox", "cell_bacteria_bbox", "cell_ctc_bbox", "cell_deepcell_bbox", "medqa",
        "medical_meadow_medqa","medical_meadow_cord19","medreason"
    ]
    # eval_only_tasks += [task_name + "_test" for task_name in eval_only_tasks]
    eval_only_tasks += [task_name for task_name in eval_only_tasks]
    if name == "tall_qa_count":
        task_name = "tally_qa"
    elif name in eval_only_tasks:
        task_name = name # + "_test" if not name.endswith("_test") else name
        # task_name = name + if not name.endswith("_test") else name
    else:
        task_name = name
    evaluator.num_wandb_examples = 32
    evaluator.n_to_log = 0
    evaluator.save_predictions = None
    test_eval_tasks = [ "medtrinity_report", "iuxray_report", "roco", "chexpertplus_report",
        "vqa_med_2019", "pubmed_vision", "nih_classification", "nih_bbox", "deeplesion_bbox",
        "grazpedwri_dx_bbox", "cell_bacteria_bbox", "cell_ctc_bbox", "cell_deepcell_bbox", "medqa","medical_meadow_medqa","medical_meadow_cord19","medreason"
    ]
    if split is None:
        split = "test" if task_name in test_eval_tasks else "validation"
    if name == "pointing_eval":
        split = "test"

    short_len= ["nih_bbox","deeplesion_bbox","grazpedwri_dx_bbox", "medreason", "mmmlu_anatomy", "mmmlu_clinical_knowledge", 
    "mmmlu_college_biology", "mmmlu_college_medicine", "mmmlu_medical_genetics", "mmmlu_professional_medicine",
    "medical_meadow_pubmed_causal", "medical_meadow_medqa", "chatdoc_4option", "chatdoc_5option", "medical_meadow_mmmlu",
    "chatdoc_medqa_4option", "chatdoc_medqa_5option","medqa", "nih_classification"]

    
    if name in short_len:
        max_new_tokens = 64
    else:
        max_new_tokens = 256

    ds = DataConfig(
        dataset=task_name, sequence_length=seq_len,
        for_inference=True,
        split=split, shuffle=True, drop_last=True,
        num_workers=num_workers, pad="to_max", pin_memory=True
    )

    label = "ai2_diagram" if "ai2_diagram" in name else name
    if split is not None:
        label = f"{label}-{split}"

    return DatasetEvaluatorConfig(
        max_examples=max_examples,
        max_new_tokens=max_new_tokens,
        mm_evaluator=evaluator,
        label=label,
        data=ds
    )


DEFAULT_VISION_BACKBONE = VisionBackboneConfig(
    image_model_type="openai",
    # image_default_input_size=(336, 336),
    image_default_input_size=(224, 224),
    image_patch_size=14,
    image_pos_patch_size=14,
    image_emb_dim=1024,
    image_num_heads=16,
    image_num_key_value_heads=16,
    image_num_layers=23,
    image_head_dim=64,
    image_mlp_dim=4096,
    image_mlp_activations="quick_gelu",
    image_dropout_rate=0.0,
    image_num_pos=577,
    image_norm_eps=1e-5,
    attention_dropout=0.0,
    residual_dropout=0.0,
    initializer_range=0.02,
)


# SIGLIP_VISION_BACKBONE = VisionBackboneConfig(
#     image_model_type="siglip",
#     # image_default_input_size=(448, 448),
#     image_default_input_size=(384, 384),
#     image_patch_size=14,
#     image_pos_patch_size=14,
#     image_emb_dim=1152,
#     image_num_heads=16,
#     image_num_key_value_heads=16,
#     image_num_layers=27,
#     image_head_dim=72,
#     image_mlp_dim=4304,
#     image_mlp_activations="gelu_pytorch_tanh",
#     image_dropout_rate=0.0,
#     image_num_pos=729, # no CLS token
#     image_norm_eps=1e-6,
#     attention_dropout=0.0,
#     residual_dropout=0.0,
#     initializer_range=0.02,
#     resize_mode="siglip",
# )

SIGLIP_VISION_BACKBONE = VisionBackboneConfig(
    image_model_type="siglip",
    image_default_input_size=(384, 384),
    image_patch_size=16,             
    image_pos_patch_size=16,
    image_num_pos=576,               
    image_emb_dim=1152,   
    image_num_heads=16,
    image_num_key_value_heads=16,    
    image_num_layers=27,
    image_head_dim=72,           
    image_mlp_dim=4304,
    image_mlp_activations="gelu_pytorch_tanh",
    image_dropout_rate=0.0,
    attention_dropout=0.0,
    residual_dropout=0.0,
    image_norm_eps=1e-6,
    initializer_range=0.02,
    resize_mode="siglip",
)



DINOV2_LARGE_336_VISION_BACKBONE = VisionBackboneConfig(
    image_model_type="dino",
    image_default_input_size=(336, 336),
    # image_default_input_size=(378, 378),
    image_patch_size=14,
    image_pos_patch_size=14,
    image_emb_dim=1024,
    image_num_heads=16,
    image_num_key_value_heads=16,
    image_num_layers=24,
    image_head_dim=64,
    image_mlp_dim=4096,
    image_mlp_activations="gelu",
    image_dropout_rate=0.0,
    image_num_pos=577,
    image_norm_eps=1e-6,
    attention_dropout=0.0,
    residual_dropout=0.0,
    initializer_range=0.02,
    resize_mode="dino",
)

# DINOV3_LARGE_336_VISION_BACKBONE = VisionBackboneConfig(
#     image_model_type="dino",
#     image_default_input_size=(224, 224),
#     image_patch_size=16,
#     image_pos_patch_size=16,
#     image_emb_dim=1024,
#     image_num_heads=16,
#     image_num_key_value_heads=16,
#     image_num_layers=24,
#     image_head_dim=64,
#     image_mlp_dim=4096,
#     image_mlp_activations="gelu",
#     image_dropout_rate=0.0,
#     image_num_pos=577,
#     image_norm_eps=1e-6,
#     attention_dropout=0.0,
#     residual_dropout=0.0,
#     initializer_range=0.02,
#     resize_mode="dino",
# )

# DINOv3 ViT-L/16 @ 224px (register tokens = 4)
DINOV3_LARGE_224_VISION_BACKBONE = VisionBackboneConfig(
    # image_model_type="dinov3",
    image_model_type="dino",
    # image_default_input_size=(448, 448),
    image_default_input_size=(224, 224),
    image_patch_size=16,
    image_pos_patch_size=16,

    # core dims
    image_emb_dim=1024,        # hidden_size
    image_num_heads=16,        # num_attention_heads
    image_num_key_value_heads=16,
    image_head_dim=64,         # 1024 / 16
    image_num_layers=24,       # num_hidden_layers
    image_mlp_dim=4096,        # intermediate_size
    image_mlp_activations="gelu",

    # tokens / positions
    image_num_pos=785,         # 28*28 patches + 1 CLS  (for 448/16)  ← register tokens are separate
    # image_num_register_tokens=4,

    # norms, drops, init
    image_norm_eps=1e-5,
    attention_dropout=0.0,
    residual_dropout=0.0,
    initializer_range=0.02,

    # extras (keep if your backbone wrapper supports them)
    # layerscale_value=1.0,
    # drop_path_rate=0.0,
    resize_mode="dino",        # same as your v2 config; OK to keep

    # DINOv3-specific (optional; safe to include/ignore if unused)
    # pos_embed_rescale=2.0,
    # rope_theta=100.0,          # RoPE base period used by DINOv3 ViT
)




METACLIP_L14_336_VISION_BACKBONE = VisionBackboneConfig(
    image_model_type="openai",
    image_default_input_size=(336, 336),
    image_patch_size=14,
    image_pos_patch_size=14,
    image_emb_dim=1024,
    image_num_heads=16,
    image_num_key_value_heads=16,
    image_num_layers=24,
    image_head_dim=64,
    image_mlp_dim=4096,
    image_mlp_activations="quick_gelu",
    image_dropout_rate=0.0,
    image_num_pos=577,
    image_norm_eps=1e-5,
    attention_dropout=0.0,
    residual_dropout=0.0,
    initializer_range=0.02,
    resize_mode="metaclip",
)


OLMOE = ModelConfig(
    d_model=2048,
    n_heads=16,
    n_layers=16,
    mlp_ratio=1,
    activation_type='swiglu',
    block_type='moe',
    rope=True,
    rope_full_precision=True,
    rope_theta=10000.0,
    low_cpu_fsdp=True,
    attention_type='sdpa',
    attention_layer_norm=True,
    residual_dropout=0.1,
    response_residual_dropout=0.0,
    embedding_dropout=0.0,
    layer_norm_type='rms',
    layer_norm_with_affine=True,
    layer_norm_eps=1e-05,
    attention_layer_norm_with_affine=True,
    max_sequence_length=4096,
    max_position_embeddings=32768,
    include_bias=False,
    bias_for_layer_norm=False,
    scale_logits=False,
    vocab_size=50280,
    embedding_size=50304,
    additional_vocab_size=128,
    new_embedding_init_range=0.02,
    weight_tying=False,
    init_device='meta',
    precision='amp_bf16',
    image_projector='mlp',
    image_projector2='mlp',
    normalize_input_embeds=False,
    use_position_ids=True,

    # MOE parameters
    moe_num_experts=64,
    moe_top_k=8,
    moe_mlp_impl='sparse',
    moe_log_expert_assignment=False,
    moe_shared_expert=False,
    moe_lbl_in_fp32=False,
    moe_interleave=False,
    moe_loss_weight=0.0,
    moe_zloss_weight=0.0,
    moe_dropless=True,
    moe_capacity_factor=1.25,

    tokenizer=TokenizerConfig(
        identifier='allenai/OLMoE-1B-7B-0924',
    ),
    image_pooling_2d="attention_meanq",
    image_padding_embed="pad_and_partial_pad",
)


OLMO_1024_PREVIEW = ModelConfig(
    d_model=4096,
    n_heads=32,
    n_kv_heads=None,
    clip_qkv=None,
    n_layers=32,
    mlp_ratio=4,
    mlp_hidden_size=22016,
    activation_type="swiglu",
    block_type="sequential",
    # block_type="llama",
    block_group_size=1,
    rope=True,
    rope_full_precision=True,
    rope_theta=500000,
    attention_dropout=0.0,
    attention_layer_norm=True,
    layer_norm_type="rms",
    layer_norm_with_affine=True,
    layer_norm_eps=1.0e-06,
    attention_layer_norm_with_affine=True,
    max_sequence_length=4096,
    include_bias=False,
    bias_for_layer_norm=False,
    scale_logits=False,
    vocab_size=100278,
    embedding_size=100352,
    additional_vocab_size=128,
    weight_tying=False,
    attention_type=AttentionType.sdpa,
    init_device="meta",
    init_fn="normal",
    init_std=0.02,
    init_cutoff_factor=3.0,
    precision="amp_bf16",
    norm_after=True,
    tokenizer=TokenizerConfig(
        identifier="allenai/dolma2-tokenizer",
    ),
    embedding_dropout=0,
    image_pooling_2d="attention_meanq",
    image_padding_embed="pad_and_partial_pad",
)


QWEN2_1B = ModelConfig(
    vocab_size=152064,
    max_sequence_length=4096,
    residual_dropout=0,
    embedding_dropout=0,
    response_residual_dropout=0,
    attention_dropout=0,
    rope=True,
    qkv_bias=True,
    weight_tying=False,
    include_bias=False,
    embedding_size=152064,
    d_model=1536,
    mlp_hidden_size=18944*2,
    n_layers=28,
    additional_vocab_size=128,
    n_heads=12,
    n_kv_heads=4,
    rope_theta=1000000.0,
    layer_norm_eps=1e-6,
    layer_norm_type=LayerNormType.rms,
    tokenizer=TokenizerConfig(
        # identifier="Qwen/Qwen2-7B",
        identifier="/Med_Repo_Gen/molmo/Qwen-tokenizer"
    ),
    image_pooling_2d="attention_meanq",
    image_padding_embed="pad_and_partial_pad",
)

QWEN2_7B = ModelConfig(
    vocab_size=152064,
    max_sequence_length=4096,
    residual_dropout=0,
    embedding_dropout=0,
    response_residual_dropout=0,
    attention_dropout=0,
    rope=True,
    qkv_bias=True,
    weight_tying=False,
    include_bias=False,
    embedding_size=152064,
    d_model=3584,
    mlp_hidden_size=18944*2,
    n_layers=28,
    additional_vocab_size=128,
    n_heads=28,
    n_kv_heads=4,
    rope_theta=1000000.0,
    layer_norm_eps=1e-6,
    layer_norm_type=LayerNormType.rms,
    tokenizer=TokenizerConfig(
        identifier="Qwen/Qwen2-7B",
        # identifier="/Med_Repo_Gen/molmo/Qwen-tokenizer"
    ),
    image_pooling_2d="attention_meanq",
    image_padding_embed="pad_and_partial_pad",
)


QWEN2_72B = ModelConfig(
    init_device="meta",
    low_cpu_fsdp=True,
    additional_vocab_size=128,
    vocab_size=152064,
    max_sequence_length=4096,
    residual_dropout=0,
    embedding_dropout=0,
    response_residual_dropout=0,
    attention_dropout=0,
    rope=True,
    qkv_bias=True,
    weight_tying=False,
    include_bias=False,
    embedding_size=152064,
    d_model=8192,
    mlp_hidden_size=29568*2,
    n_layers=80,
    n_heads=64,
    n_kv_heads=8,
    rope_theta=1000000.0,
    layer_norm_eps=1e-5,
    layer_norm_type=LayerNormType.rms,
    tokenizer=TokenizerConfig(
        identifier="Qwen/Qwen2-72B",
    ),
    image_pooling_2d="attention_meanq",
    image_padding_embed="pad_and_partial_pad",
)

QWEN25_3B = ModelConfig(
    vocab_size=151936,
    max_sequence_length=4096,
    residual_dropout=0,
    embedding_dropout=0,
    response_residual_dropout=0,
    attention_dropout=0,
    rope=True,
    qkv_bias=True,
    weight_tying=False,
    include_bias=False,
    embedding_size=151936,
    d_model=2048,
    mlp_hidden_size=18944*2,
    n_layers=36,
    additional_vocab_size=128,
    n_heads=16,
    n_kv_heads=4,
    rope_theta=1000000.0,
    layer_norm_eps=1e-6,
    layer_norm_type=LayerNormType.rms,
    tokenizer=TokenizerConfig(
        identifier="Qwen/Qwen2.5-3B",
        # identifier="/Med_Repo_Gen/molmo/Qwen-tokenizer"
    ),
    image_pooling_2d="attention_meanq",
    image_padding_embed="pad_and_partial_pad",
)


QWEN3_1B = ModelConfig(
    vocab_size=151936,
    max_sequence_length=4096,
    residual_dropout=0,
    embedding_dropout=0,
    response_residual_dropout=0,
    attention_dropout=0,
    rope=True,
    qkv_bias=False,
    weight_tying=False,
    include_bias=False,
    embedding_size=151936,
    d_model=2048,
    mlp_hidden_size=12288,  # 18944*2,
    n_layers=28,
    additional_vocab_size=128,
    n_heads=16,
    n_kv_heads=8,
    rope_theta=1000000.0,
    layer_norm_eps=1e-6,
    layer_norm_type=LayerNormType.rms,
    tokenizer=TokenizerConfig(
        identifier="Qwen/Qwen3-1.7B",
        # identifier="/Med_Repo_Gen/molmo/Qwen-tokenizer"
    ),
    image_pooling_2d="attention_meanq",
    image_padding_embed="pad_and_partial_pad",
)


QWEN3_4B = ModelConfig(
    vocab_size=151936,
    max_sequence_length=4096,
    residual_dropout=0,
    embedding_dropout=0,
    response_residual_dropout=0,
    attention_dropout=0,
    rope=True,
    qkv_bias=False,
    weight_tying=False,
    include_bias=False,
    # qkv_bias=True,
    # weight_tying=False,
    # include_bias=False,
    embedding_size=151936,
    d_model=2560,
    head_dim=128, 
    mlp_hidden_size=9728*2, # 18944*2,
    n_layers=36,
    additional_vocab_size=128,
    n_heads=32,
    n_kv_heads=8,
    rope_theta=1000000.0,
    layer_norm_eps=1e-6,
    layer_norm_type=LayerNormType.rms,
    tokenizer=TokenizerConfig(
        identifier="Qwen/Qwen3-4B",
        # identifier="/Med_Repo_Gen/molmo/Qwen-tokenizer"
    ),
    image_pooling_2d="attention_meanq",
    image_padding_embed="pad_and_partial_pad",
)



GPTOSS_20B = ModelConfig(
    block_type="sequential",
    vocab_size=201088,                     # updated from 151936
    max_sequence_length=4096,            # from max_position_embeddings
    residual_dropout=0,                    # not in JSON (keep 0)
    embedding_dropout=0,                   # not in JSON (keep 0)
    response_residual_dropout=0,           # not in JSON (keep 0)
    attention_dropout=0.0,                 # matches JSON
    rope=True,                             # still using rope embeddings
    qkv_bias=True,                         # attention_bias true -> enable bias
    weight_tying=False,                    # tie_word_embeddings = false
    include_bias=False,                    # still false
    embedding_size=201088,                 # match vocab size
    d_model=2880,                          # hidden_size in JSON
    head_dim=64,                           # head_dim in JSON
    mlp_hidden_size=2880,                  # intermediate_size in JSON (no *2 here)
    n_layers=24,                           # num_hidden_layers in JSON
    additional_vocab_size=128,             # not in JSON; keep original
    n_heads=64,                            # num_attention_heads in JSON
    n_kv_heads=8,                          # num_key_value_heads in JSON
    rope_theta=150000.0,                   # updated rope_theta
    layer_norm_eps=1e-5,                   # rms_norm_eps in JSON
    layer_norm_type=LayerNormType.rms,     # same type
    tokenizer=TokenizerConfig(
        identifier="openai/gpt-oss-20b"    # keep same (unless new tokenizer?)
    ),
    image_pooling_2d="attention_meanq",    # not in JSON, keep as before
    image_padding_embed="pad_and_partial_pad",  # not in JSON, keep as before

    # Additional fields specific to new architecture
    # sliding_window=128,                    # from JSON
    # experts_per_token=4,                   # from JSON
    # num_local_experts=32,                  # from JSON
    # router_aux_loss_coef=0.9,              # from JSON
    # rope_scaling={                         # from JSON (Yarn rope)
    #     "beta_fast": 32.0,
    #     "beta_slow": 1.0,
    #     "factor": 32.0,
    #     "original_max_position_embeddings": 4096,
    #     "rope_type": "yarn",
    #     "truncate": False
    # },
    # swiglu_limit=7.0                        # from JSON
)


DEFAULT_LOAD_PATHS = {
    "openai": "${oc.env:MOLMO_DATA_DIR}/pretrained_image_encoders/vit-l-14-336.pt",
    "siglip": "${oc.env:MOLMO_DATA_DIR}/pretrained_image_encoders/siglip-so400m-14-384.pt",
    "dinov2_large_336": "${oc.env:MOLMO_DATA_DIR}/pretrained_image_encoders/dinov2-large-336.pt",
    "dinov3_large_224": "${oc.env:MOLMO_DATA_DIR}/pretrained_image_encoders/dinov3-large-224.pt",
    "metaclip_l14_336": "${oc.env:MOLMO_DATA_DIR}/pretrained_image_encoders/metaclip-l14-336.pt",
    "olmoe": "${oc.env:MOLMO_DATA_DIR}/pretrained_llms/olmoe.pt",
    "olmo_1024_preview": "${oc.env:MOLMO_DATA_DIR}/pretrained_llms/olmo-1024-preview.pt",
    "qwen2_7b": "${oc.env:MOLMO_DATA_DIR}/pretrained_llms/qwen2-7b.pt",
    "qwen2_72b": "${oc.env:MOLMO_DATA_DIR}/pretrained_llms/qwen2-70b.pt",
    "qwen2.5_3b": "${oc.env:MOLMO_DATA_DIR}/pretrained_llms/qwen2.5-3b.pt",
    "qwen2.5_7b": "${oc.env:MOLMO_DATA_DIR}/pretrained_llms/qwen2.5-7b.pt",
    "qwen2.5_72b": "${oc.env:MOLMO_DATA_DIR}/pretrained_llms/qwen2.5-70b.pt",

    "qwen3_1b": "${oc.env:MOLMO_DATA_DIR}/pretrained_llms/qwen3-1b.pt",
    "qwen3_4b": "${oc.env:MOLMO_DATA_DIR}/pretrained_llms/qwen3-4b.pt",

    "gptoss_20b": "${oc.env:MOLMO_DATA_DIR}/pretrained_llms/gptoss-20b.pt",
}


VISION_BACKBONES: Dict[str, VisionBackboneConfig] = {
    "openai": DEFAULT_VISION_BACKBONE,
    "siglip": SIGLIP_VISION_BACKBONE,
    "dinov2_large_336": DINOV2_LARGE_336_VISION_BACKBONE,
    "dinov3_large_224": DINOV3_LARGE_224_VISION_BACKBONE,
    "metaclip_l14_336": METACLIP_L14_336_VISION_BACKBONE,
}


LLMS: Dict[str, ModelConfig] = {
    "olmoe": OLMOE,
    "olmo_1024_preview": OLMO_1024_PREVIEW,
    "qwen2_7b": QWEN2_7B,
    "qwen2_72b": QWEN2_72B,
    "qwen2.5_3b": QWEN25_3B,
    # "qwen2.5_7b": QWEN25_7B,
    # "qwen2.5_72b": QWEN25_72B,

    "qwen3_1b": QWEN3_1B,
    "qwen3_4b": QWEN3_4B,

    "gptoss_20b": GPTOSS_20B,
}


def select_checkpoint(checkpoint):
    checkpoint_dir = Path(checkpoint)
    if not (checkpoint_dir / "model.pt").exists():
        candidates = []
        for file in checkpoint_dir.iterdir():
            match = re.match("^step([0-9]+)-unsharded.*", file.name)
            if match:
                candidates.append((file, int(match.group(1))))
        if len(candidates) == 0:
            raise FileNotFoundError(f"{checkpoint_dir} is a directory but it did not "
                                    f"contain any unsharded checkpoints")
        checkpoint_dir = max(candidates, key=lambda x: x[1])[0].absolute().as_posix()
        logging.info(f"Selected {checkpoint_dir} as oldest checkpoint in {checkpoint_dir}")
        return checkpoint_dir
    else:
        return checkpoint
