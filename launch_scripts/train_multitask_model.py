import argparse
import logging
from os.path import join, exists
from typing import cast, List

import omegaconf
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf

from launch_scripts.utils import get_evaluation, DEBUG_MODEL
from olmo import TrainConfig
from olmo.config import DataConfig, \
    ModelConfig, WandbConfig, OptimizerConfig, OptimizerType, SchedulerConfig, SchedulerType, \
    BatchDivisor, SpeedMonitorConfig, ActivationCheckpointingStrategy, FSDPConfig, FSDPWrapStrategy, \
    FSDPPrecision, RootSizeMixture
from olmo.torch_util import get_world_size
from olmo.util import (
    add_cached_path_clients,
    clean_opt,
    prepare_cli_environment,
)
from scripts.train import main as train

log = logging.getLogger("train")

AUX = [
#     # Supervised datasets we want eval on
#     # "coco_2014_vqa_multi",
#     # "text_vqa",
#     # "okvqa",
#     "chart_qa_weighted",
#     # "countbench_qa"
#     "doc_qa",
#     "info_qa",
#     # "ai2_diagram_v2_mix_transparent",
#     # "a_okvqa_mc",
#     # "a_okvqa_da",
#     # "android_control",

#     # # Some other datasets we might want to eval on
#     # "science_qa_img",
#     # "tabwmp_da",
#     # "st_qa",
#     # "tally_qa",

#     # # ("clocks", 250000),  # Downsample since it is huge
    "refcoco",
    "pixmo_docs_charts",
    "pixmo_docs_tables",
    "pixmo_docs_other",
    "pixmo_docs_diagrams",
#     # "coco_bbox",

#     # # # Other synthetic data, also downsampled since they are huge
#     # ("dv_qa", 10000),
#     # ("figure_qa", 10000),
#     # ("plot_qa", 20000),
#     # "math_vista_v2",
#     "countbench_qa",
#     # "mmmu",
#     # "okvqa",
]

# AUX = [
#     "refcoco"
# ]



def get_training_mixture(submixture):
    resolved_weights = {}
    for task_name in submixture:
        mix = {}
        if isinstance(task_name, tuple):
            task_name, size = task_name
        else:
            size = None
        resolved_weights[task_name] = size
    return resolved_weights


def expand_refcoco_eval_tasks(tasks):
    expanded = []
    for task in tasks:
        if task == "refcoco":
            expanded.extend([
                "refcoco:validation",
                "refcoco:testA",
                "refcoco:testB",
            ])
        else:
            expanded.append(task)
    return expanded


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError as e:
        print(f"failed to set multiprocessing start method: {e}")
    log.info(f"Multiprocessing start method set to '{mp.get_start_method()}'")

    # Initialize process group.
    dist.init_process_group(backend="nccl")
    log.info("Process group initialized")

    prepare_cli_environment()
    log.info("CLI environment prepared")

    add_cached_path_clients()

    parser = argparse.ArgumentParser(prog="Train a multitask model")
    parser.add_argument("mixture", help="Name of datset mixture to train on")
    parser.add_argument("checkpoint", help="Path to checkpoint to start from")
    parser.add_argument("--seq_len", default=2304, type=int)
    parser.add_argument("--inf_seq_len", default=1792, type=int)
    parser.add_argument("--max_inf_examples", default=2048, type=int)
    parser.add_argument("--global_batch_size", default=24, type=int)
    parser.add_argument("--device_eval_batch_size", default=3, type=int)
    parser.add_argument("--device_inf_batch_size", default=3, type=int)
    parser.add_argument("--device_train_batch_size", default=3, type=int)
    # parser.add_argument("--global_batch_size", default=128, type=int)
    # parser.add_argument("--device_eval_batch_size", default=2, type=int)
    # parser.add_argument("--device_inf_batch_size", default=2, type=int)
    # parser.add_argument("--device_train_batch_size", default=2, type=int)
    args, other_args = parser.parse_known_args()

    if args.mixture.startswith("single"):
        task_name = args.mixture.split("_", 1)[1]
        eval_tasks = [task_name,]
        tasks = [["eval", eval_tasks, 1.0]]
    elif args.mixture == "android":
        eval_tasks = ["android_control_ll"]
        tasks = [["eval", ["android_control"], 1.0]]
    elif args.mixture in ["small1", "debug"]:
        eval_tasks = ["chart_qa", "doc_qa"]
        tasks = [["aux", ["chart_qa", "doc_qa"], 1.0]]
    elif args.mixture == "small2":
        eval_tasks = ["chart_qa", "doc_qa", "info_qa"]
        tasks = [["aux", [("chart_qa", 4*4),
                          ("doc_qa", 2*2), ("info_qa", 1)], 1.0]]
    elif args.mixture in ["3.2-synthetic"]:
        aux = list(AUX)
        eval_tasks = [
            
            # "coco_bbox",
            # "chart_qa",
            # "mmmu_test",
            # "countbench_qa"
            # "info_qa",
            # "doc_qa",
            # "ai2_diagram_v2_mix_transparent",
            # "coco_2014_vqa_multi",
            # # "clocks",
            # "android_control_ll",
            # "pointing_eval:test",
            # "countbench_qa:huggingface"
            
            # "a_okvqa_mc",
            # "a_okvqa_da",
            # "refcoco"
            "pixmo_docs_charts",
        ]
        tasks = [
            # ["demo", [
            #     "pixmo_ask_model_anything",
            #     # ("pixmo_cap", 50000),
            #     "pixmo_cap",
            #     "pixmo_cap_qa",
            #     "pixmo_pointing_explanations"
            # ], 0.25],
            ["aux", aux, 0.6],
            ["pointing", [
                "pixmo_count_counting",
                "pixmo_points",
                "pixmo_count",
                # "pixmo_points_high_freq",
                "pixmo_points_counting",
                # "pixmo_points_high_freq_counting",
                
            ], 0.4]
        ]
    else:
        raise NotImplementedError(args.mixture)

    debug = args.checkpoint in ["debug", "debug2"]
    if debug:
        model_cfg = DEBUG_MODEL
        if args.checkpoint == "debug2":
            model_cfg.max_crops = 12
            model_cfg.crop_mode = "overlap-and-resize-c2"
            model_cfg.tokenizer.identifier = "mm:hf-Qwen/Qwen2-7B"
            model_cfg.embedding_size = 152064
            model_cfg.vocab_size = 152064
            model_cfg.pad_tokenizer = True
        global_batch_size = 8
        model_init = None
        inf_eval_interval = 20
        eval_interval = 20
        log_interval = 5
        eval_examples = 16
        max_inf_examples = 16
        duration = 1000
        eval_subset_batches = 4
    else:
        eval_examples = 32
        max_inf_examples = args.max_inf_examples
        log_interval = 20
        global_batch_size = args.global_batch_size
        inf_eval_interval = 120000
        eval_interval = 120000
        duration = 30000

        # log_interval = 2
        # global_batch_size = args.global_batch_size
        # inf_eval_interval = 16
        # eval_interval = 4
        # duration = 2


        model_init = args.checkpoint
        if exists(join(args.checkpoint, "model.yaml")):
            model_cfg = ModelConfig.load(join(args.checkpoint, "model.yaml"))
        else:
            model_cfg = ModelConfig.load(join(args.checkpoint, "config.yaml"), key="model")

        eval_subset_batches = eval_examples//(args.device_eval_batch_size*get_world_size())
        logging.info(f"Setting eval subset batches to {eval_subset_batches}")
        assert eval_subset_batches > 0

    # Fine-tuning settings
    model_cfg.residual_dropout = 0.1
    model_cfg.response_residual_dropout = 0.0
    model_cfg.prompt_type = "uber_model"
    model_cfg.message_formatting = "role"
    model_cfg.system_prompt_kind = "demo_or_style"
    model_cfg.multi_annotation_weighting = "root_subsegments"

    root_size_mixture: List[RootSizeMixture] = []
    for name, submixture, rate in tasks:
        submixture = get_training_mixture(submixture)
        root_size_mixture.append(RootSizeMixture(rate, submixture))

    evaluations = []
    eval_tasks = expand_refcoco_eval_tasks(eval_tasks)
    for task in eval_tasks:
        evaluation = get_evaluation(
            task,
            args.inf_seq_len,
            batch_size=get_world_size()*args.device_inf_batch_size,
            max_examples=max_inf_examples,
            num_workers=2
        )
        evaluation.data.persistent_workers = True
        evaluations.append(evaluation)

    cfg = TrainConfig(
        run_name="multitask_train",
        no_pre_train_checkpoint=True,
        save_folder="debug_run" if debug else omegaconf.MISSING,
        seed=6198,
        dry_run=False,
        wandb=None if debug else WandbConfig(
            name="${run_name}",
            project="${oc.env:WANDB_PROJECT}",
            group=None,
            entity="${oc.env:WANDB_ENTITY}",
            log_interval=log_interval
        ),
        allow_resume=True,
        # allow_resume=False,
        model=model_cfg,
        save_overwrite=debug,
        save_dataloader_state=False,
        data=DataConfig(
            root_size_mixture=root_size_mixture,
            for_inference=False,
            shuffle=True,
            split="train",
            drop_last=True,
            sequence_length=args.seq_len,
            num_workers=2,
            pad="to_max",
            shuffle_messages=True,
            pin_memory=True,
            seed=50189
        ),
        ft_connector=True,
        ft_llm=True,
        ft_vit=True,
        # ft_llm=False,
        # ft_vit=False,
        ft_vit2=False,

        optimizer=OptimizerConfig(
            name=OptimizerType.adamw,
            connector_learning_rate=1e-5,
            vit_learning_rate=1e-5,
            llm_learning_rate=1e-5,
            connector_weight_decay=0.0,
            vit_weight_decay=0.0,
            llm_weight_decay=0.0,
            connector_betas=[0.9, 0.95],
            vit_betas=[0.9, 0.95],
            llm_betas=[0.9, 0.95],
            connector_eps=1e-6,
            vit_eps=1e-6,
            llm_eps=1e-6,
            metrics_log_interval=20
        ),
        
        scheduler=SchedulerConfig(
            name=SchedulerType.multimodal,
            connector_t_warmup=200,
            vit_t_warmup=200,
            llm_t_warmup=200,
            alpha_f=0.1,
            warmup_min_lr=0.0
        ),
        fsdp=FSDPConfig(
            use_orig_params=True,
            wrapping_strategy=FSDPWrapStrategy.by_block_and_size,
            precision=FSDPPrecision.float
        ),
        load_path=None,
        initial_model_checkpoint=None if "debug" in args.checkpoint else args.checkpoint,
        save_interval=300000,
        # save_interval=20,
        save_num_checkpoints_to_keep=0,
        save_num_unsharded_checkpoints_to_keep=1,
        save_interval_unsharded=2000,
        global_train_batch_size=global_batch_size,
        device_inf_eval_batch_size=args.device_inf_batch_size,
        device_eval_batch_size=args.device_eval_batch_size,
        device_train_microbatch_size=args.device_train_batch_size,
        time_limit=None,
        max_duration=duration,
        stop_at="${max_duration}",
        max_grad_norm=1,
        batch_divisor=BatchDivisor.global_batch,
        precision="amp_bf16",
        console_log_interval=log_interval,
        speed_monitor=SpeedMonitorConfig(window_size=20),
        softmax_auxiliary_loss=True,
        softmax_auxiliary_loss_scale=1e-4,
        activation_checkpointing=ActivationCheckpointingStrategy.whole_layer,
        eval_interval=eval_interval,
        inf_eval_interval=inf_eval_interval,
        inf_evaluators=evaluations,
        eval_subset_num_batches=eval_subset_batches,
        evaluators=[]
    )

    conf = OmegaConf.create(cfg)
    if other_args:
        overrides = [clean_opt(arg) for arg in other_args]
        conf = OmegaConf.merge(conf, OmegaConf.from_dotlist(overrides))
    cfg = cast(TrainConfig, OmegaConf.to_object(conf))
    train(cfg)
