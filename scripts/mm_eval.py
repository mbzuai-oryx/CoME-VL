"""Run this script with 'torchrun'."""
import dataclasses
import json
import logging
import re
import sys
from datetime import datetime
from os import makedirs
from os.path import dirname, join, exists
from typing import Optional

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb
from packaging import version
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
from torch.distributed.fsdp import ShardingStrategy
from transformers import AutoModelForCausalLM

from olmo.checkpoint import load_model_state
from olmo.config import EvalConfig, TokenizerConfig, ModelConfig, DatasetEvaluatorConfig, \
    VisionBackboneConfig
from olmo.data import build_torch_mm_eval_dataloader
from olmo.eval.inf_evaluator import InfDatasetEvaluator, build_inf_evaluator
from olmo.exceptions import OLMoCliError
from olmo.model import Molmo
from olmo.torch_util import (
    barrier,
    get_default_device,
    get_global_rank,
    get_local_rank,
    peak_gpu_memory,
    seed_all, get_world_size,
)
from olmo.util import (
    add_cached_path_clients,
    clean_opt,
    prepare_cli_environment, resource_path, log_metrics_to_console,
)

log = logging.getLogger(__name__)


def get_float_dtype_by_name(dtype):
    return {
        'bf16': torch.bfloat16,
        'bfloat16': torch.bfloat16,
        'fp16': torch.float16,
        'float16': torch.float16,
        'fp32': torch.float32,
        'float32': torch.float32,
        'fp64': torch.float64,
        'float64': torch.float64,
    }[dtype]


def cast_float_dtype(t: torch.Tensor, dtype: str):
    if t.dtype in (torch.bfloat16, torch.float16, torch.float32, torch.float64):
        t = t.to(get_float_dtype_by_name(dtype))
    return t


def get_gcs_url(output_file):
    assert output_file.startswith("gs://")
    return f"https://storage.cloud.google.com/{output_file[5:]}?authuser=1"


@dataclasses.dataclass
class ModelEvaluator:
    """Evaluates a model on possibly multiple tasks"""
    config: EvalConfig

    def get_save_dir(self, cfg: DatasetEvaluatorConfig) -> Optional[str]:
        """Get directory to save the eval results"""
        if not cfg.save_dir and not cfg.save_to_checkpoint_dir:
            return None

        if cfg.save_to_checkpoint_dir:
            base = dirname(self.config.load_path)
        else:
            base = cfg.save_dir

        # If the load path has a step indicator, use it in the save dir name
        step_match = re.match(".*/step([0-9]+).*",  self.config.load_path)
        if step_match is not None:
            step = int(step_match.group(1))
        else:
            step = None

        mixture_or_task_name = cfg.data.dataset
        split = cfg.data.split
        if step is not None:
            name = f"predictions-ck{step}-{mixture_or_task_name}-{split}"
        else:
            name = f"predictions-{mixture_or_task_name}-{split}"
        if cfg.eval_name:
            name += "-" + cfg.eval_name
        default_prediction_dir = join(base, name)
        return default_prediction_dir

    def get_metric_file(self, cfg: DatasetEvaluatorConfig):
        save_dir = self.get_save_dir(cfg)
        if save_dir:
            return join(self.get_save_dir(cfg), "metrics.json")
        else:
            return None

    def initialize_and_load_model(self) -> Molmo:
        cfg = self.config
        torch.cuda.set_device(f"cuda:{get_local_rank()}")
        device = torch.device("cuda")

        if cfg.load_path == "debug":
            logging.warning("Loading debugging model")
            model_cfg = ModelConfig(
                d_model=128,
                n_heads=2,
                n_layers=1,
                max_sequence_length=4096,
                additional_vocab_size=128,
                vocab_size=50280,
                embedding_size=50304,
                rope=True,
                weight_tying=False,
                vision_backbone=VisionBackboneConfig(
                    image_num_layers=1,
                ),
                pad_tokenizer=True,
                crop_mode="resize",
                tokenizer=TokenizerConfig(
                    identifier='allenai/OLMoE-1B-7B-0924'
                )
            )
            olmo_model = Molmo(model_cfg).to(device)
            olmo_model.reset_parameters()
        elif cfg.load_path.startswith("hf-"):
            hf_model = AutoModelForCausalLM.from_pretrained(
                cfg.load_path[3:], trust_remote_code=True, torch_dtype='fp32', device_map='cpu')
            import pdb; pdb.set_trace()
        elif cfg.fsdp is None:
            log.info("Loading model without FSDP...")
            olmo_model = Molmo.from_checkpoint(cfg.load_path, device=device)
            model_cfg = olmo_model.config
        else:
            log.info("Building FSDP model...")
            model_cfg_path = resource_path(cfg.load_path, "config.yaml")
            model_cfg = ModelConfig.load(model_cfg_path, key="model", validate_paths=False)
            olmo_model = Molmo(model_cfg)

            # We always have only rank0 load the checkpoint, and then use `sync_module_states`
            # in FSDP to broadcast the weights to the other processes
            if get_global_rank() == 0:
                is_unsharded = resource_path(cfg.load_path, "model.pt").is_file()
                if is_unsharded:
                    log.info("Loading state dict...")
                    state_dict_path = resource_path(cfg.load_path, "model.pt")
                    olmo_model.to_empty(device="cpu")
                    state_dict = torch.load(state_dict_path, map_location="cpu")
                    olmo_model.load_state_dict(state_dict, assign=True)
                else:
                    olmo_model.to_empty(device="cpu")
                    load_model_state(cfg.load_path, olmo_model)

            log.info("Wrapping model with FDSP...")
            wrap_policy = olmo_model.get_fsdp_wrap_policy(cfg.fsdp.wrapping_strategy)
            hybrid_sharding_fsdp_kwargs = {}
            if cfg.fsdp.sharding_strategy in (ShardingStrategy.HYBRID_SHARD, ShardingStrategy._HYBRID_SHARD_ZERO2):
                raise NotImplementedError()
            if version.parse(torch.__version__) < version.parse("2.1.0"):
                raise NotImplementedError()

            def dummy_init_fn(module: torch.nn.Module) -> None:
                # Prevent FSDP from re-initializing the parameters
                module.to_empty(device=get_default_device(), recurse=False)

            param_init_fn = dummy_init_fn
            olmo_model = FSDP(
                olmo_model,
                sharding_strategy=cfg.fsdp.sharding_strategy,
                mixed_precision=MixedPrecision(
                    param_dtype=cfg.autocast_precision,
                    buffer_dtype=cfg.autocast_precision
                ),
                auto_wrap_policy=wrap_policy,
                use_orig_params=False,
                limit_all_gathers=True,
                device_id=get_local_rank(),
                sync_module_states=True,
                param_init_fn=param_init_fn,
                **hybrid_sharding_fsdp_kwargs,
            )
            olmo_model.eval()
            torch.cuda.empty_cache()  # For the 70B this can prevent OOMs by reduce memory fragmentation

        if self.config.max_crops_override:
            logging.info(f"Overriding max crops from {olmo_model.config.max_crops} to {self.config.max_crops_override}")
            olmo_model.config.max_crops = self.config.max_crops_override

        seed_all(cfg.seed)

        dtype = olmo_model.transformer.wte.embedding.dtype
        log.info(f"Model weight dtype: {dtype}")
        log.info(f"Total number of parameters: {olmo_model.num_params():,d}")
        log.info(f"Number of non-embedding parameters: {olmo_model.num_params(include_embedding=False):,d}")
        log.info(f"Peak GPU Memory (MB) before FSDP: {int(peak_gpu_memory() or 0)}")
        barrier()
        return olmo_model, device

    def run(self):
        config = self.config
        # print('-----------------------------------------------------------------------------')
        # print(config)
        # print('-----------------------------------------------------------------------------')
        # print(config.evaluations)
        # print('-----------------------------------------------------------------------------')
        # print(config.evaluations[0].data)
        
        assert len(config.evaluations) > 0

        # Load any metrics that were cached
        cfg_to_metrics = {}
        for cfg in config.evaluations:
            if cfg.skip_if_metrics_cached:
                metric_file = self.get_metric_file(cfg)
                if metric_file and exists(metric_file):
                    logging.info(f"Loading pre-computed metrics for {cfg.label} from {metric_file}")
                    if get_global_rank() == 0:
                        with open(metric_file, "r") as f:
                            cfg_to_metrics[cfg.label] = json.load(f)["metrics"]
                    else:
                        # Still set with a empty dict to mark that this eval can can be skipped
                        cfg_to_metrics[cfg.label] = {}

        # Possibly return early if everything was cached
        if all(x.label in cfg_to_metrics for x in config.evaluations):
            logging.info("All metrics cached, checkpoint will not be loaded")
            all_metrics = {}
            for name, metrics in cfg_to_metrics.items():
                all_metrics.update({f"{name}/{k}": v for k, v in metrics.items()})
            to_print = {k: v for k, v in all_metrics.items() if isinstance(v, (int, float, str))}
            log_metrics_to_console("all-metrics", to_print)
            return all_metrics

        # Initialize the model
        model, device = self.initialize_and_load_model()
        # breakpoint()

        all_metrics = {}
        inference_warmup = True
        for eval_ix, evaluation in enumerate(config.evaluations):
            if evaluation.label in cfg_to_metrics:
                continue

            if len(config.evaluations) == 1:
                logging.info(f"Starting inference {evaluation.label}")
            else:
                logging.info(f"Starting inference {evaluation.label} ({eval_ix+1}/{len(config.evaluations)})")

            metrics_file = self.get_metric_file(evaluation)
            if metrics_file and exists(metrics_file):
                assert not evaluation.skip_if_metrics_cached
                logging.warning(f"{metrics_file} already exists! File will be overwritten")

            device_batch_size = evaluation.device_eval_batch_size or config.device_inf_eval_batch_size
            global_batch_size = device_batch_size * get_world_size()
            if evaluation.max_examples is not None and evaluation.max_examples >= 0:
                max_steps = max(evaluation.max_examples // global_batch_size, 1)
            elif evaluation.subset_num_batches:
                max_steps = evaluation.subset_num_batches
            else:
                max_steps = None

            if evaluation.data.multi_modal == "torch":
                dataloader = build_torch_mm_eval_dataloader(
                    device_batch_size,
                    config.seed,
                    model.config,
                    evaluation.data,
                    config.fsdp is not None,
                    max_steps=max_steps
                )
            else:
                raise NotImplementedError()
            mm_evaluation = InfDatasetEvaluator(
                dataloader,
                build_inf_evaluator(evaluation.mm_evaluator, self.get_save_dir(evaluation)),
                label=evaluation.label,
                n_steps=max_steps,
                max_new_tokens=evaluation.max_new_tokens,
                console_log_interval=config.console_log_interval
            )
            metrics = mm_evaluation.evaluate_model(
                model,
                device,
                autocast_precision=self.config.autocast_precision,
                is_distributed=self.config.fsdp is not None,
                pbar=self.config.pbar,
                inference_warmup=inference_warmup
            )
            inference_warmup = False

            # Post-process the metrics by saving the wandb.Html outputs to disk
            save_dir = self.get_save_dir(evaluation)

            if save_dir and get_global_rank() == 0:
                if not save_dir.startswith("gs://"):
                    makedirs(save_dir, exist_ok=True)

                for k, v in list(metrics.items()):
                    if isinstance(v, wandb.Html):
                        file_name = join(save_dir, f"{evaluation.label}-{k}.html")
                        with open(file_name, "w") as f:
                            f.write(v.html)
                        if file_name.startswith("gs://"):
                            metrics[k] = get_gcs_url(file_name)
                        else:
                            metrics[k] = file_name

            to_print = {k: v for k, v in metrics.items() if isinstance(v, (int, float, str))}
            if metrics_file and get_global_rank() == 0:
                to_save = dict(
                    metrics=metrics,
                    date=datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                    eval_config=dataclasses.asdict(evaluation),
                )
                with open(metrics_file, "w") as f:
                    json.dump(to_save, f, indent=2)
            log_metrics_to_console(evaluation.label, to_print)
            cfg_to_metrics[evaluation.label] = metrics

        all_metrics = {}
        for name, metrics in cfg_to_metrics.items():
            all_metrics.update({f"{name}/{k}": v for k, v in metrics.items()})

        if len(config.evaluations) > 1:   # print aggregated metrics if doing multiple evaluations
            to_print = {k: v for k, v in all_metrics.items() if isinstance(v, (int, float, str))}
            log_metrics_to_console("all-metrics", to_print)
        return all_metrics


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

    try:
        yaml_path, args_list = sys.argv[1], sys.argv[2:]
    except IndexError:
        raise OLMoCliError(f"Usage: {sys.argv[0]} [CONFIG_PATH] [OPTIONS]")

    eval_config = EvalConfig.load(yaml_path, [clean_opt(s) for s in args_list])
    ModelEvaluator(eval_config).run()
