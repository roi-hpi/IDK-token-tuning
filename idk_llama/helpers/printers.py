import os
from typing import TYPE_CHECKING

import torch
from lightning import Fabric
from print_on_steroids import logger

# from  import num_parameters

if TYPE_CHECKING:
    from llama_transfer import Args


def pretty_print_important_args(fabric: Fabric, args: "Args"):
    logger.debug("-----Training Duration-----")
    logger.info(
        f"Training for {args.training_goal} steps, "
        f"eval every {args.eval_interval} steps ({args.eval_interval * args.gradient_accumulation_steps} iters), "
        f"save every {args.save_interval} steps ({args.save_interval * args.gradient_accumulation_steps} iters), "
        f"log every {args.log_interval} steps, "
        f"model profiling every {args.model_profiling_interval} steps, "
        if args.model_profiling_interval
        else "model proflining disabled, ",
        f"warmup for {args.warmup_period} steps, "
        f"lr decay from {args.learning_rate} to {args.min_lr} until step {args.lr_decay_period}.",
    )
    logger.info(
        f"{args.batch_size=}, "
        f"split into {args.micro_batch_size=} on {args.num_devices=} (=> {args.micro_batch_size * args.num_devices} iter batch size). "
        f"and {args.gradient_accumulation_steps=}."
    )
    logger.info(
        f"Training for {args.training_goal} steps corresponds to "
        f"{args.training_goal * args.batch_size:,} samples, "
        f"{args.training_goal * args.batch_size * args.block_size / 1_000_000_000:,}B tokens"
    )
    logger.debug("---------------------")


def print_trainable_param_info(fabric: Fabric, model: torch.nn.Module):
    # num_total_params = num_parameters(model, requires_grad=None)
    # num_trainable_params = num_parameters(model, requires_grad=True)
    # num_nontrainable_params = num_parameters(model, requires_grad=False)

    # logger.debug("-----Param Analysis-----")
    # logger.info(f"Number of trainable parameters: {num_trainable_params:,}")
    # logger.info(f"Number of non trainable parameters: {num_nontrainable_params:,}")
    # logger.info(f"Total parameters {num_total_params:,}")
    # logger.info(
    #     f"Percentage of trainable parameters: {100 * num_trainable_params / (num_nontrainable_params + num_trainable_params):.2f}%"
    # )
    # logger.debug("---------------------")
    pass


def print_and_log_eval_results(
    fabric: Fabric,
    state: dict,
    val_metrics: dict[str, float],
    val_time: float = 0,
    preserve_tqdm=False,
):
    logger.success(
        f"Eval step {state['iter_num']}: {val_metrics['loss']=:.4f}, ",
        f"{val_metrics['perplexity']=:.2f}, "
        f"{val_metrics['per_token_nll']=:.2f}, "
        f"{val_metrics['per_doc_nll']=:.2f}, "
        f"val time: {val_time * 1000:.2f}ms",
    )

    fabric.log_dict(
        {
            "val/loss": val_metrics["loss"],
            "val/per_token_nll": val_metrics["per_token_nll"],
            "val/per_doc_nll": val_metrics["per_doc_nll"],
            "val/ppl": val_metrics["perplexity"],
            "trainer/optimizer_step": state["step_count"],
            "trainer/iter": state["iter_num"],
        }
    )


def print_and_log_train_results(
    fabric: Fabric,
    state: dict,
    loss: float,
    grad_norm: float,
    iter_time: float = 0,
    global_step_time: float = 0,
):
    logger.info(
        f"iter {state['iter_num']} step {state['step_count']}: loss {loss:.4f}, {grad_norm=:.4f}"
        f"iter time: {(iter_time) * 1000:.2f}ms global step time: {(global_step_time):.2f}s",
    )
    fabric.log_dict(
        {
            "train/loss": loss,
            "train/grad_norm": grad_norm,
            "trainer/optimizer_step": state["step_count"],
            "trainer/iter": state["iter_num"],
        }
    )


def log_slurm_info():
    # The info doesn't always seem to be in the same environment variable, so we just check all of them
    gpu_identifiers = (
        os.environ.get("SLURM_GPUS")
        or os.environ.get("SLURM_GPUS_PER_TASK")
        or os.environ.get("SLURM_JOB_GPUS")
        or os.environ.get("SLURM_STEP_GPUS")
        or len(os.environ.get("CUDA_VISIBLE_DEVICES", []))
    )
    logger.debug("-----SLURM Info-----")
    logger.info(
        f"Detected SLURM environment. SLURM Job ID: {os.environ.get('SLURM_JOB_ID')}, "
        f"SLURM Host Name: {os.environ.get('SLURM_JOB_NODELIST')}, "
        f"SLURM Job Name: {os.environ.get('SLURM_JOB_NAME')}, "
        f"SLURM GPUS: {gpu_identifiers}"
    )
    logger.debug("---------------------")
