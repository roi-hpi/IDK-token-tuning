import dataclasses
import os
import time
from pathlib import Path

import lightning as L
import numpy as np
import simple_parsing
import torch
import torch.distributed.checkpoint as dist_cp

# import transformers.models.mistral.modeling_mistral as hf_mistral
import transformers.models.gpt_neox.modeling_gpt_neox as hf_gptneox
from lightning.fabric.plugins.environments import LightningEnvironment, SLURMEnvironment
from lightning.fabric.strategies import FSDPStrategy
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.loggers import WandbLogger as PytorchLightningWandbLogger
from print_on_steroids import logger as printer
from print_on_steroids.print import graceful_exceptions
from torch import nn
from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict
from torch.distributed.fsdp import FullyShardedDataParallel, StateDictType
from torch.distributed.fsdp.api import (
    ShardedOptimStateDictConfig,
    ShardedStateDictConfig,
)
from torch.nn.functional import smooth_l1_loss
from torch.utils.data import DataLoader
from tqdm.asyncio import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTNeoXForCausalLM, PreTrainedModel

import wandb
from idk_llama.args import TrainingArgs as Args
from idk_llama.data.dataset import get_dataloaders
from idk_llama.dlib import (
    SpeedMonitorFabric,
    get_lr_with_cosine_schedule,
    log_model_stats_to_wandb,
    log_slurm_info,
    measure_model_flops,
    pretty_str_from_dict,
    wait_for_debugger,
)
from idk_llama.dlib.frameworks.fabric_lightning import State, dlib_save_checkpoint_hf, get_checkpoint_type
from idk_llama.helpers.printers import pretty_print_important_args, print_trainable_param_info

WANDB_PROJECT = "idk"
WANDB_ENTITY = "kjd-hpi"

print("import done")


def setup(args: Args) -> None:
    print("setup", os.environ.get("LOCAL_RANK"))
    args.out_dir = (args.out_dir / args.run_name).resolve()
    if args.smart_cuda_alloc:
        # Explanation: By setting PYTORCH_CUDA_ALLOC_CONF to "caching_allocator",
        # we enable the caching memory allocator, which improves memory management efficiency.
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "caching_allocator"

    IS_ON_SLURM = SLURMEnvironment().detect()
    cluster_environment = None
    if IS_ON_SLURM:
        # do this as workaround check, since fabric.local_rank is not available yet
        if os.environ.get("LOCAL_RANK") is None:
            printer.info("Disabling SLURMEnvironment (we use lightning's native DDP launcher)")
            log_slurm_info()
        cluster_environment = LightningEnvironment()

    # Distributed setup
    precision = args.precision
    if args.num_devices >= 1:
        assert args.accelerator == "cuda"
        from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXLayer
        
        hf_gptneox.GPTNeoXAttention.reset_parameters = lambda self: None # need to provide for FSDP
        hf_gptneox.GPTNeoXFlashAttention2.reset_parameters = lambda self: None # need to provide for FSDP
        hf_gptneox.GPTNeoXRotaryEmbedding.reset_parameters = lambda self: None # need to provide for FSDP

        activation_checkpointing_policy = {GPTNeoXLayer} if args.activation_checkpointing else None
        # # need to create for FSDP meta device init because it's not already implemented for MistralRMSNorm / MistralRotaryEmbedding
        # MistralRMSNorm.reset_parameters = lambda self: None
        # MistralRotaryEmbedding.reset_parameters = lambda self: None

        # if args.activation_checkpointing and args.optimized_activation_checkpointing_policy:
        #     # optimized because flashattention has it's own checkpointing for attn
        #     # leave out lm_head for now
        #     activation_checkpointing_policy = {MistralMLP, nn.Embedding}
        strategy = FSDPStrategy(
            auto_wrap_policy={GPTNeoXLayer},
            activation_checkpointing_policy=activation_checkpointing_policy,
            state_dict_type="full",
            limit_all_gathers=args.fsdp_limit_all_gathers,
            cpu_offload=args.fsdp_cpu_offload,
            sync_module_states=True,  # Make sure all ranks have the same model weights
            use_orig_params=True,
            sharding_strategy=args.fsdp_sharding_strategy,
            cluster_environment=cluster_environment,
        )
    else:
        strategy = "auto"

    csv_logger = CSVLogger(
        args.out_dir.parent,
        args.out_dir.name,
        flush_logs_every_n_steps=args.gradient_accumulation_steps * 10,
    )

    ############# Construct W&B Logger ##############
    if args.offline or args.data_preprocessing_only:
        os.environ["WANDB_MODE"] = "dryrun"

    wandb_logger = PytorchLightningWandbLogger(
        name=args.run_name,
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        # log_model="all", # fails with FSDP
        tags=args.wandb_tags,
    )

    fabric = L.Fabric(
        devices=args.num_devices,
        strategy=strategy,
        precision=precision,
        loggers=[wandb_logger, csv_logger],
    )
    with graceful_exceptions(extra_message=f"Rank: {fabric.global_rank}"):
        fabric.launch(main, args)


def main(fabric: L.Fabric, args: Args):
    if args.debug and fabric.local_rank == 0:
        wait_for_debugger()
        fabric.barrier()
    if fabric.global_rank == 0:
        fabric.logger.log_hyperparams(dataclasses.asdict(args))
        fabric.logger.experiment.log_code(".")
        if not args.offline:
            if args.run_name is None:
                printer.warning("No run name specified with `--run_name`. Using W&B default (randomly generated name).")
            else:
                assert fabric.logger.version is not None
                # Append id to name for easier recognition in W&B UI
                fabric.logger.experiment.name = args.run_name + "-" + fabric.logger.version
    printer.config(mode="dev", verbosity="debug", rank=fabric.global_rank, print_rank0_only=True)
    printer.debug(args)
    pretty_print_important_args(fabric, args)

    if args.use_lora:
        raise NotImplementedError("LoRA is not supported yet")

    # if args.block_size != config.block_size:
    #     printer.warning(f"Using custom block_size={args.block_size} instead of {config.block_size} from model config.")

    if fabric.global_rank == 0:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    fabric.seed_everything(args.seed)  # same seed for every process to init model (FSDP)

    t0 = time.perf_counter()
    param_precision = torch.bfloat16 if args.precision == "bf16-true" else torch.float32
    init_device = torch.device("cuda:0") if fabric.is_global_zero else torch.device("meta")
    if args.saved_checkpoint_path and not Path(args.saved_checkpoint_path).exists():
        # download from HF Hub
        from huggingface_hub import snapshot_download

        local_ckpt_path = args.out_dir / "hfhub" / args.saved_checkpoint_path
        if fabric.is_global_zero:
            printer.info(f"Downloading checkpoint from HF Hub: {args.saved_checkpoint_path} to {local_ckpt_path}")
            snapshot_download(
                args.saved_checkpoint_path,
                local_dir=local_ckpt_path,
                local_dir_use_symlinks=False,
                max_workers=args.preprocessing_workers,
            )
        args.saved_checkpoint_path = str(local_ckpt_path)
        fabric.barrier()

    load_from_path = args.saved_checkpoint_path or args.model_path
    with init_device:
        # PyPI FA2 breaks compile
        # Pythia (GPTXNeo) not yet supported by PyTorch native FA2 instead (default fallback)
        attn_impl = "mem-efficient" if args.compile else "flash_attention_2"
        # attn_impl = "sdpa"
        need_attn_impl_monkeypatch = attn_impl == "flash_attention_2" and args.precision in ["bf16-mixed", "16-mixed"]
        if need_attn_impl_monkeypatch:
            # transformers bug: PyPI FA2 asserts not float32 weights, but we may use bf16-mixed later
            # https://github.com/huggingface/transformers/issues/28052#issuecomment-1870034307
            def _autoset_attn_implementation_monkeypatch(cls, config, *args, **kwargs):  # type: ignore
                config._attn_implementation = attn_impl
                return config

            old_autoset_attn_implementation = PreTrainedModel._autoset_attn_implementation
            PreTrainedModel._autoset_attn_implementation = classmethod(_autoset_attn_implementation_monkeypatch)
        if args.use_additional_flash_attn_kernels and not args.compile:
            # raise NotImplementedError
            from flash_attn.ops.layer_norm import (
                LayerNorm as FlashLayerNorm,  # TODO: doesn't work yet, need to integrate DropoutAddLayerNorm somehow
            )

            prevGPTNeoxLayerNorm = hf_gptneox.nn.LayerNorm
            hf_gptneox.nn.LayerNorm = FlashLayerNorm
            printer.success("Using FlashLayerNorm instead of LayerNorm.")

            # NOTE: debug prints to ensure that the monkeypatching worked
            printer.debug("nn.LayerNorm", nn.LayerNorm.__class__, nn.LayerNorm)
            printer.debug("torch.nn.LayerNorm", torch.nn.LayerNorm.__class__, torch.nn.LayerNorm)
            from torch.nn import LayerNorm as TorchLayerNorm

            printer.debug("TorchLayerNorm", TorchLayerNorm.__class__, TorchLayerNorm)
            printer.debug("hf_gptneox.nn.LayerNorm", hf_gptneox.nn.LayerNorm.__class__, hf_gptneox.nn.LayerNorm)

        if attn_impl == "sdpa":
            printer.warning("Using torch-native SDPA instead of FlashAttention2.")
            raise NotImplementedError("SDPA is not supported by Pythia.")
            # Force FlashAttention in torch-native SDPA
            torch.backends.cuda.enable_math_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_flash_sdp(True)
        elif attn_impl == "mem-efficient":
            printer.warning("Using mem-efficient instead of FlashAttention2.")
            # Force FlashAttention in torch-native SDPA
            attn_impl = "eager"
            torch.backends.cuda.enable_math_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_flash_sdp(False)
        else:
            printer.info(f"Using Attention {attn_impl} implementation.")

        model: GPTNeoXForCausalLM = GPTNeoXForCausalLM.from_pretrained(
            load_from_path,
            attn_implementation=attn_impl,
            torch_dtype=param_precision,
            low_cpu_mem_usage=init_device.type != "meta",
            use_cache=False,
            return_dict=True,
        )
        if need_attn_impl_monkeypatch:
            PreTrainedModel._autoset_attn_implementation = old_autoset_attn_implementation
        if args.use_additional_flash_attn_kernels and not args.compile:
            # raise NotImplementedError
            assert isinstance(model.gpt_neox.final_layer_norm, FlashLayerNorm)
            hf_gptneox.nn.LayerNorm = prevGPTNeoxLayerNorm  # undo monkeypatch after model creation
        printer.debug(model.config)

    printer.success(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
    printer.debug(model)

    fabric.barrier()
    ######### Model Modifications ##########
    if not args.saved_checkpoint_path:
        if args.tokenizer_path != args.model_path:
            new_vocab_size = len(AutoTokenizer.from_pretrained(args.tokenizer_path))

            # if new_vocab_size <= len(model.get_input_embeddings().weight):
            #     # NOTE: pythia already has builtin oversized vocab for better tiling, no need to resize
            #     printer.warning(f"Tokenizer vocab size {new_vocab_size} is smaller or equal to model vocab size, no need to resize.")
            # else:
            print(f"Resizing model to new vocab size: {new_vocab_size} from {len(model.get_input_embeddings().weight)}")
            model.resize_token_embeddings(new_vocab_size)

    fabric.barrier()
    printer.debug(model)

    checkpoint_parameter_keys = {k for k, p in model.named_parameters() if p.requires_grad}
    print_trainable_param_info(fabric, model)
    parameter_lookup = {k: (p.shape, p.requires_grad) for k, p in model.named_parameters()}
    fabric.barrier()

    fwd_bwd_flops = 0.0
    if fabric.is_global_zero:
        printer.info("------TFLOP & Mem Stats------")
        printer.debug("model num layers", len(model.gpt_neox.layers), rank0_only=False)
        printer.debug("model hidden size", model.gpt_neox.embed_in.weight.shape[-1], rank0_only=False)
        fwd_bwd_flops = measure_model_flops(
            fabric,
            args.micro_batch_size,
            args.block_size,
            lambda: AutoModelForCausalLM.from_pretrained(
                load_from_path,
                torch_dtype=param_precision,
                use_cache=False,
                return_dict=True,
            ),
            parameter_lookup=parameter_lookup,
            num_layers=len(model.gpt_neox.layers),
            hidden_size=model.gpt_neox.embed_in.weight.shape[-1],
        )[0]
    fabric.broadcast(fwd_bwd_flops)
    fabric.barrier()
    speed_monitor = SpeedMonitorFabric(
        fabric,
        world_size=fabric.world_size,
        model_flops_fwd_bwd=fwd_bwd_flops,
        window_size=1,
    )
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    printer.info(f"expected bf16 memory usage from params: {num_params * 2 / 1e9:.2f} GB")
    bf16_bytes = num_params * 2
    fp32_bytes = num_params * 4
    if args.precision == "bf16-true":
        total = bf16_bytes + bf16_bytes + bf16_bytes * 2
        if args.fsdp_sharding_strategy == "FULL_SHARD":
            sharded = total / fabric.world_size
        elif args.fsdp_sharding_strategy == "SHARD_GRAD_OP":
            sharded = (bf16_bytes + bf16_bytes * 2) / fabric.world_size + bf16_bytes
        else:
            sharded = total
        optim_state_ckpt_bytes = bf16_bytes * 2
    if args.precision in ["bf16-mixed", "16-mixed"]:
        # master weights + bf16 weights + fp32 adam states + fp32 grads (converted from bf16)
        total = fp32_bytes + bf16_bytes + fp32_bytes * 2 + fp32_bytes
        if args.fsdp_sharding_strategy == "FULL_SHARD":
            sharded = total / fabric.world_size
        elif args.fsdp_sharding_strategy == "SHARD_GRAD_OP":
            sharded = (fp32_bytes * 2 + fp32_bytes) / fabric.world_size + bf16_bytes + fp32_bytes
        else:
            sharded = total
        optim_state_ckpt_bytes = fp32_bytes * 2
    printer.info(f"Expected {args.precision} total memory usage from params + grad + adam state: {total / 1e9:.2f} GB")
    printer.info(
        f"Expected {args.precision} {args.fsdp_sharding_strategy} sharded memory usage from params + grad + adam state: {sharded / 1e9:.2f} GB"
    )
    printer.info(f"Expected {args.precision} peak checkpointing CPU RAM usage: {optim_state_ckpt_bytes / 1e9:.2f} GB")
    printer.info(f"TFLOP / sec available: {speed_monitor.hardware_flops_per_sec_promised / 1e12:.2f}")
    printer.info(f"Device type: {torch.cuda.get_device_name(fabric.device).lower()}")
    printer.info(f"Device memory: {torch.cuda.get_device_properties(fabric.device).total_memory / 1e9:.2f} GB")
    printer.info("----------------------------")

    # if args.compile:
    #     13/01/24: compile is bugged w/ pypi FA2 => don't use
    #     printer.debug("Running `torch.compile` on  model...", rank0_only=False)
    #     Debug History:
    #     - # ----> not actually!? Crucial: we need to unwrap the model before compiling (only for bf16-true for some reason)
    #     - # Error message from dynamo: Global state changed ... Likely caused by some setattr in the _FabricModule forward
    #     - # related: https://github.com/pytorch/pytorch/issues/112787
    #     - # moved before FSDP because of this https://discuss.pytorch.org/t/torch-compile-what-is-the-best-scope-of-compilation/185442/3
    #     - # not sure if recommendation is still up to date though
    #     model = torch.compile(model)

    model = fabric.setup_module(model)
    fabric.print(model)
    printer.info(f"current memory usage with (sharded) model on device {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    optimizer = fabric.setup_optimizers(get_optimizer(args, model))
    printer.info(f"Peak memory usage after optimizer setup: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    state: State = {
        "model": model,
        "optimizer": optimizer,
        "hparams": dataclasses.asdict(args),
        "iter_num": 0,
        "step_count": 0,
    }

    resume_from_sample_idx = None
    if args.saved_checkpoint_path:
        load_optimizer_checkpoint(fabric, args, model, optimizer)

        metadata = torch.load(Path(args.saved_checkpoint_path) / "metadata.pt")
        state["iter_num"] = metadata["iter_num"] + 1
        state["step_count"] = metadata["step_count"]
        state["hparams"] = metadata["hparams"]
        resume_from_sample_idx = state["step_count"] * args.batch_size
        # resume_from_sample_training_order_seed
        speed_monitor.step = state["iter_num"] - 1
        printer.success(f"Resuming from step {state['step_count']} (sample idx={resume_from_sample_idx})", rank0_only=False)
        printer.debug(state["hparams"])
        # TODO: use correct seed for training_order when resuming from ckpt in 2nd+ epoch
    train_dataloader, val_dataloader = get_dataloaders(
        data_dir=args.data_dir,
        block_size=args.block_size,
        batch_size=args.micro_batch_size,
        workers=args.workers,
        tokenizer_path=args.tokenizer_path,
        use_clipped_val=args.cross_tokenizer_val,
        val_batch_size=args.eval_micro_batch_size,
        resume_from_sample_idx=resume_from_sample_idx,
    )

    train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)
    fabric.barrier()
    fabric.seed_everything(args.seed + fabric.global_rank)
    printer.debug(f"Starting training: {fabric.global_rank}, seed: {args.seed +  fabric.global_rank}", rank0_only=False)

    try:
        printer.info(f"peak memory usage before training {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

        train_time = time.perf_counter()
        train(
            fabric,
            args,
            state,
            train_dataloader,
            val_dataloader,
            speed_monitor,
            checkpoint_parameter_keys,
        )
        printer.success(f"Training time: {(time.perf_counter()-train_time):.2f}s")
        future = dlib_save_checkpoint_hf(
            fabric,
            state,
            args.out_dir,
            tags=["final"],
            state_dict_type=StateDictType.FULL_STATE_DICT,
        )
        if fabric.is_global_zero:
            printer.success(f"Saved final checkpoint to {future.result()}")  # future.result() waits until checkpoint is saved
        fabric.barrier()

    except KeyboardInterrupt:
        printer.error("Detected KeyboardInterrupt, stopping training...")


def ce_loss_vector_target(logits, targets):
    """
    had to implement on our own because torch ce doesnt support targets that are like one-hot vectors, only indices
    :param logits: shape (num_masked_tokens, vocab_size)
    :param targets: shape (num_masked_tokens, vocab_size)
    :return: ce loss between logits and targets


    """
    log_sm_logits = nn.functional.log_softmax(logits, dim=-1)  # shape (num_masked_tokens, vocab_size)
    ce_loss = -torch.sum(log_sm_logits * targets, dim=-1)  # shape (num_masked_tokens)
    return ce_loss


def ce_loss_vector_target_correct_pred_reg(logits, targets):
    """
    same as "ce_loss_vector_target" but with additional binary CE loss between the IDK logit and it's target is 0 (the prediction was correct)
    :param logits: shape (num_masked_tokens, vocab_size)
    :param targets: shape (num_masked_tokens, vocab_size)
    :return: ce loss between logits and targets


    """
    # sm_logits = nn.functional.softmax(logits, dim=-1)  # shape (num_masked_tokens, vocab_size)
    # log_sm_logits = torch.log(sm_logits)  # shape (num_masked_tokens, vocab_size)

    # CAREFUL, do this instead if commented out part for numerical stability
    # NOTE: sm_logits is a prob distribution, no logits??
    log_sm_logits = nn.functional.log_softmax(logits, dim=-1)  # shape (num_masked_tokens, vocab_size)
    sm_logits = torch.exp(log_sm_logits)  # shape (num_masked_tokens, vocab_size)

    ce_loss = -torch.sum(log_sm_logits * targets, dim=-1)  # shape (num_masked_tokens)

    # additional BCE for IDK token
    p_IDK_token = sm_logits[:, 50277]  # shape (num_masked_tokens)
    bce_idk = torch.nn.functional.binary_cross_entropy(
        p_IDK_token, torch.zeros_like(p_IDK_token), reduction="none"
    )  # shape (num_masked_tokens)

    # add ce_loss and bce_idk
    total_loss = ce_loss + bce_idk  # shape (num_masked_tokens)
    return total_loss


def train(
    fabric: L.Fabric,
    args: Args,
    state: State,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    speed_monitor: SpeedMonitorFabric,
    checkpoint_parameter_keys: set[str],
):
    model = state["model"].train()
    optimizer = state["optimizer"]

    if val_dataloader is not None:
        do_and_log_eval(fabric, args, state, val_dataloader, speed_monitor)
        if args.only_val:
            exit(0)

    train_iter = iter(train_dataloader)

    # print bar only on rank0
    step_bar = range(state["step_count"], args.training_goal)
    if fabric.global_rank == 0:
        step_bar = tqdm(step_bar, desc="Such adaptation much wow...")

    global_step_end_t = last_global_step_end_t = speed_monitor_end_t = time.perf_counter()
    for i in step_bar:
        iter_bar = range(args.gradient_accumulation_steps)
        if fabric.global_rank == 0:
            iter_bar = tqdm(iter_bar, desc=f"Step {i+1}...", leave=False)

        # iter until effective batch size is reached with gradient accumulation
        for j in iter_bar:
            iter_start_t = time.perf_counter()
            state["iter_num"] += 1
            input_ids, targets = next(train_iter, (None, None))
            if input_ids is None:
                printer.info("Reached end of dataset, starting from beginning.")
                # Re-shuffle dataset w/ reproducible seed
                train_dataloader.dataset.training_order = train_dataloader.dataset.get_reproducible_shuffled_training_order(
                    seed=state["step_count"]
                )

                train_iter = iter(train_dataloader)
                input_ids, targets = next(train_iter)
            do_optimizer_step = j == (args.gradient_accumulation_steps - 1)
            with fabric.no_backward_sync(model, enabled=not do_optimizer_step):
                logits = model(input_ids)["logits"]
                IDK_TOKEN = 50277
                t_loss_start = time.perf_counter()

                if args.loss_type == "ce":
                    # logits = model(input_ids)["logits"]
                    logits = logits.reshape(-1, logits.size(-1))
                    targets = targets.reshape(-1)
                    loss = torch.nn.functional.cross_entropy(logits, targets, ignore_index=-1)
                    t_loss_end = time.perf_counter()

                    # fabric.backward(loss / args.gradient_accumulation_steps)
                elif args.loss_type == "idk":
                    # logits = logits.reshape(-1, logits.size(-1))
                    # targets = targets.reshape(-1)
                    # ------------------------------
                    # batch_size = logits.shape[0]
                    # sequence_length = logits.shape[1]
                    vocab_size = logits.shape[2]
                    # print(f"batch_size: {batch_size}")
                    # print(f"sequence_length: {sequence_length}")
                    # print(f"vocab_size: {vocab_size}")
                    # num_masked_tokens = (labels != -100).sum()
                    # ------------------------------
                    masked_tokens_msk = targets != -1  # shape (batch_size, sequence_length) with True/False values

                    # logits for masekd tokens:
                    masked_logits = logits[masked_tokens_msk]  # shape (num_masked_tokens, vocab_size)
                    predicted_logits = masked_logits.argmax(dim=-1)  # shape (num_masked_tokens)
                    # gt token indices of masked tokens:
                    mask_token_gts = targets[masked_tokens_msk]  # shape (num_masked_tokens)

                    correct_predictions_msk = (
                        predicted_logits == mask_token_gts
                    )  # shape (num_masked_tokens) with True/False values
                    masked_logits_correct_pred = masked_logits[
                        correct_predictions_msk
                    ]  # shape (num_correct_predictions, vocab_size)
                    masked_logits_wrong_pred = masked_logits[
                        ~correct_predictions_msk
                    ]  # shape (num_wrong_predictions, vocab_size)

                    onehot_labels = nn.functional.one_hot(
                        mask_token_gts, num_classes=vocab_size
                    )  # shape (num_masked_tokens, vocab_size) with 1s in the correct token index and 0s elsewhere

                    # --------loss for correct predictions- regular cross entropy-----------------:
                    # todo label_smoothing?
                    onehot_labels_correct_pred = onehot_labels[
                        correct_predictions_msk
                    ]  # shape (num_correct_predictions, vocab_size)
                    # if self.correct_pred_reg:
                    correct_pred_loss = ce_loss_vector_target_correct_pred_reg(
                        masked_logits_correct_pred, onehot_labels_correct_pred
                    )
                    # else:
                    #     correct_pred_loss = self.ce_loss_vector_target(masked_logits_correct_pred,
                    #                                             onehot_labels_correct_pred)  # shape (num_correct_predictions)

                    # --------loss for wrong predictions- IDK loss--------------------------------:
                    onehot_labels_wrong_pred = onehot_labels[
                        ~correct_predictions_msk
                    ]  # shape (num_wrong_predictions, vocab_size)
                    # if self.IDK_weight_schedule=='adaptive':
                    p_preds_wrong = torch.nn.functional.softmax(masked_logits_wrong_pred, dim=-1)
                    # -------------------
                    gold_token_index = mask_token_gts[~correct_predictions_msk]  # shape (num_wrong_predictions)
                    p_gold_token = p_preds_wrong[
                        torch.arange(p_preds_wrong.shape[0]), gold_token_index
                    ]  # shape (num_wrong_predictions)
                    p_top_token = p_preds_wrong.max(dim=-1)[0]  # shape (num_wrong_predictions)

                    # [KD 14/3] this loss did not converge for pythia-160m and pythia-410m, trying with **2
                    # idk_weights = (
                    #     torch.ones_like(p_top_token) - p_gold_token**2 / p_top_token**2
                    # )  # shape (num_wrong_predictions)
                    idk_weights = 0.5 * (
                        torch.ones_like(p_top_token) - p_gold_token / p_top_token
                    )  # shape (num_wrong_predictions)

                    # -------------------
                    onehot_labels_wrong_pred = onehot_labels_wrong_pred * (1 - idk_weights.unsqueeze(1))  # replace 1s
                    onehot_labels_wrong_pred[:, IDK_TOKEN] = idk_weights
                    wrong_pred_loss = ce_loss_vector_target(
                        masked_logits_wrong_pred, onehot_labels_wrong_pred
                    )  # shape (num_wrong_predictions)

                    # for logging:
                    idk_weight = idk_weights.mean().detach().cpu().numpy()

                    # else:
                    #     idk_weight = self.cur_IDK_weight
                    #     # put 0.2 in the IDK token index and 0.8 in the correct token index (where was 1 before)
                    #     onehot_labels_wrong_pred = onehot_labels_wrong_pred * (1 - idk_weight)  # replace 1s
                    #     onehot_labels_wrong_pred[:, self.IDK_token_index] = idk_weight
                    #     wrong_pred_loss = self.ce_loss_vector_target(masked_logits_wrong_pred,
                    #                                                 onehot_labels_wrong_pred)  # shape (num_wrong_predictions)

                    # if there are no correct predictions, the loss is only the IDK loss, and vice-versa
                    if masked_logits_correct_pred.shape[0] == 0:
                        combined_loss = wrong_pred_loss
                    elif masked_logits_wrong_pred.shape[0] == 0:
                        combined_loss = correct_pred_loss
                    else:  # if there are both correct and wrong predictions
                        combined_loss = torch.cat((correct_pred_loss, wrong_pred_loss), dim=0)  # shape (num_masked_tokens)
                    loss = combined_loss.mean()
                    t_loss_end = time.perf_counter()

                fabric.backward(loss / args.gradient_accumulation_steps)
                if args.loss_type == "idk":
                    with torch.no_grad():
                        probs = nn.functional.softmax(logits, dim=-1)
                        is_target_not_minus_one = targets != -1
                        hotfixed_targets = targets.clone()[is_target_not_minus_one]
                        gold_probs = (
                            probs[is_target_not_minus_one].gather(dim=-1, index=hotfixed_targets.unsqueeze(-1)).squeeze(-1)
                        )
                if fabric.is_global_zero:
                    if args.loss_type == "idk":
                        log_dict = {
                            "trainer/idk_weight": idk_weight,
                            "trainer/pgold-div-ptop": torch.mean(p_gold_token / p_top_token).item(),
                            # "train/thinking-objective-probs": torch.mean(thinking_weights).item(),
                            "train/pred-probs-idk": torch.mean(probs[:, :, IDK_TOKEN]).item(),
                            "train/wrongpred-probs-gold": torch.mean(p_gold_token).item(),
                            "train/pred-probmass-gold": torch.mean(gold_probs).item(),
                            "train/wrongpred-probs-top": torch.mean(p_top_token).item(),
                            "train/wrong-pred-loss": wrong_pred_loss.mean().item(),
                            "train/correct-pred-loss": correct_pred_loss.mean().item(),
                        }
                        wandb.run.log(
                            log_dict,
                            commit=False,
                        )
            iter_end_t = time.perf_counter()

            # Log performance stats
            speed_monitor.on_train_batch_end(
                args.micro_batch_size,
                iter_end_t - iter_start_t,
                iter_end_t - speed_monitor_end_t,
                # this assumes that device FLOPs are the same and that all devices have the same batch size
                tokens=input_ids.numel(),
                compute=True,
                step_kwargs={
                    "trainer/optimizer_step": state["step_count"],
                    "trainer/iter": state["iter_num"],
                },
            )
            speed_monitor_end_t = time.perf_counter()

        ###########################
        ####### OPTIM STEP ########
        ###########################
        opt_step_t0 = time.perf_counter()
        # determine and set the learning rate for this optimizer step
        lr = (
            get_lr_with_cosine_schedule(
                state["step_count"] + 1,  # avoid skipping first step due to resulting lr = 0
                args.learning_rate,
                args.warmup_period,
                args.lr_decay_period,
                args.min_lr,
            )
            if args.decay_lr
            else args.learning_rate
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if args.grad_clip != -1:
            pre_clip_grad_norm = fabric.clip_gradients(
                model,
                optimizer,
                max_norm=args.grad_clip,
                error_if_nonfinite=False,  # NOTE: had to disable error on inf for pythia (only for idk loss)
            ).item()

        # Gradient & param tracking
        stat_tracking_elapsed_t = 0
        if (
            state["step_count"] % args.model_profiling_interval == 0
            and args.model_profiling
            and not np.isinf(pre_clip_grad_norm)
            and not np.isnan(pre_clip_grad_norm)
        ):
            # [KD] NOTE: we get inf grad norm in the first few opt steps, I think it's fine, we just need to skip the grad logging code because it errors in that case
            stat_tracking_elapsed_t = log_model_stats_to_wandb(model, log_weights=True, log_grads=True)

        optimizer.step()
        optimizer.zero_grad()
        last_global_step_end_t = global_step_end_t
        global_step_end_t = time.perf_counter()
        state["step_count"] += 1

        # Also log first opt step, do -1. Do after optimizer.step & zero_grad to log timings
        if (state["step_count"] - 1) % args.log_interval == 0:
            metrics = {
                "trainer/optimizer_step": state["step_count"],
                "trainer/iter": state["iter_num"],
                "trainer/tokens": state["step_count"] * args.batch_size * args.block_size,
                "trainer/samples": state["step_count"] * args.batch_size,
                "train/loss": loss.item(),
                "train/grad_norm": pre_clip_grad_norm,
                "trainer/lr": lr,
            }
            timings = {
                "iter_time": iter_end_t - iter_start_t,
                "global_step_time": global_step_end_t - last_global_step_end_t,
                "opt_step_time": (global_step_end_t - opt_step_t0) - stat_tracking_elapsed_t,
                "grad_tracking_time": stat_tracking_elapsed_t,
                "speed_monitor_time": speed_monitor_end_t - iter_end_t,
                "loss_time": t_loss_end - t_loss_start,
                "max_cuda_ram": f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB",
            }
            torch.cuda.reset_peak_memory_stats()
            printer.info(pretty_str_from_dict(metrics | timings, prefix="Step stats:"))
            fabric.log_dict(metrics)

        if val_dataloader is not None and state["step_count"] % args.eval_interval == 0:
            do_and_log_eval(fabric, args, state, val_dataloader, speed_monitor)
            fabric.barrier()

        if state["step_count"] % args.save_interval == 0:
            dlib_save_checkpoint_hf(
                fabric,
                state,
                args.out_dir,
                state_dict_type=StateDictType.FULL_STATE_DICT,
            )


def load_optimizer_checkpoint(
    fabric: L.Fabric,
    args: Args,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    """
    Note that after loading an optimizer checkpoint with FSDP, memory usage seems to be higher than before resuming the training.
    https://github.com/huggingface/transformers/issues/26186#issuecomment-1725035726
    """
    local_checkpoint_path = Path(args.saved_checkpoint_path)
    printer.info(f"Resuming training from {local_checkpoint_path}")
    checkpoint_type = get_checkpoint_type(local_checkpoint_path)

    if checkpoint_type == StateDictType.FULL_STATE_DICT:
        raise NotImplementedError("Should have used sharded state dict implementation")
    elif checkpoint_type == StateDictType.SHARDED_STATE_DICT:
        t1 = time.perf_counter()

        from lightning.fabric.wrappers import _unwrap_objects
        from torch.distributed.checkpoint.state_dict import StateDictOptions

        unwrapped_model, unwrapped_optim = _unwrap_objects([model, optimizer])
        # NOTE: SUPER IMPORTANT to use cpu_offload=True, otherwise we take ~>10GB more CUDA RAM during training
        optim_sd = get_optimizer_state_dict(
            unwrapped_model, unwrapped_optim, options=StateDictOptions(cpu_offload=True, full_state_dict=False)
        )

        printer.info("Pre-C", optim_sd["state"].keys())
        printer.info("Pre-C", optim_sd["param_groups"])
        dist_cp.state_dict_loader.load(
            optim_sd,
            storage_reader=dist_cp.filesystem.FileSystemReader(local_checkpoint_path / "optimizer"),
        )
        printer.info("load done", rank0_only=False)
        with FullyShardedDataParallel.state_dict_type(
            model,
            StateDictType.SHARDED_STATE_DICT,
            ShardedStateDictConfig(offload_to_cpu=True),
            ShardedOptimStateDictConfig(offload_to_cpu=True),
        ):
            optim_sd = FullyShardedDataParallel.optim_state_dict_to_load(
                unwrapped_model, unwrapped_optim, optim_state_dict=optim_sd
            )
        printer.info("done", rank0_only=False)
        printer.info("A", optimizer.state_dict()["state"].keys())
        printer.info("B", unwrapped_optim.state_dict()["state"].keys())
        printer.info("C", optim_sd["state"].keys())

        printer.info("A", optimizer.state_dict()["param_groups"])
        printer.info("B", unwrapped_optim.state_dict()["param_groups"])
        printer.info("C", optim_sd["param_groups"])
        optimizer.load_state_dict(optim_sd)
        printer.info(f"Time to load optimizer state dict: {time.perf_counter() - t1:.02f} seconds.")
        # set_optimizer_state_dict is bugged with KeyError when using activation checkpointing ~ 13/01/24
        # set_optimizer_state_dict(unwrapped_model, unwrapped_optim, optim_state_dict=optim_sd)
        fabric.barrier()


def get_optimizer(args: Args, model: torch.nn.Module) -> torch.optim.Optimizer:
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    trainable_named_parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    assert len(trainable_parameters) == len(trainable_named_parameters)

    ### Do not include RMSNorm and embs for weight decay https://forums.fast.ai/t/is-weight-decay-applied-to-the-bias-term/73212/6
    no_decay = ["lm_head", "wte", "embed_tokens", "ln_f", "norm"]
    trainable_parameters = [
        {
            "params": [p for n, p in trainable_named_parameters if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in trainable_named_parameters if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    # printer.info(f"no weight decay for: {[n for n, p in trainable_named_parameters if any(nd in n for nd in no_decay)]}")
    # printer.info(f"weight decay for: {[n for n, p in trainable_named_parameters if not any(nd in n for nd in no_decay)]}")

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        foreach=args.adamw_foreach,
    )
    return optimizer


def do_and_log_eval(
    fabric: L.Fabric,
    args: Args,
    state: dict,
    val_dataloader: DataLoader,
    speed_monitor: SpeedMonitorFabric,
):
    state["model"].eval()
    t0 = time.perf_counter()
    val_metrics = validate(fabric, args, state["model"], val_dataloader)
    t1 = time.perf_counter() - t0
    speed_monitor.eval_end(t1)
    metrics = {
        "trainer/optimizer_step": state["step_count"],
        "trainer/iter": state["iter_num"],
        "val/loss": val_metrics["loss"].item(),
        "val/per_token_nll": val_metrics["per_token_nll"].item(),
        "val/per_doc_nll": val_metrics["per_doc_nll"].item(),
        "val/ppl": val_metrics["perplexity"].item(),
    }
    printer.info(pretty_str_from_dict(metrics | {"val/time": t1}, prefix="Eval Stats:"))
    fabric.log_dict(metrics)
    state["model"].train()


@torch.no_grad()
def validate(
    fabric: L.Fabric,
    args: Args,
    model: GPTNeoXForCausalLM,
    val_dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, float]:
    model.eval()
    val_iter = iter(val_dataloader)
    eval_iter_batch_size = args.eval_micro_batch_size * args.num_devices
    max_iters_in_dataloader = len(val_dataloader)
    iters = args.eval_samples // eval_iter_batch_size if args.eval_samples != -1 else max_iters_in_dataloader
    iters = min(iters, max_iters_in_dataloader)

    num_non_pad_tokens = torch.tensor(0, device=fabric.device)
    losses = torch.zeros(iters, device=fabric.device)
    logprob_accumulator = torch.zeros(iters, device=fabric.device)

    for i in tqdm(range(iters), desc="Validating...", leave=False):
        input_ids, targets = next(val_iter)
        logits = model(input_ids)["logits"]
        logits = logits.reshape(-1, logits.size(-1))
        targets = targets.reshape(-1)
        summed_loss = torch.nn.functional.cross_entropy(logits, targets, ignore_index=-1, reduction="sum")

        # Count num of non pad *labels* (since they count for loss). Assumes ignore idx == -1.
        non_pad_targets_in_batch = (targets != -1).sum()
        num_non_pad_tokens += non_pad_targets_in_batch
        losses[i] = summed_loss / non_pad_targets_in_batch  # equivalent to nn.cross_entropy w/ reduction="mean"
        logprob_accumulator[i] = summed_loss

    avg_loss = losses.mean()
    summed_corpus_nll = logprob_accumulator.sum()

    # Reduce across all processes
    avg_loss = fabric.all_reduce(avg_loss, reduce_op="mean")
    summed_corpus_nll = fabric.all_reduce(summed_corpus_nll, reduce_op="sum")
    num_non_pad_tokens = fabric.all_reduce(num_non_pad_tokens, reduce_op="sum")

    per_token_perplexity = torch.exp(avg_loss)
    per_token_nll = summed_corpus_nll / num_non_pad_tokens
    num_documents = iters * eval_iter_batch_size
    per_doc_nll = summed_corpus_nll / num_documents

    model.train()
    return {
        "loss": avg_loss,
        "perplexity": per_token_perplexity,
        "per_token_nll": per_token_nll,
        "per_doc_nll": per_doc_nll,
    }


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    setup(simple_parsing.parse(Args, add_config_path_arg=True, argument_generation_mode=""))
