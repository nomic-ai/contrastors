import json
import logging
import os
from argparse import ArgumentParser

import numpy as np
import torch
import torch.distributed as dist
import yaml
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import DummyOptim, set_seed
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric
from tqdm import tqdm
from transformers import AutoTokenizer, get_scheduler

from contrastors.dataset.torch_loader import StreamingShardDataset, collate_fn, get_local_dataloader
from contrastors.distributed import gather
from contrastors.loss import clip_loss, grad_cache_loss_biencoder
from contrastors.models.biencoder import BiEncoder, BiEncoderConfig
from contrastors.read import read_config

logger = logging.getLogger('s3fs')
logger.setLevel(logging.DEBUG)  # Set the logging level to DEBUG

# Create a console handler and set its level to DEBUG
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# Create a formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add the formatter to the console handler
ch.setFormatter(formatter)

# Add the console handler to the logger
logger.addHandler(ch)


def format_metrics(metrics, split, prefix=""):
    log = f"[{split}]" + prefix
    log += " ".join([f"{key}: {value:.4f}" for key, value in metrics.items()])

    return log


# adapted from https://github.com/karpathy/minGPT/commit/bbbdac74fa9b2e55574d70056163ffbae42310c1#diff-2075fa9c224b395be5bda85544dd36572b59c76c54562819eadadbf268602834R157s
# and using similar logic from openclip
def configure_optimizer(modules, args):
    decay = set()
    no_decay = set()
    blacklist_weight_modules = (torch.nn.LayerNorm,)
    named_parameters = [(name, param) for model in modules for name, param in model.named_parameters()]
    for name, param in named_parameters:
        if not param.requires_grad:
            continue
        # YUCK!!!
        if param.squeeze().ndim < 2:
            no_decay.add(name)
        elif "bias" in name:
            no_decay.add(name)
        elif isinstance(param, blacklist_weight_modules):
            no_decay.add(name)
        elif "logit_scale" in name:
            no_decay.add(name)
        else:
            decay.add(name)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in named_parameters if p.requires_grad}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
    assert (
        len(param_dict.keys() - union_params) == 0
    ), "parameters %s were not separated into either decay/no_decay set!" % (str(param_dict.keys() - union_params),)

    # create the pytorch optimizer object
    optim_groups = [
        {
            "params": [param_dict[pn] for pn in sorted(list(decay))],
            "weight_decay": args.weight_decay,
            "lr": args.learning_rate,
        },
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0, "lr": args.learning_rate},
        # if we want different learning rates for the projection layer and encoder
    ]

    optimizer = AdamW(optim_groups, betas=(args.adam_beta1, args.adam_beta2))
    return optimizer


def train(accelerator, args):
    train_args = args.train_args

    data_config = args.data_args
    model_args = args.model_args

    set_seed(data_config.seed)

    accelerator.print(json.dumps(args.dict(), indent=4))
    accelerator.print(f"Using {accelerator.num_processes} GPUs")

    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name)
    tokenizer.model_max_length = model_args.seq_len
    # if no pad token, set it to eos
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.cls_token is None:
        tokenizer.add_special_tokens({"cls_token": "<s>"})

    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": "<mask>"})

    if model_args.pretrained is None:
        config = BiEncoderConfig(
            model_name=model_args.model_name,
            pooling=model_args.pooling,
            logit_scale=train_args.logit_scale,
            encoder=model_args.encoder,
        )
        model = BiEncoder(config)
        # DDP requires all trainable params to be used in forward pass
        # we pull the logit scale out of the module since it's not explicitly used in every forward for grad cache
        # and we will get an error otherwise
        scale = train_args.logit_scale
    # load finetuned model to cooldown lr to 0
    else:
        accelerator.print(f"Loading pretrained model from {model_args.pretrained}")
        config = BiEncoderConfig.from_pretrained(f"{model_args.pretrained}_model")
        if model_args.projection_dim is not None:
            config.projection_dim = model_args.projection_dim
        model = BiEncoder.from_pretrained(f"{model_args.pretrained}_model", config=config)
        scale = train_args.logit_scale

    accelerator.print(f"Training a {model.num_parameters():,} parameter model")
    accelerator.print(f"Trainable parameters: {model.num_parameters(only_trainable=True):,}")
    accelerator.print(model)

    checkpoint = train_args.gradient_checkpointing

    if checkpoint:
        model.gradient_checkpointing_enable()

    if (
        accelerator.state.deepspeed_plugin is None
        or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        optimizer = configure_optimizer([model], train_args)
        # optimizer = Adafactor(model.parameters(), lr=train_args.learning_rate, scale_parameter=False, relative_step=False, warmup_init=False)
    else:
        optimizer = DummyOptim(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=train_args.learning_rate,
            weight_decay=train_args.weight_decay,
        )

    gradient_accumulation_steps = accelerator.gradient_accumulation_steps

    if accelerator.state.deepspeed_plugin is not None:
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = (
            data_config.batch_size // accelerator.num_processes
        )

    if data_config.streaming:
        train_dataset = StreamingShardDataset(
            data_config.input_shards,
            data_config.batch_size,
            tokenizer,
            seed=data_config.seed,
            add_eos=model_args.encoder != True,
            add_prefix=model_args.add_prefix,
            num_negatives=model_args.num_negatives,
            download_locally=data_config.download,
            process_one_shard=data_config.process_one_shard,
            weighted_sampling=data_config.weighted_sampling,
            verbose=data_config.verbose,
        )
        if train_args.checkpoint is not None:
            print(f"Loading dataloader state from {train_args.checkpoint}")
            train_dataset.load_state(train_args.checkpoint)

        train_dataloader = DataLoader(train_dataset, batch_size=1, collate_fn=collate_fn, num_workers=0)
        accelerator.print(f"Len of train_dataloader: {len(train_dataset)}")
        # round down in case
        total_num_steps = int(len(train_dataset) / gradient_accumulation_steps // data_config.batch_size)
    else:
        # config defines global batch size
        if data_config.batch_size % accelerator.num_processes != 0:
            raise ValueError(
                f"Batch size {data_config.batch_size} must be divisible by accelerator.num_processes {accelerator.num_processes}"
            )

        batch_size = int(data_config.batch_size / accelerator.num_processes)
        train_dataloader = get_local_dataloader(
            data_config.input_shards,
            batch_size,
            tokenizer,
            seed=data_config.seed,
            num_negatives=model_args.num_negatives,
            add_prefix=model_args.add_prefix,
            num_workers=data_config.workers,
            epoch=0,
        )

        accelerator.print(f"Len of train_dataloader: {len(train_dataloader.dataset)}")
        total_num_steps = int(len(train_dataloader.dataset) / gradient_accumulation_steps // data_config.batch_size)

    scheduler = get_scheduler(
        name=train_args.schedule_type,
        optimizer=optimizer,
        num_warmup_steps=train_args.warmup_steps * accelerator.num_processes,
        # this is only used for linear decay when we decay from last lr to 0
        num_training_steps=(
            total_num_steps * accelerator.num_processes * train_args.num_epochs
            if train_args.schedule_type != "inverse_sqrt"
            else None
        ),
    )

    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    # setup for saving training states in case preemption
    accelerator.register_for_checkpointing(scheduler)

    if train_args.checkpoint is not None:
        accelerator.load_state(train_args.checkpoint)
        accelerator.print(f"Resumed from checkpoint: {train_args.checkpoint}")
        path = os.path.basename(train_args.checkpoint)
        training_difference = os.path.splitext(path)[0]
        resume_step = int(training_difference.replace("checkpoint_", ""))
        accelerator.print(f"Resuming from step {resume_step}")
    else:
        resume_step = 0

    accelerator.wait_for_everyone()

    if train_args.loss_fn not in ["clip", "gte"]:
        raise ValueError(f"Invalid loss function: {train_args.loss_fn}")

    for epoch in range(0, train_args.num_epochs):
        if epoch > 0 and data_config.streaming is False:
            train_dataloader.sampler.set_epoch(epoch)
        elif epoch > 0 and data_config.streaming is True:
            train_dataset = StreamingShardDataset(
                data_config.input_shards,
                data_config.batch_size,
                tokenizer,
                seed=data_config.seed + epoch,
                add_eos=model_args.encoder != True,
                add_prefix=model_args.add_prefix,
                num_negatives=model_args.num_negatives,
                download_locally=data_config.download,
                process_one_shard=data_config.process_one_shard,
                weighted_sampling=data_config.weighted_sampling,
                verbose=data_config.verbose,
            )

            train_dataloader = DataLoader(train_dataset, batch_size=1, collate_fn=collate_fn, num_workers=0)

        train_loss = MeanMetric(nan_strategy="error").to(model.device)
        accelerator.print(f"Total training steps: {total_num_steps}")

        progbar = tqdm(
            train_dataloader, desc=f"Epoch {epoch} step {resume_step}", initial=resume_step, total=total_num_steps
        )

        offset = resume_step
        if train_args.checkpoint is not None and resume_step > 0:
            offset += 1

        for step, batch in enumerate(progbar):
            # if using deepspeed, it handles gradient accumulation
            curr_step = epoch * total_num_steps + step + offset

            model.train()

            # if mixing datasets, we can't rely on this as each row will be different
            # we could make our code better and handle this, but it's a bit of a pain
            dataset_name = batch.pop("dataset_name", "")

            if train_args.grad_cache is False:
                query_outputs = model(
                    input_ids=batch["query_input_ids"].to(model.device),
                    attention_mask=batch["query_attention_mask"].to(model.device),
                )
                document_outputs = model(
                    input_ids=batch["document_input_ids"].to(model.device),
                    attention_mask=batch["document_attention_mask"].to(model.device),
                )
                if "negative_input_ids" in batch:
                    negative_outputs = model(
                        input_ids=batch["negative_input_ids"].to(model.device),
                        attention_mask=batch["negative_attention_mask"].to(model.device),
                    )

                queries = gather(query_outputs["embedding"])
                documents = gather(document_outputs["embedding"])
                if "negative_input_ids" in batch:
                    negatives = gather(negative_outputs["embedding"])
                else:
                    negatives = None

                # gather across all processes
                # this concats on first dim
                # and preserves gradients
                loss = clip_loss(
                    queries,
                    documents,
                    step=step + epoch * total_num_steps,
                    logit_scale=scale,
                    negatives=negatives,
                    kd_scores=batch.get("kd_scores", None),
                    tokenizer=tokenizer,
                    accelerator=accelerator,
                    inputs=batch,
                    dataset=dataset_name,
                )
                accelerator.backward(loss)

            else:
                # this will calculate the loss, handle gathering across multi-gpu
                # and accumulate gradients and also calculate the backwards pass
                batch = {k: v.to(model.device) for k, v in batch.items()}
                loss = grad_cache_loss_biencoder(
                    model, batch, train_args.chunk_size, logit_scale=scale, loss_fn_name=train_args.loss_fn
                )

            # clip gradients
            if accelerator.sync_gradients:
                if train_args.max_grad_norm is not None and train_args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), train_args.max_grad_norm)

            if (step + 1) % gradient_accumulation_steps == 0 or step == total_num_steps - 1:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            # clamp like original clip
            if train_args.clamp_logits:
                with torch.no_grad():
                    scale.module.logit_scale.clamp_(0, np.log(train_args.logit_max))

            loss_values = accelerator.gather_for_metrics({"loss": loss.detach().float()})
            if train_args.wandb:
                accelerator.log({"loss": torch.mean(loss_values["loss"]).item()}, step=curr_step)
            else:
                accelerator.print(f'Loss: {torch.mean(loss_values["loss"]).item()}')
                accelerator.print(f"LR: {scheduler.get_last_lr()[0]}")

            train_loss.update(loss_values["loss"])

            # log LR in case something weird happens
            if step > 0 and step % (train_args.log_lr_every) == 0:
                if train_args.wandb:
                    accelerator.log({"lr": scheduler.get_last_lr()[0]}, step=curr_step)
                    if isinstance(scale, torch.nn.Module):
                        accelerator.log({"logit_scale": scale.module.logit_scale.exp().item()}, step=curr_step)

            if step > 0 and train_args.save_every > 0 and step % train_args.save_every == 0:
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(
                    f"{train_args.output_dir}/step_{curr_step}_model",
                    is_main_process=accelerator.is_main_process,
                    save_function=accelerator.save,
                    state_dict=accelerator.get_state_dict(model),
                )
                if isinstance(scale, torch.nn.Module):
                    unwrapped_scale = accelerator.unwrap_model(scale)
                    unwrapped_scale.save_pretrained(
                        f"{train_args.output_dir}/step_{curr_step}_logit",
                        is_main_process=accelerator.is_main_process,
                        save_function=accelerator.save,
                        state_dict=accelerator.get_state_dict(scale),
                    )
                accelerator.save_state(f"{train_args.output_dir}/checkpoint_{curr_step}")

                # save the files we've seen already in case of preemption
                ds_state = data_config.input_shards.replace(".yaml", "")
                with open(f"{ds_state}/rank_{dist.get_rank()}_processed.json", "r") as f:
                    processed = json.load(f)

                with open(
                    f"{train_args.output_dir}/checkpoint_{curr_step}/rank_{dist.get_rank()}_processed.json", "w"
                ) as f:
                    json.dump(processed, f, indent=3)

        accelerator.print(f"Epoch {epoch} finished")

        if train_args.save_every > 0:
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                f"{train_args.output_dir}/epoch_{epoch}_model",
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
                state_dict=accelerator.get_state_dict(model),
            )
            if isinstance(scale, torch.nn.Module):
                unwrapped_scale = accelerator.unwrap_model(scale)
                unwrapped_scale.save_pretrained(
                    f"{train_args.output_dir}/epoch_{epoch}_logit",
                    is_main_process=accelerator.is_main_process,
                    save_function=accelerator.save,
                    state_dict=accelerator.get_state_dict(scale),
                )
            accelerator.save_state(f"{train_args.output_dir}/epoch_{epoch}")

    if train_args.num_epochs > 0:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            f"{train_args.output_dir}/final_model",
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=accelerator.get_state_dict(model),
        )
        if isinstance(scale, torch.nn.Module):
            unwrapped_scale = accelerator.unwrap_model(scale)
            unwrapped_scale.save_pretrained(
                f"{train_args.output_dir}/final_logit",
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
                state_dict=accelerator.get_state_dict(scale),
            )
        accelerator.save_state(f"{train_args.output_dir}/final")

    accelerator.end_training()


if __name__ == "__main__":
    # parse arguments by reading in a config
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")

    args = parser.parse_args()

    config = read_config(args.config)

    ddp_kwargs = DistributedDataParallelKwargs(broadcast_buffers=False)

    if config.train_args.wandb:
        # turn nested dict into flat dict for better logging
        hyperparams = {}
        for key, params in config.dict().items():
            if params is None:
                continue
            for k, v in params.items():
                hyperparams[f"{key}_{k}"] = v

        data_file = config.data_args.input_shards
        data_mix = yaml.safe_load(data_file)
        hyperparams["data_mix"] = data_mix
        accelerator = Accelerator(
            log_with="wandb",
            kwargs_handlers=[ddp_kwargs],
            gradient_accumulation_steps=getattr(config.train_args, "gradient_accumulation_steps", 1),
        )
        accelerator.init_trackers(
            project_name=config.train_args.wandb_project_name,
            config=hyperparams,
            init_kwargs={"wandb": {"entity": config.train_args.wandb_entity}},
        )
    else:
        accelerator = Accelerator(
            kwargs_handlers=[ddp_kwargs],
            gradient_accumulation_steps=getattr(config.train_args, "gradient_accumulation_steps", 1),
        )

    train(accelerator, args=config)
