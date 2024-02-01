import json
import os
from argparse import ArgumentParser

import torch
import yaml
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import DummyOptim, DummyScheduler, convert_model, set_seed
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric
from tqdm import tqdm
from transformers import AutoTokenizer, BertConfig, DataCollatorForLanguageModeling, get_scheduler

from contrastors.models.encoder import NomicBertForPreTraining, bert_config_to_nomic_config
from contrastors.read import read_config


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

    optimizer = AdamW(optim_groups, betas=(args.adam_beta1, args.adam_beta2), eps=args.eps)
    return optimizer


def train(accelerator, args):
    train_args = args.train_args

    data_config = args.data_args
    model_args = args.model_args

    set_seed(data_config.seed)

    accelerator.print(json.dumps(args.dict(), indent=4))
    accelerator.print(f"Using {accelerator.num_processes} GPUs")

    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, add_cls_token=True)
    tokenizer.model_max_length = model_args.seq_len
    # if no pad token, set it to eos
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.cls_token is None:
        tokenizer.add_special_tokens({"cls_token": "<s>"})

    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": "<mask>"})

    hf_config = BertConfig.from_pretrained(model_args.model_name)
    if hf_config.vocab_size != len(tokenizer):
        accelerator.print(f"Resizing model vocab from {hf_config.vocab_size} to {len(tokenizer)}")
        hf_config.vocab_size = len(tokenizer)
    hf_config.max_position_embeddings = model_args.seq_len
    hf_config.rotary_emb_fraction = model_args.rotary_emb_fraction

    hf_config.pad_vocab_to_multiple_of = model_args.pad_vocab_to_multiple_of
    # use rmsnorm instead of layernorm
    hf_config.use_rms_norm = model_args.use_rms_norm
    hf_config.hidden_act = model_args.activation_function
    hf_config.qkv_proj_bias = model_args.qkv_proj_bias
    hf_config.mlp_fc1_bias = model_args.mlp_fc1_bias
    hf_config.mlp_fc2_bias = model_args.mlp_fc2_bias
    hf_config.attention_probs_dropout_prob = model_args.attn_pdrop

    config = bert_config_to_nomic_config(hf_config)
    model = NomicBertForPreTraining(config)

    if train_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    accelerator.print(f"Training a {model.num_parameters():,} parameter model")
    accelerator.print(f"Trainable parameters: {model.num_parameters(only_trainable=True):,}")
    accelerator.print(model)

    checkpoint = train_args.gradient_checkpointing

    with accelerator.main_process_first():
        tokenized_datasets = load_dataset(data_config.tokenized_dataset, split="train")
        # shuffle dataset
        tokenized_datasets = tokenized_datasets.shuffle(seed=data_config.seed)
        split = tokenized_datasets.train_test_split(test_size=data_config.val_pct, seed=data_config.seed)
        train_tokenized, val_tokenized = split["train"], split["test"]

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=data_config.mlm_prob)

    train_dataloader = DataLoader(
        train_tokenized,
        batch_size=data_config.batch_size // accelerator.num_processes,
        shuffle=True,
        num_workers=4,
        collate_fn=data_collator,
        drop_last=True,
        persistent_workers=True,
    )

    # eval with lower mlm prob
    eval_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=data_config.val_mlm_prob)
    val_dataloader = DataLoader(
        val_tokenized,
        batch_size=data_config.batch_size // accelerator.num_processes,
        shuffle=False,
        num_workers=4,
        collate_fn=eval_collator,
        drop_last=True,
        persistent_workers=True,
    )

    accelerator.print(f"Train dataset: {len(train_tokenized):,}")
    accelerator.print(f"Total Tokens: {len(train_tokenized)*tokenizer.model_max_length:,}")

    if checkpoint:
        model.gradient_checkpointing_enable()

    if (
        accelerator.state.deepspeed_plugin is None
        or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        accelerator.print(f"Using AdamW optimizer")
        optimizer = configure_optimizer([model], train_args)
    else:
        accelerator.print(f"Using DeepSpeed optimizer")
        optimizer = DummyOptim(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=train_args.learning_rate,
            weight_decay=train_args.weight_decay,
            eps=train_args.eps,
        )

    gradient_accumulation_steps = accelerator.gradient_accumulation_steps

    if getattr(train_args, "warmup_steps", None):
        warmup_steps = train_args.warmup_steps
    else:
        warmup_steps = int(
            len(train_dataloader) // gradient_accumulation_steps * train_args.num_epochs * train_args.warmup_pct
        )

    accelerator.print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    accelerator.print(f"Dataloader length: {len(train_dataloader)}")
    accelerator.print(f"Warmup steps: {warmup_steps}")
    accelerator.print(
        f"Total batches: {len(train_dataloader) // gradient_accumulation_steps // accelerator.num_processes * train_args.num_epochs}"
    )

    if (
        accelerator.state.deepspeed_plugin is None
        or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        scheduler = get_scheduler(
            name=train_args.schedule_type,
            optimizer=optimizer,
            num_warmup_steps=(
                warmup_steps
                if accelerator.state.deepspeed_plugin is None
                else warmup_steps // accelerator.num_processes
            ),
            # this is only used for linear decay when we decay from last lr to 0
            num_training_steps=(
                len(train_dataloader) * train_args.num_epochs // accelerator.gradient_accumulation_steps
                if accelerator.state.deepspeed_plugin is None
                else len(train_dataloader)
                * train_args.num_epochs
                // accelerator.num_processes
                // accelerator.gradient_accumulation_steps
            ),
        )
    else:
        scheduler = DummyScheduler(
            optimizer, total_num_steps=len(train_dataloader), warmup_num_steps=warmup_steps // accelerator.num_processes
        )

    model, train_dataloader, val_dataloader, optimizer, scheduler = accelerator.prepare(
        model, train_dataloader, val_dataloader, optimizer, scheduler
    )

    # setup for saving training states in case preemption
    accelerator.register_for_checkpointing(scheduler)

    if train_args.checkpoint is not None:
        accelerator.load_state(train_args.checkpoint)
        accelerator.print(f"Resumed from checkpoint: {train_args.checkpoint}")
        path = os.path.basename(train_args.checkpoint)
        training_difference = os.path.splitext(path)[0]
        resume_epoch = int(training_difference.replace("epoch_", "")) + 1
        resume_step = 0
        accelerator.print(f"Resuming from step {resume_epoch}")
    else:
        resume_step = 0
        resume_epoch = 0

    accelerator.wait_for_everyone()

    for epoch in range(resume_epoch, train_args.num_epochs):
        accelerator.print(f"Len of train_dataloader: {len(train_dataloader)}")
        total_num_steps = len(train_dataloader) * (train_args.num_epochs)
        accelerator.print(f"Total training steps: {total_num_steps}")

        max_steps = len(train_dataloader)

        progbar = tqdm(train_dataloader, desc=f"Epoch {epoch} step {resume_step}", initial=resume_step, total=max_steps)

        offset = resume_step
        if train_args.checkpoint is not None and resume_step > 0:
            offset += 1

        for step, batch in enumerate(progbar):
            # if using deepspeed, it handles gradient accumulation
            curr_step = epoch * len(train_dataloader) + step + offset

            model.train()
            with accelerator.accumulate(model):
                output = model(**batch)
                loss = output.loss
                accelerator.backward(loss)

                # clip gradients
                if accelerator.sync_gradients:
                    if train_args.max_grad_norm is not None and train_args.max_grad_norm > 0:
                        accelerator.clip_grad_norm_(model.parameters(), train_args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                loss_values = accelerator.gather_for_metrics({"loss": loss.detach().float()})

            if train_args.wandb:
                accelerator.log({"loss": torch.mean(loss_values["loss"]).item()}, step=curr_step)
            else:
                accelerator.print(torch.mean(loss_values["loss"]).item())
                accelerator.print(f"LR: {scheduler.get_last_lr()[0]}")

            # log LR in case something weird happens
            if step > 0 and step % (train_args.log_lr_every) == 0:
                if train_args.wandb:
                    accelerator.log({"lr": scheduler.get_last_lr()[0]}, step=curr_step)

            if step > 0 and train_args.save_every > 0 and step % train_args.save_every == 0:
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(
                    f"{train_args.output_dir}/step_{curr_step}_model",
                    is_main_process=accelerator.is_main_process,
                    save_function=accelerator.save,
                    state_dict=accelerator.get_state_dict(model),
                )
                accelerator.save_state(f"{train_args.output_dir}/checkpoint_{curr_step}")

            if step > 0 and train_args.eval_every > 0 and step % train_args.eval_every == 0:
                val_loss = MeanMetric(nan_strategy="error").to(model.device)
                model.eval()
                for step, batch in enumerate(tqdm(val_dataloader, desc=f"Eval epoch {epoch} step {step}")):
                    with torch.no_grad():
                        outputs = model(**batch)

                    loss = outputs.loss
                    loss_values = accelerator.gather_for_metrics({"loss": loss.detach().float()})
                    val_loss.update(loss_values["loss"])

                val_loss = val_loss.compute()
                ppl = torch.exp(val_loss)
                if train_args.wandb:
                    accelerator.log({"val_loss": val_loss, "val_ppl": ppl}, step=curr_step)
                else:
                    accelerator.print({"val_loss": val_loss, "val_ppl": ppl})

        accelerator.print(f"Epoch {epoch} finished")
        if train_args.eval_every > 0:
            val_loss = MeanMetric(nan_strategy="error").to(model.device)
            model.eval()
            for step, batch in enumerate(tqdm(val_dataloader, desc=f"Eval end of epoch {epoch}")):
                with torch.no_grad():
                    outputs = model(**batch)

                loss = outputs.loss
                loss_values = accelerator.gather_for_metrics({"loss": loss.detach().float()})
                val_loss.update(loss_values["loss"])

            val_loss = val_loss.compute()
            ppl = torch.exp(val_loss)
            if train_args.wandb:
                accelerator.log({"val_loss": val_loss, "val_ppl": ppl}, step=curr_step)
            else:
                accelerator.print({"val_loss": val_loss, "val_ppl": ppl})

        if train_args.save_every > 0:
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                f"{train_args.output_dir}/epoch_{epoch}_model",
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
                state_dict=accelerator.get_state_dict(model),
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
            for k, v in params.items():
                hyperparams[f"{key}_{k}"] = v

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
