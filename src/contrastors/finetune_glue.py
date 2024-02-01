import json
import os
from argparse import ArgumentParser

import evaluate
import torch
import yaml
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import DummyOptim, DummyScheduler, set_seed
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertConfig,
    DataCollatorWithPadding,
    get_scheduler,
)

from contrastors.models.encoder import NomicBertConfig, NomicBertForSequenceClassification
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


task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def train(accelerator, args):
    train_args = args.train_args

    data_config = args.data_args
    model_args = args.model_args

    set_seed(data_config.seed)

    accelerator.print(json.dumps(args.dict(), indent=4))
    accelerator.print(f"Using {accelerator.num_processes} GPUs")

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name,
    )
    tokenizer.model_max_length = model_args.seq_len
    # if no pad token, set it to eos
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.cls_token is None:
        tokenizer.add_special_tokens({"cls_token": "<s>"})

    raw_datasets = load_dataset("glue", data_config.task_name)
    raw_datasets.pop("test", None)
    if data_config.task_name == "mnli":
        raw_datasets.pop("test_matched", None)
        raw_datasets.pop("test_mismatched", None)
    is_regression = data_config.task_name == "stsb"
    if not is_regression:
        label_list = raw_datasets["train"].unique("label")
        label_list.sort()
        num_labels = len(label_list)
    else:
        num_labels = 1

    config = NomicBertConfig.from_pretrained(model_args.pretrained)
    config.num_labels = num_labels
    config.problem_type = "regression" if is_regression else "single_label_classification"

    model = NomicBertForSequenceClassification.from_pretrained(
        model_args.pretrained, config=config, ignore_mismatched_sizes=True, strict=False
    )

    with accelerator.main_process_first():
        sentence1_key, sentence2_key = task_to_keys[data_config.task_name]
        label_to_id = None
        if not is_regression:
            label_to_id = {v: i for i, v in enumerate(label_list)}

        padding = False

        def preprocess_function(examples):
            # Tokenize the texts
            texts = (
                (examples[sentence1_key],)
                if sentence2_key is None
                else (examples[sentence1_key], examples[sentence2_key])
            )
            result = tokenizer(*texts, padding=padding, max_length=model_args.seq_len, truncation=True)

            if "label" in examples:
                if label_to_id is not None:
                    # Map labels to IDs (not necessary for GLUE tasks)
                    result["labels"] = [label_to_id[l] for l in examples["label"]]
                else:
                    # In all cases, rename the column to labels because the model will expect that.
                    result["labels"] = examples["label"]
            return result

        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
        )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation_matched" if data_config.task_name == "mnli" else "validation"]

    data_collator = DataCollatorWithPadding(tokenizer)

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=data_config.batch_size
    )
    val_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=data_config.batch_size)

    if data_config.task_name == "mnli":
        eval_mm_dataset = processed_datasets["validation_mismatched"]
        val_mm_dataloader = DataLoader(eval_mm_dataset, collate_fn=data_collator, batch_size=data_config.batch_size)

    accelerator.print(f"Training a {model.num_parameters():,} parameter model")
    accelerator.print(f"Trainable parameters: {model.num_parameters(only_trainable=True):,}")
    accelerator.print(model)

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

    if accelerator.state.deepspeed_plugin is not None:
        gradient_accumulation_steps = accelerator.gradient_accumulation_steps
    else:
        gradient_accumulation_steps = getattr(train_args, "gradient_accumulation_steps", 1)

    if getattr(train_args, "warmup_steps", None):
        warmup_steps = train_args.warmup_steps
    else:
        warmup_steps = int(len(train_dataloader) * train_args.num_epochs * train_args.warmup_pct)

    accelerator.print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    accelerator.print(f"Warmup steps: {warmup_steps}")
    accelerator.print(f"Total steps: {len(train_dataloader) * train_args.num_epochs}")

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
            num_training_steps=len(train_dataloader) * train_args.num_epochs,
        )
    else:
        scheduler = DummyScheduler(optimizer, total_num_steps=len(train_dataloader), warmup_num_steps=warmup_steps)

    if data_config.task_name == "mnli":
        model, train_dataloader, val_dataloader, val_mm_dataloader, optimizer, scheduler = accelerator.prepare(
            model, train_dataloader, val_dataloader, val_mm_dataloader, optimizer, scheduler
        )
    else:
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
        resume_step = int(training_difference.replace("checkpoint_", ""))
        accelerator.print(f"Resuming from step {resume_step}")
    else:
        resume_step = 0

    accelerator.wait_for_everyone()

    metric = evaluate.load("glue", data_config.task_name)

    for epoch in range(0, train_args.num_epochs):
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

            output = model(**batch)
            loss = output.loss
            accelerator.backward(loss)

            # clip gradients
            if accelerator.sync_gradients:
                if train_args.max_grad_norm is not None and train_args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), train_args.max_grad_norm)

            if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
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

        accelerator.print(f"Epoch {epoch} finished")
        samples_seen = 0
        model.eval()
        for step, batch in enumerate(val_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
            predictions, references = accelerator.gather((predictions, batch["labels"]))
            # If we are in a multiprocess environment, the last batch has duplicates
            if accelerator.num_processes > 1:
                if step == len(val_dataloader) - 1:
                    predictions = predictions[: len(val_dataloader.dataset) - samples_seen]
                    references = references[: len(val_dataloader.dataset) - samples_seen]
                else:
                    samples_seen += references.shape[0]
            metric.add_batch(
                predictions=predictions,
                references=references,
            )

        if data_config.task_name == "mnli":
            mm_metric = evaluate.load("glue", "mnli")
            for step, batch in enumerate(val_mm_dataloader):
                with torch.no_grad():
                    outputs = model(**batch)
                predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
                predictions, references = accelerator.gather((predictions, batch["labels"]))
                # If we are in a multiprocess environment, the last batch has duplicates
                if accelerator.num_processes > 1:
                    if step == len(val_mm_dataloader) - 1:
                        predictions = predictions[: len(val_mm_dataloader.dataset) - samples_seen]
                        references = references[: len(val_mm_dataloader.dataset) - samples_seen]
                    else:
                        samples_seen += references.shape[0]
                mm_metric.add_batch(
                    predictions=predictions,
                    references=references,
                )

        val_metric = metric.compute()
        if train_args.wandb:
            if data_config.task_name == "mnli":
                accelerator.log({"val_metric": val_metric, "val_mm_metric": mm_metric.compute(), "epoch": epoch})
            else:
                accelerator.log({"val_metric": val_metric, "epoch": epoch})
        else:
            accelerator.print({"val_metric": val_metric})

        if train_args.save_every > 0:
            # discard classifier head for other tasks that have more labels than this task
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                f"ckpts/{train_args.run_name}/epoch_{epoch}",
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
                state_dict=accelerator.get_state_dict(model),
            )

    if train_args.wandb:
        accelerator.end_training()


if __name__ == "__main__":
    # parse arguments by reading in a config
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--batch_size", type=int)

    args = parser.parse_args()

    config = read_config(args.config)

    if args.seed is not None:
        config.data_args.seed = args.seed
    if args.learning_rate is not None:
        config.train_args.learning_rate = args.learning_rate

    if args.batch_size is not None:
        config.data_args.batch_size = args.batch_size

    task = config.data_args.task_name
    model_name = config.model_args.pretrained.split("/")[-2]
    run_name = f"{task}_{model_name}_seed_{config.data_args.seed}_lr_{config.train_args.learning_rate}_bs_{config.data_args.batch_size}"
    config.train_args.run_name = run_name

    ddp_kwargs = DistributedDataParallelKwargs(broadcast_buffers=False)

    if config.train_args.wandb:
        # turn nested dict into flat dict for better logging
        hyperparams = {}
        for key, params in config.dict().items():
            for k, v in params.items():
                hyperparams[f"{key}_{k}"] = v

        accelerator = Accelerator(log_with="wandb", kwargs_handlers=[ddp_kwargs])
        accelerator.init_trackers(
            project_name=config.train_args.wandb_project_name,
            config=hyperparams,
            init_kwargs={"wandb": {"entity": config.train_args.wandb_entity, "name": run_name}},
        )
    else:
        accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    train(accelerator, args=config)
