import evaluate
import torch
import torch.distributed as dist
from datasets import load_dataset
from torch.utils.data import DataLoader, DistributedSampler
from transformers import DataCollatorWithPadding

from contrastors.distributed import gather
from contrastors.models import NomicBertConfig, NomicBertForSequenceClassification

from .base import BaseTrainer

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

task_to_problem_type = {
    "cola": "single_label_classification",
    "mnli": "single_label_classification",
    "mrpc": "single_label_classification",
    "qnli": "single_label_classification",
    "qqp": "single_label_classification",
    "rte": "single_label_classification",
    "sst2": "single_label_classification",
    "stsb": "regression",
}

task_to_num_labels = {
    "cola": 2,
    "mnli": 3,
    "mrpc": 2,
    "qnli": 2,
    "qqp": 2,
    "rte": 2,
    "sst2": 2,
    "stsb": 1,
}


class GlueTrainer(BaseTrainer):
    def __init__(self, config, dtype):
        self.task = config.mlm_data_args.task_name
        self.is_regression = task_to_problem_type[self.task] == "regression"
        super(GlueTrainer, self).__init__(config, dtype)

    def get_model(self, config):
        model_config = NomicBertConfig.from_pretrained(config.pretrained)
        model_config.num_labels = task_to_num_labels[self.task]
        model_config.problem_type = "regression" if self.is_regression else "single_label_classification"

        model = NomicBertForSequenceClassification.from_pretrained(
            config.pretrained, config=model_config, ignore_mismatched_sizes=True, strict=False
        )

        if config.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        if self.distributed and not self.deepspeed:
            model = model.to("cuda")
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[dist.get_rank()],
            )

        return {"model": model}

    def get_dataloaders(self, config):
        data_config = config.mlm_data_args
        raw_datasets = load_dataset("glue", data_config.task_name)
        raw_datasets.pop("test", None)

        if data_config.task_name == "mnli":
            raw_datasets.pop("test_matched", None)
            raw_datasets.pop("test_mismatched", None)

        is_regression = data_config.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()

        with self.main_process_first():
            sentence1_key, sentence2_key = task_to_keys[self.task]
            label_to_id = None
            is_regression = self.is_regression
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
                result = self.tokenizer(*texts, padding=padding, max_length=config.model_args.seq_len, truncation=True)

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

        data_collator = DataCollatorWithPadding(self.tokenizer)

        if self.num_processes > 1:
            train_sampler = DistributedSampler(train_dataset, num_replicas=self.num_processes, rank=self.process_index)
            val_sampler = DistributedSampler(eval_dataset, num_replicas=self.num_processes, rank=self.process_index)
        else:
            train_sampler = None
            val_sampler = None

        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=data_collator,
            batch_size=data_config.batch_size,
            sampler=train_sampler,
        )
        val_dataloader = DataLoader(
            eval_dataset, collate_fn=data_collator, batch_size=data_config.batch_size, sampler=val_sampler
        )

        if data_config.task_name == "mnli":
            eval_mm_dataset = processed_datasets["validation_mismatched"]
            val_mm_dataloader = DataLoader(eval_mm_dataset, collate_fn=data_collator, batch_size=data_config.batch_size)

        self.total_num_steps = int(len(train_dataloader) // config.train_args.gradient_accumulation_steps)

        return {
            "train": train_dataloader,
            "val": (val_dataloader, val_mm_dataloader) if data_config.task_name == "mnli" else val_dataloader,
            "test": None,
        }

    def forward_step(self, model, inputs, **kwargs):
        model.train()
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        output = model(**inputs)

        loss = output.loss

        return loss

    def eval_step(self, model, batch, **kwargs):
        batch = {k: v.to(model.device) for k, v in batch.items()}
        output = model(**batch)

        loss = output.loss

        return loss

    def eval_loop(self, model, dataloader, step, val_mm_dataloader=None):
        model.eval()
        data_config = self.config.mlm_data_args
        metric = evaluate.load("glue", data_config.task_name)

        # hacky way to get mnli to work
        if isinstance(dataloader, tuple):
            dataloader, val_mm_dataloader = dataloader

        for curr_step, batch in enumerate(dataloader):
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=self.dtype):
                    outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1) if not self.is_regression else outputs.logits.squeeze()
            predictions = gather(predictions)
            references = gather(batch["labels"])
            # If we are in a multiprocess environment, the last batch has duplicates
            if self.num_processes > 1:
                if curr_step == len(dataloader) - 1:
                    predictions = predictions[: len(dataloader.dataset) - samples_seen]
                    references = references[: len(dataloader.dataset) - samples_seen]
                else:
                    samples_seen += references.shape[0]
            metric.add_batch(
                predictions=predictions,
                references=references,
            )

        if data_config.task_name == "mnli":
            mm_metric = evaluate.load("glue", "mnli")
            for curr_step, batch in enumerate(val_mm_dataloader):
                with torch.no_grad():
                    with torch.autocast(device_type="cuda", dtype=self.dtype):
                        outputs = model(**batch)
                predictions = outputs.logits.argmax(dim=-1) if not self.is_regression else outputs.logits.squeeze()
                predictions = gather(predictions)
                references = gather(batch["labels"])
                # If we are in a multiprocess environment, the last batch has duplicates
                if self.num_processes > 1:
                    if curr_step == len(val_mm_dataloader) - 1:
                        predictions = predictions[: len(val_mm_dataloader.dataset) - samples_seen]
                        references = references[: len(val_mm_dataloader.dataset) - samples_seen]
                    else:
                        samples_seen += references.shape[0]
                mm_metric.add_batch(
                    predictions=predictions,
                    references=references,
                )

        val_metric = metric.compute()
        if self.config.train_args.wandb:
            if data_config.task_name == "mnli":
                self.log({"val_metric": val_metric, "val_mm_metric": mm_metric.compute(), "epoch": step})
            else:
                self.log({"val_metric": val_metric, "epoch": step})
        else:
            self.print({"val_metric": val_metric})
