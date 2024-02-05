import torch
import torch.distributed as dist
from datasets import load_dataset
from torch.utils.data import DataLoader, DistributedSampler
from torchmetrics import MeanMetric
from tqdm import tqdm
from transformers import DataCollatorForLanguageModeling

from contrastors.distributed import gather
from contrastors.models import BertConfig, NomicBertForPreTraining, bert_config_to_nomic_config

from .base import BaseTrainer


# TODO: add deepspeed support/check that it works and then train a mlm bert
class MLMTrainer(BaseTrainer):
    def __init__(self, config, dtype):
        super(MLMTrainer, self).__init__(config, dtype)

    def get_model(self, config):
        hf_config = BertConfig.from_pretrained(config.model_name)
        if hf_config.vocab_size != len(self.tokenizer):
            self.print(f"Resizing model vocab from {hf_config.vocab_size} to {len(self.tokenizer)}")
            hf_config.vocab_size = len(self.tokenizer)
        hf_config.max_position_embeddings = config.seq_len
        hf_config.rotary_emb_fraction = config.rotary_emb_fraction

        hf_config.pad_vocab_to_multiple_of = config.pad_vocab_to_multiple_of
        # use rmsnorm instead of layernorm
        hf_config.use_rms_norm = config.use_rms_norm
        hf_config.hidden_act = config.activation_function
        hf_config.qkv_proj_bias = config.qkv_proj_bias
        hf_config.mlp_fc1_bias = config.mlp_fc1_bias
        hf_config.mlp_fc2_bias = config.mlp_fc2_bias
        hf_config.attention_probs_dropout_prob = config.attn_pdrop

        model_config = bert_config_to_nomic_config(hf_config)
        model = NomicBertForPreTraining(model_config)

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
        train_args = config.train_args
        data_config = config.mlm_data_args
        with self.main_process_first():
            dataset = load_dataset(data_config.tokenized_dataset, split="train")
            tokenized_datasets = dataset.shuffle(seed=data_config.seed)
            split = tokenized_datasets.train_test_split(test_size=data_config.val_pct, seed=data_config.seed)
            train_tokenized, val_tokenized = split["train"], split["test"]

        if self.num_processes > 1:
            train_sampler = DistributedSampler(
                train_tokenized, num_replicas=self.num_processes, rank=self.process_index
            )
            val_sampler = DistributedSampler(val_tokenized, num_replicas=self.num_processes, rank=self.process_index)
        else:
            train_sampler = None
            val_sampler = None

        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=data_config.mlm_prob)

        train_dataloader = DataLoader(
            train_tokenized,
            batch_size=data_config.batch_size // self.num_processes,
            shuffle=True if train_sampler is None else False,
            num_workers=data_config.workers,
            collate_fn=data_collator,
            drop_last=True,
            persistent_workers=True,
            sampler=train_sampler,
        )

        eval_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm_probability=data_config.val_mlm_prob
        )
        val_dataloader = DataLoader(
            val_tokenized,
            batch_size=data_config.batch_size // self.num_processes,
            shuffle=False,
            num_workers=data_config.workers,
            collate_fn=eval_collator,
            drop_last=True,
            persistent_workers=True,
            sampler=val_sampler,
        )

        self.total_num_steps = int(len(train_dataloader) // train_args.gradient_accumulation_steps)

        return {"train": train_dataloader, "val": val_dataloader, "test": None}

    def forward_step(self, model, inputs, **kwargs):
        model.train()
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        output = model(**inputs)

        loss = output.loss

        return loss

    def eval_step(self, model, batch, **kwargs):
        batch = {k: v.to(model.device) for k, v in batch.items()}
        output = model(**batch, **kwargs)

        loss = output.loss

        return loss

    def eval_loop(self, model, dataloader, step):
        train_args = self.config.train_args
        val_loss = MeanMetric(nan_strategy="error").to(model.device)
        model.eval()
        for batch in tqdm(dataloader, desc=f"Eval epoch step {step}"):
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=self.dtype):
                    loss = self.eval_step(model, batch)

            loss = gather(loss.detach().float())
            val_loss.update(loss)

        val_loss = val_loss.compute()
        ppl = torch.exp(val_loss)
        if train_args.wandb:
            self.log({"val_loss": val_loss, "val_ppl": ppl}, step=step)
        else:
            self.print({"val_loss": val_loss, "val_ppl": ppl})
