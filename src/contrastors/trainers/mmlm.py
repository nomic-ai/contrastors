import torch
import torch.distributed as dist
from datasets import load_dataset
from torch.utils.data import DataLoader, DistributedSampler
from torchmetrics import MeanMetric
from tqdm import tqdm
from transformers import DataCollatorForLanguageModeling

from contrastors.distributed import gather, print_in_order
from contrastors.models import BertConfig, NomicBertForPreTraining, bert_config_to_nomic_config
from contrastors.dataset.multilingual import DistributedIterableMLMDataset, EvalDistributedIterableMLMDataset

from .base import BaseTrainer


# TODO: add deepspeed support/check that it works and then train a mlm bert
class MMLMTrainer(BaseTrainer):
    def __init__(self, config, dtype):
        super(MMLMTrainer, self).__init__(config, dtype)

    def get_model(self, config):
        config = config.model_args
        hf_config = BertConfig.from_pretrained(config.model_name)
        if hf_config.vocab_size != len(self.tokenizer):
            self.print(f"Resizing model vocab from {hf_config.vocab_size} to {len(self.tokenizer)}")
            hf_config.vocab_size = len(self.tokenizer)
        hf_config.max_position_embeddings = config.seq_len
        hf_config.rotary_emb_fraction = config.rotary_emb_fraction
        hf_config.rotary_emb_base = config.rotary_emb_base

        hf_config.pad_vocab_to_multiple_of = config.pad_vocab_to_multiple_of
        # use rmsnorm instead of layernorm
        hf_config.use_rms_norm = config.use_rms_norm
        hf_config.hidden_act = config.activation_function
        hf_config.qkv_proj_bias = config.qkv_proj_bias
        hf_config.mlp_fc1_bias = config.mlp_fc1_bias
        hf_config.mlp_fc2_bias = config.mlp_fc2_bias
        hf_config.attention_probs_dropout_prob = config.attn_pdrop

        model_config = bert_config_to_nomic_config(hf_config)
        model = NomicBertForPreTraining.from_pretrained(config.model_name, config=model_config)

        if config.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        model = model.to("cuda")
        if self.distributed and not self.deepspeed:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[dist.get_rank()],
            )

        return {"model": model}

    def get_dataloaders(self, config, epoch=0):
        data_config = config.data_args
        model_config = config.model_args
        with self.main_process_first():
            dataset = DistributedIterableMLMDataset(
                dataset_name=data_config.tokenized_dataset,
                # use default languages
                languages=None,
                max_length=model_config.seq_len,
                seed=data_config.seed,
                global_batch_size=data_config.batch_size,
            )

            eval_dataset = EvalDistributedIterableMLMDataset(
                dataset_name=data_config.tokenized_dataset,
                # use default languages
                languages=["en"],
                max_length=model_config.seq_len,
                seed=data_config.seed,
                global_batch_size=data_config.eval_batch_size,
                mlm_probability=data_config.val_mlm_prob,
            )

        collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=True, mlm_probability=data_config.mlm_prob,
        )

        def collate_fn(batch):
            batches = batch[0]
            for b in batches:
                lang = b.pop("lang", None)
            mlm_batch = collator(batches)
            if lang:
                mlm_batch["lang"] = lang
            return mlm_batch

        train_dataloader = DataLoader(
            dataset,
            collate_fn=collate_fn,
        )

        eval_dataloader = DataLoader(
            eval_dataset,
            collate_fn=collate_fn,
        )

        self.total_num_steps = int(config.train_args.num_train_steps)
        self.total_training_steps = int(self.total_num_steps * config.train_args.gradient_accumulation_steps)

        return {"train": train_dataloader, "val": eval_dataloader, "test": None}

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
        for batch in tqdm(dataloader, desc=f"Eval epoch step {step}", total=dataloader.dataset.num_batches):
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

    def clip_gradients(self, max_grad_norm):
        super().clip_gradients(max_grad_norm)

    def training_step(
        self, model, batch, optimizer, scheduler, step, train_args, total_num_steps, gradient_accumulation_steps
    ):
        language = batch.pop("lang")
        loss = super().training_step(
            model=model,
            batch=batch,
            optimizer=optimizer,
            scheduler=scheduler,
            step=step,
            train_args=train_args,
            total_num_steps=total_num_steps,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

        return {f"{language}_loss": loss, "loss": loss}