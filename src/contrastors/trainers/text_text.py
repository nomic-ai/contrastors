import torch.distributed
import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import deepspeed
from torch.utils.data import DataLoader

from contrastors.dataset.text_text_loader import StreamingShardDataset, collate_fn, get_local_dataloader
from contrastors.distributed import gather_with_grad, print_in_order
from contrastors.loss import clip_loss, grad_cache_loss, calculate_auxiliary_loss
from contrastors.models import BiEncoder, BiEncoderConfig, LogitScale
from megablocks.layers import moe
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import NanoBEIREvaluator

from .base import BaseTrainer

from torch import nn
from typing import List, Dict, Union, Tuple

class SentenceTransformerModule(nn.Module):
    """
    Wrapper class to make a custom embedding model compatible with SentenceTransformers.
    
    Args:
        custom_model: Your custom PyTorch module that generates embeddings
        model_name_or_path: Name/path of the tokenizer to use (e.g., 'bert-base-uncased')
        max_seq_length: Maximum sequence length for the tokenizer
        do_lower_case: Whether to lowercase input text
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        max_seq_length: int = 128,
        pooling: str = None
    ):
        # Initialize as nn.Sequential with your custom model as the only module
        super().__init__()
        self.model = model
        
        # Store configuration
        self.max_seq_length = max_seq_length
        self.do_lower_case = False
        
        # Initialize tokenizer
        self.tokenizer = tokenizer

        self.pooling = pooling

    def mean_pooling(self, inputs, attention_mask):
        token_embeddings = inputs[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
    def forward(self, features: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass that matches SentenceTransformers expected format"""
        # Get embeddings from your custom model
        keep_keys = ["input_ids", "attention_mask", "token_type_ids"]
        features = {k: features[k] for k in keep_keys if k in features}
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                embeddings = self.model(**features)

        if self.pooling == "cls":
            # hack to get cls pooling
            emb = embeddings[0][:, 0]
            emb = F.normalize(emb, p=2, dim=1)
            embeddings = {"embedding": emb}
        elif self.pooling == "mean":
            emb = self.mean_pooling(embeddings, features["attention_mask"])
            emb = F.normalize(emb, p=2, dim=1)
            embeddings = {"embedding": emb}

        return {"sentence_embedding": embeddings["embedding"]}
        
    
    def tokenize(
        self,
        texts: Union[List[str], List[Dict], List[Tuple[str, str]]],
        padding: Union[str, bool] = True
    ) -> Dict[str, torch.Tensor]:
        """Tokenizes text input in the format expected by SentenceTransformers"""
        output = {}
        
        # Handle different input formats
        if isinstance(texts[0], str):
            to_tokenize = [texts]
        elif isinstance(texts[0], dict):
            to_tokenize = []
            output["text_keys"] = []
            for lookup in texts:
                text_key, text = next(iter(lookup.items()))
                to_tokenize.append(text)
                output["text_keys"].append(text_key)
            to_tokenize = [to_tokenize]
        else:
            batch1, batch2 = [], []
            for text_tuple in texts:
                batch1.append(text_tuple[0])
                batch2.append(text_tuple[1])
            to_tokenize = [batch1, batch2]

        # Preprocess text
        to_tokenize = [[str(s).strip() for s in col] for col in to_tokenize]
        if self.do_lower_case:
            to_tokenize = [[s.lower() for s in col] for col in to_tokenize]

        # Perform tokenization
        output.update(
            self.tokenizer(
                *to_tokenize,
                padding=padding,
                truncation="longest_first",
                return_tensors="pt",
                max_length=self.max_seq_length
            )
        )
        return output
    

class TextTextTrainer(BaseTrainer):
    def __init__(self, config, dtype):
        super(TextTextTrainer, self).__init__(config, dtype)
        self.use_grad_cache = config.train_args.grad_cache
        self.matryoshka_dims = config.train_args.matryoshka_dims
        if self.matryoshka_dims:
            self.matryoshka_loss_weights = (
                config.train_args.matryoshka_loss_weights
                if config.train_args.matryoshka_dims and config.train_args.matryoshka_loss_weights
                else [1] * len(config.train_args.matryoshka_dims)
            )
        else:
            self.matryoshka_loss_weights = None

    def get_model(self, config):
        model_config = config.model_args
        if model_config.checkpoint is None:
            model_config_dict = model_config.dict()
            if config.train_args.router_aux_loss_coef is not None:
                model_config_dict["router_aux_loss_coef"] = config.train_args.router_aux_loss_coef
            config = BiEncoderConfig(**model_config_dict)
            model = BiEncoder(config)
        else:
            self.print(f"Loading model from {model_config.checkpoint}")
            loaded_config = BiEncoderConfig.from_pretrained(model_config.checkpoint)
            if model_config.projection_dim is not None:
                loaded_config.projection_dim = model_config.projection_dim
            if model_config.gradient_checkpointing:
                loaded_config.gradient_checkpointing = True
            if model_config.num_experts is not None:
                loaded_config.num_experts = model_config.num_experts
            if model_config.moe_top_k is not None:
                loaded_config.moe_top_k = model_config.moe_top_k
            if config.train_args.router_aux_loss_coef is not None:
                loaded_config.router_aux_loss_coef = config.train_args.router_aux_loss_coef
            model = BiEncoder.from_pretrained(model_config.checkpoint, config=loaded_config)
            config = loaded_config

        if self.distributed and not self.deepspeed:
            model = model.to("cuda")
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.process_index],
                # find_unused_parameters=True,
                broadcast_buffers=False,
            )

        scale = LogitScale(config)

        if self.distributed and not self.deepspeed:
            scale = scale.to("cuda")
            if sum(p.requires_grad for p in scale.parameters()) > 0:
                scale = torch.nn.parallel.DistributedDataParallel(
                    scale,
                    device_ids=[self.process_index],
                )

        return {"model": model, "logit_scale": scale}

    def get_dataloaders(self, config, epoch=0):
        train_args = config.train_args
        data_config = config.data_args
        model_args = config.model_args
        gradient_accumulation_steps = train_args.gradient_accumulation_steps
        if train_args.wandb_run_name is None and train_args.wandb:
            raise ValueError("wandb_run_name must be set, got None")
        if data_config.streaming:
            train_dataset = StreamingShardDataset(
                data_config.input_shards,
                data_config.batch_size,
                self.tokenizer,
                seed=data_config.seed,
                add_eos=model_args.nomic_encoder != True,
                add_prefix=model_args.add_prefix,
                num_negatives=model_args.num_negatives,
                download_locally=data_config.download,
                process_one_shard=data_config.process_one_shard,
                weighted_sampling=data_config.weighted_sampling,
                verbose=data_config.verbose,
                sample_negatives=data_config.sample_negatives,
                run_name=train_args.wandb_run_name,
                query_max_length=data_config.query_max_length,
                document_max_length=data_config.document_max_length
            )
            # wait until every process has finished loading the dataset
            # this takes a long time to get the offsets!
            torch.distributed.barrier()
            if train_args.checkpoint is not None:
                print(f"Loading dataloader state from {train_args.checkpoint}")
                train_dataset.load_state(train_args.checkpoint)

            train_dataloader = DataLoader(train_dataset, batch_size=1, collate_fn=collate_fn, num_workers=0)
            self.print(f"Len of train_dataloader: {len(train_dataset)}")
            # round down in case

            self.total_num_steps = int(len(train_dataset) / gradient_accumulation_steps // data_config.batch_size)
        else:
            # config defines global batch size
            if data_config.batch_size % self.num_processes != 0:
                raise ValueError(
                    f"Batch size {data_config.batch_size} must be divisible by accelerator.num_processes {self.num_processes}"
                )

            batch_size = int(data_config.batch_size / self.num_processes)
            train_dataloader = get_local_dataloader(
                data_config.input_shards,
                batch_size,
                self.tokenizer,
                seed=data_config.seed,
                num_negatives=model_args.num_negatives,
                add_prefix=model_args.add_prefix,
                num_workers=data_config.workers,
                epoch=0,
            )
            self.total_num_steps = int(
                len(train_dataloader.dataset) / gradient_accumulation_steps // data_config.batch_size
            )

        nano_beir = NanoBEIREvaluator(query_prompts=model_args.query_prefix, corpus_prompts=model_args.document_prefix, show_progress_bar=True)

        return {"train": train_dataloader, "val": nano_beir, "test": None}

    def save_model(self, output_dir):
        super().save_model(output_dir)
        if self.global_rank == 0:
            logit_scale = self.model.get("logit_scale", None)
            if isinstance(logit_scale, (nn.Module, nn.DataParallel, nn.parallel.DistributedDataParallel)) and any(
                p.requires_grad for p in logit_scale.parameters()
            ):
                unwrapped_scale = self.unwrap(logit_scale)
                torch.save(unwrapped_scale.state_dict(), f"{output_dir}/logit_scale.pt")

                
    def load_model(self, model_path):
        config = BiEncoderConfig.from_pretrained(model_path)
        loaded_model = BiEncoder.from_pretrained(model_path, config=config)
        loaded_model = loaded_model.to("cuda")
        if isinstance(self.model["model"],(nn.parallel.DistributedDataParallel, nn.DataParallel, deepspeed.DeepSpeedEngine)):
            torch.distributed.barrier()
            loaded_model = torch.nn.parallel.DistributedDataParallel(
                loaded_model,
                device_ids=[self.process_index],
                # find_unused_parameters=True,
                broadcast_buffers=False,
            )

        return loaded_model

    def clip_gradients(self, max_grad_norm):
        super().clip_gradients(max_grad_norm)

    def forward_step(self, model, inputs, logit_scale, **kwargs):
        model.train()
        if self.use_grad_cache:
            loss = self._grad_cache_forward_step(model, inputs, logit_scale, **kwargs)
        else:
            loss = self._forward_step(
                model=model,
                batch=inputs,
                logit_scale=logit_scale,
                matryoshka_dims=self.matryoshka_dims,
                matroyshka_loss_weights=self.matryoshka_loss_weights,
                **kwargs,
            )

        return loss

    def backward(self, loss):
        if isinstance(loss, dict):
            loss = loss["loss"]

        if self.deepspeed:
            self.engine.backward(loss)
            self.engine.step()
        else:
            # grad cache backprops in the loss function, becomes a noop
            if not self.use_grad_cache:
                loss.backward()

    def _grad_cache_forward_step(self, model, batch, logit_scale, **kwargs):
        # TODO: could pass this to grad cache loss and log?
        batch.pop("dataset_name")
        kwargs.pop("step")
        batch = {k: v.to(model.device) for k, v in batch.items()}
        query_inputs = {k.replace("query_", ""): v for k, v in batch.items() if "query" in k}
        document_inputs = {k.replace("document_", ""): v for k, v in batch.items() if "document" in k}
        aux_coeff = getattr(self.config.train_args, "router_aux_loss_coef", None)
        loss = grad_cache_loss(
            tower1=model,
            tower2=model,
            t1_inputs=query_inputs,
            t2_inputs=document_inputs,
            chunk_size=self.config.train_args.chunk_size,
            logit_scale=logit_scale,
            router_aux_coeff=aux_coeff,
            **kwargs,
        )
        return {"loss": loss}

    def _forward_step(self, model, batch, logit_scale, matryoshka_dims=None, matroyshka_loss_weights=None, **kwargs):
        normalize = True if matryoshka_dims is None else False
        dataset_name = batch.pop("dataset_name")
        if self.config.model_args.num_experts > 0 and self.config.train_args.router_aux_loss_coef > 0:
            # if using gradient checkpointing, need to reclear the list
            moe.clear_load_balancing_loss()

        max_length = batch["document_input_ids"].shape[1] 
        padded_query_inputs = {
                "input_ids": batch["query_input_ids"].to(model.device), 
                "attention_mask": batch["query_attention_mask"].to(model.device),
        }

        query_outputs = model(
            **padded_query_inputs,
            normalize=normalize,
        )
        document_outputs = model(
            input_ids=batch["document_input_ids"].to(model.device),
            attention_mask=batch["document_attention_mask"].to(model.device),
            normalize=normalize,
        )
        if "negative_input_ids" in batch:
            raise NotImplementedError("Negative sampling not supported for text-text models")

        queries = query_outputs["embedding"]
        all_documents = gather_with_grad(document_outputs["embedding"])

        if matryoshka_dims:
            loss = 0.0
            for loss_weight, dim in zip(matroyshka_loss_weights, matryoshka_dims):
                reduced_q = F.normalize(queries[:, :dim], dim=-1)
                reduced_d = F.normalize(all_documents[:, :dim], dim=-1)

                name_with_dim = f"{dataset_name}_matryoshka_{dim}"

                dim_loss = clip_loss(
                    query=reduced_q,
                    document=reduced_d,
                    logit_scale=logit_scale,
                    tracker=self.tracker,
                    dataset=name_with_dim,
                    **kwargs,
                )

                loss += loss_weight * dim_loss
        else:
            loss = clip_loss(
                query=queries,
                document=all_documents,
                logit_scale=logit_scale,
                tracker=self.tracker,
                dataset=dataset_name,
                **kwargs,
            )

        if self.config.model_args.num_experts > 0 and self.config.train_args.router_aux_loss_coef > 0:
            if query_outputs["router_loss"] is not None and document_outputs["router_loss"] is not None:
                # megablocks scales loss already
                aux_loss = query_outputs["router_loss"] + document_outputs["router_loss"]

                if self.config.train_args.wandb and kwargs.get("step") % 1000 == 0:
                    query_tokens_per_expert = query_outputs["tokens_per_expert"].sum(dim=0)
                    query_expert_pct = query_tokens_per_expert / query_tokens_per_expert.sum()


                    document_tokens_per_expert = document_outputs["tokens_per_expert"].sum(dim=0)
                    document_expert_pct = document_tokens_per_expert / document_tokens_per_expert.sum()

                    table = wandb.Table(columns=[f"expert_{i}" for i in range(query_expert_pct.shape[0])],
                                        data=[query_expert_pct.tolist(), document_expert_pct.tolist()],
                                        )
                    self.log({"expert_pct": table}, step=kwargs["step"])

            else:
                num_experts = self.config.model_args.num_experts
                top_k = self.config.model_args.moe_top_k
                aux_coeff = self.config.train_args.router_aux_loss_coef
                query_aux_loss = calculate_auxiliary_loss(
                    router_logits=query_outputs["router_logits"],
                    num_experts=num_experts,
                    top_k=top_k,
                    tracker=self.tracker,
                    name="query",
                    **kwargs,
                )
                document_aux_loss = calculate_auxiliary_loss(
                    router_logits=document_outputs["router_logits"],
                    num_experts=num_experts,
                    top_k=top_k,
                    tracker=self.tracker,
                    name="document",
                    **kwargs
                )

                aux_loss = (query_aux_loss + document_aux_loss) * aux_coeff
                # aux_loss = gradnorm(aux_loss)

            # loss = gradnorm(loss)
            total_loss = loss + aux_loss

            loss = {"loss": total_loss, "aux_loss": aux_loss, "infonce_loss": loss}

        return loss

    def training_step(
        self, model, batch, optimizer, scheduler, step, train_args, total_num_steps, gradient_accumulation_steps
    ):
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

        if train_args.clamp_logits:
            with torch.no_grad():
                self.model["scale"].module.logit_scale.clamp_(0, np.log(train_args.logit_max))

        if train_args.wandb:
            if isinstance(loss, dict):
                self.log({k: v.detach().cpu().item() for k, v in loss.items()}, step=step)

        return loss

    def eval_loop(self, model, dataloader, step, **kwargs):
        model.eval()
        train_args = self.config.train_args
        model_args = self.config.model_args
        if self.process_index == 0:
            original_model = model.module
            module = nn.Sequential(SentenceTransformerModule(model=original_model, max_seq_length=model_args.seq_len, tokenizer=self.tokenizer))
            emb = SentenceTransformer(modules=module, similarity_fn_name="cosine")
            results = dataloader(emb) 

            ndcg = {f'beir/{k.replace("Nano", "").replace("_cosine", "").lower()}': v for k, v in results.items() if "ndcg@10" in k}

            if train_args.wandb:
                self.log(ndcg, step=step) 
            else:
                self.print(ndcg)
            
            
        torch.distributed.barrier()
            
