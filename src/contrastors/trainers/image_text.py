from functools import partial
from pathlib import Path

import datasets
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from clip_benchmark.metrics import zeroshot_retrieval as zsr
from deepspeed.checkpoint.utils import clone_tensors_for_torch_save
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from contrastors.dataset.image_text_loader import get_imagenet, get_wds_image_text_dataset
from contrastors.dataset.transform import image_transform
from contrastors.distributed import gather
from contrastors.eval.datacomp.retr_eval import RetrievalDataset, image_captions_collage_fn_prefix
from contrastors.eval.imagenet import evaluate_imagenet
from contrastors.models.dual_encoder import DualEncoder, DualEncoderConfig

from .text_text import TextTextTrainer


class ImageTextTrainer(TextTextTrainer):
    def __init__(self, config, dtype):
        self.transforms = self.get_transforms(config.transforms)
        super(ImageTextTrainer, self).__init__(config, dtype)

    def get_transforms(self, transforms):
        train_transforms = image_transform(**transforms.dict(), is_train=True)
        val_transforms = image_transform(
            **transforms.dict(exclude={"aug_cfg", "resize_longest_max", "fill_color"}), is_train=False
        )

        return {"train": train_transforms, "val": val_transforms}

    def get_tokenizer(self, config):
        config = config.text_model_args
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        tokenizer.model_max_length = config.seq_len

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if tokenizer.cls_token is None:
            tokenizer.add_special_tokens({"cls_token": "<s>"})

        if tokenizer.mask_token is None:
            tokenizer.add_special_tokens({"mask_token": "<mask>"})

        return tokenizer

    def get_model(self, config):
        model_config = DualEncoderConfig(config)
        model = DualEncoder(model_config)

        has_trainable_params = sum(p.requires_grad for p in model.parameters()) > 0
        model = model.to(f"cuda:{self.process_index}")

        if self.distributed and not self.deepspeed and has_trainable_params:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[dist.get_rank()],
            )

        models = {"model": model}

        return models

    def get_dataloaders(self, config, epoch=0):
        train_args = config.train_args
        data_config = config.data_args
        train_transforms = self.transforms["train"]
        val_transforms = self.transforms["val"]
        tokenizer = self.tokenizer
        gradient_accumulation_steps = train_args.gradient_accumulation_steps

        text_args = config.text_model_args
        train_data_info = get_wds_image_text_dataset(
            data_config,
            train_transforms,
            tokenizer=tokenizer,
            is_train=True,
            epoch=epoch,
            add_eos=text_args.nomic_encoder == False,
            add_prefix=text_args.add_prefix,
            precomputed_text=text_args.precomputed,
        )
        train_dataloader = train_data_info.dataloader

        self.total_num_steps = int(len(train_data_info) / gradient_accumulation_steps)

        if data_config.imagenet_val_path is not None:
            val_data_info = get_imagenet(data_config, transforms=val_transforms)
            val_dataloader = val_data_info.dataloader
        else:
            val_dataloader = None

        dataloaders = {"train": train_dataloader, "train_sampler": train_data_info, "val": val_dataloader, "test": None}

        data_config = config.data_args
        if data_config.eval_flickr:
            val_transforms = self.transforms["val"]

            dataset = RetrievalDataset(
                datasets.load_dataset(
                    f"nlphuji/flickr_1k_test_image_text_retrieval",
                    split="test",
                ),
                transform=val_transforms,
            )

            dataloader = DataLoader(
                dataset,
                batch_size=128,
                shuffle=False,
                num_workers=0,
                collate_fn=lambda x: image_captions_collage_fn_prefix(
                    x, prefix="search_query" if self.config.text_model_args.add_prefix else None
                ),
            )

            imagenet_dataloader = dataloaders["val"]
            dataloaders["val"] = {"imagenet": imagenet_dataloader, "flickr": dataloader}

        return dataloaders

    def save_model(self, output_dir):
        super().save_model(output_dir)

        vision_output_dir = Path(f"{output_dir}/vision")
        if not vision_output_dir.exists():
            vision_output_dir.mkdir(parents=True, exist_ok=True)

        if self.global_rank == 0:
            if self.config.vision_model_args.freeze is False and "vision" in self.model:
                unwrapped = self.unwrap(self.model["vision"])
                if self.deepspeed:
                    # TODO: need to add zero3 support
                    # reduces bloat when saving with deepsped
                    state_dict = clone_tensors_for_torch_save(unwrapped.state_dict())
                else:
                    state_dict = None

                unwrapped.save_pretrained(vision_output_dir, state_dict=state_dict)

            logit_scale = self.model.get("logit_scale", None)
            if isinstance(logit_scale, (nn.Module, nn.DataParallel, nn.parallel.DistributedDataParallel)) and any(
                p.requires_grad for p in logit_scale.parameters()
            ):
                unwrapped_scale = self.unwrap(logit_scale)
                torch.save(unwrapped_scale.state_dict(), f"{output_dir}/logit_scale.pt")

    def forward_step(self, model, inputs, **kwargs):
        model.train()
        if self.use_grad_cache:
            raise NotImplementedError("Grad cache not supported for three towers")
        else:
            loss = self._forward_step(model, inputs, **kwargs)

        return loss

    def backward(self, loss):
        # grad cache backprops in the loss function, becomes a noop
        if not self.use_grad_cache:
            if self.deepspeed:
                self.engine.backward(loss["loss"])
                self.engine.step()
            else:
                loss["loss"].backward()

    def _forward_step(self, model, batch, **kwargs):
        text_inputs = {k: v.to(model.device) for k, v in batch["text"].items()}
        vision_inputs = {k: v.to(model.device) for k, v in batch["vision"].items()}

        outputs = model(text_inputs, vision_inputs)

        return outputs

    def training_step(
        self, model, batch, optimizer, scheduler, step, train_args, total_num_steps, gradient_accumulation_steps
    ):
        loss = super().training_step(
            model, batch, optimizer, scheduler, step, train_args, total_num_steps, gradient_accumulation_steps
        )

        if train_args.clamp_logits:
            if sum(p.requires_grad for p in model["model"].logit_scale.parameters()) > 0:
                with torch.no_grad():
                    torch.clamp_(model["model"].logit_scale.logit_scale, 0, np.log(train_args.logit_max))

        if train_args.wandb:
            if sum(p.requires_grad for p in model["model"].logit_scale.parameters()) > 0:
                self.log({"logit_scale": model["model"].logit_scale.logit_scale.exp().item()}, step=step)

        return loss

    def _eval_imagenet(self, model, dataloader, step, **kwargs):
        train_args = self.config.train_args

        text = model.text
        text.eval()
        vision = model.vision
        vision.eval()

        with torch.autocast(device_type="cuda", dtype=self.dtype):
            top1, top5, _, _ = evaluate_imagenet(
                text=text,
                vision=vision,
                tokenizer=self.tokenizer,
                dataloader=dataloader,
                return_embeddings=False,
                prefix="search_query" if self.config.text_model_args.add_prefix else None,
            )

        top1 = gather(top1)
        top5 = gather(top5)

        top1 = torch.mean(top1).item()
        top5 = torch.mean(top5).item()

        log_val = {"top1_acc": top1, "top5_acc": top5}

        if train_args.wandb:
            self.log(log_val, step=step)
        else:
            self.print(f"Step: {step} Top1: {top1} Top5: {top5}")

    def _eval_flickr(self, model, dataloader, step, **kwargs):
        tokenizer = self.tokenizer
        if self.global_rank == 0:
            tokenizer.model_max_length = 77
            tokenizer = partial(tokenizer, return_tensors="pt", truncation=True, padding="max_length")
            device = model.device
            metrics = zsr.evaluate(model, dataloader, tokenizer, recall_k_list=[1, 5, 10], device=device, amp=False)
            metrics["mean_recall@1"] = 0.5 * (metrics["text_retrieval_recall@1"] + metrics["image_retrieval_recall@1"])

            log_flickr = {"flickr/" + k: v for k, v in metrics.items()}
            if self.config.train_args.wandb:
                self.log(log_flickr, step=step)
            else:
                self.print(f"Step: {step} Flickr: {metrics}")

        dist.barrier()

    def eval_loop(self, model, dataloader, step, **kwargs):
        if isinstance(dataloader, dict):
            imagenet_dataloader = dataloader["imagenet"]
        else:
            imagenet_dataloader = dataloader
        self._eval_imagenet(model, imagenet_dataloader, step, **kwargs)

        if isinstance(dataloader, dict):
            self._eval_flickr(model, dataloader["flickr"], step, **kwargs)
