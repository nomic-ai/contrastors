import json
import random
from abc import ABCMeta, abstractmethod
from contextlib import contextmanager
from pathlib import Path

import deepspeed
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import wandb
from deepspeed.checkpoint.utils import clone_tensors_for_torch_save
from torch.nn.utils import clip_grad_norm_
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer, get_scheduler

from contrastors.dataset.torch_loader import StreamingShardDataset
from contrastors.distributed import DistributedWandbTracker, gather, print_rank_zero
from contrastors.optimizer import configure_optimizer


class BaseTrainer(metaclass=ABCMeta):
    def __init__(self, config, dtype=torch.float32):
        self.config = config
        self.distributed = dist.is_initialized()
        self.print = print_rank_zero if self.distributed else print

        seed = config.mlm_data_args.seed if config.mlm_data_args else config.contrastive_data_args.seed
        self.set_seed(seed)

        if config.train_args.wandb:
            self.tracker = self.get_trackers(config)
        else:
            self.tracker = None

        self.print(json.dumps(config.dict(), indent=3))

        self.print(f"Using dtype: {dtype}")

        self.deepspeed = config.deepspeed

        self.tokenizer = self.get_tokenizer(config.model_args)

        self.model_type = config.model_args.model_type
        self.model = self.get_model(config.model_args)
        self.print(f"Model: {self.model}")
        self.print(
            f"Trainable parameters: {sum(p.numel() for p in self.model['model'].parameters() if p.requires_grad):,}"
        )

        self.dataloaders = self.get_dataloaders(config)
        self.optimizer = self.get_optimizer(config.train_args)
        self.scheduler = self.get_scheduler(config.train_args, self.optimizer)

        if self.deepspeed:
            engine, optimizer, dataloader, lr_scheduler = self.initialize_deepspeed(
                rank=dist.get_rank(), ds_config_path=config.deepspeed_config
            )

            self.engine = engine
            if optimizer is not None:
                self.optimizer = optimizer
            if dataloader is not None:
                self.dataloaders["train"] = dataloader
            if lr_scheduler is not None:
                self.scheduler = lr_scheduler

        self.dtype = dtype

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    @property
    def num_processes(self):
        return dist.get_world_size() if self.distributed else 1

    @property
    def process_index(self):
        return dist.get_rank() if self.distributed else 0

    @contextmanager
    def _goes_first(self, is_main):
        if not is_main:
            dist.barrier()

        yield

        if is_main:
            dist.barrier()

    @contextmanager
    def main_process_first(self):
        with self._goes_first(self.process_index == 0):
            yield

    def log(self, metrics, step=None):
        if self.process_index == 0:
            self.tracker.log(metrics, step=step)

    def initialize_deepspeed(self, rank, ds_config_path):
        # TODO handle grad cache?
        ds_config = json.load(open(ds_config_path))

        # don't let deepspeed print to stdout
        ds_config["steps_per_print"] = float("inf")

        if ds_config["gradient_accumulation_steps"] == "auto":
            ds_config["gradient_accumulation_steps"] = self.config.train_args.gradient_accumulation_steps

        # todo this could use some love
        if ds_config["train_micro_batch_size_per_gpu"] == "auto":
            data_args = self.config.mlm_data_args if self.config.mlm_data_args else self.config.contrastive_data_args
            ds_config["train_micro_batch_size_per_gpu"] = data_args.batch_size // self.num_processes
            ds_config["train_batch_size"] = data_args.batch_size * ds_config["gradient_accumulation_steps"]

        if self.config.train_args.max_grad_norm != ds_config["gradient_clipping"]:
            ds_config["gradient_clipping"] = self.config.train_args.max_grad_norm

        return deepspeed.initialize(
            rank, model=self.model["model"], optimizer=self.optimizer, lr_scheduler=self.scheduler, config=ds_config
        )

    def get_trackers(self, config):
        tracker = None
        if self.process_index == 0:
            project_name = config.train_args.wandb_project_name
            entity = config.train_args.wandb_entity
            run_name = config.train_args.wandb_run_name
            if run_name is None:
                run_name = config.train_args.output_dir.replace("ckpts/", "")

            hyperparams = {}
            for key, params in config.dict().items():
                if params is None or not isinstance(params, dict):
                    continue
                for k, v in params.items():
                    hyperparams[f"{key}_{k}"] = v

            tracker = wandb.init(project=project_name, entity=entity, name=run_name, config=hyperparams)

            if self.num_processes > 1:
                tracker = DistributedWandbTracker(tracker)

        return tracker

    def get_tokenizer(self, config):
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        tokenizer.model_max_length = config.seq_len

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if tokenizer.cls_token is None:
            tokenizer.add_special_tokens({"cls_token": "<s>"})

        if tokenizer.mask_token is None:
            tokenizer.add_special_tokens({"mask_token": "<mask>"})

        return tokenizer

    @abstractmethod
    def get_model(self, config):
        pass

    @abstractmethod
    def get_dataloaders(self, config):
        pass

    def get_optimizer(self, config):
        models = [model for model in self.model.values()]
        optimizer = configure_optimizer(models, config)

        return optimizer

    def get_scheduler(self, config, optimizer):
        if hasattr(config, "warmup_steps") and getattr(config, "warmup_steps") is not None:
            total_num_steps = self.total_num_steps * config.num_epochs
            warmup_steps = config.warmup_steps

        elif hasattr(config, "warmup_pct") and getattr(config, "warmup_pct") is not None:
            total_num_steps = self.total_num_steps * config.num_epochs
            warmup_steps = int(total_num_steps * config.warmup_pct)

        else:
            warmup_steps = 0

        self.print("*" * 50 + " SCHEDULER " + "*" * 50)
        self.print(f"Using {config.schedule_type} learning rate schedule")
        self.print(f"Warmup steps: {warmup_steps}")
        self.print(f"Total num steps: {total_num_steps}")
        scheduler = get_scheduler(
            name=config.schedule_type,
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=(total_num_steps if config.schedule_type != "inverse_sqrt" else None),
        )

        return scheduler

    def unwrap(self, model):
        # TODO: add deepspeed unwrapping support
        model_types = (nn.parallel.DistributedDataParallel, nn.DataParallel, deepspeed.DeepSpeedEngine)
        while isinstance(model, model_types):
            model = model.module

        return model

    def save_model(self, output_dir):
        if self.process_index == 0:
            unwrapped = self.unwrap(self.model["model"])
            if self.deepspeed:
                # TODO: need to add zero3 support
                # reduces bloat when saving with deepsped
                state_dict = clone_tensors_for_torch_save(unwrapped.state_dict())
            else:
                state_dict = None

            unwrapped.save_pretrained(output_dir, state_dict=state_dict)

    def load_model(self, model_path):
        loaded_model = self.model["model"].load_pretrained(model_path)

        return loaded_model

    def load_state(self, input_dir):
        if self.deepspeed:
            self.engine.load_checkpoint(input_dir)

        else:
            self.print(f"Loading model from {input_dir}/model")
            self.model = self.load_model(f"{input_dir}/model")

            self.print(f"Loading optimizer and scheduler state from {input_dir}/optimizer.pt")
            self.optimizer.load_state_dict(torch.load(f"{input_dir}/optimizer.pt"))

            self.print(f"Loading optimizer and scheduler state from {input_dir}/scheduler.pt")
            self.scheduler.load_state_dict(torch.load(f"{input_dir}/scheduler.pt"))

            # todo check this works!
            if hasattr(self.dataloaders["train"], "sampler") and isinstance(
                self.dataloaders["train"].sampler, DistributedSampler
            ):
                self.print(f"Loading sampler state from {input_dir}/sampler.pt")
                self.dataloaders["train"].sampler.load_state_dict(torch.load(f"{input_dir}/sampler.pt"))

        self.print(f"Loading random states from {input_dir}/random_states_{self.process_index}.pt")
        random_states = torch.load(f"{input_dir}/random_states_{self.process_index}.pt")

        torch.set_rng_state(random_states["torch"])
        np.random.set_state(random_states["numpy"])
        random.setstate(random_states["random"])
        torch.cuda.set_rng_state_all(random_states["cuda"])

    def save_state(self, output_dir):
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        self.save_model(f"{output_dir}/model")

        if self.deepspeed:
            self.engine.save_checkpoint(output_dir)

        else:
            opt_state_dict = self.optimizer.state_dict()
            torch.save(opt_state_dict, f"{output_dir}/optimizer.pt")

            schedulr_state_dict = self.scheduler.state_dict()
            torch.save(schedulr_state_dict, f"{output_dir}/scheduler.pt")

            if hasattr(self.dataloaders["train"], "sampler") and isinstance(
                self.dataloaders["train"].sampler, DistributedSampler
            ):
                sampler = self.dataloaders["train"].sampler
                torch.save(sampler.state_dict(), f"{output_dir}/sampler.pt")
            elif isinstance(self.dataloaders["train"], StreamingShardDataset):
                data_config = (
                    self.config.mlm_data_args if self.config.mlm_data_args else self.config.contrastive_data_args
                )
                ds_state = data_config.input_shards.replace(".yaml", "")
                with open(f"{ds_state}/rank_{dist.get_rank()}_processed.json", "r") as f:
                    processed = json.load(f)

                with open(f"{output_dir}/rank_{dist.get_rank()}_processed.json", "w") as f:
                    json.dump(processed, f, indent=3)

        random_states = {}
        random_states["torch"] = torch.get_rng_state()
        random_states["numpy"] = np.random.get_state()
        random_states["random"] = random.getstate()
        random_states["cuda"] = torch.cuda.get_rng_state_all()

        torch.save(random_states, f"{output_dir}/random_states_{self.process_index}.pt")

    def backward(self, loss):
        if self.deepspeed:
            self.engine.backward(loss)
            self.engine.step()
        else:
            loss.backward()

    @abstractmethod
    def eval_loop(self, model, dataloader, step):
        pass

    @abstractmethod
    def forward_step(self, model, inputs, **kwargs):
        pass

    def training_step(
        self, model, batch, optimizer, scheduler, step, train_args, total_num_steps, gradient_accumulation_steps
    ):
        with torch.autocast(device_type="cuda", dtype=self.dtype):
            loss = self.forward_step(inputs=batch, **model, step=step)

        self.backward(loss)

        # clip gradients
        if step % gradient_accumulation_steps == 0:
            if train_args.max_grad_norm is not None and train_args.max_grad_norm > 0:
                if not self.deepspeed:
                    clip_grad_norm_(model["model"].parameters(), train_args.max_grad_norm)

        # all of this is handled by the deepspeed engine
        if not self.deepspeed:
            if (step + 1) % gradient_accumulation_steps == 0 or step == total_num_steps - 1:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        return loss

    def train(self):
        train_args = self.config.train_args
        data_config = self.config.mlm_data_args if self.config.mlm_data_args else self.config.contrastive_data_args

        dataloaders = self.dataloaders
        train_dataloader = dataloaders["train"]
        val_dataloader = dataloaders.get("val", None)

        model = self.model
        optimizer = self.optimizer
        scheduler = self.scheduler

        total_num_steps = self.total_num_steps
        gradient_accumulation_steps = train_args.gradient_accumulation_steps

        if train_args.checkpoint:
            self.load_state(train_args.checkpoint)

        for epoch in range(0, train_args.num_epochs):
            if (
                epoch > 0
                and getattr(train_dataloader, "sampler", None) is not None
                and isinstance(train_dataloader.sampler, DistributedSampler)
            ):
                train_dataloader.sampler.set_epoch(epoch)
            elif epoch > 0 and getattr(data_config, "streaming", False):
                temp_config = self.config.copy(deep=True)
                temp_config.contrastive_data_args.seed = data_config.seed + epoch
                train_dataloader = self.get_dataloaders(temp_config)["train"]

            self.print(f"Total training steps: {self.total_num_steps}")

            progbar = tqdm(
                train_dataloader, desc=f"Epoch {epoch}", disable=not self.process_index == 0, total=self.total_num_steps
            )

            # TODO: fix resuming from a checkpoint

            for step, batch in enumerate(progbar):
                # if using deepspeed, it handles gradient accumulation
                curr_step = epoch * total_num_steps + step  # + offset

                loss = self.training_step(
                    model,
                    batch,
                    optimizer,
                    scheduler,
                    curr_step,
                    train_args,
                    total_num_steps,
                    gradient_accumulation_steps,
                )

                loss = gather(loss.detach().float())

                if train_args.wandb:
                    self.log({"loss": torch.mean(loss).item()}, step=curr_step)
                else:
                    self.print(f'Loss: {torch.mean(loss).item()}')
                    self.print(f"LR: {scheduler.get_last_lr()[0]}")

                if val_dataloader:
                    if (
                        step > 0
                        and train_args.eval_strategy == "steps"
                        and train_args.eval_every > 0
                        and step % train_args.eval_every == 0
                    ):
                        self.eval_loop(dataloader=val_dataloader, step=curr_step, **model)

                # log LR in case something weird happens
                if step > 0 and step % (train_args.log_lr_every) == 0:
                    if train_args.wandb:
                        self.log({"lr": scheduler.get_last_lr()[0]}, step=curr_step)

                if step > 0 and train_args.save_every > 0 and step % train_args.save_every == 0:
                    self.save_state(f"{train_args.output_dir}/step_{curr_step}")

            if val_dataloader and train_args.eval_strategy == "epochs":
                self.eval_loop(dataloader=val_dataloader, step=epoch, **model)

            if train_args.save_every > 0:
                self.save_model(f"{train_args.output_dir}/epoch_{epoch}_model")
                if train_args.num_epochs > 1:
                    self.save_state(f"{train_args.output_dir}/epoch_{epoch}")

        if train_args.num_epochs > 0 and train_args.save_every > 0:
            torch.distributed.barrier()
            self.save_model(f"{train_args.output_dir}/final_model")
