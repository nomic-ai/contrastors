from typing import Any, Dict, Optional, Tuple, Union

from pydantic import BaseModel, validator

from contrastors.dataset.constants import OPENAI_IMAGE_DATASET_MEAN, OPENAI_IMAGE_DATASET_STD


class TrainArgs(BaseModel):
    num_epochs: int
    learning_rate: float
    weight_decay: float
    eps: Optional[float]
    warmup_steps: Optional[int]
    warmup_pct: Optional[float]
    cooldown_steps: Optional[int]
    checkpoint: Optional[str]
    wandb: bool
    wandb_project_name: str
    wandb_entity: str
    log_grads_every: int
    log_lr_every: int
    save_every: Optional[int]
    eval_every: Optional[int]
    output_dir: Optional[str]
    gradient_checkpointing: Optional[bool]
    gradient_accumulation_steps: Optional[int] = 1
    # if using deepspeed, this will be ignored
    schedule_type: str
    max_grad_norm: float
    adam_beta1: float
    adam_beta2: float
    loss_fn: Optional[str]
    grad_cache: Optional[bool]
    chunk_size: Optional[int]
    logit_scale: Optional[float] = 1 / 0.07
    clamp_logits: Optional[bool] = True
    logit_max: Optional[float] = 100.0
    add_l2_loss: Optional[bool] = False

    class Config:
        validate_assignment = True

    @validator('logit_scale')
    def set_logit_scale(cls, scale):
        return scale or 1 / 0.07

    @validator('logit_max')
    def set_logic_max(cls, max):
        return max or 100.0


class DataArgs(BaseModel):
    input_shards: Optional[str]
    tokenized_dataset: Optional[str]
    task_name: Optional[Optional[str]]
    image_text_shards: Optional[str]
    workers: int
    batch_size: int
    seed: int
    train_num_samples: Optional[int]
    shuffle: bool
    mlm_prob: Optional[float]
    val_mlm_prob: Optional[float]
    val_pct: Optional[float]
    download: Optional[bool] = False
    process_one_shard: Optional[bool] = False
    streaming: Optional[bool] = True
    weighted_sampling: Optional[bool] = False
    verbose: Optional[bool] = False
    imagenet_val_path: Optional[str] = None


class ModelArgs(BaseModel):
    seq_len: Optional[int]
    rotary_emb_fraction: Optional[float]
    pad_vocab_to_multiple_of: Optional[int]
    use_rms_norm: Optional[bool]
    pretrained: Optional[str]
    model_name: Optional[str]
    pooling: Optional[str]
    encoder: Optional[bool]
    add_prefix: Optional[bool] = False
    num_negatives: Optional[int] = 7
    tokenizer_name: Optional[str] = None
    activation_function: Optional[str] = "gelu"
    qkv_proj_bias: Optional[bool] = True
    mlp_fc1_bias: Optional[bool] = True
    mlp_fc2_bias: Optional[bool] = True
    attn_pdrop: Optional[float] = 0.0
    projection_dim: Optional[int] = None
    freeze: Optional[bool] = False


class Config(BaseModel):
    train_args: TrainArgs
    data_args: DataArgs
    model_args: Optional[ModelArgs]
