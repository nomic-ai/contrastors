from typing import List, Optional

from pydantic import BaseModel, root_validator, validator


class TrainArgs(BaseModel):
    num_epochs: int
    learning_rate: float
    weight_decay: float
    eps: Optional[float] = 1e-8
    warmup_steps: Optional[int]
    warmup_pct: Optional[float]
    cooldown_steps: Optional[int]
    checkpoint: Optional[str]
    wandb: bool
    wandb_project_name: str
    wandb_entity: str
    wandb_run_name: Optional[str]
    log_grads_every: int
    log_lr_every: int
    save_every: Optional[int]
    eval_steps: Optional[int]
    eval_strategy: Optional[str]
    output_dir: Optional[str]
    gradient_accumulation_steps: Optional[int] = 1
    # if using deepspeed, this will be ignored
    schedule_type: str
    max_grad_norm: float
    adam_beta1: float
    adam_beta2: float
    loss_fn: Optional[str]
    grad_cache: Optional[bool]
    chunk_size: Optional[int]
    clamp_logits: Optional[bool] = True
    logit_max: Optional[float] = 100.0
    add_l2_loss: Optional[bool] = False
    matryoshka_dims: Optional[List[int]] = None
    matryoshka_loss_weights: Optional[List[float]] = None

    class Config:
        validate_assignment = True

    @validator('logit_max')
    def set_logic_max(cls, max):
        return max or 100.0

    @validator("eval_strategy")
    def validate_eval_strategy(cls, strategy):
        if strategy not in ["steps", "epochs"]:
            raise ValueError(f"Eval strategy {strategy} not found in eval strategy registry")
        return strategy

    @root_validator
    def validate_steps_set(cls, values):
        # validate that eval_steps is set if eval_strategy is set to steps
        eval_steps, eval_strategy = values.get("eval_steps"), values.get("eval_strategy")
        if eval_strategy == "steps" and eval_steps is None:
            raise ValueError("Eval steps must be set if eval strategy is set to steps")

        return values

    @root_validator
    def validate_matryoshka_no_grad_cache(cls, values):
        # validate that matryoska isn't set if grad_cache is set
        matryoshka, grad_cache = values.get("matryoshka_dims"), values.get("grad_cache")
        if matryoshka is not None and grad_cache:
            raise ValueError("Matryoshka dims cannot be set if grad cache is set")

        return values


class DataArgs(BaseModel):
    shuffle: bool
    workers: int
    batch_size: int
    seed: int
    val_pct: Optional[float] = None


class MLMDataArgs(DataArgs):
    tokenized_dataset: Optional[str]
    mlm_prob: Optional[float]
    task_name: Optional[Optional[str]]
    val_mlm_prob: Optional[float]

    @root_validator
    def validate_data(cls, values):
        tokenized, task_name = values.get("tokenized_dataset"), values.get("task_name")
        if tokenized is None and task_name is None:
            raise ValueError("Either tokenized dataset or task name must be set")
        return values

    @root_validator
    def validate_mlm(cls, values):
        tokenized, mlm_prob, val_prob = (
            values.get("tokenized_dataset"),
            values.get("mlm_prob"),
            values.get("val_mlm_prob"),
        )
        # validate mlm_prob if tokenized is set
        if tokenized is not None and mlm_prob is None:
            raise ValueError("MLM probability must be set if tokenized dataset is set")
        if tokenized is not None and val_prob is None:
            raise ValueError("Validation MLM probability must be set if tokenized dataset is set")
        if mlm_prob is not None and (mlm_prob < 0 or mlm_prob > 1):
            raise ValueError("MLM probability must be between 0 and 1")
        if val_prob is not None and (val_prob < 0 or val_prob > 1):
            raise ValueError("Validation MLM probability must be between 0 and 1")
        return values


class ContrastiveDataArgs(DataArgs):
    input_shards: str
    download: Optional[bool] = False
    process_one_shard: Optional[bool] = False
    streaming: Optional[bool] = True
    weighted_sampling: Optional[bool] = False
    verbose: Optional[bool] = False


class ModelArgs(BaseModel):
    model_type: str
    logit_scale: Optional[float] = 1 / 0.07
    trainable_logit_scale: Optional[bool] = False
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
    gradient_checkpointing: Optional[bool] = False
    hamming: Optional[bool] = False

    @validator('logit_scale')
    def set_logit_scale(cls, scale):
        return scale or 1 / 0.07

    @validator('model_type')
    def validate_model_type(cls, model_type):
        if model_type not in ["encoder", "mlm", "glue"]:
            raise ValueError(f"Model type {model_type} not found in model registry")
        return model_type


class Config(BaseModel):
    train_args: TrainArgs
    mlm_data_args: Optional[MLMDataArgs]
    contrastive_data_args: Optional[ContrastiveDataArgs]
    model_args: Optional[ModelArgs]
    deepspeed: Optional[bool] = False
    deepspeed_config: Optional[dict] = None

    @root_validator
    def check_args(cls, values):
        mlm, contrastive = values.get("mlm_data_args"), values.get("contrastive_data_args")

        # Check if either arg1 or arg2 is set, but not both
        if (mlm is None and contrastive is None) or (mlm is not None and contrastive is not None):
            raise ValueError('Either arg1 or arg2 must be set, but not both')

        return values
