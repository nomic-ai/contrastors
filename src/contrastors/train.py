import logging
import os
from argparse import ArgumentParser, ArgumentTypeError

import deepspeed
import torch
import torch.distributed as dist

from contrastors.read import read_config
from contrastors.trainers import TRAINER_REGISTRY
import datetime

logger = logging.getLogger('s3fs')
logger.setLevel(logging.DEBUG)  # Set the logging level to DEBUG

# Create a console handler and set its level to DEBUG
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# Create a formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add the formatter to the console handler
ch.setFormatter(formatter)

# Add the console handler to the logger
logger.addHandler(ch)

DTYPE_MAPPING = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')



def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--local_rank", type=int, default=-1)

    parser = deepspeed.add_config_arguments(parser)

    train_args = parser.add_argument_group("train_args")
    # TODO: set wandb name and output_dir the same
    train_args.add_argument("--output_dir", type=str, help="Output directory")
    train_args.add_argument("--learning_rate", type=float, help="Learning rate")
    train_args.add_argument("--weight_decay", type=float, help="Weight decay")
    train_args.add_argument("--router_aux_loss_coef", type=float, help="Auxiliary loss coefficient for router")
    train_args.add_argument("--num_epochs", type=int, help="Number of epochs")

    model_args = parser.add_argument_group("model_args")
    model_args.add_argument("--seq_len", type=int, help="Sequence length")
    model_args.add_argument("--gradient_checkpointing", type=str2bool, help="Gradient checkpointing", nargs='?', const=True, default=None)
    model_args.add_argument("--num_experts", type=int, help="Number of experts")
    model_args.add_argument("--moe_top_k", type=int, help="Top-k for router")
    model_args.add_argument("--moe_normalize_expert_weights", type=str2bool, help="Normalize expert weights", nargs='?', const=True, default=None)
    model_args.add_argument("--expert_choice_router", type=str2bool, help="Use expert choice router", nargs='?', const=True, default=None)
    model_args.add_argument("--resid_pdrop", type=float, help="Residual dropout")
    model_args.add_argument("--ffn_div", type=int, help="Fine grained expert segmentation of ffn, how much to divide the ffn dimension")
    model_args.add_argument("--model_name", type=str, help="Model name")
    model_args.add_argument("--tokenizer_name", type=str, help="Tokenizer name")

    data_args = parser.add_argument_group("data_args")
    data_args.add_argument("--batch_size", type=int, help="Batch size")
    data_args.add_argument("--weighted_sampling", type=str2bool, help="Weighted sampling", nargs='?', const=True, default=None)
    data_args.add_argument("--seed", type=int, help="Random seed")


    return parser.parse_args()

def update_config_with_args(config, args):
    for group in ['train_args', 'model_args', 'data_args']:
        if hasattr(config, group):
            group_args = getattr(config, group)
            for key, value in vars(args).items():
                if value is not None and hasattr(group_args, key):
                    setattr(group_args, key, value)
    return config


def main(config, dtype):
    model_type = config.model_args.model_type

    trainer_cls = TRAINER_REGISTRY[model_type]

    dtype = DTYPE_MAPPING[dtype]

    trainer = trainer_cls(config, dtype)
    trainer.train()


if __name__ == "__main__":
    args = parse_args()

    if args.deepspeed_config:
        args.deepspeed = True

    if args.deepspeed:
        deepspeed.init_distributed()
    else:
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=2))
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    config = read_config(args.config)
    if args.output_dir is not None:
        args.wandb_run_name = args.output_dir.split("/")[-1]

    config = update_config_with_args(config, args)


    if args.deepspeed:
        config.deepspeed = True
        config.deepspeed_config = args.deepspeed_config

    main(config, args.dtype)
