import logging
from argparse import ArgumentParser

import deepspeed
import torch
import torch.distributed as dist

from contrastors.read import read_config
from contrastors.trainers import TRAINER_REGISTRY

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


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--local_rank", type=int, default=-1)

    parser = deepspeed.add_config_arguments(parser)

    return parser.parse_args()


def main(config, dtype):
    model_type = config.model_args.model_type

    trainer_cls = TRAINER_REGISTRY[model_type]

    dtype = DTYPE_MAPPING[dtype]

    trainer = trainer_cls(config, dtype)
    trainer.train()


if __name__ == "__main__":
    args = parse_args()

    if args.deepspeed:
        deepspeed.init_distributed()
    else:
        dist.init_process_group()
        torch.cuda.set_device(dist.get_rank())

    config = read_config(args.config)
    if args.deepspeed:
        config.deepspeed = True
        config.deepspeed_config = args.deepspeed_config

    main(config, args.dtype)
