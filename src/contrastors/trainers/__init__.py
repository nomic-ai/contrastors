from .base import *
from .glue import *
from .image_text import *
from .mlm import *
from .text_text import *
from .mmlm import *
from .distill import *

TRAINER_REGISTRY = {
    "mlm": MLMTrainer,
    "mmlm": MMLMTrainer,
    "glue": GlueTrainer,
    "encoder": TextTextTrainer,
    "clip": ImageTextTrainer,
    "locked_text": ImageTextTrainer,
    "distill": DistillTrainer,
}
