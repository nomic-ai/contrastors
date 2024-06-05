from .base import *
from .glue import *
from .image_text import *
from .mlm import *
from .text_text import *

TRAINER_REGISTRY = {
    "mlm": MLMTrainer,
    "glue": GlueTrainer,
    "encoder": TextTextTrainer,
    "clip": ImageTextTrainer,
    "locked_text": ImageTextTrainer,
}
