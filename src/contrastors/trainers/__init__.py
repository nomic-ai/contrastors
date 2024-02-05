from .base import *
from .clip import *
from .glue import *
from .mlm import *

TRAINER_REGISTRY = {
    "mlm": MLMTrainer,
    "glue": GlueTrainer,
    "encoder": CLIPTrainer,
}
