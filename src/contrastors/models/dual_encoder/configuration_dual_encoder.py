from typing import Any, Dict

from transformers.configuration_utils import PretrainedConfig

from contrastors.models.biencoder import BiEncoderConfig


class DualEncoderConfig(PretrainedConfig):
    def __init__(
        self,
        config=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if config:
            self.text_model_args = BiEncoderConfig(**config.text_model_args.dict())
            self.image_model_args = BiEncoderConfig(**config.vision_model_args.dict())
            if config.tower_model_args is not None:
                self.tower_model_args = BiEncoderConfig(**config.tower_model_args.dict())
            else:
                self.tower_model_args = None

        else:
            self.text_model_args = BiEncoderConfig()
            self.image_model_args = BiEncoderConfig()
            self.tower_model_args = None

        self.projection_dim = self.text_model_args.projection_dim

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], **kwargs) -> PretrainedConfig:
        if kwargs.get("return_unused_kwargs", False):
            config, _ = super().from_dict(config_dict, **kwargs)
        else:
            config = super().from_dict(config_dict, **kwargs)

        for modality in ["text_model_args", "image_model_args", "tower_model_args"]:
            if config_dict.get(modality):
                cleaned_config = config_dict[modality]
                setattr(config, modality, BiEncoderConfig(**cleaned_config))
            else:
                setattr(config, modality, None)

        if kwargs.get("return_unused_kwargs", False):
            return config, {}
        else:
            return config
