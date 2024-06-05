from transformers.configuration_utils import PretrainedConfig


class BiEncoderConfig(PretrainedConfig):
    def __init__(
        self,
        model_name="EleutherAI/pythia-1b",
        projection_dim=None,
        logit_scale=1 / 0.07,
        use_fused_kernels=True,
        pooling="last",
        nomic_encoder=False,
        freeze=False,
        trainable_logit_scale=False,
        hamming=False,
        pretrained=False,
        gradient_checkpointing=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.projection_dim = projection_dim
        self.logit_scale = logit_scale
        self.trainable_logit_scale = trainable_logit_scale
        self.use_fused_kernels = use_fused_kernels
        self.pooling = pooling
        self.nomic_encoder = nomic_encoder
        self.freeze = freeze
        self.hamming = hamming
        self.pretrained = pretrained
        self.gradient_checkpointing = gradient_checkpointing
