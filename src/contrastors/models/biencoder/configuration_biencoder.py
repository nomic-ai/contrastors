from transformers.configuration_utils import PretrainedConfig


class BiEncoderConfig(PretrainedConfig):
    def __init__(
        self,
        model_name="EleutherAI/pythia-1b",
        projection_dim=None,
        logit_scale=1 / 0.07,
        use_fused_kernels=True,
        pooling="last",
        encoder=False,
        freeze=False,
        trainable_logit_scale=False,
        hamming=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.projection_dim = projection_dim
        self.logit_scale = logit_scale
        self.trainable_logit_scale = trainable_logit_scale
        self.use_fused_kernels = use_fused_kernels
        self.pooling = pooling
        self.encoder = encoder
        self.freeze = freeze
        self.hamming = hamming
