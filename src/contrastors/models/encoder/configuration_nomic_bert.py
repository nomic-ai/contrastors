from transformers import GPT2Config


class NomicBertConfig(GPT2Config):
    model_type = "nomic_bert"

    def __init__(
        self,
        prenorm=False,
        parallel_block=False,
        parallel_block_tied_norm=False,
        rotary_emb_fraction=0.0,
        fused_dropout_add_ln=False,
        fused_bias_fc=False,
        use_flash_attn=False,
        use_xentropy=False,
        qkv_proj_bias=True,
        rotary_emb_base=1000,
        rotary_emb_scale_base=None,
        rotary_emb_interleaved=False,
        mlp_fc1_bias=True,
        mlp_fc2_bias=True,
        use_rms_norm=False,
        causal=False,
        type_vocab_size=2,
        dense_seq_output=True,
        pad_vocab_size_multiple=1,
        tie_word_embeddings=True,
        rotary_scaling_factor=1.0,
        max_trained_positions=2048,
        **kwargs,
    ):
        self.prenorm = prenorm
        self.parallel_block = parallel_block
        self.parallel_block_tied_norm = parallel_block_tied_norm
        self.rotary_emb_fraction = rotary_emb_fraction
        self.tie_word_embeddings = tie_word_embeddings
        self.fused_dropout_add_ln = fused_dropout_add_ln
        self.fused_bias_fc = fused_bias_fc
        self.use_flash_attn = use_flash_attn
        self.use_xentropy = use_xentropy
        self.qkv_proj_bias = qkv_proj_bias
        self.rotary_emb_base = rotary_emb_base
        self.rotary_emb_scale_base = rotary_emb_scale_base
        self.rotary_emb_interleaved = rotary_emb_interleaved
        self.mlp_fc1_bias = mlp_fc1_bias
        self.mlp_fc2_bias = mlp_fc2_bias
        self.use_rms_norm = use_rms_norm
        self.causal = causal
        self.type_vocab_size = type_vocab_size
        self.dense_seq_output = dense_seq_output
        self.pad_vocab_size_multiple = pad_vocab_size_multiple
        self.rotary_scaling_factor = rotary_scaling_factor
        self.max_trained_positions = max_trained_positions

        super().__init__(**kwargs)
