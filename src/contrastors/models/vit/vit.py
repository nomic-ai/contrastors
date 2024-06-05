import logging
import math

import torch
import torch.nn as nn
from flash_attn.ops.layer_norm import dropout_add_layer_norm
from flash_attn.ops.rms_norm import RMSNorm
from torchvision.ops import StochasticDepth
from transformers import GPT2Config, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast

from contrastors.layers import Block, PatchEmbedding
from contrastors.models.model_utils import state_dict_from_pretrained
from contrastors.models.vit.clip import remap_state_dict_hf_clip
from contrastors.models.vit.dinov2 import remap_state_dict_hf_dinov2
from contrastors.models.vit.hf_vit import remap_state_dict_hf_vit
from contrastors.models.vit.timm_vit import remap_timm_state_dict

logger = logging.getLogger(__name__)


class ViTPretrainedModel(PreTrainedModel):
    config_class = GPT2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Block"]
    _skip_keys_device_placement = "past_key_values"

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config)
        if not isinstance(config, GPT2Config):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `GPT2Config`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                )
            )
        self.config = config

    @classmethod
    def from_pretrained(
        cls,
        model_name,
        config,
        strict=True,
        dtype=None,
        **kwargs,
    ):
        """
        Instantiate a ViTPreTrainedModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.
        """
        # Instantiate model.
        safe_serialization = kwargs.pop("safe_serialization", False)
        model = cls(config, **kwargs)
        state_dict = state_dict_from_pretrained(
            model_name, device="cpu", dtype=dtype, safe_serialization=safe_serialization
        )
        if model_name.startswith("facebook/dinov2-"):
            state_dict = remap_state_dict_hf_dinov2(state_dict, config)
        elif model_name.startswith("laion/CLIP-ViT") or model_name.startswith("openai/clip-vit"):
            # TODO (zanussbaum): what should we do about the projection layer
            vison_only_state_dict = {k: v for k, v in state_dict.items() if "vision_model" in k}
            state_dict = remap_state_dict_hf_clip(vison_only_state_dict, config)
        elif (
            model_name.startswith("google/vit")
            or model_name.startswith("facebook/dino-vit")
            or model_name.startswith("facebook/vit-mae")
        ):
            state_dict = remap_state_dict_hf_vit(state_dict, config)
        elif model_name.startswith("timm/") or "eva02" in model_name:
            state_dict = remap_timm_state_dict(state_dict, config)
        else:
            print(f"WARNING: No remapping for model {model_name}")

        load_return = model.load_state_dict(state_dict, strict=strict)
        logger.info(load_return)
        return model

    # https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
    def _init_weights(self, module, initializer_range=0.02, rescale_prenorm_residual=True):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=initializer_range)

        if rescale_prenorm_residual:
            # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
            #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
            #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
            #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
            #
            # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
            for name, p in module.named_parameters():
                if name in ["out_proj.weight", "fc2.weight"]:
                    # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                    nn.init.normal_(p, mean=0.0, std=initializer_range / math.sqrt(2 * self.config.n_layer))

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, ViTModel):
            module.gradient_checkpointing = value


class ViTModel(ViTPretrainedModel):
    def __init__(self, config: GPT2Config):
        super().__init__(config)
        if config.activation_function == "gelu_python":
            config.activation_function = "gelu"
        assert config.activation_function in [
            "gelu",
            "gelu_new",
            "gelu_fast",
            "gelu_approx",
            "quick_gelu",
            "relu",
            "sqrelu",
            "glu",
            "swiglu",
            "geglu",
        ]
        self.prenorm = getattr(config, "prenorm", True)
        use_rms_norm = getattr(config, "rms_norm", False)
        self.embeddings = PatchEmbedding(config)
        norm_cls = nn.LayerNorm if not use_rms_norm else RMSNorm
        self.prepre_layernom = (
            norm_cls(config.n_embd, eps=config.layer_norm_epsilon)
            if getattr(config, "prepre_layernom", False)
            else nn.Identity()
        )

        dpr = [
            x.item() for x in torch.linspace(0, config.drop_path_rate, config.n_layer)
        ]  # stochastic depth decay rule

        # We change the order of dropout, residual and layer norm:
        # Instead of LN -> Attn / MLP -> Dropout -> Add, we do:
        # Dropout -> Add -> LN -> Attn / MLP, returning both the residual branch (output of Add) and
        # the main branch (output of MLP). The model definition is unchanged, but the mapping of the
        # nn.Dropout probabilities are changed.
        # This is for performance reason: we can fuse dropout + add + layer_norm.
        self.layers = nn.ModuleList(
            [
                Block(
                    config,
                    drop_path_rate1=dpr[i - 1] if i > 0 else 0.0,
                    drop_path_rate2=dpr[i],
                )
                for i in range(config.num_hidden_layers)
            ]
        )

        self.fused_dropout_add_ln = getattr(config, "fused_dropout_add_ln", False)

        self.dropout = nn.Dropout(p=config.resid_pdrop)
        self.drop_path = StochasticDepth(p=dpr[-1], mode="row")

        self.fused_dropout_add_ln = config.fused_dropout_add_ln
        self.global_pool = getattr(config, "global_pool", None)
        self.num_prefix_tokens = (1 if not getattr(config, "no_cls_token", False) else 0) + getattr(
            config, "register_tokens", 0
        )

        if self.prenorm and not getattr(config, "no_last_ln", False):
            norm_cls = nn.LayerNorm if not use_rms_norm else RMSNorm
            self.ln_f = norm_cls(config.n_embd, eps=config.layer_norm_epsilon)
        else:
            self.ln_f = None

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, input_ids, **kwargs):
        embeddings, rope = self.embeddings(input_ids)

        original_dtype = embeddings.dtype
        embeddings = self.prepre_layernom(embeddings).to(dtype=original_dtype)

        hidden_states = embeddings
        # unused but easier to pass to gradient checkpointing as words
        residual = None
        hidden_states2 = None
        attention_mask = None
        position_ids = None
        past_key_value = None
        is_padded_inputs = False
        output_attentions = False
        use_cache = False
        cu_seqlens = None
        max_seq_len = None
        kv_hidden_states = None
        kv_indices = None
        kv_cu_seqlens = None
        kv_max_seqlen = None
        for layer in self.layers:
            # need to pass none for backwards compatability
            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs)

                    return custom_forward

                hidden_states, _, residual = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden_states,
                    hidden_states2,
                    residual,
                    attention_mask,
                    position_ids,
                    past_key_value,
                    is_padded_inputs,
                    output_attentions,
                    use_cache,
                    cu_seqlens,
                    max_seq_len,
                    kv_hidden_states,
                    kv_indices,
                    kv_cu_seqlens,
                    kv_max_seqlen,
                    rope,
                    # if you freeze ANY layers, you need `use_reentrant=False`
                    # https://github.com/huggingface/transformers/issues/21381
                    # https://discuss.pytorch.org/t/checkpoint-with-no-grad-requiring-inputs-problem/19117/7
                    use_reentrant=False,
                )
            else:
                hidden_states, _, residual = layer(
                    hidden_states, None, residual=residual, is_padded_inputs=False, rope=rope
                )

        if self.ln_f is not None and self.global_pool is None:
            if not self.fused_dropout_add_ln:
                residual = self.drop_path(self.dropout(hidden_states)) + residual
                hidden_states = self.ln_f(residual.to(dtype=self.ln_f.weight.dtype))
            else:
                if self.drop_path.p == 0 or not self.training:
                    rowscale = None
                else:
                    rowscale = self.drop_path(
                        torch.ones(
                            hidden_states.shape[:-1],
                            device=hidden_states.device,
                            dtype=hidden_states.dtype,
                        )
                    )
                # Set prenorm=False here since we don't need to the residual
                hidden_states = dropout_add_layer_norm(
                    hidden_states,
                    residual,
                    self.ln_f.weight,
                    self.ln_f.bias,
                    self.dropout.p if self.training else 0.0,
                    self.ln_f.eps,
                    rowscale=rowscale,
                    prenorm=False,
                    residual_in_fp32=True,
                )
        else:
            # eva style models don't have this last layernorm
            hidden_states = self.drop_path(self.dropout(hidden_states)) + residual
            if self.global_pool == "avg":
                hidden_states = hidden_states[:, self.num_prefix_tokens :].mean(dim=1)

            if self.ln_f is not None:
                hidden_states = self.ln_f(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            hidden_states=None,
        )
