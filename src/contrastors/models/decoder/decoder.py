import logging
import math
from typing import List, Optional

import torch
import torch.nn as nn
from flash_attn.ops.layer_norm import dropout_add_layer_norm, dropout_add_layer_norm_parallel_residual
from flash_attn.ops.rms_norm import RMSNorm, dropout_add_rms_norm, dropout_add_rms_norm_parallel_residual
from transformers import GPT2Config, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast

from contrastors.layers import Block, ParallelBlock
from contrastors.models.decoder.gpt_neox import remap_state_dict_hf_gpt_neox
from contrastors.models.decoder.open_lm import remap_state_dict_hf_open_lm
from contrastors.models.model_utils import state_dict_from_pretrained

logger = logging.getLogger(__name__)


class DecoderPretrainedModel(PreTrainedModel):
    config_class = GPT2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["ParallelBlock"]
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
        Instantiate a GPTPreTrainedModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.
        """
        # Instantiate model.
        safe_serialization = kwargs.pop("safe_serialization", False)
        model = cls(config, **kwargs)
        # Load state_dict in cpu because we already initialized the model in GPU, and we don't
        # want extra stuff taking up more GPU memory
        state_dict = state_dict_from_pretrained(
            model_name, device="cpu", dtype=dtype, safe_serialization=safe_serialization
        )
        if (
            model_name.startswith("EleutherAI/gpt-neox-")
            or model_name.startswith("EleutherAI/pythia-")
            or model_name.startswith("nomic-ai/pythia-")
        ):
            state_dict = remap_state_dict_hf_gpt_neox(state_dict, config)
        elif model_name.startswith("nomic-ai/open_lm_"):
            state_dict = remap_state_dict_hf_open_lm(state_dict, config)
        else:
            raise NotImplementedError(f"Model {model_name} not supported")

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
        if isinstance(module, DecoderModel):
            module.gradient_checkpointing = value


class DecoderModel(DecoderPretrainedModel):
    def __init__(self, config: GPT2Config):
        super().__init__(config)
        assert config.activation_function in [
            "gelu",
            "gelu_new",
            "gelu_fast",
            "gelu_approx",
            "relu",
            "sqrelu",
            "glu",
            "swiglu",
            "geglu",
        ]
        pad_vocab_size_multiple = getattr(config, "pad_vocab_size_multiple", 1)

        vocab_size = math.ceil(config.vocab_size / pad_vocab_size_multiple) * pad_vocab_size_multiple

        # TD [2022-07-30]: Force residual in fp32, seems to make fp16 training more stable
        self.residual_in_fp32 = getattr(config, "residual_in_fp32", False)
        # These 2 options are for OPT-350m
        self.prenorm = getattr(config, "prenorm", True)
        use_rms_norm = getattr(config, "rms_norm", False)

        # For GPT-J, GPT-NeoX
        self.parallel_block = getattr(config, "parallel_block", False)

        self.embeddings = nn.Embedding(vocab_size, config.n_embd, config.pad_token_id)

        # We change the order of dropout, residual and layer norm:
        # Instead of LN -> Attn / MLP -> Dropout -> Add, we do:
        # Dropout -> Add -> LN -> Attn / MLP, returning both the residual branch (output of Add) and
        # the main branch (output of MLP). The model definition is unchanged, but the mapping of the
        # nn.Dropout probabilities are changed.
        # This is for performance reason: we can fuse dropout + add + layer_norm.
        block = ParallelBlock if self.parallel_block else Block
        self.layers = nn.ModuleList([block(config) for _ in range(config.num_hidden_layers)])

        self.fused_dropout_add_ln = getattr(config, "fused_dropout_add_ln", False)

        if self.prenorm:
            self.drop_f = nn.Dropout(config.resid_pdrop)
            norm_cls = nn.LayerNorm if not use_rms_norm else RMSNorm
            self.ln_f = norm_cls(config.n_embd, eps=config.layer_norm_epsilon)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        is_padded_inputs: Optional[bool] = True,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            _, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            _, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            # TODO: support position ids?
            inputs_embeds = self.embeddings(input_ids)

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None

        hidden_states2 = None
        residual = None

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs)

                    return custom_forward

                hidden_states, hidden_states2, residual = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden_states,
                    hidden_states2,
                    residual,
                    attention_mask,
                    position_ids,
                    None,
                    is_padded_inputs,
                    # if you freeze ANY layers, you need `use_reentrant=False`
                    # https://github.com/huggingface/transformers/issues/21381
                    # https://discuss.pytorch.org/t/checkpoint-with-no-grad-requiring-inputs-problem/19117/7
                    use_reentrant=False,
                )

            else:
                hidden_states, hidden_states2, residual = layer(
                    hidden_states,
                    hidden_states2,
                    residual,
                    attention_mask,
                    position_ids,
                    None,
                    is_padded_inputs,
                    output_attentions,
                    use_cache,
                )

        if self.prenorm:
            if not self.fused_dropout_add_ln:
                dropped = self.drop_f(hidden_states)
                if not self.parallel_block:
                    residual = (dropped + residual) if residual is not None else dropped
                else:
                    dropped2 = self.drop_f(hidden_states2)
                    residual = (residual + dropped + dropped2) if residual is not None else dropped + dropped2
                hidden_states = self.ln_f(residual.to(dtype=self.ln_f.weight.dtype))
            else:
                # Set prenorm=False here since we don't need the residual
                # TODO: what happens to hidden states 2?
                if not self.parallel_block:
                    fused_add_norm_fn = (
                        dropout_add_rms_norm if isinstance(self.ln_f, RMSNorm) else dropout_add_layer_norm
                    )
                    hidden_states = fused_add_norm_fn(
                        hidden_states,
                        residual,
                        self.ln_f.weight,
                        self.ln_f.bias,
                        self.drop_f.p if self.training else 0.0,
                        self.ln_f.eps,
                        prenorm=False,
                        residual_in_fp32=self.residual_in_fp32,
                    )
                else:
                    fused_add_norm_fn = (
                        dropout_add_rms_norm_parallel_residual
                        if isinstance(self.ln_f, RMSNorm)
                        else dropout_add_layer_norm_parallel_residual
                    )
                    hidden_states, _ = fused_add_norm_fn(
                        hidden_states,
                        hidden_states2,
                        residual,
                        self.ln_f.weight,
                        self.ln_f.bias,
                        None,
                        None,
                        self.drop_f.p if self.training else 0.0,
                        self.ln_f.eps,
                        prenorm=False,
                        residual_in_fp32=self.residual_in_fp32,
                    )

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )
