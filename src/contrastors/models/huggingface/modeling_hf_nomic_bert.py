# Copyright (c) 2022, Tri Dao.
# This BERT implementation is based on our MLPerf 2.0 and MLPerf 2.1 BERT implementation.
# https://github.com/mlcommons/training_results_v2.0/blob/main/HazyResearch/benchmarks/bert/implementations/pytorch/modeling.py
# https://github.com/mlcommons/training_results_v2.1/blob/main/Azure-HazyResearch/benchmarks/bert/implementations/ND96amsr_A100_v4/modeling.py

import logging

# Inspired by https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py
import os
import re
from collections import OrderedDict
from functools import partial
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from safetensors.torch import load_file as safe_load_file
from transformers import GPT2Config, PreTrainedModel
from transformers.models.bert.modeling_bert import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    MaskedLMOutput,
    SequenceClassifierOutput,
)
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, WEIGHTS_INDEX_NAME, WEIGHTS_NAME
from transformers.utils.hub import cached_file, get_checkpoint_shard_files

from .configuration_hf_nomic_bert import NomicBertConfig

logger = logging.getLogger(__name__)


# adapted from flash attention, added safe serialization option for hf models
def state_dict_from_pretrained(model_name, safe_serialization=False, device=None, dtype=None):
    # If not fp32, then we don't want to load directly to the GPU
    mapped_device = "cpu" if dtype not in [torch.float32, None] else device
    is_sharded = False
    load_safe = False
    resolved_archive_file = None

    weights_path = os.path.join(model_name, WEIGHTS_NAME)
    weights_index_path = os.path.join(model_name, WEIGHTS_INDEX_NAME)
    safe_weights_path = os.path.join(model_name, SAFE_WEIGHTS_NAME)
    safe_weights_index_path = os.path.join(model_name, SAFE_WEIGHTS_INDEX_NAME)

    if os.path.isfile(weights_path):
        resolved_archive_file = cached_file(model_name, WEIGHTS_NAME, _raise_exceptions_for_missing_entries=False)
    elif os.path.isfile(weights_index_path):
        resolved_archive_file = cached_file(model_name, WEIGHTS_INDEX_NAME, _raise_exceptions_for_missing_entries=False)
        is_sharded = True
    elif os.path.isfile(safe_weights_path):
        resolved_archive_file = cached_file(model_name, SAFE_WEIGHTS_NAME, _raise_exceptions_for_missing_entries=False)
        load_safe = True
    elif os.path.isfile(safe_weights_index_path):
        resolved_archive_file = cached_file(
            model_name, SAFE_WEIGHTS_INDEX_NAME, _raise_exceptions_for_missing_entries=False
        )
        is_sharded = True
        load_safe = True
    else:  # Try loading from HF hub instead of from local files
        weight_name = WEIGHTS_NAME if not safe_serialization else SAFE_WEIGHTS_NAME
        resolved_archive_file = cached_file(model_name, weight_name, _raise_exceptions_for_missing_entries=False)
        if resolved_archive_file is None:
            weight_index = WEIGHTS_INDEX_NAME if not safe_serialization else SAFE_WEIGHTS_INDEX_NAME
            resolved_archive_file = cached_file(model_name, weight_index, _raise_exceptions_for_missing_entries=False)
            if resolved_archive_file is not None:
                is_sharded = True

        load_safe = safe_serialization

    if resolved_archive_file is None:
        raise EnvironmentError(f"Model name {model_name} was not found.")

    if load_safe:
        loader = partial(safe_load_file, device=mapped_device)
    else:
        loader = partial(torch.load, map_location=mapped_device)

    if is_sharded:
        # resolved_archive_file becomes a list of files that point to the different
        # checkpoint shards in this case.
        resolved_archive_file, sharded_metadata = get_checkpoint_shard_files(model_name, resolved_archive_file)
        state_dict = {}
        for sharded_file in resolved_archive_file:
            state_dict.update(loader(sharded_file))
    else:
        state_dict = loader(resolved_archive_file)
    # Convert dtype before moving to GPU to save memory
    if dtype is not None:
        state_dict = {k: v.to(dtype=dtype) for k, v in state_dict.items()}
    state_dict = {k: v.to(device=device) for k, v in state_dict.items()}
    return state_dict


def filter_shapes(state_dict, model):
    """
    Filters the state dict to match the current model shape.
    """
    filtered_state_dict = {}
    for key, value in state_dict.items():
        if key in model.state_dict():
            if value.shape == model.state_dict()[key].shape:
                filtered_state_dict[key] = value
    return filtered_state_dict


def remap_bert_state_dict(state_dict, config, remove_bert=False, remove_cls_weights=False, add_pooling_layer=False):
    """
    Map the state_dict of a Huggingface BERT model to be flash_attn compatible.
    """

    def add_bert_prefix(key):
        # prepend bert. to the key
        if key.startswith("bert.") or key.startswith("cls."):
            return key
        return f"bert.{key}"

    state_dict = OrderedDict((add_bert_prefix(k), v) for k, v in state_dict.items())

    # LayerNorm
    def key_mapping_ln_gamma_beta(key):
        key = re.sub(r"LayerNorm.gamma$", "LayerNorm.weight", key)
        key = re.sub(r"LayerNorm.beta$", "LayerNorm.bias", key)
        return key

    state_dict = OrderedDict((key_mapping_ln_gamma_beta(k), v) for k, v in state_dict.items())

    # Layers
    def key_mapping_layers(key):
        return re.sub(r"^bert.encoder.layer\.", "bert.encoder.layers.", key)

    state_dict = OrderedDict((key_mapping_layers(k), v) for k, v in state_dict.items())

    # LayerNorm
    def key_mapping_ln(key):
        key = re.sub(r"^bert.embeddings.LayerNorm.", "bert.emb_ln.", key)
        key = re.sub(
            r"^bert.encoder.layers.(\d+).attention.output.LayerNorm.(weight|bias)",
            r"bert.encoder.layers.\1.norm1.\2",
            key,
        )
        key = re.sub(
            r"^bert.encoder.layers.(\d+).output.LayerNorm.(weight|bias)",
            r"bert.encoder.layers.\1.norm2.\2",
            key,
        )
        key = re.sub(
            r"^cls.predictions.transform.LayerNorm.(weight|bias)",
            r"cls.predictions.transform.layer_norm.\1",
            key,
        )
        return key

    state_dict = OrderedDict((key_mapping_ln(k), v) for k, v in state_dict.items())

    # MLP
    def key_mapping_mlp(key):
        key = re.sub(
            r"^bert.encoder.layers.(\d+).intermediate.dense.(weight|bias)",
            r"bert.encoder.layers.\1.mlp.fc1.\2",
            key,
        )
        key = re.sub(
            r"^bert.encoder.layers.(\d+).output.dense.(weight|bias)",
            r"bert.encoder.layers.\1.mlp.fc2.\2",
            key,
        )
        return key

    state_dict = OrderedDict((key_mapping_mlp(k), v) for k, v in state_dict.items())

    # Attention
    last_layer_subset = getattr(config, "last_layer_subset", False)
    for d in range(config.num_hidden_layers):
        if f"bert.encoder.layers.{d}.attention.self.query.weight" not in state_dict:
            continue
        Wq = state_dict.pop(f"bert.encoder.layers.{d}.attention.self.query.weight")
        Wk = state_dict.pop(f"bert.encoder.layers.{d}.attention.self.key.weight")
        Wv = state_dict.pop(f"bert.encoder.layers.{d}.attention.self.value.weight")
        bq = state_dict.pop(f"bert.encoder.layers.{d}.attention.self.query.bias")
        bk = state_dict.pop(f"bert.encoder.layers.{d}.attention.self.key.bias")
        bv = state_dict.pop(f"bert.encoder.layers.{d}.attention.self.value.bias")
        if not (last_layer_subset and d == config.num_hidden_layers - 1):
            state_dict[f"bert.encoder.layers.{d}.attn.Wqkv.weight"] = torch.cat([Wq, Wk, Wv], dim=0)
            state_dict[f"bert.encoder.layers.{d}.attn.Wqkv.bias"] = torch.cat([bq, bk, bv], dim=0)
        else:
            state_dict[f"bert.encoder.layers.{d}.attn.Wq.weight"] = Wq
            state_dict[f"bert.encoder.layers.{d}.attn.Wkv.weight"] = torch.cat([Wk, Wv], dim=0)
            state_dict[f"bert.encoder.layers.{d}.attn.Wq.bias"] = bq
            state_dict[f"bert.encoder.layers.{d}.attn.Wkv.bias"] = torch.cat([bk, bv], dim=0)

    def key_mapping_attn(key):
        return re.sub(
            r"^bert.encoder.layers.(\d+).attention.output.dense.(weight|bias)",
            r"bert.encoder.layers.\1.attn.out_proj.\2",
            key,
        )

    state_dict = OrderedDict((key_mapping_attn(k), v) for k, v in state_dict.items())

    def key_mapping_decoder_bias(key):
        return re.sub(r"^cls.predictions.bias", "cls.predictions.decoder.bias", key)

    # remove nsp weights, we don't use
    state_dict.pop("cls.seq_relationship.weight", None)
    state_dict.pop("cls.seq_relationship.bias", None)
    state_dict.pop("bert.embeddings.position_ids", None)

    state_dict = OrderedDict((key_mapping_decoder_bias(k), v) for k, v in state_dict.items())

    if remove_cls_weights:
        cls_weights = [
            "cls.predictions.decoder.bias",
            "cls.predictions.transform.dense.weight",
            "cls.predictions.transform.dense.bias",
            "cls.predictions.transform.layer_norm.weight",
            "cls.predictions.transform.layer_norm.bias",
            "cls.predictions.decoder.weight",
        ]
        for weight in cls_weights:
            state_dict.pop(weight, None)

    # Word embedding
    pad_vocab_size_multiple = getattr(config, "pad_vocab_size_multiple", 1)
    if pad_vocab_size_multiple > 1:
        word_embeddings = state_dict["bert.embeddings.word_embeddings.weight"]
        state_dict["bert.embeddings.word_embeddings.weight"] = F.pad(
            word_embeddings, (0, 0, 0, config.vocab_size - word_embeddings.shape[0])
        )
        if not remove_cls_weights:
            decoder_weight = state_dict["cls.predictions.decoder.weight"]
            state_dict["cls.predictions.decoder.weight"] = F.pad(
                decoder_weight, (0, 0, 0, config.vocab_size - decoder_weight.shape[0])
            )
            # If the vocab was padded, we want to set the decoder bias for those padded indices to be
            # strongly negative (i.e. the decoder shouldn't predict those indices).
            # TD [2022-05-09]: I don't think it affects the MLPerf training.
            if "cls.predictions.decoder.bias" in state_dict:
                decoder_bias = state_dict["cls.predictions.decoder.bias"]
                state_dict["cls.predictions.decoder.bias"] = F.pad(
                    decoder_bias, (0, config.vocab_size - decoder_bias.shape[0]), value=-100.0
                )

    if add_pooling_layer is False:
        pooler_weights = [
            "bert.pooler.dense.weight",
            "bert.pooler.dense.bias",
        ]
        for key in pooler_weights:
            state_dict.pop(key, None)

    if remove_bert:

        def remove_bert_prefix(key):
            key = re.sub(r"^bert.", "", key)
            return key

        state_dict = OrderedDict((remove_bert_prefix(k), v) for k, v in state_dict.items())

    return state_dict


class NomicBertPreTrainedModel(PreTrainedModel):
    """An abstract class to handle weights initialization and
    a simple interface for dowloading and loading pretrained models.
    """

    config_class = NomicBertConfig
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
    def from_pretrained(cls, model_name, config=None, *inputs, **kwargs):
        """
        Instantiate a NomicBertPreTrainedModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.

        Params:
            pretrained_model_name_or_path: either:
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `pytorch_model.bin` a PyTorch dump of a NomicBertForPretraining instance
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `model.chkpt` a TensorFlow checkpoint
            *inputs, **kwargs: additional input for the specific NomicBert class
                (ex: num_labels for NomicBertForSequenceClassification)
        """
        # Instantiate model.
        if config is None:
            config = cls.config_class.from_pretrained(model_name)
        remove_cls = cls != NomicBertForPreTraining
        remove_bert_prefix = cls != NomicBertForPreTraining
        ignore_mismatched_shapes = kwargs.pop("ignore_mismatched_sizes", False)
        num_labels = kwargs.pop("num_labels", None)
        rotary_scaling_factor = kwargs.pop("rotary_scaling_factor", None)
        if rotary_scaling_factor:
            config.rotary_scaling_factor = rotary_scaling_factor

        if config.n_positions <= 0 and config.rotary_emb_fraction > 0:
            config.n_positions = 2048
        if num_labels:
            config.num_labels = num_labels

        if "add_pooling_layer" in kwargs:
            model = cls(config, *inputs, add_pooling_layer=kwargs.pop("add_pooling_layer"))
        else:
            if cls == NomicBertModel:
                model = cls(config, *inputs, add_pooling_layer=False)
            else:
                model = cls(config, *inputs)
        # TODO: fix this
        # Assuming we know what we're doing when loading from disk
        # Prob a bad assumption but i'm tired and want to train this asap
        if os.path.exists(model_name):
            model_path = f"{model_name}/pytorch_model.bin"
            if os.path.exists(model_path):
                state_dict = torch.load(f"{model_name}/pytorch_model.bin")
            else:
                model_path = f"{model_name}/model.safetensors"
                if not os.path.exists(model_path):
                    raise ValueError(f"Model path {model_path} not found")
                state_dict = safe_load_file(model_path)

            if ignore_mismatched_shapes:
                state_dict = filter_shapes(state_dict, model)
            load_return = model.load_state_dict(state_dict, strict=False)
        else:
            # TODO: can probably check config class and see if we need to remap from a bert model
            state_dict = state_dict_from_pretrained(
                model_name, safe_serialization=kwargs.get("safe_serialization", False)
            )
            state_dict = remap_bert_state_dict(
                state_dict,
                config,
                remove_bert=remove_bert_prefix,
                remove_cls_weights=remove_cls,
                add_pooling_layer=getattr(config, "add_pooling_layer", False),
            )
            if ignore_mismatched_shapes:
                state_dict = filter_shapes(state_dict, model)

            load_return = model.load_state_dict(state_dict, strict=True)
        logger.warning(load_return)
        return model

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, NomicBertEncoder):
            module.gradient_checkpointing = value


# https://github.com/huggingface/transformers/blob/7032e0203262ebb2ebf55da8d2e01f873973e835/src/transformers/models/bert/modeling_bert.py#L748
def _init_weights(module, initializer_range=0.02):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=initializer_range)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)
        if module.padding_idx is not None:
            nn.init.zeros_(module.weight[module.padding_idx])


class NomicBertEmbeddings(nn.Module):
    def __init__(self, config):
        """
        If max_position_embeddings <= 0, there's no position embeddings
        If type_vocab_size <= 0, there's no token type embeddings
        """
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.max_position_embeddings = config.max_position_embeddings if config.rotary_emb_fraction <= 0 else 0
        self.type_vocab_size = config.type_vocab_size
        if self.max_position_embeddings > 0 and config.rotary_emb_fraction <= 0:
            self.position_embeddings = nn.Embedding(
                config.max_position_embeddings,
                config.hidden_size,
            )
        if self.type_vocab_size > 0:
            self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

    def forward(self, input_ids, position_ids=None, token_type_ids=None):
        """
        input_ids: (batch, seqlen)
        position_ids: (batch, seqlen)
        token_type_ids: (batch, seqlen)
        """
        batch_size, seqlen = input_ids.shape
        embeddings = self.word_embeddings(input_ids)

        if self.type_vocab_size > 0:
            if token_type_ids is None:
                token_type_ids = torch.zeros(seqlen, dtype=torch.long, device=input_ids.device)
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings = embeddings + token_type_embeddings

        if self.max_position_embeddings > 0:
            if position_ids is None:
                position_ids = torch.arange(seqlen, dtype=torch.long, device=input_ids.device)
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = embeddings + position_embeddings
        return embeddings


class NomicBertMLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        activation=F.gelu,
        bias1=True,
        bias2=True,
        return_residual=False,
        fused_bias_fc=False,
    ):
        super().__init__()
        out_features = out_features if out_features is not None else in_features
        hidden_features = hidden_features if hidden_features is not None else in_features * 4
        self.return_residual = return_residual
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias1)
        approximate = "tanh" if activation in ["gelu_new", "gelu_fast", "gelu_pytorch_tanh"] else "none"
        self.activation = nn.GELU(approximate=approximate) if activation == "gelu" else activation
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias2)

    def forward(self, x):
        y = self.fc1(x)
        y = self.activation(y)
        y = self.fc2(y)
        return y if not self.return_residual else (y, x)


class NomciBertGatedMLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        activation=F.sigmoid,
        bias1=True,
        bias2=True,
        multiple_of=256,
        return_residual=False,
        fused_bias_fc=True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        out_features = out_features if out_features is not None else in_features
        hidden_features = hidden_features if hidden_features is not None else int(8 * in_features / 3)
        hidden_features = (hidden_features + multiple_of - 1) // multiple_of * multiple_of
        self.return_residual = return_residual

        self.fc11 = nn.Linear(in_features, hidden_features, bias=bias1)
        self.fc12 = nn.Linear(in_features, hidden_features, bias=bias1)
        self.activation = activation
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias2)

    def forward(self, x):
        y = self.fc11(x)
        gate = self.fc12(x)
        if self.activation == F.sigmoid:  # Special case for GLU
            y = F.glu(torch.cat([y, gate], dim=-1), dim=-1)
        else:
            y = y * self.activation(gate)
        y = self.fc2(y)
        return y if not self.return_residual else (y, x)


def rotate_half(x, interleaved=False):
    if not interleaved:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    else:
        x1, x2 = x[..., ::2], x[..., 1::2]
        return rearrange(torch.stack((-x2, x1), dim=-1), "... d two -> ... (d two)", two=2)


def apply_rotary_emb(x, cos, sin, offset=0, interleaved=False):
    """
    x: (batch_size, seqlen, nheads, headdim)
    cos, sin: (seqlen, rotary_dim / 2) or (batch_size, seqlen, rotary_dim / 2)
    """
    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= x.shape[-1]
    cos, sin = (
        cos[offset : offset + x.shape[1]],
        sin[offset : offset + x.shape[1]],
    )
    cos = repeat(cos, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")
    sin = repeat(sin, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")
    return torch.cat(
        [x[..., :ro_dim] * cos + rotate_half(x[..., :ro_dim], interleaved) * sin, x[..., ro_dim:]],
        dim=-1,
    )


class NomicBertRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        base=10000.0,
        interleaved=False,
        scale_base=None,
        pos_idx_in_fp32=True,
        device=None,
    ):
        """
        interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead
            of 1st half and 2nd half (GPT-NeoX style).
        pos_idx_in_fp32: if True, the position indices [0.0, ..., seqlen - 1] are in fp32,
            otherwise they might be in lower precision.
            This option was added because previously (before 2023-07-02), when we construct
            the position indices, we use the dtype of self.inv_freq. In most cases this would
            be fp32, but if the model is trained in pure bf16 (not mixed precision), then
            self.inv_freq would be bf16, and the position indices are also in bf16.
            Because of the limited precision of bf16 (e.g. 1995.0 is rounded to 2000.0), the
            embeddings for some positions will coincide.
            To maintain compatibility with models previously trained in pure bf16,
            we add this option.
        """
        super().__init__()
        self.dim = dim
        self.base = float(base)
        self.pos_idx_in_fp32 = pos_idx_in_fp32
        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = self._compute_inv_freq(device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.interleaved = interleaved
        self.scale_base = scale_base
        scale = (
            (torch.arange(0, dim, 2, device=device, dtype=torch.float32) + 0.4 * dim) / (1.4 * dim)
            if scale_base is not None
            else None
        )
        self.register_buffer("scale", scale, persistent=False)

        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        self._cos_k_cached = None
        self._sin_k_cached = None

    def _compute_inv_freq(self, device=None):
        return 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim))

    def _update_cos_sin_cache(self, seqlen, device=None, dtype=None):
        # Reset the tables if the sequence length has changed,
        # if we're on a new device (possibly due to tracing for instance),
        # or if we're switching from inference mode to training
        if (
            seqlen > self._seq_len_cached
            or self._cos_cached is None
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
            or (self.training and self._cos_cached.is_inference())
        ):
            self._seq_len_cached = seqlen
            # We want fp32 here, not self.inv_freq.dtype, since the model could be loaded in bf16
            # And the output of arange can be quite large, so bf16 would lose a lot of precision.
            # However, for compatibility reason, we add an option to use the dtype of self.inv_freq.
            if self.pos_idx_in_fp32:
                t = torch.arange(seqlen, device=device, dtype=torch.float32)
                # We want fp32 here as well since inv_freq will be multiplied with t, and the output
                # will be large. Having it in bf16 will lose a lot of precision and cause the
                # cos & sin output to change significantly.
                # We want to recompute self.inv_freq if it was not loaded in fp32
                if self.inv_freq.dtype != torch.float32:
                    inv_freq = self._compute_inv_freq(device=device)
                else:
                    inv_freq = self.inv_freq
            else:
                t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
                inv_freq = self.inv_freq
            # Don't do einsum, it converts fp32 to fp16 under AMP
            # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            freqs = torch.outer(t, inv_freq)
            self._cos_cached = torch.cos(freqs).to(dtype)
            self._sin_cached = torch.sin(freqs).to(dtype)

    def forward(
        self,
        qkv: torch.Tensor,
        kv: Optional[torch.Tensor] = None,
        seqlen_offset: Union[int, torch.Tensor] = 0,
        max_seqlen: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        qkv: (batch, seqlen, 3, nheads, headdim) if kv is none,
             else it's just q of shape (batch, seqlen, nheads, headdim)
        kv: (batch, seqlen, 2, nheads, headdim)
        seqlen_offset: (batch_size,) or int. Each sequence in x is shifted by this amount.
            Most commonly used in inference when we have KV cache.
            If it's a tensor of shape (batch_size,), then to update the cos / sin cache, one
            should pass in max_seqlen, which will update the cos / sin cache up to that length.
        Apply rotary embedding *inplace* to qkv and / or kv.
        """
        seqlen = qkv.shape[1]
        if seqlen > self._seq_len_cached:
            self._update_cos_sin_cache(seqlen, device=qkv.device, dtype=qkv.dtype)
        elif max_seqlen is not None:
            self._update_cos_sin_cache(max_seqlen, device=qkv.device, dtype=qkv.dtype)
        elif isinstance(seqlen_offset, int):
            self._update_cos_sin_cache(seqlen + seqlen_offset, device=qkv.device, dtype=qkv.dtype)

        q_rot = apply_rotary_emb(qkv[:, :, 0], self._cos_cached, self._sin_cached, seqlen_offset, self.interleaved)
        k_rot = apply_rotary_emb(qkv[:, :, 1], self._cos_cached, self._sin_cached, seqlen_offset, self.interleaved)
        return torch.stack((q_rot, k_rot, qkv[:, :, 2]), dim=2)


class NomicBertDynamicNTKRotaryEmbedding(NomicBertRotaryEmbedding):
    def __init__(self, rotary_scaling_factor, max_position_embeddings, **kwargs):
        super().__init__(**kwargs)
        self.rotary_scaling_factor = rotary_scaling_factor
        self.max_position_embeddings = max_position_embeddings

    def _compute_inv_freq(self, base=None, device=None):
        if base is None:
            base = self.base
        return 1.0 / (base ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim))

    def _update_cos_sin_cache(self, seqlen, device=None, dtype=None):
        # Reset the tables if the sequence length has changed,
        # if we're on a new device (possibly due to tracing for instance),
        # or if we're switching from inference mode to training
        if seqlen > self.max_position_embeddings:
            base = self.base * (
                (self.rotary_scaling_factor * seqlen / self.max_position_embeddings) - (self.rotary_scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = self._compute_inv_freq(base=base, device=device)
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        if (
            seqlen > self._seq_len_cached
            or self._cos_cached is None
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
            or (self.training and self._cos_cached.is_inference())
        ):
            self._seq_len_cached = seqlen
            # We want fp32 here, not self.inv_freq.dtype, since the model could be loaded in bf16
            # And the output of arange can be quite large, so bf16 would lose a lot of precision.
            # However, for compatibility reason, we add an option to use the dtype of self.inv_freq.
            if self.pos_idx_in_fp32:
                t = torch.arange(seqlen, device=device, dtype=torch.float32)
                # We want fp32 here as well since inv_freq will be multiplied with t, and the output
                # will be large. Having it in bf16 will lose a lot of precision and cause the
                # cos & sin output to change significantly.
                # We want to recompute self.inv_freq if it was not loaded in fp32
                if self.inv_freq.dtype != torch.float32:
                    if seqlen > self.max_position_embeddings:
                        base = self.base * (
                            (self.scaling_factor * seqlen / self.max_position_embeddings) - (self.scaling_factor - 1)
                        ) ** (self.dim / (self.dim - 2))
                    else:
                        base = self.base
                    inv_freq = self._compute_inv_freq(device=device, base=base)
                else:
                    inv_freq = self.inv_freq
            else:
                t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
                inv_freq = self.inv_freq
            # Don't do einsum, it converts fp32 to fp16 under AMP
            # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            freqs = torch.outer(t, inv_freq)
            if self.scale is None:
                self._cos_cached = torch.cos(freqs).to(dtype)
                self._sin_cached = torch.sin(freqs).to(dtype)
            else:
                power = (
                    torch.arange(seqlen, dtype=self.scale.dtype, device=self.scale.device) - seqlen // 2
                ) / self.scale_base
                scale = self.scale.to(device=power.device) ** rearrange(power, "s -> s 1")
                # We want the multiplication by scale to happen in fp32
                self._cos_cached = (torch.cos(freqs) * scale).to(dtype)
                self._sin_cached = (torch.sin(freqs) * scale).to(dtype)
                self._cos_k_cached = (torch.cos(freqs) / scale).to(dtype)
                self._sin_k_cached = (torch.sin(freqs) / scale).to(dtype)


class NomicBertAttention(nn.Module):
    """Multi-head self-attention and cross-attention"""

    def __init__(
        self,
        config,
    ) -> None:
        """
        num_heads_kv: can be used to toggle MQA / GQA. If None, use num_heads.
        return_residual: whether to return the input x along with the output. This is for
            performance reason: for post-norm architecture, returning the input allows us
            to fuse the backward of nn.Linear with the residual connection.
        """
        super().__init__()
        self.embed_dim = config.n_embd
        self.use_flash_attn = config.use_flash_attn
        self.fused_bias_fc = config.fused_bias_fc

        self.num_heads = config.n_head
        self.num_heads_kv = config.num_heads_kv if getattr(config, "num_heads_kv", None) is not None else self.num_heads
        assert self.embed_dim % self.num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = self.embed_dim // self.num_heads
        # we don't really support mqa / gqa for now
        qkv_dim = self.head_dim * (self.num_heads + 2 * self.num_heads_kv)

        self.register_buffer(
            "norm_factor",
            torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)).to(torch.get_default_dtype()),
            persistent=False,
        )

        self.rotary_emb_dim = self.head_dim * config.rotary_emb_fraction
        if self.rotary_emb_dim > 0:
            if config.rotary_scaling_factor:
                self.rotary_emb = NomicBertDynamicNTKRotaryEmbedding(
                    dim=self.rotary_emb_dim,
                    base=config.rotary_emb_base,
                    scale_base=config.rotary_emb_scale_base,
                    interleaved=config.rotary_emb_interleaved,
                    rotary_scaling_factor=config.rotary_scaling_factor,
                    max_position_embeddings=config.max_trained_positions,
                )
            else:
                self.rotary_emb = NomicBertRotaryEmbedding(
                    dim=self.rotary_emb_dim,
                    base=config.rotary_emb_base,
                    scale_base=config.rotary_emb_scale_base,
                    interleaved=config.rotary_emb_interleaved,
                )
            # bug in xformers: https://github.com/facebookresearch/xformers/issues/841
            # uses the head dimension instead of the sequence dimension
            self.rotary_head_dim = getattr(config, "rotary_head_dim", False)

        self.Wqkv = nn.Linear(self.embed_dim, qkv_dim, bias=config.qkv_proj_bias)

        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.qkv_proj_bias)
        self.causal = config.causal
        self.drop = nn.Dropout(config.attn_pdrop)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        is_padded_inputs: Optional[bool] = True,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seq_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        has_layer_past = past_key_value is not None

        if has_layer_past:
            past_key_value = past_key_value[0]
            past_len = past_key_value[1]
        else:
            past_len = 0

        qkv = self.Wqkv(hidden_states)
        qkv = rearrange(qkv, "... (three h d) -> ... three h d", three=3, d=self.head_dim)

        past_key_value = (past_key_value, past_len + qkv.size(1)) if use_cache else None

        if self.rotary_emb_dim > 0:
            if self.rotary_head_dim:
                qkv = rearrange(qkv, "b s three h d -> b h three s d")
            qkv = self.rotary_emb(qkv, seqlen_offset=past_len)

            if self.rotary_head_dim:
                qkv = rearrange(qkv, "b h three s d -> b s three h d")

        query, key, value = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)

        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / self.norm_factor
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attentions_probs = F.softmax(attention_scores, dim=-1)
        attentions_probs = self.drop(attentions_probs)

        attn_output = torch.matmul(attentions_probs, value)
        attn_output = rearrange(attn_output.permute(0, 2, 1, 3), "... h d -> ... (h d)")

        attn_output = self.out_proj(attn_output)

        return attn_output


class NomicBertBlock(nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.prenorm = config.prenorm
        self.fused_dropout_add_ln = config.fused_dropout_add_ln

        self.attn = NomicBertAttention(config)
        activation = (
            F.sigmoid
            if config.activation_function == "glu"
            else (F.silu if config.activation_function == "swiglu" else F.gelu)
        )
        if config.activation_function in ["glu", "swiglu", "geglu"]:
            self.mlp = NomciBertGatedMLP(
                config.n_embd,
                hidden_features=config.n_inner,
                bias1=config.mlp_fc1_bias,
                bias2=config.mlp_fc2_bias,
                activation=activation,
                fused_bias_fc=config.fused_bias_fc,
            )
        else:
            self.mlp = NomicBertMLP(
                config.n_embd,
                hidden_features=config.n_inner,
                bias1=config.mlp_fc1_bias,
                bias2=config.mlp_fc2_bias,
                activation=activation,
                fused_bias_fc=config.fused_bias_fc,
            )

        self.dropout1 = nn.Dropout(config.resid_pdrop)
        self.norm1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.norm2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.dropout2 = nn.Dropout(config.resid_pdrop)

    def forward(
        self,
        hidden_states: torch.Tensor,
        hidden_states2: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        is_padded_inputs: Optional[bool] = True,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seq_len: Optional[int] = None,
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: if postnorm, residual=None, If prenorm, hidden_states = Attn/MLP(LN(residual))
            mixer_subset: for cross-attention only. If not None, will take a subset of x
                before applying the query projection. Useful for e.g., ViT where we only care
                about the CLS token in the last layer.
        """
        if self.prenorm:
            dropped = self.dropout1(hidden_states)
            residual = (dropped + residual) if residual is not None else dropped
            hidden_states = self.norm1(residual.to(dtype=self.norm1.weight.dtype))
            hidden_states = self.attn(
                hidden_states,
                attention_mask=attention_mask,
                is_padded_inputs=is_padded_inputs,
                cu_seqlens=cu_seqlens,
                max_seq_len=max_seq_len,
            )

            dropped = self.dropout2(hidden_states)
            residual = (dropped + residual) if residual is not None else dropped
            hidden_states = self.norm2(residual.to(dtype=self.norm2.weight.dtype))
            hidden_states = self.mlp(hidden_states)

            return hidden_states, None, residual
        else:
            assert residual is None
            attn_outputs = self.attn(
                hidden_states,
                attention_mask=attention_mask,
                is_padded_inputs=is_padded_inputs,
                cu_seqlens=cu_seqlens,
                max_seq_len=max_seq_len,
            )
            hidden_states = self.norm1((self.dropout1(attn_outputs) + hidden_states).to(dtype=self.norm1.weight.dtype))
            mlp_out = self.mlp(hidden_states)

            hidden_states = self.norm2((self.dropout2(mlp_out) + hidden_states).to(dtype=self.norm2.weight.dtype))
            return hidden_states, None, None


class NomicBertEncoder(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.layers = nn.ModuleList([NomicBertBlock(config) for _ in range(config.n_layer)])
        self.gradient_checkpointing = False
        self.config = config

    def forward(
        self,
        hidden_states: torch.LongTensor = None,
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
        """If subset_mask is not None, we only want output for the subset of the sequence.
        This means that we only compute the last layer output for these tokens.
        subset_mask: (batch, seqlen), dtype=torch.bool
        """
        hidden_states2 = None
        residual = None

        for _, layer in enumerate(self.layers):
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
                    None,
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
        return hidden_states


class NomicBertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.n_embd, config.n_embd)
        self.activation = nn.Tanh()

    def forward(self, hidden_states, pool=True):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0] if pool else hidden_states
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class NomicBertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.n_embd, config.n_embd, bias=config.mlp_fc1_bias)
        approximate = "tanh" if config.activation_function in ["gelu_new", "gelu_fast", "gelu_pytorch_tanh"] else "none"
        if config.activation_function == "swiglu":
            self.transform_act_fn = F.silu
        else:
            self.transform_act_fn = nn.GELU(approximate=approximate)

        self.layer_norm = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.layer_norm(hidden_states)

        return hidden_states


class NomicBertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.transform = NomicBertPredictionHeadTransform(config)

        self.decoder = nn.Linear(config.n_embd, config.vocab_size, bias=config.mlp_fc1_bias)

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class NomicBertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = NomicBertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class NomicBertModel(NomicBertPreTrainedModel):
    def __init__(self, config: GPT2Config, add_pooling_layer=True):
        super().__init__(config)
        self.pad_vocab_size_multiple = getattr(config, "pad_vocab_size_multiple", 1)
        if config.vocab_size % self.pad_vocab_size_multiple != 0:
            config.vocab_size += self.pad_vocab_size_multiple - (config.vocab_size % self.pad_vocab_size_multiple)

        assert config.activation_function in [
            "gelu",
            "gelu_new",
            "gelu_fast",
            "gelu_pytorch_tanh",
            "swiglu",
            "geglu",
            "glu",
        ]

        self.embeddings = NomicBertEmbeddings(config)
        self.emb_drop = nn.Dropout(config.resid_pdrop)
        self.emb_ln = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.encoder = NomicBertEncoder(config)
        self.pooler = NomicBertPooler(config) if add_pooling_layer else None

        self.apply(partial(_init_weights, initializer_range=config.initializer_range))

    def forward(
        self,
        input_ids,
        position_ids=None,
        token_type_ids=None,
        attention_mask=None,
        return_dict=None,
        matryoshka_dim=None,
    ):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        hidden_states = self.embeddings(input_ids, position_ids=position_ids, token_type_ids=token_type_ids)
        hidden_states = self.emb_ln(hidden_states)
        hidden_states = self.emb_drop(hidden_states)

        attention_mask = self.get_extended_attention_mask(attention_mask, input_ids.shape)
        sequence_output = self.encoder(hidden_states, attention_mask=attention_mask, return_dict=return_dict)

        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if matryoshka_dim:
            sequence_output = sequence_output[:, :matryoshka_dim]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
        )


class NomicBertForPreTraining(NomicBertPreTrainedModel):
    _tied_weights_keys = ["predictions.decoder.bias", "cls.predictions.decoder.weight"]

    def __init__(self, config: GPT2Config):
        super().__init__(config)

        self.bert = NomicBertModel(config, add_pooling_layer=getattr(config, "add_pooling_layer", False))
        self.cls = NomicBertPreTrainingHeads(config)
        self.mlm_loss = nn.CrossEntropyLoss()

        # Initialize weights and apply final processing
        self.apply(partial(_init_weights, initializer_range=config.initializer_range))
        self.tie_weights()

    def tie_weights(self):
        self.cls.predictions.decoder.weight = self.bert.embeddings.word_embeddings.weight

    def forward(
        self,
        input_ids,
        position_ids=None,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
    ):
        """
        If labels are provided, they must be -100 for masked out tokens (as specified in the attention
        mask).
        Outputs:
            if `labels` and `next_sentence_label` are not `None`:
                Outputs the total_loss which is the sum of the masked language modeling loss and the next
                sentence classification loss.
            if `labels` or `next_sentence_label` is `None`:
                Outputs a tuple comprising
                - the masked language modeling logits of shape [batch_size, sequence_length, vocab_size], and
                - the next sentence classification logits of shape [batch_size, 2].

        """
        outputs = self.bert(
            input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask.bool() if attention_mask is not None else None,
        )
        sequence_output, _ = outputs.last_hidden_state, outputs.pooler_output

        prediction_scores = self.cls(sequence_output)

        total_loss = None
        if labels is not None:
            masked_lm_loss = self.mlm_loss(
                rearrange(prediction_scores, "... v -> (...) v"),
                rearrange(labels, "... -> (...)"),
            )
            total_loss = masked_lm_loss.float()

        return MaskedLMOutput(
            loss=total_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=None,
        )


class NomicBertForSequenceClassification(NomicBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = NomicBertModel(config)
        classifier_dropout = getattr(config, "classifier_dropout", config.embd_pdrop)
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.n_embd, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.bert(
            input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask.bool() if attention_mask is not None else None,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
