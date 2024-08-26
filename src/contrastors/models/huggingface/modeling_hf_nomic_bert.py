# Copyright (c) 2022, Tri Dao.
# This BERT implementation is based on our MLPerf 2.0 and MLPerf 2.1 BERT implementation.
# https://github.com/mlcommons/training_results_v2.0/blob/main/HazyResearch/benchmarks/bert/implementations/pytorch/modeling.py
# https://github.com/mlcommons/training_results_v2.1/blob/main/Azure-HazyResearch/benchmarks/bert/implementations/ND96amsr_A100_v4/modeling.py

import collections
import logging

# Inspired by https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py
import math
import os
import re
from collections import OrderedDict
from functools import partial
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from safetensors.torch import load_file as safe_load_file
from torch.nn.modules.utils import _pair
from transformers import GPT2Config, PreTrainedModel, ViTConfig, ViTModel
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.bert.modeling_bert import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    MaskedLMOutput,
    SequenceClassifierOutput,
)
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, WEIGHTS_INDEX_NAME, WEIGHTS_NAME
from transformers.utils.hub import cached_file, get_checkpoint_shard_files

from .configuration_hf_nomic_bert import NomicBertConfig

try:
    from torch.nn.functional import scaled_dot_product_attention
except ImportError:
    scaled_dot_product_attention = None

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
        resolved_archive_file = None
        for weight_name in [WEIGHTS_NAME, SAFE_WEIGHTS_NAME, WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_INDEX_NAME]:
            resolved_archive_file = cached_file(model_name, weight_name, _raise_exceptions_for_missing_entries=False)
            if resolved_archive_file is not None:
                if weight_name in [SAFE_WEIGHTS_NAME, SAFE_WEIGHTS_INDEX_NAME]:
                    load_safe = True
                if weight_name in [WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_INDEX_NAME]:
                    is_sharded = True
                break

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


def remap_bert_state_dict(
    state_dict,
    config,
    remove_bert=False,
    remove_cls_weights=False,
    add_pooling_layer=False,
):
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


def _trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        print(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    # Values are generated by using a truncated uniform distribution and
    # then using the inverse CDF for the normal distribution.
    # Get upper and lower cdf values
    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)

    # Uniformly fill tensor with values from [l, u], then translate to
    # [2l-1, 2u-1].
    tensor.uniform_(2 * l - 1, 2 * u - 1)

    # Use inverse cdf transform for normal distribution to get truncated
    # standard normal
    tensor.erfinv_()

    # Transform to proper mean, std
    tensor.mul_(std * math.sqrt(2.0))
    tensor.add_(mean)

    # Clamp to ensure it's in the proper range
    tensor.clamp_(min=a, max=b)
    return tensor


def trunc_normal_tf_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    NOTE: this 'tf' variant behaves closer to Tensorflow / JAX impl where the
    bounds [a, b] are applied when sampling the normal distribution with mean=0, std=1.0
    and the result is subsquently scaled and shifted by the mean and std args.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    with torch.no_grad():
        _trunc_normal_(tensor, 0, 1.0, a, b)
        tensor.mul_(std).add_(mean)
    return tensor


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
        remove_bert_prefix = cls != NomicBertForPreTraining and cls != NomicBertForSequenceClassification
        ignore_mismatched_shapes = kwargs.pop("ignore_mismatched_sizes", False)
        num_labels = kwargs.pop("num_labels", None)
        rotary_scaling_factor = kwargs.pop("rotary_scaling_factor", None)
        strict = kwargs.pop("strict", True)
        dtype = kwargs.pop("torch_dtype", None)
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

        if dtype is not None:
            model = model.to(dtype=dtype)
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
            state_dict = state_dict_from_pretrained(model_name, dtype=dtype)
            state_dict = remap_bert_state_dict(
                state_dict,
                config,
                remove_bert=remove_bert_prefix,
                remove_cls_weights=remove_cls,
                add_pooling_layer=getattr(config, "add_pooling_layer", False),
            )
            if ignore_mismatched_shapes:
                state_dict = filter_shapes(state_dict, model)

            load_return = model.load_state_dict(state_dict, strict=strict)
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


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def get_2d_sincos_pos_embed(embed_dim, grid_size, add_cls_token=False):
    """
    Create 2D sin/cos positional embeddings.

    Args:
        embed_dim (`int`):
            Embedding dimension.
        grid_size (`int`):
            The grid height and width.
        add_cls_token (`bool`, *optional*, defaults to `False`):
            Whether or not to add a classification (CLS) token.

    Returns:
        (`torch.FloatTensor` of shape (grid_size*grid_size, embed_dim) or (1+grid_size*grid_size, embed_dim): the
        position embeddings (with or without classification token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)

    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if add_cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position pos: a list of positions to be encoded: size (M,) out: (M, D)
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")

    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def ndgrid(*tensors) -> Tuple[torch.Tensor, ...]:
    """generate N-D grid in dimension order.

    The ndgrid function is like meshgrid except that the order of the first two input arguments are switched.

    That is, the statement
    [X1,X2,X3] = ndgrid(x1,x2,x3)

    produces the same result as

    [X2,X1,X3] = meshgrid(x2,x1,x3)

    This naming is based on MATLAB, the purpose is to avoid confusion due to torch's change to make
    torch.meshgrid behaviour move from matching ndgrid ('ij') indexing to numpy meshgrid defaults of ('xy').

    """
    try:
        return torch.meshgrid(*tensors, indexing='ij')
    except TypeError:
        # old PyTorch < 1.10 will follow this path as it does not have indexing arg,
        # the old behaviour of meshgrid was 'ij'
        return torch.meshgrid(*tensors)


def build_fourier_pos_embed(
    feat_shape: List[int],
    bands: Optional[torch.Tensor] = None,
    num_bands: int = 64,
    max_res: int = 224,
    temperature: float = 10000.0,
    linear_bands: bool = False,
    include_grid: bool = False,
    in_pixels: bool = True,
    ref_feat_shape: Optional[List[int]] = None,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> List[torch.Tensor]:
    """

    Args:
        feat_shape: Feature shape for embedding.
        bands: Pre-calculated frequency bands.
        num_bands: Number of frequency bands (determines output dim).
        max_res: Maximum resolution for pixel based freq.
        temperature: Temperature for non-pixel freq.
        linear_bands: Linear band spacing for pixel based freq.
        include_grid: Include the spatial grid in output.
        in_pixels: Output in pixel freq.
        ref_feat_shape: Reference feature shape for resize / fine-tune.
        dtype: Output dtype.
        device: Output device.

    Returns:

    """
    if bands is None:
        if in_pixels:
            bands = pixel_freq_bands(
                num_bands,
                float(max_res),
                linear_bands=linear_bands,
                device=device,
            )
        else:
            bands = freq_bands(
                num_bands,
                temperature=temperature,
                step=1,
                device=device,
            )
    else:
        if device is None:
            device = bands.device
        if dtype is None:
            dtype = bands.dtype

    if in_pixels:
        t = [torch.linspace(-1.0, 1.0, steps=s, device=device, dtype=torch.float32) for s in feat_shape]
    else:
        t = [torch.arange(s, device=device, dtype=torch.int64).to(torch.float32) for s in feat_shape]

    if ref_feat_shape is not None:
        # eva's scheme for resizing rope embeddings (ref shape = pretrain)
        t = [x / f * r for x, f, r in zip(t, feat_shape, ref_feat_shape)]

    grid = torch.stack(ndgrid(t), dim=-1)
    grid = grid.unsqueeze(-1)
    pos = grid * bands

    pos_sin, pos_cos = pos.sin().to(dtype=dtype), pos.cos().to(dtype)
    out = [grid, pos_sin, pos_cos] if include_grid else [pos_sin, pos_cos]
    return out


def build_rotary_pos_embed(
    feat_shape: List[int],
    bands: Optional[torch.Tensor] = None,
    dim: int = 64,
    max_res: int = 224,
    temperature: float = 10000.0,
    linear_bands: bool = False,
    in_pixels: bool = True,
    ref_feat_shape: Optional[List[int]] = None,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
):
    """

    Args:
        feat_shape: Spatial shape of the target tensor for embedding.
        bands: Optional pre-generated frequency bands
        dim: Output dimension of embedding tensor.
        max_res: Maximum resolution for pixel mode.
        temperature: Temperature (inv freq) for non-pixel mode
        linear_bands: Linearly (instead of log) spaced bands for pixel mode
        in_pixels: Pixel vs language (inv freq) mode.
        dtype: Output dtype.
        device: Output device.

    Returns:

    """
    sin_emb, cos_emb = build_fourier_pos_embed(
        feat_shape,
        bands=bands,
        num_bands=dim // 4,
        max_res=max_res,
        temperature=temperature,
        linear_bands=linear_bands,
        in_pixels=in_pixels,
        ref_feat_shape=ref_feat_shape,
        device=device,
        dtype=dtype,
    )
    num_spatial_dim = 1
    # this would be much nicer as a .numel() call to torch.Size(), but torchscript sucks
    for x in feat_shape:
        num_spatial_dim *= x
    sin_emb = sin_emb.reshape(num_spatial_dim, -1).repeat_interleave(2, -1)
    cos_emb = cos_emb.reshape(num_spatial_dim, -1).repeat_interleave(2, -1)
    return sin_emb, cos_emb


def freq_bands(
    num_bands: int,
    temperature: float = 10000.0,
    step: int = 2,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    exp = torch.arange(0, num_bands, step, dtype=torch.int64, device=device).to(torch.float32) / num_bands
    bands = 1.0 / (temperature**exp)
    return bands


def pixel_freq_bands(
    num_bands: int,
    max_freq: float = 224.0,
    linear_bands: bool = True,
    device: Optional[torch.device] = None,
):
    if linear_bands:
        bands = torch.linspace(1.0, max_freq / 2, num_bands, dtype=torch.float32, device=device)
    else:
        bands = 2 ** torch.linspace(0, math.log(max_freq, 2) - 1, num_bands, dtype=torch.float32, device=device)
    return bands * torch.pi


def rot(x):
    return torch.stack([-x[..., 1::2], x[..., ::2]], -1).reshape(x.shape)


def apply_rot_embed_cat(x: torch.Tensor, emb):
    sin_emb, cos_emb = emb.tensor_split(2, -1)
    if sin_emb.ndim == 3:
        return x * cos_emb.unsqueeze(1).expand_as(x) + rot(x) * sin_emb.unsqueeze(1).expand_as(x)
    return x * cos_emb + rot(x) * sin_emb


# taken from https://github.com/huggingface/pytorch-image-models/blob/cb0e4391beedcc5ac3ae4bce16561b95c326f32c/timm/layers/pos_embed_sincos.py#L363
class NomicVisionRotaryEmbeddingCat(nn.Module):
    """Rotary position embedding w/ concatenatd sin & cos

    The following impl/resources were referenced for this impl:
    * https://github.com/lucidrains/vit-pytorch/blob/6f3a5fcf0bca1c5ec33a35ef48d97213709df4ba/vit_pytorch/rvt.py
    * https://blog.eleuther.ai/rotary-embeddings/
    """

    def __init__(
        self,
        dim,
        max_res=224,
        temperature=10000,
        in_pixels=True,
        linear_bands: bool = False,
        feat_shape: Optional[List[int]] = None,
        ref_feat_shape: Optional[List[int]] = None,
    ):
        super().__init__()
        self.dim = dim
        self.max_res = max_res
        self.temperature = temperature
        self.in_pixels = in_pixels
        self.feat_shape = feat_shape
        self.ref_feat_shape = ref_feat_shape

        if feat_shape is None:
            # only cache bands
            if in_pixels:
                bands = pixel_freq_bands(
                    dim // 4,
                    float(max_res),
                    linear_bands=linear_bands,
                )
            else:
                bands = freq_bands(
                    dim // 4,
                    temperature=temperature,
                    step=1,
                )
            self.register_buffer(
                'bands',
                bands,
                persistent=False,
            )
            self.pos_embed = None
        else:
            # cache full sin/cos embeddings if shape provided up front
            embeds = build_rotary_pos_embed(
                feat_shape=feat_shape,
                dim=dim,
                max_res=max_res,
                linear_bands=linear_bands,
                in_pixels=in_pixels,
                ref_feat_shape=self.ref_feat_shape,
            )
            self.bands = None
            self.register_buffer(
                'pos_embed',
                torch.cat(embeds, -1),
                persistent=False,
            )

    def get_embed(self, shape: Optional[List[int]] = None):
        if self.bands is not None and shape is not None:
            # rebuild embeddings every call, use if target shape changes
            embeds = build_rotary_pos_embed(
                shape,
                self.bands,
                in_pixels=self.in_pixels,
                ref_feat_shape=self.ref_feat_shape,
            )
            return torch.cat(embeds, -1)
        elif self.pos_embed is not None:
            return self.pos_embed
        else:
            assert False, "get_embed() requires pre-computed pos_embed or valid shape w/ pre-computed bands"

    def forward(self, x):
        # assuming channel-first tensor where spatial dim are >= 2
        pos_embed = self.get_embed(x.shape[2:])
        return apply_rot_embed_cat(x, pos_embed)


class NomicVisionPatchEmbeddings(nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()
        img_size = _pair(config.img_size)
        patch_size = _pair(config.patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Linear(
            config.num_channels * patch_size[0] * patch_size[1], config.n_embd, bias=config.patch_embed_bias
        )

        self.learned_pos_embedding = False
        self.sinusoidal_pos_embedding = False
        self.no_embed_class = getattr(config, "no_embed_class", False)

        self.cls_token = (
            nn.Parameter(torch.zeros(1, 1, config.n_embd)) if not getattr(config, "no_cls_token", False) else None
        )
        if config.learned_pos_embedding:
            # this is the default in DINO
            self.learned_pos_embedding = True
            # hack for timm dinov2 with registers
            num_patches = self.num_patches if getattr(config, "register_tokens", 0) > 0 else self.num_patches + 1
            self.pos_embed = (
                nn.Parameter(torch.randn(1, num_patches, config.n_embd) * 0.02)
                if getattr(config, "use_pos_embed", True)
                else None
            )
        elif getattr(config, "sinusoidal_pos_embedding", False):
            self.sinusoidal_pos_embedding = True
            if getattr(config, "use_pos_embed", True):
                self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, config.n_embd), requires_grad=False)
                pos_embed = get_2d_sincos_pos_embed(config.n_embd, self.grid_size[0], add_cls_token=True)
                self.pos_embed.data.copy_(torch.from_numpy(pos_embed).to(self.pos_embed))
            else:
                self.pos_embed = None
        else:
            self.pos_embed = (
                nn.Parameter(torch.randn(1, self.num_patches + 1, config.n_embd) * 0.02)
                if getattr(config, "use_pos_embed", True)
                else None
            )

        if getattr(config, "register_tokens", 0) > 0:
            self.reg_token = nn.Parameter(torch.randn(1, config.register_tokens, config.n_embd) * 0.02)
        else:
            self.reg_token = None

        if config.mask_token:
            self.mask_token = nn.Parameter(torch.zeros(1, config.n_embd))

        self.patch_dropout = nn.Identity()

        if getattr(config, "use_rotary_pos_emb", False):
            ref_feat_shape = getattr(config, "ref_feat_shape", None)
            ref_feat_shape = to_2tuple(ref_feat_shape) if ref_feat_shape is not None else None
            self.rope = NomicVisionRotaryEmbeddingCat(
                config.n_embd // config.n_head,
                in_pixels=False,
                feat_shape=self.grid_size,
                ref_feat_shape=ref_feat_shape,
            )
        else:
            self.rope = None

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images.

        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """
        num_patches = embeddings.shape[1] - 1
        num_positions = self.pos_embed.shape[1] - 1
        if num_patches == num_positions and height == width:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = embeddings.shape[-1]
        height = height // self.patch_size[0]
        width = width // self.patch_size[1]
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        height, width = height + 0.1, width + 0.1
        patch_pos_embed = patch_pos_embed.reshape(1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            scale_factor=(height / math.sqrt(num_positions), width / math.sqrt(num_positions)),
            mode="bicubic",
            align_corners=False,
        )
        if int(height) != patch_pos_embed.shape[-2] or int(width) != patch_pos_embed.shape[-1]:
            raise ValueError("Width or height does not match with the interpolated position embeddings")
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward(self, x):
        # deepspeed case where the input is in fp32
        if x.dtype != self.proj.weight.dtype:
            x = x.to(dtype=self.proj.weight.dtype)

        _, _, height, width = x.shape
        x = self.proj(
            rearrange(
                x,
                "b c (h p1) (w p2) -> b h w (c p1 p2)",
                p1=self.patch_size[0],
                p2=self.patch_size[1],
            )
        )
        embeddings = rearrange(x, "b h w c -> b (h w) c")

        to_cat = []
        if self.cls_token is not None:
            if self.sinusoidal_pos_embedding:
                cls_token = self.cls_token + self.pos_embed[:, 0]
                cls_token = cls_token.expand(embeddings.shape[0], -1, -1)
                to_cat += [cls_token]
            else:
                cls_token = self.cls_token.expand(embeddings.shape[0], 1, -1)
                to_cat += [cls_token]

        if self.reg_token is not None:
            to_cat += [self.reg_token.expand(embeddings.shape[0], -1, -1)]

        rot_pos_embed = self.rope.get_embed() if self.rope is not None else None

        if self.no_embed_class:
            if self.learned_pos_embedding:
                embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
            else:
                if self.pos_embed is not None:
                    embeddings = embeddings + self.pos_embed
            if to_cat:
                embeddings = torch.cat(to_cat + [embeddings], dim=1)
        else:
            if to_cat:
                embeddings = torch.cat(to_cat + [embeddings], dim=1)
            if self.learned_pos_embedding:
                if self.pos_embed is not None:
                    embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
            else:
                if self.pos_embed is not None:
                    embeddings = embeddings + self.pos_embed

        embeddings = self.patch_dropout(embeddings)

        return embeddings, rot_pos_embed


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
        norm_layer=False,
    ):
        super().__init__()
        out_features = out_features if out_features is not None else in_features
        hidden_features = hidden_features if hidden_features is not None else int(8 * in_features / 3)
        hidden_features = int((hidden_features + multiple_of - 1) // multiple_of * multiple_of)
        self.return_residual = return_residual

        self.fc11 = nn.Linear(in_features, hidden_features, bias=bias1)
        self.fc12 = nn.Linear(in_features, hidden_features, bias=bias1)
        self.activation = activation
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias2)
        self.norm = nn.LayerNorm(hidden_features) if norm_layer else nn.Identity()

    def forward(self, x):
        y = self.fc11(x)
        gate = self.fc12(x)
        if self.activation == F.sigmoid:  # Special case for GLU
            y = F.glu(torch.cat([y, gate], dim=-1), dim=-1)
        else:
            y = y * self.activation(gate)

        # eva uses layer norm after the activation
        y = self.norm(y)

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
            if getattr(config, "rotary_scaling_factor", None):
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
        self.num_prefix_tokens = max(getattr(config, "register_tokens", 1), 1)

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
        rope: Optional[torch.Tensor] = None,
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
        elif rope is not None:
            q, k, v = qkv.permute(0, 3, 1, 2, 4).unbind(dim=-2)
            q = torch.cat(
                [q[:, :, : self.num_prefix_tokens], apply_rot_embed_cat(q[:, :, self.num_prefix_tokens :], rope)], dim=2
            ).type_as(q)
            k = torch.cat(
                [k[:, :, : self.num_prefix_tokens], apply_rot_embed_cat(k[:, :, self.num_prefix_tokens :], rope)], dim=2
            ).type_as(q)

            qkv = torch.stack([q, k, v], dim=-2)
            qkv = rearrange(qkv, "b h s three d -> b s three h d")

        query, key, value = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)
        if scaled_dot_product_attention is not None:
            attn_output = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=self.drop.p, is_causal=False
            )
        else:
            attention_scores = torch.matmul(query, key.transpose(-1, -2)) / self.norm_factor
            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask

            attentions_probs = F.softmax(attention_scores, dim=-1)
            attentions_probs = self.drop(attentions_probs)

            attn_output = torch.matmul(attentions_probs, value)

        attn_output = rearrange(attn_output.permute(0, 2, 1, 3), "... h d -> ... (h d)")

        attn_output = self.out_proj(attn_output)

        return attn_output


class NomicBertBlock(NomicBertPreTrainedModel):
    def __init__(
        self,
        config,
    ):
        super().__init__(config=config)
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
                norm_layer=getattr(config, "norm_mlp", False),
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
        rope: Optional[torch.Tensor] = None,
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
                rope=rope,
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
                rope=rope,
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
        rope: Optional[torch.Tensor] = None,
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
                    position_ids,
                    past_key_values,
                    is_padded_inputs,
                    output_attentions,
                    use_cache,
                    None,
                    None,
                    rope,
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
                    rope=rope,
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
        attention_mask=None,
        position_ids=None,
        token_type_ids=None,
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


def hf_vit_config_to_vit_config(vit_config: ViTConfig) -> GPT2Config:
    return GPT2Config(
        n_embd=vit_config.hidden_size,
        n_layer=vit_config.num_hidden_layers,
        n_head=vit_config.num_attention_heads,
        n_inner=vit_config.intermediate_size,
        activation_function=vit_config.hidden_act,
        vocab_size=0,  # no vocab since using patches
        n_positions=0,  # No absolute position embedding
        resid_pdrop=0.0,  # No dropout
        embd_pdrop=getattr(vit_config, "dropout", 0.0),
        attn_pdrop=vit_config.attention_probs_dropout_prob,
        layer_norm_epsilon=vit_config.layer_norm_eps,
        initializer_range=vit_config.initializer_range,
        bos_token_id=None,
        eos_token_id=None,
        # These are new arguments not in the original GPT2Config
        drop_path_rate=0.0,
        # Why is there double layer norm??
        prepre_layernom=False,
        layer_scale=False,
        layer_scale_init=None,
        img_size=vit_config.image_size,
        patch_size=vit_config.patch_size,
        num_channels=vit_config.num_channels,
        prenorm=True,
        parallel_block=False,
        parallel_block_tied_norm=False,
        rotary_emb_fraction=0,
        tie_word_embeddings=False,
        fused_dropout_add_ln=True,
        fused_bias_fc=True,
        patch_embed_bias=True,
        use_flash_attn=True,
        qkv_proj_bias=True,
        mlp_fc1_bias=getattr(vit_config, "mlp_fc1_bias", True),
        mlp_fc2_bias=getattr(vit_config, "mlp_fc2_bias", True),
        use_rms_norm=False,
        causal=False,
        hidden_features_scaling_factor=1.0,
        mask_token=False,
        learned_pos_embedding=False,
        patch_dropout=0,
        sinusoidal_pos_embedding=vit_config.model_type == "vit_mae",
    )


class NomicAttentionPooling(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.n_embd
        self.use_flash_attn = config.use_flash_attn
        self.fused_bias_fc = config.fused_bias_fc

        self.num_heads = config.n_head
        self.num_heads_kv = config.num_heads_kv if getattr(config, "num_heads_kv", None) is not None else self.num_heads
        assert self.embed_dim % self.num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = self.embed_dim // self.num_heads
        # we don't really support mqa / gqa for now
        kv_dim = 2 * self.head_dim * self.num_heads_kv

        self.register_buffer(
            "norm_factor",
            torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)).to(torch.get_default_dtype()),
            persistent=False,
        )

        self.Wq = nn.Linear(self.embed_dim, self.embed_dim, bias=config.qkv_proj_bias)
        self.Wkv = nn.Linear(self.embed_dim, kv_dim, bias=config.qkv_proj_bias)

        self.latent = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.qkv_proj_bias)
        self.causal = config.causal
        self.drop = nn.Dropout(config.attn_pdrop)

    def init_weights(self):
        trunc_normal_tf_(self.latent, std=self.embed_dim**-0.5)

    def forward(
        self,
        kv,
        attention_mask=None,
        cu_seqlens_k=None,
        max_seqlen_k=None,
        is_padded_inputs: Optional[bool] = True,
        output_attentions: bool = False,
    ):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q: The tensor containing the query. (B, Sq, H, D)
            kv: The tensor containing the key and value. (B, Sk, 2, H_k, D)
            causal: if passed, will override self.causal
            cu_seqlens: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
                of the sequences in the batch, used to index into q.
            max_seqlen: int. Maximum sequence length in the batch of q.
            cu_seqlens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
                of the sequences in the batch, used to index into kv.
            max_seqlen_k: int. Maximum sequence length in the batch of k and v.
        """
        q_latent = self.latent.expand(kv.size(0), -1, -1)
        q = self.Wq(q_latent)
        bsz, q_len, h_size = q.shape
        kv = self.Wkv(kv)
        query = rearrange(q, "... (h d) -> ... h d", d=self.head_dim)
        kv = rearrange(kv, "... (two hkv d) -> ... two hkv d", two=2, d=self.head_dim)

        key, value = kv[:, :, 0], kv[:, :, 1]

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


class NomicMultiHeadAttentionPooling(nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.prenorm = config.prenorm
        self.fused_dropout_add_ln = config.fused_dropout_add_ln

        self.attn = NomicAttentionPooling(config)
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
        self.dropout2 = nn.Dropout(config.resid_pdrop)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: if postnorm, residual=None, If prenorm, hidden_states = Attn/MLP(LN(residual))
            mixer_subset: for cross-attention only. If not None, will take a subset of x
                before applying the query projection. Useful for e.g., ViT where we only care
                about the CLS token in the last layer.
        """

        attn_outputs = self.attn(
            hidden_states,
            attention_mask=attention_mask,
        )

        normed = self.norm1(attn_outputs)
        hidden_states = hidden_states + self.mlp(normed)

        return hidden_states


class NomicVisionPreTrainedModel(PreTrainedModel):
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


class NomicVisionModel(NomicVisionPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embeddings = NomicVisionPatchEmbeddings(config)
        self.layers = nn.ModuleList([NomicBertBlock(config) for _ in range(config.n_layer)])

        self.selector = NomicMultiHeadAttentionPooling(config)

        self.global_pool = getattr(config, "global_pool", None)
        self.num_prefix_tokens = (1 if not getattr(config, "no_cls_token", False) else 0) + getattr(
            config, "register_tokens", 0
        )

        self.apply(partial(_init_weights, initializer_range=config.initializer_range))

    def forward(
        self,
        pixel_values,
        attention_mask=None,
        position_ids=None,
        token_type_ids=None,
        return_dict=None,
        matryoshka_dim=None,
    ):
        embeddings, rope = self.embeddings(pixel_values)

        original_dtype = embeddings.dtype

        hidden_states = embeddings
        # unused but easier to pass to gradient checkpointing as words
        residual = None
        for layer in self.layers:
            # need to pass none for backwards compatability
            hidden_states, _, residual = layer(
                hidden_states, None, residual=residual, is_padded_inputs=False, rope=rope
            )

        hidden_states = hidden_states + residual
        if self.global_pool == "avg":
            hidden_states = hidden_states[:, self.num_prefix_tokens :].mean(dim=1)

        pooled_output = self.selector(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=pooled_output,
            hidden_states=hidden_states,
        )
