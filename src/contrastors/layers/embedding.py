import collections
import math
from itertools import repeat
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from flash_attn.layers.rotary import RotaryEmbedding, apply_rotary_emb_func, apply_rotary_emb_kv_, apply_rotary_emb_qkv_
from flash_attn.ops.fused_dense import FusedDense
from torch.nn.modules.utils import _pair


# From PyTorch internals
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
class RotaryEmbeddingCat(nn.Module):
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


class PatchEmbedding(nn.Module):
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

        linear_cls = nn.Linear if not config.fused_bias_fc else FusedDense
        self.proj = linear_cls(
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

        if config.patch_dropout > 0.0:
            self.patch_dropout = PatchDropout(config.patch_dropout, exclude_first_token=True)
        else:
            self.patch_dropout = nn.Identity()

        if getattr(config, "use_rotary_pos_emb", False):
            ref_feat_shape = getattr(config, "ref_feat_shape", None)
            ref_feat_shape = to_2tuple(ref_feat_shape) if ref_feat_shape is not None else None
            self.rope = RotaryEmbeddingCat(
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


class PatchDropout(nn.Module):
    """
    Allegedly speeds up training 2x-3x
    https://arxiv.org/abs/2212.00794
    """

    def __init__(self, prob, exclude_first_token=True):
        super().__init__()
        assert 0 <= prob < 1.0
        self.prob = prob
        self.exclude_first_token = exclude_first_token  # exclude CLS token

    def forward(self, x):
        if not self.training or self.prob == 0.0:
            return x

        if self.exclude_first_token:
            cls_tokens, x = x[:, :1], x[:, 1:]
        else:
            cls_tokens = torch.jit.annotate(torch.Tensor, x[:, :1])

        batch = x.size()[0]
        num_tokens = x.size()[1]

        batch_indices = torch.arange(batch)
        batch_indices = batch_indices[..., None]

        keep_prob = 1 - self.prob
        num_patches_keep = max(1, int(num_tokens * keep_prob))

        rand = torch.randn(batch, num_tokens)
        patch_indices_keep = rand.topk(num_patches_keep, dim=-1).indices

        x = x[batch_indices, patch_indices_keep]

        if self.exclude_first_token:
            x = torch.cat((cls_tokens, x), dim=1)

        return x


class BertEmbeddings(nn.Module):
    def __init__(self, config):
        """
        If max_position_embeddings <= 0, there's no position embeddings
        If type_vocab_size <= 0, there's no token type embeddings
        """
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.max_position_embeddings = config.max_position_embeddings if config.rotary_emb_fraction <= 0 else 0
        self.type_vocab_size = getattr(config, "type_vocab_size", 0)
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
        if self.max_position_embeddings > 0:
            if position_ids is None:
                position_ids = torch.arange(seqlen, dtype=torch.long, device=input_ids.device)
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = embeddings + position_embeddings
        if self.type_vocab_size > 0:
            if token_type_ids is None:
                token_type_ids = torch.zeros(seqlen, dtype=torch.long, device=input_ids.device)
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings = embeddings + token_type_embeddings
        return embeddings


class VarLengthRotaryEmbedding(RotaryEmbedding):
    """
    The rotary position embeddings from RoFormer_ (Su et. al).
    A crucial insight from the method is that the query and keys are
    transformed by rotation matrices which depend on the relative positions.

    Other implementations are available in the Rotary Transformer repo_ and in
    GPT-NeoX_, GPT-NeoX was an inspiration

    .. _RoFormer: https://arxiv.org/abs/2104.09864
    .. _repo: https://github.com/ZhuiyiTechnology/roformer
    .. _GPT-NeoX: https://github.com/EleutherAI/gpt-neox

    If scale_base is not None, this implements XPos (Sun et al., https://arxiv.org/abs/2212.10554).
    A recommended value for scale_base is 512: https://github.com/HazyResearch/flash-attention/issues/96
    Reference: https://github.com/sunyt32/torchscale/blob/main/torchscale/component/xpos_relative_position.py
    """

    def __init__(self, **kwargs):
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
        super().__init__(**kwargs)

    def forward(
        self,
        qkv: torch.Tensor,
        kv: Optional[torch.Tensor] = None,
        seqlen_offset: Union[int, torch.Tensor] = 0,
        max_seqlen: Optional[int] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
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
        if max_seqlen is not None:
            self._update_cos_sin_cache(max_seqlen, device=qkv.device, dtype=qkv.dtype)
        elif isinstance(seqlen_offset, int):
            self._update_cos_sin_cache(seqlen + seqlen_offset, device=qkv.device, dtype=qkv.dtype)
        if kv is None:
            if self.scale is None and cu_seqlens is None and max_seqlen is None:
                return apply_rotary_emb_qkv_(
                    qkv,
                    self._cos_cached,
                    self._sin_cached,
                    interleaved=self.interleaved,
                    seqlen_offsets=seqlen_offset,
                )
            elif cu_seqlens is not None and max_seqlen is not None:
                q = qkv[:, 0]
                k = qkv[:, 1]
                q_rot = apply_rotary_emb_func(
                    q,
                    self._cos_cached,
                    self._sin_cached,
                    interleaved=self.interleaved,
                    seqlen_offsets=seqlen_offset,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                )
                k_rot = apply_rotary_emb_func(
                    k,
                    self._cos_cached,
                    self._sin_cached,
                    interleaved=self.interleaved,
                    seqlen_offsets=seqlen_offset,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                )
                return torch.stack([q_rot, k_rot, qkv[:, 2]], dim=1)
            else:
                return apply_rotary_emb_qkv_(
                    qkv,
                    self._cos_cached,
                    self._sin_cached,
                    self._cos_k_cached,
                    self._sin_k_cached,
                    interleaved=self.interleaved,
                    seqlen_offsets=seqlen_offset,
                )
        else:
            q = qkv
            q = apply_rotary_emb_func(
                q,
                self._cos_cached,
                self._sin_cached,
                interleaved=self.interleaved,
                inplace=True,
                seqlen_offsets=seqlen_offset,
                cu_seqlens=cu_seqlens,
                max_seq_len=max_seqlen,
            )
            if self.scale is None:
                kv = apply_rotary_emb_kv_(
                    kv,
                    self._cos_cached,
                    self._sin_cached,
                    interleaved=self.interleaved,
                    seqlen_offsets=seqlen_offset,
                )
            else:
                kv = apply_rotary_emb_kv_(
                    kv,
                    self._cos_k_cached,
                    self._sin_k_cached,
                    interleaved=self.interleaved,
                    seqlen_offsets=seqlen_offset,
                )
            return q, kv


class DynamicNTKRotaryEmbedding(VarLengthRotaryEmbedding):
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
