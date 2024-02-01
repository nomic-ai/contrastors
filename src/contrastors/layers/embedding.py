import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange
from flash_attn.layers.rotary import RotaryEmbedding, apply_rotary_emb_func, apply_rotary_emb_kv_, apply_rotary_emb_qkv_
from flash_attn.ops.fused_dense import FusedDense
from torch.nn.modules.utils import _pair


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

        linear_cls = nn.Linear if not config.fused_bias_fc or not config.patch_embed_bias else FusedDense
        self.proj = linear_cls(
            config.num_channels * patch_size[0] * patch_size[1], config.n_embd, bias=config.patch_embed_bias
        )

        if config.learned_pos_embedding:
            # this is the default in DINO
            self.learned_pos_embedding = True
            self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, config.n_embd) * 0.02)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.n_embd))
        else:
            self.learned_pos_embedding = False
            self.pos_embed = nn.Embedding(self.num_patches + 1, config.n_embd)
            self.cls_token = nn.Parameter(torch.zeros(config.n_embd))
            self.register_buffer("position_ids", torch.arange(self.num_patches + 1).expand((1, -1)), persistent=False)

        if config.mask_token:
            self.mask_token = nn.Parameter(torch.zeros(1, config.n_embd))

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
        batch_size, _, height, width = x.shape
        x = self.proj(
            rearrange(
                x,
                "b c (h p1) (w p2) -> b h w (c p1 p2)",
                p1=self.patch_size[0],
                p2=self.patch_size[1],
            )
        )
        x = rearrange(x, "b h w c -> b (h w) c")
        embeddings = torch.cat((self.cls_token.expand(x.shape[0], 1, -1), x), dim=1)
        if self.learned_pos_embedding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.pos_embed(self.position_ids)
        return embeddings


class BertEmbeddings(nn.Module):
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
