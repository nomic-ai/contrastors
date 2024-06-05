import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from flash_attn import (
    flash_attn_kvpacked_func,
    flash_attn_qkvpacked_func,
    flash_attn_varlen_kvpacked_func,
    flash_attn_varlen_qkvpacked_func,
)
from flash_attn.bert_padding import pad_input, unpad_input
from flash_attn.ops.fused_dense import FusedDense

from contrastors.layers.embedding import DynamicNTKRotaryEmbedding, VarLengthRotaryEmbedding, apply_rot_embed_cat


class FlashAttention(nn.Module):
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
                self.rotary_emb = DynamicNTKRotaryEmbedding(
                    dim=self.rotary_emb_dim,
                    base=config.rotary_emb_base,
                    scale_base=config.rotary_emb_scale_base,
                    interleaved=config.rotary_emb_interleaved,
                    rotary_scaling_factor=config.rotary_scaling_factor,
                    max_position_embeddings=config.max_trained_positions,
                )
            else:
                self.rotary_emb = VarLengthRotaryEmbedding(
                    dim=self.rotary_emb_dim,
                    base=config.rotary_emb_base,
                    scale_base=config.rotary_emb_scale_base,
                    interleaved=config.rotary_emb_interleaved,
                )
            # bug in xformers: https://github.com/facebookresearch/xformers/issues/841
            # uses the head dimension instead of the sequence dimension
            self.rotary_head_dim = getattr(config, "rotary_head_dim", False)

        linear_cls = nn.Linear if not config.fused_bias_fc else FusedDense
        self.Wqkv = linear_cls(self.embed_dim, qkv_dim, bias=config.qkv_proj_bias)

        self.out_proj = linear_cls(self.embed_dim, self.embed_dim, bias=config.qkv_proj_bias)
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
            if cu_seqlens is None and max_seq_len is None:
                if self.rotary_head_dim:
                    qkv = rearrange(qkv, "b s three h d -> b h three s d")
                # TODO 1/3/2024: rotary_embedding for unpadded inputs -> more efficient: https://github.com/Dao-AILab/flash-attention/issues/177
                qkv = self.rotary_emb(qkv, seqlen_offset=past_len)

                if self.rotary_head_dim:
                    qkv = rearrange(qkv, "b h three s d -> b s three h d")
            else:
                qkv = self.rotary_emb(qkv, cu_seqlens=cu_seqlens, max_seqlen=max_seq_len, seqlen_offset=past_len)
        elif rope is not None:
            # TODO there's no way this is efficient
            # b, s, 3, h, d -> b h s 3 d
            q, k, v = qkv.permute(0, 3, 1, 2, 4).unbind(dim=-2)
            q = torch.cat(
                [q[:, :, : self.num_prefix_tokens], apply_rot_embed_cat(q[:, :, self.num_prefix_tokens :], rope)], dim=2
            ).type_as(q)
            k = torch.cat(
                [k[:, :, : self.num_prefix_tokens], apply_rot_embed_cat(k[:, :, self.num_prefix_tokens :], rope)], dim=2
            ).type_as(q)

            qkv = torch.stack([q, k, v], dim=-2)
            qkv = rearrange(qkv, "b h s three d -> b s three h d")

        if attention_mask is not None:
            # varlen, ignore padding tokens, efficient for large batch with many paddings
            assert attention_mask is not None
            if cu_seqlens is None and max_seq_len is None:
                bsz, q_len, h_size = hidden_states.shape
                unpadded_qkv, indices, cu_seqlens, max_seq_len = unpad_input(qkv, attention_mask)

                attn_outputs = flash_attn_varlen_qkvpacked_func(
                    unpadded_qkv,
                    cu_seqlens,
                    max_seq_len,
                    dropout_p=self.drop.p if self.training else 0.0,
                    softmax_scale=1.0 / self.norm_factor,
                    causal=self.causal,
                    return_attn_probs=output_attentions,
                )

                attn_output = attn_outputs[0] if output_attentions else attn_outputs

                attn_output = pad_input(attn_output, indices, bsz, q_len).reshape(bsz, q_len, h_size)
            else:
                attn_outputs = flash_attn_varlen_qkvpacked_func(
                    qkv,
                    cu_seqlens,
                    max_seq_len,
                    dropout_p=self.drop.p if self.training else 0.0,
                    softmax_scale=1.0 / self.norm_factor,
                    causal=self.causal,
                    return_attn_probs=output_attentions,
                )
                attn_output = attn_outputs[0] if output_attentions else attn_outputs
                attn_output = rearrange(attn_output, "... h d -> ... (h d)")

        else:
            bsz, q_len, h_size = hidden_states.shape
            # no padding tokens, more efficient
            attn_outputs = flash_attn_qkvpacked_func(
                qkv,
                dropout_p=self.drop.p if self.training else 0.0,
                softmax_scale=1.0 / self.norm_factor,
                causal=self.causal,
                return_attn_probs=output_attentions,
            )

            attn_output = attn_outputs[0] if output_attentions else attn_outputs
            attn_output = attn_output.reshape(bsz, q_len, h_size)

        attn_output = self.out_proj(attn_output)

        return attn_output


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


class FlashAttentionPooling(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """

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

        linear_cls = nn.Linear if not config.fused_bias_fc else FusedDense
        self.Wq = linear_cls(self.embed_dim, self.embed_dim, bias=config.qkv_proj_bias)
        self.Wkv = linear_cls(self.embed_dim, kv_dim, bias=config.qkv_proj_bias)

        self.latent = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        self.out_proj = linear_cls(self.embed_dim, self.embed_dim, bias=config.qkv_proj_bias)
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
        q = rearrange(q, "... (h d) -> ... h d", d=self.head_dim)
        kv = rearrange(kv, "... (two hkv d) -> ... two hkv d", two=2, d=self.head_dim)

        if attention_mask is not None:
            # varlen, ignore padding tokens, efficient for large batch with many paddings
            assert attention_mask is not None
            unpadded_q, _, cu_seqlens, max_seqlen = unpad_input(q, torch.ones_like(attention_mask))
            if cu_seqlens_k is None and max_seqlen_k is None:
                unpadded_kv, indices_kv, cu_seqlens_k, max_seqlen_k = unpad_input(kv, attention_mask)

                attn_outputs = flash_attn_varlen_kvpacked_func(
                    unpadded_q,
                    unpadded_kv,
                    cu_seqlens,
                    cu_seqlens_k,
                    max_seqlen,
                    max_seqlen_k,
                    self.drop.p if self.training else 0.0,
                    softmax_scale=1.0 / self.norm_factor,
                    causal=self.causal,
                )

                attn_output = attn_outputs[0] if output_attentions else attn_outputs
                attn_output = rearrange(attn_output, "... h d -> ... (h d)")
                attn_output = pad_input(attn_output, indices_kv, bsz, q_len).reshape(bsz, q_len, h_size)
            else:
                attn_outputs = flash_attn_varlen_kvpacked_func(
                    unpadded_q,
                    kv,
                    cu_seqlens_q=cu_seqlens,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen=max_seqlen,
                    max_seqlen_k=max_seqlen_k,
                    dropout_p=self.drop.p if self.training else 0.0,
                    softmax_scale=1.0 / self.norm_factor,
                    causal=self.causal,
                )
                attn_output = attn_outputs[0] if output_attentions else attn_outputs
                attn_output = rearrange(attn_output, "... h d -> ... (h d)")
        else:
            attn_outputs = flash_attn_kvpacked_func(
                q,
                kv,
                self.drop.p if self.training else 0.0,
                causal=self.causal,
                softmax_scale=1.0 / self.norm_factor,
            )
            attn_output = attn_outputs[0] if output_attentions else attn_outputs
            attn_output = attn_output.reshape(bsz, q_len, h_size)

        attn_output = self.out_proj(attn_output)
        return attn_output
