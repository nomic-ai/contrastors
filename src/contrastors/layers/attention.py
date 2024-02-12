from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from flash_attn import flash_attn_qkvpacked_func, flash_attn_varlen_qkvpacked_func
from flash_attn.bert_padding import pad_input, unpad_input
from flash_attn.layers.rotary import RotaryEmbedding, apply_rotary
from flash_attn.ops.fused_dense import FusedDense

from contrastors.layers.embedding import DynamicNTKRotaryEmbedding, VarLengthRotaryEmbedding


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
            if config.rotary_scaling_factor:
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
            if cu_seqlens is None and max_seq_len is None:
                if self.rotary_head_dim:
                    qkv = rearrange(qkv, "b s three h d -> b h three s d")
                # TODO 1/3/2024: rotary_embedding for unpadded inputs -> more efficient: https://github.com/Dao-AILab/flash-attention/issues/177
                qkv = self.rotary_emb(qkv, seqlen_offset=past_len)

                if self.rotary_head_dim:
                    qkv = rearrange(qkv, "b h three s d -> b s three h d")
            else:
                qkv = self.rotary_emb(qkv, cu_seqlens=cu_seqlens, max_seqlen=max_seq_len, seqlen_offset=past_len)

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
