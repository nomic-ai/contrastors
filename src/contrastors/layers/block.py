from functools import partial
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn.ops.layer_norm import dropout_add_layer_norm, dropout_add_layer_norm_parallel_residual
from flash_attn.ops.rms_norm import RMSNorm, dropout_add_rms_norm, dropout_add_rms_norm_parallel_residual
from torchvision.ops import StochasticDepth

from contrastors.layers.attention import FlashAttention
from contrastors.layers.mlp import MLP, GatedMLP


class ParallelBlock(nn.Module):
    """The attention (mixer) and MLP blocks are done in parallel, similar to GPT-J, GPT-NeoX,
    and PaLM.
    """

    def __init__(
        self,
        config,
    ):
        """
        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA / MLP -> Dropout -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Dropout -> Add -> LN -> MHA / MLP, returning both
        the hidden_states (output1 of the MHA / MLP) and the residual.
        This is for performance reasons, as we can fuse the dropout, add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.prenorm = config.prenorm
        self.fused_dropout_add_ln = config.fused_dropout_add_ln
        self.residual_in_fp32 = getattr(config, "residual_in_fp32", False)

        self.attn = FlashAttention(config)
        activation = (
            F.sigmoid
            if config.activation_function == "glu"
            else (F.silu if config.activation_function == "swiglu" else F.gelu)
        )
        if config.activation_function in ["glu", "swiglu"]:
            self.mlp = GatedMLP(
                config.n_embd,
                hidden_features=config.n_inner,
                bias1=config.mlp_fc1_bias,
                bias2=config.mlp_fc2_bias,
                activation=activation,
                fused_bias_fc=config.fused_bias_fc,
            )
        else:
            self.mlp = MLP(
                config.n_embd,
                hidden_features=config.n_inner,
                bias1=config.mlp_fc1_bias,
                bias2=config.mlp_fc2_bias,
                activation=activation,
                fused_bias_fc=config.fused_bias_fc,
            )
        self.dropout1 = nn.Dropout(config.resid_pdrop)
        norm_cls = partial(
            nn.LayerNorm if not config.use_rms_norm else RMSNorm,
            eps=config.layer_norm_epsilon,
        )
        self.norm1 = norm_cls(config.n_embd)
        self.norm2 = norm_cls(config.n_embd)
        self.dropout2 = nn.Dropout(config.resid_pdrop)

        self.tied_norm = False

    def forward(
        self,
        hidden_states1: torch.Tensor,
        hidden_states2: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        is_padded_inputs: Optional[bool] = True,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states1: the output of the previous attention (mixer) or embedding layer.
            hidden_states2: the output of the previous MLP layer (if None, will use hidden_states1).
            residual.
        """
        fused_add_norm_fn = (
            dropout_add_rms_norm_parallel_residual
            if isinstance(self.norm1, RMSNorm)
            else dropout_add_layer_norm_parallel_residual
        )
        if not self.fused_dropout_add_ln:
            dropped1 = self.dropout1(hidden_states1)
            # For the very 1st block, we only want 1 dropout, not two different dropouts
            if hidden_states2 is not None:
                dropped2 = self.dropout2(hidden_states2)
                residual = (residual + dropped1 + dropped2) if residual is not None else dropped1 + dropped2
            else:
                residual = (residual + dropped1) if residual is not None else dropped1
            hidden_states1 = self.norm1(residual.to(dtype=self.norm1.weight.dtype))
            hidden_states2 = (
                self.norm2(residual.to(dtype=self.norm2.weight.dtype)) if not self.tied_norm else hidden_states1
            )
        else:
            weight2, bias2 = (self.norm2.weight, self.norm2.bias) if not self.tied_norm else (None, None)
            hidden_states1, hidden_states2, residual = fused_add_norm_fn(
                hidden_states1,
                hidden_states2,
                residual,
                self.norm1.weight,
                self.norm1.bias,
                weight2,
                bias2,
                self.dropout1.p if self.training else 0.0,
                self.norm1.eps,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
            )
            if self.tied_norm:
                hidden_states2 = hidden_states1

        hidden_states1 = self.attn(
            hidden_states1,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,
            is_padded_inputs=is_padded_inputs,
        )
        hidden_states2 = self.mlp(hidden_states2)
        return hidden_states1, hidden_states2, residual


class Block(nn.Module):
    def __init__(
        self,
        config,
        drop_path_rate1=0.0,
        drop_path_rate2=0.0,
    ):
        """
        For prenorm=True, this Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA -> Dropout -> Add -> LN -> MLP -> Dropout -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Dropout -> Add -> LN -> MHA -> Dropout -> Add -> LN -> MLP, returning both
        the hidden_states (output of the MLP) and the residual.
        This is for performance reasons, as we can fuse the dropout, add and LayerNorm.
        The residual needs to be provided (except for the very first block).

        For prenorm=False, this Block has the same structure as a regular postnorm Transformer
        block: MHA -> Dropout -> Add -> LN -> MLP -> Dropout -> Add -> LN.
        See more: https://github.com/Dao-AILab/flash-attention/issues/216#issuecomment-1546638138
        """
        super().__init__()
        self.prenorm = config.prenorm
        self.fused_dropout_add_ln = config.fused_dropout_add_ln

        self.attn = FlashAttention(config)
        activation = (
            F.sigmoid
            if config.activation_function == "glu"
            else (F.silu if config.activation_function == "swiglu" else F.gelu)
        )
        if config.activation_function in ["glu", "swiglu", "geglu"]:
            self.mlp = GatedMLP(
                config.n_embd,
                hidden_features=config.n_inner,
                bias1=config.mlp_fc1_bias,
                bias2=config.mlp_fc2_bias,
                activation=activation,
                fused_bias_fc=config.fused_bias_fc,
            )
        else:
            self.mlp = MLP(
                config.n_embd,
                hidden_features=config.n_inner,
                bias1=config.mlp_fc1_bias,
                bias2=config.mlp_fc2_bias,
                activation=activation,
                fused_bias_fc=config.fused_bias_fc,
            )

        self.dropout1 = nn.Dropout(config.resid_pdrop)
        norm_cls = partial(
            nn.LayerNorm if not config.use_rms_norm else RMSNorm,
            eps=config.layer_norm_epsilon,
        )
        self.norm1 = norm_cls(config.n_embd)
        self.norm2 = norm_cls(config.n_embd)
        self.dropout2 = nn.Dropout(config.resid_pdrop)
        self.residual_in_fp32 = getattr(config, "residual_in_fp32", False)

        self.drop_path1 = StochasticDepth(drop_path_rate1, mode="row") if drop_path_rate1 > 0.0 else nn.Identity()
        self.drop_path2 = StochasticDepth(drop_path_rate2, mode="row") if drop_path_rate2 > 0.0 else nn.Identity()

        self.layer_scale = getattr(config, "layer_scale", False)
        if self.layer_scale:
            self.ls1 = nn.Parameter(config.layer_scale_init * torch.ones(config.n_embd))
            self.ls2 = nn.Parameter(config.layer_scale_init * torch.ones(config.n_embd))

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
        fused_add_norm_fn = (
            dropout_add_rms_norm if RMSNorm and isinstance(self.norm1, RMSNorm) else dropout_add_layer_norm
        )
        if self.prenorm:
            if not self.fused_dropout_add_ln:
                dropped = self.drop_path1(self.dropout1(hidden_states))
                residual = (dropped + residual) if residual is not None else dropped
                hidden_states = self.norm1(residual.to(dtype=self.norm1.weight.dtype))
            else:
                if isinstance(self.drop_path1, nn.Identity) or self.drop_path1.p == 0 or not self.training:
                    rowscale1 = None
                else:
                    rowscale1 = self.drop_path1(
                        torch.ones(
                            hidden_states.shape[:-1],
                            device=hidden_states.device,
                            dtype=hidden_states.dtype,
                        )
                    )
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    residual,
                    self.norm1.weight,
                    self.norm1.bias,
                    self.dropout1.p if self.training else 0.0,
                    self.norm1.eps,
                    rowscale=rowscale1,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                )
            hidden_states = self.attn(
                hidden_states,
                attention_mask=attention_mask,
                is_padded_inputs=is_padded_inputs,
                cu_seqlens=cu_seqlens,
                max_seq_len=max_seq_len,
            )
            if not self.fused_dropout_add_ln:
                if self.layer_scale:
                    hidden_states = hidden_states * self.ls1

                dropped = self.drop_path2(self.dropout2(hidden_states))
                residual = (dropped + residual) if residual is not None else dropped
                hidden_states = self.norm2(residual.to(dtype=self.norm2.weight.dtype))
            else:
                if isinstance(self.drop_path2, nn.Identity) or self.drop_path2.p == 0 or not self.training:
                    rowscale2 = None
                else:
                    rowscale2 = self.drop_path2(
                        torch.ones(
                            hidden_states.shape[:-1],
                            device=hidden_states.device,
                            dtype=hidden_states.dtype,
                        )
                    )
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    residual,
                    self.norm2.weight,
                    self.norm2.bias,
                    self.dropout2.p if self.training else 0.0,
                    self.norm2.eps,
                    rowscale=rowscale2,
                    layerscale=None if not self.layer_scale else self.ls1,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                )
            hidden_states = self.mlp(hidden_states)

            if self.layer_scale:
                hidden_states = hidden_states * self.ls2

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
            if not self.fused_dropout_add_ln:
                hidden_states = self.norm1(
                    (self.drop_path1(self.dropout1(attn_outputs)) + hidden_states).to(dtype=self.norm1.weight.dtype)
                )
            else:
                if isinstance(self.drop_path1, nn.Identity) or self.drop_path1.p == 0 or not self.training:
                    rowscale1 = None
                else:
                    rowscale1 = self.drop_path1(
                        torch.ones(attn_outputs.shape[:-1], device=attn_outputs.device, dtype=attn_outputs.dtype)
                    )
                hidden_states = fused_add_norm_fn(
                    attn_outputs,
                    hidden_states,
                    self.norm1.weight,
                    self.norm1.bias,
                    self.dropout1.p if self.training else 0.0,
                    self.norm1.eps,
                    rowscale=rowscale1,
                    prenorm=False,
                )
            mlp_out = self.mlp(hidden_states)
            if not self.fused_dropout_add_ln:
                hidden_states = self.norm2(
                    (self.drop_path2(self.dropout2(mlp_out)) + hidden_states).to(dtype=self.norm2.weight.dtype)
                )
            else:
                if isinstance(self.drop_path2, nn.Identity) or self.drop_path2.p == 0 or not self.training:
                    rowscale2 = None
                else:
                    rowscale2 = self.drop_path2(
                        torch.ones(mlp_out.shape[:-1], device=mlp_out.device, dtype=mlp_out.dtype)
                    )
                hidden_states = fused_add_norm_fn(
                    mlp_out,
                    hidden_states,
                    self.norm2.weight,
                    self.norm2.bias,
                    self.dropout2.p if self.training else 0.0,
                    self.norm2.eps,
                    rowscale=rowscale2,
                    prenorm=False,
                )
            return hidden_states, None, None
