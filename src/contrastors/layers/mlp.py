import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn.ops.activations import swiglu
from flash_attn.ops.fused_dense import FusedDense


class MLP(nn.Module):
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
        linear_cls = nn.Linear if not fused_bias_fc else FusedDense
        self.fc1 = linear_cls(in_features, hidden_features, bias=bias1)
        approximate = "tanh" if activation in ["gelu_new", "gelu_fast", "gelu_pytorch_tanh"] else "none"
        self.activation = nn.GELU(approximate=approximate) if activation == "gelu" else activation
        self.fc2 = linear_cls(hidden_features, out_features, bias=bias2)

    def forward(self, x, attention_mask=None):
        y = self.fc1(x)
        y = self.activation(y)
        y = self.fc2(y)
        return y if not self.return_residual else (y, x)


class GatedMLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        activation=F.sigmoid,
        bias1=True,
        bias2=True,
        multiple_of=256,
        hidden_features_scaling_factor=1,
        return_residual=False,
        fused_bias_fc=True,
        norm_layer=False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        out_features = out_features if out_features is not None else in_features
        hidden_features = hidden_features if hidden_features is not None else int(8 * in_features / 3)
        hidden_features = int(
            (hidden_features_scaling_factor * hidden_features + multiple_of - 1) // multiple_of * multiple_of
        )
        self.return_residual = return_residual
        linear_cls = nn.Linear if not fused_bias_fc else FusedDense
        self.fc11 = linear_cls(in_features, hidden_features, bias=bias1)
        self.fc12 = linear_cls(in_features, hidden_features, bias=bias1)
        self.activation = activation
        self.fc2 = linear_cls(hidden_features, out_features, bias=bias2)
        self.norm = nn.LayerNorm(hidden_features) if norm_layer else nn.Identity()

    def forward(self, x, attention_mask=None):
        y = self.fc11(x)
        gate = self.fc12(x)
        if self.activation == F.sigmoid:  # Special case for GLU
            y = F.glu(torch.cat([y, gate], dim=-1), dim=-1)
        elif self.activation == F.silu and swiglu is not None:  # Special case for SwiGLU
            # y, gate = y.chunk(2, dim=-1)
            y = swiglu(gate, y)
        else:
            y = y * self.activation(gate)

        # eva uses layer norm after the activation
        y = self.norm(y)

        y = self.fc2(y)
        return y if not self.return_residual else (y, x)
