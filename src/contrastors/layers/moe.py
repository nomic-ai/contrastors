import torch
import torch.nn as nn
import torch.nn.functional as F
from contrastors.layers.mlp import MLP, GatedMLP
from contrastors.layers.activations import quick_gelu
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
from functools import partial



class Router(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False)

        
    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.config.hidden_size)
        logits = self.router(hidden_states)

        return logits

        
class MoEBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        
        self.router = Router(config)
        self.top_k = config.top_k
        
        activation = (
            F.sigmoid
            if config.activation_function == "glu"
            else (
                F.silu
                if config.activation_function == "swiglu"
                else (quick_gelu if config.activation_function == "quick_gelu" else F.gelu)
            )
        )

        if config.activation_function in ["glu", "swiglu"]:
            expert = partial(GatedMLP, in_features=config.n_embd,
                hidden_features=config.n_inner,
                bias1=config.mlp_fc1_bias,
                bias2=config.mlp_fc2_bias,
                activation=activation,
                # for whatever reason, flash attention barfs if true
                fused_bias_fc=False,
                norm_layer=getattr(config, "norm_mlp", False),
            )
        else:
            expert = partial(MLP, in_features=config.n_embd,
                hidden_features=config.n_inner,
                bias1=config.mlp_fc1_bias,
                bias2=config.mlp_fc2_bias,
                activation=activation,
                # for whatever reason, flash attention barfs if true
                fused_bias_fc=False,
            )

            
        self.experts = nn.ModuleList([expert() for _ in range(self.num_experts)])

        self.bias = nn.Parameter(torch.zeros(config.hidden_size))

        
    def forward(self, hidden_states, attention_mask=None):
        input_shape = hidden_states.shape
        batch, seqlen = hidden_states.shape[:2]
        if attention_mask is not None:
            hidden_states, indices, _, _ = unpad_input(hidden_states, attention_mask)
        hidden_states = hidden_states.reshape(-1, input_shape[-1])
        # (bs * s, num_experts)
        # TODO: could use fp32 here as switch transformer found it didn't diverge
        router_logits = self.router(hidden_states)
        router_weights = F.softmax(router_logits, dim=-1)

        weights, experts = torch.topk(router_weights, self.top_k, dim=-1)

        # (bs * s, hidden_size)
        output_shape = (*router_logits.shape[:-1], input_shape[-1])
        results = torch.zeros(output_shape, device=hidden_states.device, dtype=hidden_states.dtype)
        for i, expert in enumerate(self.experts):
            batch_idx, nth_expert = torch.where(experts == i)
            # add the weighted experts output to the results
            results[batch_idx] += weights[batch_idx, nth_expert].unsqueeze(-1) * expert(hidden_states[batch_idx])

        results += self.bias

        if attention_mask is not None:
            results = pad_input(results, indices, batch, seqlen)

        return results.reshape(input_shape), router_logits
        