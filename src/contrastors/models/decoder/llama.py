import torch
import math
import re
from collections import OrderedDict

import torch.nn.functional as F
from einops import rearrange
from transformers import GPT2Config, LlamaConfig


def remap_state_dict_hf_llama(state_dict, config):
    def key_mapping_layers(key):
        return re.sub(r"^model.", "", key)

    state_dict = OrderedDict((key_mapping_layers(k), v) for k, v in state_dict.items())

    # Word embedding
    def key_mapping_emb(key):
        return re.sub(r"^embed_tokens.", "embeddings.", key)

    state_dict = OrderedDict((key_mapping_emb(k), v) for k, v in state_dict.items())
    
    # LayerNorm
    def key_mapping_ln(key):
        key = re.sub(r"^norm.", r"ln_f.", key)
        key = re.sub(r"^layers.(\d+).input_layernorm.", r"layers.\1.norm1.", key)
        key = re.sub(r"^layers.(\d+).post_attention_layernorm.", r"layers.\1.norm2.", key)
        return key

    state_dict = OrderedDict((key_mapping_ln(k), v) for k, v in state_dict.items())

    # MLP
    def key_mapping_mlp(key):
        key = re.sub(r"^layers.(\d+).mlp.gate_proj.", r"layers.\1.mlp.fc12.", key)
        key = re.sub(r"^layers.(\d+).mlp.up_proj.", r"layers.\1.mlp.fc11.", key)
        key = re.sub(r"^layers.(\d+).mlp.down_proj.", r"layers.\1.mlp.fc2.", key)
        return key

    state_dict = OrderedDict((key_mapping_mlp(k), v) for k, v in state_dict.items())

    # Attention
    for l in range(config.num_hidden_layers):
        # Combine q, k, v projections into a single Wqkv
        q_weight = state_dict.pop(f"layers.{l}.self_attn.q_proj.weight")
        k_weight = state_dict.pop(f"layers.{l}.self_attn.k_proj.weight")
        v_weight = state_dict.pop(f"layers.{l}.self_attn.v_proj.weight")
        
        # Assuming the hidden size is the same for q, k, and v
        hidden_size = q_weight.size(0)
        
        # Combine the weights
        combined_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
        state_dict[f"layers.{l}.attn.Wqkv.weight"] = combined_weight
        
        # Remove rotary embedding parameters if they exist
        state_dict.pop(f"layers.{l}.self_attn.rotary_emb.inv_freq", None)

    def key_mapping_attn(key):
        key = re.sub(r"^layers.(\d+).self_attn.o_proj.", r"layers.\1.attn.out_proj.", key)
        return key

    state_dict = OrderedDict((key_mapping_attn(k), v) for k, v in state_dict.items())

    return state_dict


def llama_config_to_gpt2_config(config: LlamaConfig) -> GPT2Config:

    return GPT2Config(
        vocab_size=config.vocab_size,
        n_positions=0,  # No absolute position embedding
        n_embd=config.hidden_size,
        n_layer=config.num_hidden_layers,
        n_head=config.num_attention_heads,
        num_heads_kv=config.num_key_value_heads,
        n_inner=config.intermediate_size,
        activation_function="swiglu",
        resid_pdrop=0.0,  # No dropout
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        layer_norm_epsilon=config.rms_norm_eps,
        initializer_range=config.initializer_range,
        bos_token_id=config.bos_token_id,
        eos_token_id=config.eos_token_id,
        # These are new arguments not in the original GPT2Config
        prenorm=True,
        # TODO: rope_scaling?
        rotary_emb_fraction=1.0,
        tie_word_embeddings=config.tie_word_embeddings,
        fused_dropout_add_ln=True,
        fused_bias_fc=True,
        use_flash_attn=True,
        qkv_proj_bias=getattr(config, "attention_bias", True),
        rotary_emb_base=config.rope_theta,
        rotary_emb_scale_base=getattr(config, "rotary_emb_scale_base", None),
        rotary_emb_interleaved=getattr(config, "rotary_emb_interleaved", False),
        mlp_fc1_bias=getattr(config, "mlp_bias", True),
        mlp_fc2_bias=getattr(config, "mlp_bias", True),
        rope_scaling=config.rope_scaling,
        max_trained_positions=getattr(config, "max_trained_positions", None),
        ln_f_bias=False,
        use_rms_norm=True,
        causal=True,
    )
