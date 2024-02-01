import math
import re
from collections import OrderedDict

import torch.nn.functional as F
from einops import rearrange
from transformers import GPT2Config, GPTNeoXConfig


def remap_state_dict_hf_gpt_neox(state_dict, config):
    def key_mapping_layers(key):
        return re.sub(r"^gpt_neox.", "", key)

    state_dict = OrderedDict((key_mapping_layers(k), v) for k, v in state_dict.items())

    # Word embedding
    def key_mapping_emb(key):
        return re.sub(r"^embed_in.", "embeddings.", key)

    state_dict = OrderedDict((key_mapping_emb(k), v) for k, v in state_dict.items())
    word_embeddings = state_dict.pop("embeddings.weight")
    # It's possible that vocab_size is padded to be a multiple of 8, for example.
    pad_vocab_size_multiple = getattr(config, "pad_vocab_size_multiple", 1)
    vocab_size = math.ceil(config.vocab_size / pad_vocab_size_multiple) * pad_vocab_size_multiple
    state_dict["embeddings.weight"] = F.pad(word_embeddings, (0, 0, 0, vocab_size - word_embeddings.shape[0]))
    if getattr(config, "tie_word_embeddings"):
        state_dict["lm_head.weight"] = state_dict["embeddings.weight"]
    else:
        output_embeddings = state_dict.pop("embed_out.weight")
        # It's possible that vocab_size is padded to be a multiple of 8, for example.
        state_dict["lm_head.weight"] = F.pad(output_embeddings, (0, 0, 0, vocab_size - output_embeddings.shape[0]))

    # LayerNorm
    def key_mapping_ln(key):
        key = re.sub(r"^final_layer_norm.", r"ln_f.", key)
        key = re.sub(r"^layers.(\d+).input_layernorm.", r"layers.\1.norm1.", key)
        key = re.sub(
            r"^layers.(\d+).post_attention_layernorm.",
            r"layers.\1.norm2.",
            key,
        )
        return key

    state_dict = OrderedDict((key_mapping_ln(k), v) for k, v in state_dict.items())

    # MLP
    def key_mapping_mlp(key):
        key = re.sub(r"^layers.(\d+).mlp.dense_h_to_4h.", r"layers.\1.mlp.fc1.", key)
        key = re.sub(r"^layers.(\d+).mlp.dense_4h_to_h.", r"layers.\1.mlp.fc2.", key)
        return key

    state_dict = OrderedDict((key_mapping_mlp(k), v) for k, v in state_dict.items())

    # Attention
    for l in range(config.n_layer):
        # We don't store these biases
        state_dict.pop(f"layers.{l}.attention.bias")
        state_dict.pop(f"layers.{l}.attention.masked_bias")
        state_dict.pop(f"layers.{l}.attention.rotary_emb.inv_freq", None)
        # GPT-NeoX stores Wqkv as ((nheads 3 headdim), hidden_dim)
        # while we store Wqkv as ((3 nheads headdim), hidden_dim)
        headdim = config.hidden_size // config.num_attention_heads
        Wqkv = state_dict.pop(f"layers.{l}.attention.query_key_value.weight")
        state_dict[f"layers.{l}.attn.Wqkv.weight"] = rearrange(
            Wqkv,
            "(nheads three headdim) ... -> (three nheads headdim) ...",
            three=3,
            headdim=headdim,
        )
        bqkv = state_dict.pop(f"layers.{l}.attention.query_key_value.bias")
        state_dict[f"layers.{l}.attn.Wqkv.bias"] = rearrange(
            bqkv, "(nheads three headdim) -> (three nheads headdim)", three=3, headdim=headdim
        )

    def key_mapping_attn(key):
        key = re.sub(
            r"^layers.(\d+).attention.dense.",
            r"layers.\1.attn.out_proj.",
            key,
        )
        key = re.sub(
            r"^layers.(\d+).attention.rotary_emb.",
            r"layers.\1.attn.rotary_emb.",
            key,
        )
        return key

    state_dict.pop("lm_head.weight")

    state_dict = OrderedDict((key_mapping_attn(k), v) for k, v in state_dict.items())

    return state_dict


def gpt_neox_config_to_gpt2_config(gpt_neox_config: GPTNeoXConfig) -> GPT2Config:
    assert gpt_neox_config.rotary_emb_base == 10000
    return GPT2Config(
        vocab_size=gpt_neox_config.vocab_size,
        n_positions=0,  # No absolute position embedding
        n_embd=gpt_neox_config.hidden_size,
        n_layer=gpt_neox_config.num_hidden_layers,
        n_head=gpt_neox_config.num_attention_heads,
        n_inner=gpt_neox_config.intermediate_size,
        activation_function=gpt_neox_config.hidden_act,
        resid_pdrop=0.0,  # No dropout
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        layer_norm_epsilon=gpt_neox_config.layer_norm_eps,
        initializer_range=gpt_neox_config.initializer_range,
        bos_token_id=gpt_neox_config.bos_token_id,
        eos_token_id=gpt_neox_config.eos_token_id,
        # These are new arguments not in the original GPT2Config
        prenorm=True,
        parallel_block=gpt_neox_config.use_parallel_residual,
        parallel_block_tied_norm=False,
        rotary_emb_fraction=gpt_neox_config.rotary_pct,
        tie_word_embeddings=gpt_neox_config.tie_word_embeddings,
        fused_dropout_add_ln=True,
        fused_bias_fc=True,
        use_flash_attn=True,
        qkv_proj_bias=getattr(gpt_neox_config, "qkv_proj_bias", True),
        rotary_emb_base=gpt_neox_config.rotary_emb_base,
        rotary_emb_scale_base=getattr(gpt_neox_config, "rotary_emb_scale_base", None),
        rotary_emb_interleaved=getattr(gpt_neox_config, "rotary_emb_interleaved", False),
        mlp_fc1_bias=getattr(gpt_neox_config, "mlp_fc1_bias", True),
        mlp_fc2_bias=getattr(gpt_neox_config, "mlp_fc2_bias", True),
        use_rms_norm=False,
        causal=True,
    )
