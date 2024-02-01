import math
import re
from collections import OrderedDict

import torch
import torch.nn.functional as F
from transformers import AutoConfig, GPT2Config


def remap_state_dict_hf_open_lm(state_dict, config):
    def key_mapping_layers(key):
        return re.sub(r"^model.", "", key)

    state_dict = OrderedDict((key_mapping_layers(k), v) for k, v in state_dict.items())

    # Word embedding
    def key_mapping_emb(key):
        return re.sub(r"^tok_embeddings.", "embeddings.", key)

    state_dict = OrderedDict((key_mapping_emb(k), v) for k, v in state_dict.items())

    def key_mapping_lm_head(key):
        return re.sub(r"^output.", "lm_head.", key)

    state_dict = OrderedDict((key_mapping_lm_head(k), v) for k, v in state_dict.items())

    word_embeddings = state_dict.pop("embeddings.weight")
    # It's possible that vocab_size is padded to be a multiple of 8, for example.
    pad_vocab_size_multiple = getattr(config, "pad_vocab_size_multiple", 1)
    vocab_size = math.ceil(config.vocab_size / pad_vocab_size_multiple) * pad_vocab_size_multiple
    state_dict["embeddings.weight"] = F.pad(word_embeddings, (0, 0, 0, vocab_size - word_embeddings.shape[0]))
    if getattr(config, "tie_word_embeddings"):
        state_dict["lm_head.weight"] = state_dict["embeddings.weight"]
    else:
        output_embeddings = state_dict.pop("lm_head.weight")
        # It's possible that vocab_size is padded to be a multiple of 8, for example.
        state_dict["lm_head.weight"] = F.pad(output_embeddings, (0, 0, 0, vocab_size - output_embeddings.shape[0]))

    # LayerNorm
    def key_mapping_ln(key):
        key = re.sub(r"^norm.", r"ln_f.", key)
        key = re.sub(r"^layers.(\d+).attention_norm.", r"layers.\1.norm1.", key)
        key = re.sub(
            r"^layers.(\d+).ffn_norm.",
            r"layers.\1.norm2.",
            key,
        )
        return key

    state_dict = OrderedDict((key_mapping_ln(k), v) for k, v in state_dict.items())

    # MLP
    def key_mapping_mlp(key):
        key = re.sub(r"^layers.(\d+).feed_forward.w12.", r"layers.\1.mlp.fc1.", key)
        key = re.sub(r"^layers.(\d+).feed_forward.w3.", r"layers.\1.mlp.fc2.", key)
        return key

    state_dict = OrderedDict((key_mapping_mlp(k), v) for k, v in state_dict.items())

    # ordering of weight makes difference for gatedmlp
    for layer in range(config.n_layer):
        fc1 = state_dict.pop(f"layers.{layer}.mlp.fc1.weight")
        w1, w2 = fc1.chunk(2, dim=0)
        fc1 = torch.cat([w2, w1], dim=0)
        state_dict[f"layers.{layer}.mlp.fc1.weight"] = fc1

    state_dict = {k: v for k, v in state_dict.items() if "inv_freq" not in k}

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
        key = re.sub(
            r"^layers.(\d+).attention.in_proj.",
            r"layers.\1.attn.Wqkv.",
            key,
        )
        key = re.sub(
            r"^layers.(\d+).attention.out_proj.",
            r"layers.\1.attn.out_proj.",
            key,
        )
        return key

    state_dict.pop("lm_head.weight")

    state_dict = OrderedDict((key_mapping_attn(k), v) for k, v in state_dict.items())

    return state_dict


def open_lm_config_to_gpt2_config(open_lm_config: AutoConfig) -> GPT2Config:
    # NOTE: rotary is applied to the head dimension instead of the sequence dimension (accident by open_lm team)
    return GPT2Config(
        vocab_size=open_lm_config.vocab_size,
        n_positions=0,  # No absolute position embedding
        n_embd=open_lm_config.hidden_dim,
        n_layer=open_lm_config.n_layers,
        n_head=open_lm_config.n_heads,
        n_inner=256 * ((int(2 * 4 * open_lm_config.hidden_dim / 3) + 256 - 1) // 256),
        activation_function="swiglu",
        resid_pdrop=0.0,  # No dropout
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        layer_norm_epsilon=1e-5,
        # NOTE: this isn't correct, should look at code as it's scaled by depth according to: https://arxiv.org/abs/1908.11365
        initializer_range=0.02,
        bos_token_id=0,
        eos_token_id=0,
        # These are new arguments not in the original GPT2Config
        prenorm=True,
        parallel_block=False,
        parallel_block_tied_norm=False,
        rotary_emb_fraction=1,
        tie_word_embeddings=open_lm_config.weight_tying,
        fused_dropout_add_ln=True,
        fused_bias_fc=True,
        use_flash_attn=True,
        qkv_proj_bias=getattr(open_lm_config, "qkv_proj_bias", False),
        rotary_emb_base=10000,
        rotary_emb_scale_base=getattr(open_lm_config, "rotary_emb_scale_base", None),
        rotary_emb_interleaved=getattr(open_lm_config, "rotary_emb_interleaved", False),
        rotary_head_dim=getattr(open_lm_config, "rotary_old", False),
        mlp_fc1_bias=getattr(open_lm_config, "mlp_fc1_bias", False),
        mlp_fc2_bias=getattr(open_lm_config, "mlp_fc2_bias", False),
        use_rms_norm=False,
        causal=True,
    )
