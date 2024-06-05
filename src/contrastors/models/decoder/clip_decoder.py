import math
import re
from collections import OrderedDict

import torch
import torch.nn.functional as F
from transformers import CLIPConfig, GPT2Config


def remap_state_dict_hf_clip_text(state_dict, config):
    state_dict = {k: v for k, v in state_dict.items() if "vision_model" not in k and "visual_"}

    def key_mapping_layers(key):
        key = re.sub(r"^text_model.", "", key)
        key = re.sub(r"^encoder.", "", key)
        return key

    state_dict = OrderedDict((key_mapping_layers(k), v) for k, v in state_dict.items())

    # Word embedding
    def key_mapping_emb(key):
        key = re.sub(r"token_embedding", "word_embeddings", key)
        key = re.sub(r"position_embedding", "position_embeddings", key)

        return key

    state_dict = OrderedDict((key_mapping_emb(k), v) for k, v in state_dict.items())
    word_embeddings = state_dict.pop("embeddings.word_embeddings.weight")
    # It's possible that vocab_size is padded to be a multiple of 8, for example.
    pad_vocab_size_multiple = getattr(config, "pad_vocab_size_multiple", 1)
    vocab_size = math.ceil(config.vocab_size / pad_vocab_size_multiple) * pad_vocab_size_multiple
    state_dict["embeddings.word_embeddings.weight"] = F.pad(
        word_embeddings, (0, 0, 0, vocab_size - word_embeddings.shape[0])
    )
    state_dict.pop("embeddings.position_ids", None)

    # LayerNorm
    def key_mapping_ln(key):
        key = re.sub(r"^final_layer_norm.", r"ln_f.", key)

        return key

    state_dict = OrderedDict((key_mapping_ln(k), v) for k, v in state_dict.items())

    def layer_norm_mapping(key):
        return re.sub("layer_norm", "norm", key)

    state_dict = OrderedDict((layer_norm_mapping(k), v) for k, v in state_dict.items())

    # Attention
    for l in range(config.n_layer):
        Wq = state_dict.pop(f"layers.{l}.self_attn.q_proj.weight")
        bq = state_dict.pop(f"layers.{l}.self_attn.q_proj.bias")

        Wk = state_dict.pop(f"layers.{l}.self_attn.k_proj.weight")
        bk = state_dict.pop(f"layers.{l}.self_attn.k_proj.bias")

        Wv = state_dict.pop(f"layers.{l}.self_attn.v_proj.weight")
        bv = state_dict.pop(f"layers.{l}.self_attn.v_proj.bias")

        Wqkv = torch.cat([Wq, Wk, Wv], dim=0)
        bqkv = torch.cat([bq, bk, bv], dim=0)
        state_dict[f"layers.{l}.attn.Wqkv.weight"] = Wqkv
        state_dict[f"layers.{l}.attn.Wqkv.bias"] = bqkv

    def key_mapping_attn(key):
        key = re.sub(
            r"^layers.(\d+).self_attn",
            r"layers.\1.attn",
            key,
        )
        return key

    state_dict = OrderedDict((key_mapping_attn(k), v) for k, v in state_dict.items())

    return state_dict


def clip_config_to_gpt2_config(clip_config: CLIPConfig) -> GPT2Config:
    text_config = clip_config.text_config
    return GPT2Config(
        vocab_size=text_config.vocab_size,
        n_positions=text_config.max_position_embeddings,
        n_embd=text_config.hidden_size,
        n_layer=text_config.num_hidden_layers,
        n_head=text_config.num_attention_heads,
        n_inner=text_config.intermediate_size,
        activation_function=text_config.hidden_act,
        resid_pdrop=0.0,  # No dropout
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        layer_norm_epsilon=text_config.layer_norm_eps,
        initializer_range=text_config.initializer_range,
        bos_token_id=text_config.bos_token_id,
        eos_token_id=text_config.eos_token_id,
        # These are new arguments not in the original GPT2Config
        prenorm=True,
        parallel_block=False,
        parallel_block_tied_norm=False,
        rotary_emb_fraction=0.0,
        tie_word_embeddings=text_config.tie_word_embeddings,
        fused_dropout_add_ln=True,
        fused_bias_fc=True,
        use_flash_attn=True,
        qkv_proj_bias=getattr(text_config, "qkv_proj_bias", True),
        rotary_emb_base=None,
        rotary_emb_scale_base=None,
        rotary_emb_interleaved=getattr(text_config, "rotary_emb_interleaved", False),
        mlp_fc1_bias=getattr(text_config, "mlp_fc1_bias", True),
        mlp_fc2_bias=getattr(text_config, "mlp_fc2_bias", True),
        use_rms_norm=False,
        causal=True,
    )
