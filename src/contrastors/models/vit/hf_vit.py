import re
from collections import OrderedDict

import torch
from einops import rearrange
from transformers import GPT2Config, ViTConfig


def hf_vit_config_to_vit_config(vit_config: ViTConfig) -> GPT2Config:
    return GPT2Config(
        n_embd=vit_config.hidden_size,
        n_layer=vit_config.num_hidden_layers,
        n_head=vit_config.num_attention_heads,
        n_inner=vit_config.intermediate_size,
        activation_function=vit_config.hidden_act,
        vocab_size=0,  # no vocab since using patches
        n_positions=0,  # No absolute position embedding
        resid_pdrop=0.0,  # No dropout
        embd_pdrop=getattr(vit_config, "dropout", 0.0),
        attn_pdrop=vit_config.attention_probs_dropout_prob,
        layer_norm_epsilon=vit_config.layer_norm_eps,
        initializer_range=vit_config.initializer_range,
        bos_token_id=None,
        eos_token_id=None,
        # These are new arguments not in the original GPT2Config
        drop_path_rate=0.0,
        # Why is there double layer norm??
        prepre_layernom=False,
        layer_scale=False,
        layer_scale_init=None,
        img_size=vit_config.image_size,
        patch_size=vit_config.patch_size,
        num_channels=vit_config.num_channels,
        prenorm=True,
        parallel_block=False,
        parallel_block_tied_norm=False,
        rotary_emb_fraction=0,
        tie_word_embeddings=False,
        fused_dropout_add_ln=True,
        fused_bias_fc=True,
        patch_embed_bias=True,
        use_flash_attn=True,
        qkv_proj_bias=True,
        mlp_fc1_bias=getattr(vit_config, "mlp_fc1_bias", True),
        mlp_fc2_bias=getattr(vit_config, "mlp_fc2_bias", True),
        use_rms_norm=False,
        causal=False,
        hidden_features_scaling_factor=1.0,
        mask_token=False,
        learned_pos_embedding=False,
        patch_dropout=0,
        sinusoidal_pos_embedding=vit_config.model_type == "vit_mae",
    )


def remap_state_dict_hf_vit(state_dict, config):
    def remove_vision_prefix(key):
        return re.sub(r"^vit.", "", key)

    state_dict = OrderedDict((remove_vision_prefix(k), v) for k, v in state_dict.items())

    def key_mapping_layers(key):
        return re.sub(r"^encoder.", "", key)

    state_dict = OrderedDict((key_mapping_layers(k), v) for k, v in state_dict.items())

    def key_mapping_ln(key):
        key = re.sub(r"^post_layernorm.", r"ln_f.", key)
        return key

    state_dict = OrderedDict((key_mapping_ln(k), v) for k, v in state_dict.items())

    def key_mapping_ln(key):
        key = re.sub(r"^pre_layrnorm.", r"prepre_layernom.", key)
        return key

    state_dict = OrderedDict((key_mapping_ln(k), v) for k, v in state_dict.items())

    def key_mapping_rename(key):
        return re.sub(r"^layer\.", "layers.", key)

    state_dict = OrderedDict((key_mapping_rename(k), v) for k, v in state_dict.items())

    def key_mapping_embeddings(key):
        key = re.sub(r"^embeddings.position_embeddings", r"embeddings.pos_embed.weight", key)
        key = re.sub(r"^embeddings.patch_embeddings.projection.", r"embeddings.proj.", key)
        key = re.sub(r"^embeddings.class_embedding", r"embeddings.cls_token", key)
        return key

    state_dict = OrderedDict((key_mapping_embeddings(k), v) for k, v in state_dict.items())

    state_dict["embeddings.cls_token"] = state_dict["embeddings.cls_token"]
    state_dict["embeddings.pos_embed"] = state_dict.pop("embeddings.pos_embed.weight")

    patch_embed_weight = state_dict["embeddings.proj.weight"]
    if patch_embed_weight.dim() == 4:
        # convert from Conv2d to Linear
        state_dict["embeddings.proj.weight"] = rearrange(patch_embed_weight, "o c h w -> o (c h w)")

    def attention_mapping_layers(key):
        return re.sub("attention.attention.", "attn.", key)

    state_dict = OrderedDict((attention_mapping_layers(k), v) for k, v in state_dict.items())

    def layer_norm_mapping(key):
        key = re.sub("^layers.(\d+).layernorm_before", "layers.\\1.norm1", key)
        key = re.sub("^layers.(\d+).layernorm_after", "layers.\\1.norm2", key)

        return key

    state_dict = OrderedDict((layer_norm_mapping(k), v) for k, v in state_dict.items())

    def key_mapping_attn(key):
        key = re.sub(
            r"^layers.(\d+).attention.output.dense.",
            r"layers.\1.attn.out_proj.",
            key,
        )
        return key

    state_dict = OrderedDict((key_mapping_attn(k), v) for k, v in state_dict.items())

    def key_mapping_ffn(key):
        key = re.sub(
            r"^layers.(\d+).intermediate.dense.",
            r"layers.\1.mlp.fc1.",
            key,
        )
        key = re.sub(
            r"^layers.(\d+).output.dense.",
            r"layers.\1.mlp.fc2.",
            key,
        )
        return key

    state_dict = OrderedDict((key_mapping_ffn(k), v) for k, v in state_dict.items())

    for l in range(config.n_layer):
        # We don't store these biases
        qw = state_dict.pop(f"layers.{l}.attn.query.weight")
        kw = state_dict.pop(f"layers.{l}.attn.key.weight")
        vw = state_dict.pop(f"layers.{l}.attn.value.weight")

        qb = state_dict.pop(f"layers.{l}.attn.query.bias")
        kb = state_dict.pop(f"layers.{l}.attn.key.bias")
        vb = state_dict.pop(f"layers.{l}.attn.value.bias")

        Wqkv_weight = torch.cat([qw, kw, vw], dim=0)
        Wqkv_bias = torch.cat([qb, kb, vb], dim=0)

        state_dict[f"layers.{l}.attn.Wqkv.weight"] = Wqkv_weight
        state_dict[f"layers.{l}.attn.Wqkv.bias"] = Wqkv_bias

    # some models are stored as ViTModelForImageClassification, which has a classifier
    # while others are just ViTModel
    state_dict.pop("classifier.weight", None)
    state_dict.pop("classifier.bias", None)
    state_dict.pop("pooler.dense.weight", None)
    state_dict.pop("pooler.dense.bias", None)

    state_dict["ln_f.weight"] = state_dict.pop("layernorm.weight")
    state_dict["ln_f.bias"] = state_dict.pop("layernorm.bias")

    # remove all layers that start with decoder (e.g. vit-mae)
    state_dict = OrderedDict((k, v) for k, v in state_dict.items() if not k.startswith("decoder"))

    return state_dict


def inverse_remap_state_dict_hf_vit(state_dict, config):
    def add_encoder_prefix(key):
        return re.sub(r"^layers.", "encoder.layers.", key)

    state_dict = OrderedDict((add_encoder_prefix(k), v) for k, v in state_dict.items())

    # Reverse the attention weights and biases concatenation
    for l in range(config.n_layer):
        Wqkv_weight = state_dict.pop(f"encoder.layers.{l}.attn.Wqkv.weight")
        Wqkv_bias = state_dict.pop(f"encoder.layers.{l}.attn.Wqkv.bias")

        split_dim = Wqkv_weight.size(0) // 3
        qw, kw, vw = torch.split(Wqkv_weight, split_dim, dim=0)
        qb, kb, vb = torch.split(Wqkv_bias, split_dim, dim=0)

        state_dict[f"encoder.layers.{l}.attn.query.weight"] = qw
        state_dict[f"encoder.layers.{l}.attn.key.weight"] = kw
        state_dict[f"encoder.layers.{l}.attn.value.weight"] = vw

        state_dict[f"encoder.layers.{l}.attn.query.bias"] = qb
        state_dict[f"encoder.layers.{l}.attn.key.bias"] = kb
        state_dict[f"encoder.layers.{l}.attn.value.bias"] = vb

    # Reverse the key mappings applied last to first
    def key_mapping_reverse_ffn(key):
        key = re.sub(r"layers.(\d+).mlp.fc2", r"layers.\1.output.dense", key)
        key = re.sub(r"layers.(\d+).mlp.fc1", r"layers.\1.intermediate.dense", key)
        return key

    state_dict = OrderedDict((key_mapping_reverse_ffn(k), v) for k, v in state_dict.items())

    def key_mapping_reverse_attn(key):
        key = re.sub(r"layers.(\d+).attn.out_proj", r"layers.\1.attention.output.dense", key)
        return key

    state_dict = OrderedDict((key_mapping_reverse_attn(k), v) for k, v in state_dict.items())

    def layer_norm_reverse_mapping(key):
        key = re.sub("layers.(\d+).norm2", "layers.\\1.layernorm_after", key)
        key = re.sub("layers.(\d+).norm1", "layers.\\1.layernorm_before", key)
        return key

    state_dict = OrderedDict((layer_norm_reverse_mapping(k), v) for k, v in state_dict.items())

    def attention_reverse_mapping_layers(key):
        return re.sub("attn.", "attention.attention.", key)

    state_dict = OrderedDict((attention_reverse_mapping_layers(k), v) for k, v in state_dict.items())

    def key_mapping_embeddings_reverse(key):
        key = re.sub(r"embeddings.pos_embed.weight", r"embeddings.position_embeddings.weight", key)
        key = re.sub(r"embeddings.proj.", r"embeddings.patch_embeddings.projection.", key)
        key = re.sub(r"embeddings.cls_token", r"embeddings.class_embedding", key)
        return key

    state_dict = OrderedDict((key_mapping_embeddings_reverse(k), v) for k, v in state_dict.items())

    def key_mapping_rename_reverse(key):
        return re.sub(r"layers.", "layer.", key)

    state_dict = OrderedDict((key_mapping_rename_reverse(k), v) for k, v in state_dict.items())

    # Reverse post-processing of embeddings
    state_dict["embeddings.cls_token"] = state_dict.pop("embeddings.class_embedding")
    state_dict["embeddings.position_embeddings"] = state_dict.pop("embeddings.pos_embed")

    state_dict["layernorm.weight"] = state_dict.pop("ln_f.weight")
    state_dict["layernorm.bias"] = state_dict.pop("ln_f.bias")

    if state_dict["embeddings.patch_embeddings.projection.weight"].dim() == 2:
        # convert from Linear to Conv2d
        state_dict["embeddings.patch_embeddings.projection.weight"] = rearrange(
            state_dict["embeddings.patch_embeddings.projection.weight"],
            "o (c h w) -> o c h w",
            c=3,
            h=config.patch_size,
            w=config.patch_size,
        )

    return state_dict
