import re
from collections import OrderedDict

import torch
from einops import rearrange
from transformers import CLIPConfig, GPT2Config


def clip_config_to_vit_config(clip_config: CLIPConfig) -> GPT2Config:
    clip_config = clip_config.vision_config
    return GPT2Config(
        n_embd=clip_config.hidden_size,
        n_layer=clip_config.num_hidden_layers,
        n_head=clip_config.num_attention_heads,
        n_inner=clip_config.intermediate_size,
        activation_function=clip_config.hidden_act,
        vocab_size=0,  # no vocab since using patches
        n_positions=0,  # No absolute position embedding
        resid_pdrop=0.0,  # No dropout
        embd_pdrop=getattr(clip_config, "dropout", 0.0),
        attn_pdrop=clip_config.attention_dropout,
        layer_norm_epsilon=clip_config.layer_norm_eps,
        initializer_range=clip_config.initializer_range,
        bos_token_id=None,
        eos_token_id=None,
        # These are new arguments not in the original GPT2Config
        drop_path_rate=0.0,
        # Why is there double layer norm??
        prepre_layernom=True,
        layer_scale=False,
        layer_scale_init=None,
        img_size=clip_config.image_size,
        patch_size=clip_config.patch_size,
        num_channels=clip_config.num_channels,
        prenorm=True,
        parallel_block=False,
        parallel_block_tied_norm=False,
        rotary_emb_fraction=0,
        tie_word_embeddings=False,
        fused_dropout_add_ln=True,
        fused_bias_fc=True,
        patch_embed_bias=False,
        use_flash_attn=True,
        qkv_proj_bias=True,
        mlp_fc1_bias=getattr(clip_config, "mlp_fc1_bias", True),
        mlp_fc2_bias=getattr(clip_config, "mlp_fc2_bias", True),
        use_rms_norm=False,
        causal=False,
        hidden_features_scaling_factor=2.0 / 3.0 if getattr(clip_config, "use_swiglu_ffn", False) else 1.0,
        mask_token=False,
        learned_pos_embedding=False,
        patch_dropout=0.0,
    )


def remap_state_dict_hf_clip(state_dict, config):
    def remove_vision_prefix(key):
        return re.sub(r"^vision_model.", "", key)

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
        key = re.sub(r"^embeddings.position_embedding", r"embeddings.pos_embed", key)
        key = re.sub(r"^embeddings.pos_embed.weight", r"embeddings.pos_embed", key)
        key = re.sub(r"^embeddings.patch_embedding.", r"embeddings.proj.", key)
        key = re.sub(r"^embeddings.class_embedding", r"embeddings.cls_token", key)
        return key

    state_dict = OrderedDict((key_mapping_embeddings(k), v) for k, v in state_dict.items())
    state_dict.pop("embeddings.position_ids", None)
    state_dict["embeddings.pos_embed"] = state_dict["embeddings.pos_embed"].unsqueeze(0)

    patch_embed_weight = state_dict["embeddings.proj.weight"]
    if patch_embed_weight.dim() == 4:
        # convert from Conv2d to Linear
        state_dict["embeddings.proj.weight"] = rearrange(patch_embed_weight, "o c h w -> o (c h w)")

    def attention_mapping_layers(key):
        return re.sub("self_attn.", "attn.", key)

    state_dict = OrderedDict((attention_mapping_layers(k), v) for k, v in state_dict.items())

    def layer_norm_mapping(key):
        return re.sub("layer_norm", "norm", key)

    state_dict = OrderedDict((layer_norm_mapping(k), v) for k, v in state_dict.items())

    def key_mapping_attn(key):
        key = re.sub(
            r"^layers.(\d+).attention.output.dense.",
            r"layers.\1.attn.out_proj.",
            key,
        )
        return key

    state_dict = OrderedDict((key_mapping_attn(k), v) for k, v in state_dict.items())

    def key_mapping_ls(key):
        key = re.sub(
            r"^layers.(\d+).layer_scale1.lambda1",
            r"layers.\1.ls1",
            key,
        )
        key = re.sub(
            r"^layers.(\d+).layer_scale2.lambda1",
            r"layers.\1.ls2",
            key,
        )
        return key

    state_dict = OrderedDict((key_mapping_ls(k), v) for k, v in state_dict.items())

    def key_mapping_swiglu(key):
        key = re.sub(
            r"^layers.(\d+).mlp.weights_out",
            r"layers.\1.mlp.fc2",
            key,
        )
        return key

    state_dict = OrderedDict((key_mapping_swiglu(k), v) for k, v in state_dict.items())

    use_swiglu = config.activation_function == "swiglu"
    for l in range(config.n_layer):
        # We don't store these biases
        qw = state_dict.pop(f"layers.{l}.attn.q_proj.weight")
        kw = state_dict.pop(f"layers.{l}.attn.k_proj.weight")
        vw = state_dict.pop(f"layers.{l}.attn.v_proj.weight")

        qb = state_dict.pop(f"layers.{l}.attn.q_proj.bias")
        kb = state_dict.pop(f"layers.{l}.attn.k_proj.bias")
        vb = state_dict.pop(f"layers.{l}.attn.v_proj.bias")

        Wqkv_weight = torch.cat([qw, kw, vw], dim=0)
        Wqkv_bias = torch.cat([qb, kb, vb], dim=0)

        state_dict[f"layers.{l}.attn.Wqkv.weight"] = Wqkv_weight
        state_dict[f"layers.{l}.attn.Wqkv.bias"] = Wqkv_bias

        if use_swiglu:
            weights_in = state_dict.pop(f'layers.{l}.mlp.weights_in.weight')
            weights_in_bias = state_dict.pop(f'layers.{l}.mlp.weights_in.bias')
            # we've reorderd the weights in the model definition
            fc12, fc11 = weights_in.chunk(2, dim=0)
            b12, b11 = weights_in_bias.chunk(2, dim=0)
            state_dict[f"layers.{l}.mlp.fc11.weight"] = fc12
            state_dict[f"layers.{l}.mlp.fc12.weight"] = fc11
            state_dict[f"layers.{l}.mlp.fc11.bias"] = b12
            state_dict[f"layers.{l}.mlp.fc12.bias"] = b11

    return state_dict
