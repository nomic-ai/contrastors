import re
from collections import OrderedDict

import torch
from einops import rearrange
from transformers import Dinov2Config, GPT2Config


def dino_config_to_vit_config(dino_config: Dinov2Config) -> GPT2Config:
    return GPT2Config(
        n_embd=dino_config.hidden_size,
        n_layer=dino_config.num_hidden_layers,
        n_head=dino_config.num_attention_heads,
        n_inner=dino_config.mlp_ratio * dino_config.hidden_size,
        activation_function=(
            dino_config.hidden_act if getattr(dino_config, "use_swiglu_ffn", False) is False else "swiglu"
        ),
        vocab_size=0,  # no vocab since using patches
        n_positions=0,  # No absolute position embedding
        resid_pdrop=0.0,  # No dropout
        embd_pdrop=dino_config.hidden_dropout_prob,
        attn_pdrop=dino_config.attention_probs_dropout_prob,
        layer_norm_epsilon=dino_config.layer_norm_eps,
        initializer_range=dino_config.initializer_range,
        bos_token_id=None,
        eos_token_id=None,
        # These are new arguments not in the original GPT2Config
        drop_path_rate=dino_config.drop_path_rate,
        layer_scale=True,
        layer_scale_init=dino_config.layerscale_value,
        img_size=dino_config.image_size,
        patch_size=dino_config.patch_size,
        num_channels=dino_config.num_channels,
        prenorm=True,
        parallel_block=False,
        parallel_block_tied_norm=False,
        rotary_emb_fraction=0,
        tie_word_embeddings=False,
        fused_dropout_add_ln=True,
        fused_bias_fc=True,
        patch_embed_bias=True,
        use_flash_attn=True,
        qkv_proj_bias=dino_config.qkv_bias,
        mlp_fc1_bias=getattr(dino_config, "mlp_fc1_bias", True),
        mlp_fc2_bias=getattr(dino_config, "mlp_fc2_bias", True),
        use_rms_norm=False,
        causal=False,
        hidden_features_scaling_factor=2.0 / 3.0 if getattr(dino_config, "use_swiglu_ffn", False) else 1.0,
        mask_token=True,
        learned_pos_embedding=True,
        patch_dropout=0.0,
    )


def remap_state_dict_hf_dinov2(state_dict, config):
    def key_mapping_layers(key):
        return re.sub(r"^encoder.", "", key)

    state_dict = OrderedDict((key_mapping_layers(k), v) for k, v in state_dict.items())

    def key_mapping_ln(key):
        key = re.sub(r"^layernorm.", r"ln_f.", key)
        return key

    state_dict = OrderedDict((key_mapping_ln(k), v) for k, v in state_dict.items())

    def key_mapping_rename(key):
        return re.sub(r"^layer.", "layers.", key)

    state_dict = OrderedDict((key_mapping_rename(k), v) for k, v in state_dict.items())

    def key_mapping_embeddings(key):
        key = re.sub(r"^embeddings.position_embeddings", r"embeddings.pos_embed", key)
        key = re.sub(r"^embeddings.patch_embeddings.projection.", r"embeddings.proj.", key)
        return key

    state_dict = OrderedDict((key_mapping_embeddings(k), v) for k, v in state_dict.items())

    patch_embed_weight = state_dict["embeddings.proj.weight"]
    if patch_embed_weight.dim() == 4:
        # convert from Conv2d to Linear
        state_dict["embeddings.proj.weight"] = rearrange(patch_embed_weight, "o c h w -> o (c h w)")

    def attention_mapping_layers(key):
        return re.sub("attention.attention.", "attn.", key)

    state_dict = OrderedDict((attention_mapping_layers(k), v) for k, v in state_dict.items())

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
