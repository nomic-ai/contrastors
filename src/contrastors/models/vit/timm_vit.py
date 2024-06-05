import re
from collections import OrderedDict

import torch
import torch.nn as nn
from transformers import GPT2Config


def get_vit_base_patch16_224_params():
    return {
        "patch_size": 16,
        "n_embd": 768,
        "n_layer": 12,
        "n_head": 12,
        "mlp_ratio": 4,
        "activation": "gelu",
        "image_size": 224,
    }


def get_vit_base_patch14_reg4_dinov2_params():
    return {
        "patch_size": 14,
        "n_embd": 768,
        "n_layer": 12,
        "n_head": 12,
        "mlp_ratio": 4,
        "activation": "gelu",
        "dropout": 0.0,
        "attention_dropout": 0.0,
        "eps": 1e-6,
        "init_std": 0.02,
        "layer_scale": True,
        "layer_scale_init": 1.0e-5,
        "register_tokens": 4,
        "learned_pos_embedding": True,
        "image_size": 518,
        "no_embed_class": True,
    }


def get_vit_base_patch16_rope_reg1_gap_256_params():
    return {
        "patch_size": 16,
        "n_embd": 768,
        "n_layer": 12,
        "n_head": 12,
        "mlp_ratio": 4,
        "activation": "gelu",
        "dropout": 0.0,
        "attention_dropout": 0.0,
        "eps": 1e-6,
        "init_std": 0.02,
        "layer_scale": True,
        "layer_scale_init": 1.0e-5,
        "register_tokens": 1,
        "learned_pos_embedding": False,
        "use_rotary_pos_emb": True,
        # to use 256, change ref_feat_shape to (16, 16)
        "image_size": 224,
        "no_embed_class": True,
        "ref_feat_shape": (14, 14),
        "no_cls_token": True,
        "use_pos_embed": False,
        "eva_qkv_bias": True,
        "no_last_ln": True,
        # "global_pool": "avg",
    }


def get_eva_base_patch16_224_params():
    return {
        "patch_size": 16,
        "n_embd": 768,
        "n_layer": 12,
        "n_head": 12,
        "mlp_ratio": 4 * 2 / 3,
        "activation": "swiglu",
        "dropout": 0.0,
        "attention_dropout": 0.0,
        "eps": 1e-6,
        "init_std": 0.02,
        "learned_pos_embedding": False,
        "use_rotary_pos_emb": True,
        # to use 256, change ref_feat_shape to (16, 16)
        "image_size": 224,
        "ref_feat_shape": (14, 14),
        # IDK why they use both rotary and pos_embed
        "use_pos_embed": True,
        "no_last_ln": True,
        # "global_pool": "avg",
        "norm_mlp": True,
    }


TIMM_NAME_TO_PARAMS = {
    "vit_base_patch16_224": get_vit_base_patch16_224_params,
    "vit_base_patch14_reg4_dinov2": get_vit_base_patch14_reg4_dinov2_params,
    "vit_base_patch16_rope_reg1_gap_256": get_vit_base_patch16_rope_reg1_gap_256_params,
    "vit_eva02_base_patch16_224": get_eva_base_patch16_224_params,
}


def normalize_name(timm_name: str) -> str:
    name = timm_name.replace("timm/", "").replace("hf-hub:", "").replace("nomic-ai/", "")
    base, model_type = name.split(".")
    return base, model_type


def timm_name_to_vit_config(timm_name: str) -> GPT2Config:
    normalized_name, _ = normalize_name(timm_name)

    vit_config = TIMM_NAME_TO_PARAMS[normalized_name]()
    return GPT2Config(
        n_embd=vit_config["n_embd"],
        n_layer=vit_config["n_layer"],
        n_head=vit_config["n_head"],
        n_inner=vit_config.get("intermediate_size", vit_config["mlp_ratio"] * vit_config["n_embd"]),
        activation_function=vit_config["activation"],
        vocab_size=0,  # no vocab since using patches
        n_positions=0,  # No absolute position embedding
        resid_pdrop=0.0,  # No dropout
        embd_pdrop=getattr(vit_config, "dropout", 0.0),
        attn_pdrop=vit_config.get("attention_dropout", 0.0),
        layer_norm_epsilon=vit_config.get("eps", 1e-6),
        initializer_range=vit_config.get("init_std", 0.02),
        bos_token_id=None,
        eos_token_id=None,
        # These are new arguments not in the original GPT2Config
        drop_path_rate=0.0,
        # Why is there double layer norm??
        prepre_layernom=vit_config.get("prenorm", False),
        layer_scale=vit_config.get("layer_scale", False),
        layer_scale_init=vit_config.get("layer_scale_init", 1.0),
        img_size=vit_config["image_size"],
        patch_size=vit_config["patch_size"],
        num_channels=vit_config.get("num_channels", 3),
        prenorm=True,
        parallel_block=False,
        parallel_block_tied_norm=False,
        rotary_emb_fraction=vit_config.get("rotary_emb_fraction", 0),
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
        mask_token=vit_config.get("mask_token", False),
        learned_pos_embedding=vit_config.get("learned_pos_embedding", False),
        patch_dropout=0,
        sinusoidal_pos_embedding=vit_config.get("model_type", "vit") == "mae",
        register_tokens=vit_config.get("register_tokens", 0),
        no_cls_token=vit_config.get("no_cls_token", False),
        no_embed_class=vit_config.get("no_embed_class", False),
        use_rotary_pos_emb=vit_config.get("use_rotary_pos_emb", False),
        ref_feat_shape=vit_config.get("ref_feat_shape", None),
        use_pos_embed=vit_config.get("use_pos_embed", True),
        eva_qkv_bias=vit_config.get("eva_qkv_bias", False),
        no_last_ln=vit_config.get("no_last_ln", False),
        norm_mlp=vit_config.get("norm_mlp", False),
        global_pool=vit_config.get("global_pool", None),
    )


def remap_timm_state_dict(state_dict, config):
    def key_mapping_embeddings(key):
        key = re.sub(r"^cls_token", r"embeddings.cls_token", key)
        key = re.sub(r"^pos_embed", r"embeddings.pos_embed", key)
        key = re.sub(r"^reg_token", r"embeddings.reg_token", key)
        key = re.sub(r"^patch_embed.proj.weight", r"embeddings.proj.weight", key)
        key = re.sub(r"^patch_embed.proj.bias", r"embeddings.proj.bias", key)
        return key

    state_dict = OrderedDict((key_mapping_embeddings(k), v) for k, v in state_dict.items())

    def key_mapping_layers(key):
        key = re.sub(r"^blocks\.(\d+)\.norm1", r"layers.\1.norm1", key)
        key = re.sub(r"^blocks\.(\d+)\.norm2", r"layers.\1.norm2", key)
        key = re.sub(r"^blocks\.(\d+)\.attn\.qkv\.weight", r"layers.\1.attn.Wqkv.weight", key)
        key = re.sub(r"^blocks\.(\d+)\.attn\.qkv\.bias", r"layers.\1.attn.Wqkv.bias", key)
        key = re.sub(r"^blocks\.(\d+)\.attn\.proj\.weight", r"layers.\1.attn.out_proj.weight", key)
        key = re.sub(r"^blocks\.(\d+)\.attn\.q_proj\.bias", r"blocks.\1.attn.q_bias", key)
        key = re.sub(r"^blocks\.(\d+)\.attn\.k_proj\.bias", r"blocks.\1.attn.k_bias", key)
        key = re.sub(r"^blocks\.(\d+)\.attn\.v_proj\.bias", r"blocks.\1.attn.v_bias", key)
        key = re.sub(r"^blocks\.(\d+)\.attn\.proj\.bias", r"layers.\1.attn.out_proj.bias", key)
        key = re.sub(r"^blocks\.(\d+)\.mlp\.fc1\.weight", r"layers.\1.mlp.fc1.weight", key)
        key = re.sub(r"^blocks\.(\d+)\.mlp\.fc1\.bias", r"layers.\1.mlp.fc1.bias", key)
        key = re.sub(r"^blocks\.(\d+)\.mlp\.fc1_x\.weight", r"layers.\1.mlp.fc11.weight", key)
        key = re.sub(r"^blocks\.(\d+)\.mlp\.fc1_x\.bias", r"layers.\1.mlp.fc11.bias", key)
        key = re.sub(r"^blocks\.(\d+)\.mlp\.fc1_g\.weight", r"layers.\1.mlp.fc12.weight", key)
        key = re.sub(r"^blocks\.(\d+)\.mlp\.fc1_g\.bias", r"layers.\1.mlp.fc12.bias", key)
        key = re.sub(r"^blocks\.(\d+)\.mlp\.fc2\.weight", r"layers.\1.mlp.fc2.weight", key)
        key = re.sub(r"^blocks\.(\d+)\.mlp\.fc2\.bias", r"layers.\1.mlp.fc2.bias", key)
        key = re.sub(r"^blocks\.(\d+)\.mlp\.norm\.weight", r"layers.\1.mlp.norm.weight", key)
        key = re.sub(r"^blocks\.(\d+)\.mlp\.norm\.bias", r"layers.\1.mlp.norm.bias", key)
        key = re.sub(r"^blocks\.(\d+)\.ls1\.gamma", r"layers.\1.ls1", key)
        key = re.sub(r"^blocks\.(\d+)\.ls2\.gamma", r"layers.\1.ls2", key)
        key = re.sub(r"^blocks\.(\d+)\.gamma_1", r"layers.\1.ls1", key)
        key = re.sub(r"^blocks\.(\d+)\.gamma_2", r"layers.\1.ls2", key)

        # map fc_norm to ln_f
        # key = re.sub(r"^fc_norm.", "ln_f.", key)

        return key

    state_dict = OrderedDict((key_mapping_layers(k), v) for k, v in state_dict.items())

    concat_bias = getattr(config, "eva_qkv_bias", False) or "blocks.0.attn.q_bias" in state_dict
    zero_k_bias = "blocks.0.attn.k_bias" not in state_dict
    concat_weight = "blocks.0.attn.q_proj.weight" in state_dict
    if concat_bias or concat_weight:
        for layer in range(config.n_layer):
            if concat_bias:
                q_bias = state_dict.pop(f"blocks.{layer}.attn.q_bias")
                if zero_k_bias:
                    k_bias = nn.Parameter(torch.zeros_like(q_bias), requires_grad=False)
                else:
                    k_bias = state_dict.pop(f"blocks.{layer}.attn.k_bias")
                v_bias = state_dict.pop(f"blocks.{layer}.attn.v_bias")

                qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)
                state_dict[f"layers.{layer}.attn.Wqkv.bias"] = qkv_bias

            if concat_weight:
                q_proj_weight = state_dict.pop(f"blocks.{layer}.attn.q_proj.weight")
                k_proj_weight = state_dict.pop(f"blocks.{layer}.attn.k_proj.weight")
                v_proj_weight = state_dict.pop(f"blocks.{layer}.attn.v_proj.weight")

                qkv_proj_weight = torch.cat([q_proj_weight, k_proj_weight, v_proj_weight], dim=0)
                state_dict[f"layers.{layer}.attn.Wqkv.weight"] = qkv_proj_weight

    def key_mapping_norm(key):
        key = re.sub(r"^norm\.", r"ln_f.", key)
        return key

    state_dict = OrderedDict((key_mapping_norm(k), v) for k, v in state_dict.items())

    state_dict["embeddings.proj.weight"] = state_dict["embeddings.proj.weight"].reshape(
        config.n_embd, config.patch_size * config.patch_size * 3
    )

    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("head")}
    # ignore register token global pooling used for classification?
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("fc_norm")}

    return state_dict
