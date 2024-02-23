import re
from collections import OrderedDict

import torch
import torch.nn.functional as F
from transformers import BertConfig, GPT2Config, PretrainedConfig

from .configuration_nomic_bert import NomicBertConfig


def bert_config_to_nomic_config(bert_config: BertConfig) -> NomicBertConfig:
    return NomicBertConfig(
        vocab_size=bert_config.vocab_size,
        n_positions=bert_config.max_position_embeddings,  # No absolute position embedding
        n_embd=bert_config.hidden_size,
        n_layer=bert_config.num_hidden_layers,
        n_head=bert_config.num_attention_heads,
        n_inner=bert_config.intermediate_size,
        activation_function=bert_config.hidden_act,
        resid_pdrop=bert_config.hidden_dropout_prob,
        embd_pdrop=bert_config.hidden_dropout_prob,
        attn_pdrop=bert_config.attention_probs_dropout_prob,
        layer_norm_epsilon=bert_config.layer_norm_eps,
        initializer_range=bert_config.initializer_range,
        bos_token_id=None,  # TODO: check this
        eos_token_id=None,
        # These are new arguments not in the original GPT2Config
        prenorm=False,
        parallel_block=False,
        parallel_block_tied_norm=False,
        rotary_emb_fraction=getattr(bert_config, "rotary_emb_fraction", 0),
        tie_word_embeddings=True,
        fused_dropout_add_ln=True,
        fused_bias_fc=True,
        use_flash_attn=True,
        use_xentropy=True,
        qkv_proj_bias=getattr(bert_config, "qkv_proj_bias", True),
        rotary_emb_base=getattr(bert_config, "rotary_emb_base", 1000),
        rotary_emb_scale_base=getattr(bert_config, "rotary_emb_scale_base", None),
        rotary_emb_interleaved=getattr(bert_config, "rotary_emb_interleaved", False),
        mlp_fc1_bias=getattr(bert_config, "mlp_fc1_bias", True),
        mlp_fc2_bias=getattr(bert_config, "mlp_fc2_bias", True),
        use_rms_norm=getattr(bert_config, "use_rms_norm", False),
        causal=False,
        type_vocab_size=bert_config.type_vocab_size,
        dense_seq_output=True,
        pad_vocab_size_multiple=getattr(bert_config, "pad_vocab_to_multiple_of", 1),
        rotary_scaling_factor=getattr(bert_config, "rotary_scaling_factor", None),
    )


def nomic_config_to_bert_config(gpt2_config: NomicBertConfig) -> BertConfig:
    return BertConfig(
        vocab_size=gpt2_config.vocab_size,
        hidden_size=gpt2_config.n_embd,
        num_hidden_layers=gpt2_config.n_layer,
        num_attention_heads=gpt2_config.n_head,
        intermediate_size=gpt2_config.n_inner,
        hidden_act=gpt2_config.activation_function,
        hidden_dropout_prob=gpt2_config.resid_pdrop,
        attention_probs_dropout_prob=gpt2_config.attn_pdrop,
        max_position_embeddings=gpt2_config.n_positions,
        type_vocab_size=gpt2_config.type_vocab_size,
        initializer_range=gpt2_config.initializer_range,
        layer_norm_eps=gpt2_config.layer_norm_epsilon,
        # The following attributes do not have a direct equivalent in GPT2Config
        # and are set to commonly used defaults for BertConfig
        pad_token_id=0,
        position_embedding_type="absolute",
        use_cache=True,
    )


def remap_bert_state_dict(
    state_dict,
    config: PretrainedConfig,
    remove_bert=False,
    remove_cls_weights=False,
    add_pooling_layer=False,
):
    """
    Map the state_dict of a Huggingface BERT model to be flash_attn compatible.
    """

    def add_bert_prefix(key):
        # prepend bert. to the key
        if key.startswith("bert.") or key.startswith("cls."):
            return key
        return f"bert.{key}"

    state_dict = OrderedDict((add_bert_prefix(k), v) for k, v in state_dict.items())

    # LayerNorm
    def key_mapping_ln_gamma_beta(key):
        key = re.sub(r"LayerNorm.gamma$", "LayerNorm.weight", key)
        key = re.sub(r"LayerNorm.beta$", "LayerNorm.bias", key)
        return key

    state_dict = OrderedDict((key_mapping_ln_gamma_beta(k), v) for k, v in state_dict.items())

    # Layers
    def key_mapping_layers(key):
        return re.sub(r"^bert.encoder.layer\.", "bert.encoder.layers.", key)

    state_dict = OrderedDict((key_mapping_layers(k), v) for k, v in state_dict.items())

    # LayerNorm
    def key_mapping_ln(key):
        key = re.sub(r"^bert.embeddings.LayerNorm.", "bert.emb_ln.", key)
        key = re.sub(
            r"^bert.encoder.layers.(\d+).attention.output.LayerNorm.(weight|bias)",
            r"bert.encoder.layers.\1.norm1.\2",
            key,
        )
        key = re.sub(
            r"^bert.encoder.layers.(\d+).output.LayerNorm.(weight|bias)",
            r"bert.encoder.layers.\1.norm2.\2",
            key,
        )
        key = re.sub(
            r"^cls.predictions.transform.LayerNorm.(weight|bias)",
            r"cls.predictions.transform.layer_norm.\1",
            key,
        )
        return key

    state_dict = OrderedDict((key_mapping_ln(k), v) for k, v in state_dict.items())

    # MLP
    def key_mapping_mlp(key):
        key = re.sub(
            r"^bert.encoder.layers.(\d+).intermediate.dense.(weight|bias)",
            r"bert.encoder.layers.\1.mlp.fc1.\2",
            key,
        )
        key = re.sub(
            r"^bert.encoder.layers.(\d+).output.dense.(weight|bias)",
            r"bert.encoder.layers.\1.mlp.fc2.\2",
            key,
        )
        return key

    state_dict = OrderedDict((key_mapping_mlp(k), v) for k, v in state_dict.items())

    # Attention
    last_layer_subset = getattr(config, "last_layer_subset", False)
    for d in range(config.num_hidden_layers):
        if f"bert.encoder.layers.{d}.attention.self.query.weight" not in state_dict:
            continue
        Wq = state_dict.pop(f"bert.encoder.layers.{d}.attention.self.query.weight")
        Wk = state_dict.pop(f"bert.encoder.layers.{d}.attention.self.key.weight")
        Wv = state_dict.pop(f"bert.encoder.layers.{d}.attention.self.value.weight")
        bq = state_dict.pop(f"bert.encoder.layers.{d}.attention.self.query.bias")
        bk = state_dict.pop(f"bert.encoder.layers.{d}.attention.self.key.bias")
        bv = state_dict.pop(f"bert.encoder.layers.{d}.attention.self.value.bias")
        if not (last_layer_subset and d == config.num_hidden_layers - 1):
            state_dict[f"bert.encoder.layers.{d}.attn.Wqkv.weight"] = torch.cat([Wq, Wk, Wv], dim=0)
            state_dict[f"bert.encoder.layers.{d}.attn.Wqkv.bias"] = torch.cat([bq, bk, bv], dim=0)
        else:
            state_dict[f"bert.encoder.layers.{d}.attn.Wq.weight"] = Wq
            state_dict[f"bert.encoder.layers.{d}.attn.Wkv.weight"] = torch.cat([Wk, Wv], dim=0)
            state_dict[f"bert.encoder.layers.{d}.attn.Wq.bias"] = bq
            state_dict[f"bert.encoder.layers.{d}.attn.Wkv.bias"] = torch.cat([bk, bv], dim=0)

    def key_mapping_attn(key):
        return re.sub(
            r"^bert.encoder.layers.(\d+).attention.output.dense.(weight|bias)",
            r"bert.encoder.layers.\1.attn.out_proj.\2",
            key,
        )

    state_dict = OrderedDict((key_mapping_attn(k), v) for k, v in state_dict.items())

    def key_mapping_decoder_bias(key):
        return re.sub(r"^cls.predictions.bias", "cls.predictions.decoder.bias", key)

    # remove nsp weights, we don't use
    state_dict.pop("cls.seq_relationship.weight", None)
    state_dict.pop("cls.seq_relationship.bias", None)
    state_dict.pop("bert.embeddings.position_ids", None)

    state_dict = OrderedDict((key_mapping_decoder_bias(k), v) for k, v in state_dict.items())

    if remove_cls_weights:
        cls_weights = [
            "cls.predictions.decoder.bias",
            "cls.predictions.transform.dense.weight",
            "cls.predictions.transform.dense.bias",
            "cls.predictions.transform.layer_norm.weight",
            "cls.predictions.transform.layer_norm.bias",
            "cls.predictions.decoder.weight",
        ]
        for weight in cls_weights:
            state_dict.pop(weight, None)

    # Word embedding
    pad_vocab_size_multiple = getattr(config, "pad_vocab_size_multiple", 1)
    if pad_vocab_size_multiple > 1:
        word_embeddings = state_dict["bert.embeddings.word_embeddings.weight"]
        state_dict["bert.embeddings.word_embeddings.weight"] = F.pad(
            word_embeddings, (0, 0, 0, config.vocab_size - word_embeddings.shape[0])
        )
        if not remove_cls_weights:
            decoder_weight = state_dict["cls.predictions.decoder.weight"]
            state_dict["cls.predictions.decoder.weight"] = F.pad(
                decoder_weight, (0, 0, 0, config.vocab_size - decoder_weight.shape[0])
            )
            # If the vocab was padded, we want to set the decoder bias for those padded indices to be
            # strongly negative (i.e. the decoder shouldn't predict those indices).
            # TD [2022-05-09]: I don't think it affects the MLPerf training.
            if "cls.predictions.decoder.bias" in state_dict:
                decoder_bias = state_dict["cls.predictions.decoder.bias"]
                state_dict["cls.predictions.decoder.bias"] = F.pad(
                    decoder_bias, (0, config.vocab_size - decoder_bias.shape[0]), value=-100.0
                )

    if add_pooling_layer is False:
        pooler_weights = [
            "bert.pooler.dense.weight",
            "bert.pooler.dense.bias",
        ]
        for key in pooler_weights:
            state_dict.pop(key, None)

    if remove_bert:

        def remove_bert_prefix(key):
            key = re.sub(r"^bert.", "", key)
            return key

        state_dict = OrderedDict((remove_bert_prefix(k), v) for k, v in state_dict.items())

    return state_dict


def inv_remap_state_dict(state_dict, config: PretrainedConfig):
    """
    Map the state_dict of a flash_attn model to be Huggingface BERT compatible.

    This function is meant to be the inverse of remap_state_dict.
    """
    if isinstance(config, GPT2Config):
        config = nomic_config_to_bert_config(config)
    # Word embedding
    pad_vocab_size_multiple = getattr(config, "pad_vocab_size_multiple", 1)
    if pad_vocab_size_multiple > 1:
        word_embeddings = state_dict["bert.embeddings.word_embeddings.weight"]
        decoder_weight = state_dict["cls.predictions.decoder.weight"]
        decoder_bias = state_dict["cls.predictions.decoder.bias"]
        # unpad embeddings
        state_dict["bert.embeddings.word_embeddings.weight"] = word_embeddings[: config.orig_vocab_size, :]
        state_dict["cls.predictions.decoder.weight"] = decoder_weight[: config.orig_vocab_size, :]
        state_dict["cls.predictions.decoder.bias"] = decoder_bias[: config.orig_vocab_size]

    for d in range(config.num_hidden_layers):
        last_layer_subset = getattr(config, "last_layer_subset", False)
        if not last_layer_subset or d != (config.num_hidden_layers - 1):
            Wqkv_weights = state_dict.pop(f"bert.encoder.layers.{d}.attn.Wqkv.weight")
            Wqkv_biases = state_dict.pop(f"bert.encoder.layers.{d}.attn.Wqkv.bias")
            state_dict[f"bert.encoder.layers.{d}.attention.self.query.weight"] = Wqkv_weights[
                : Wqkv_weights.shape[0] // 3, :
            ]
            state_dict[f"bert.encoder.layers.{d}.attention.self.key.weight"] = Wqkv_weights[
                Wqkv_weights.shape[0] // 3 : 2 * Wqkv_weights.shape[0] // 3, :
            ]
            state_dict[f"bert.encoder.layers.{d}.attention.self.value.weight"] = Wqkv_weights[
                2 * Wqkv_weights.shape[0] // 3 :, :
            ]
            state_dict[f"bert.encoder.layers.{d}.attention.self.query.bias"] = Wqkv_biases[: Wqkv_biases.shape[0] // 3]
            state_dict[f"bert.encoder.layers.{d}.attention.self.key.bias"] = Wqkv_biases[
                Wqkv_biases.shape[0] // 3 : 2 * Wqkv_biases.shape[0] // 3
            ]
            state_dict[f"bert.encoder.layers.{d}.attention.self.value.bias"] = Wqkv_biases[
                2 * Wqkv_biases.shape[0] // 3 :
            ]
        else:
            Wq_weight = state_dict.pop(f"bert.encoder.layers.{d}.attn.Wq.weight")
            Wkv_weights = state_dict.pop(f"bert.encoder.layers.{d}.attn.Wkv.weight")
            Wq_bias = state_dict.pop(f"bert.encoder.layers.{d}.attn.Wq.bias")
            Wkv_biases = state_dict.pop(f"bert.encoder.layers.{d}.attn.Wkv.bias")
            state_dict[f"bert.encoder.layers.{d}.attention.self.query.weight"] = Wq_weight
            state_dict[f"bert.encoder.layers.{d}.attention.self.key.weight"] = Wkv_weights[
                : Wkv_weights.shape[0] // 2, :
            ]
            state_dict[f"bert.encoder.layers.{d}.attention.self.value.weight"] = Wkv_weights[
                Wkv_weights.shape[0] // 2 :, :
            ]
            state_dict[f"bert.encoder.layers.{d}.attention.self.query.bias"] = Wq_bias
            state_dict[f"bert.encoder.layers.{d}.attention.self.key.bias"] = Wkv_biases[: Wkv_biases.shape[0] // 2]
            state_dict[f"bert.encoder.layers.{d}.attention.self.value.bias"] = Wkv_biases[Wkv_biases.shape[0] // 2 :]

    def inv_key_mapping_ln(key):
        key = re.sub(r"bert.emb_ln.", "bert.embeddings.LayerNorm.", key)
        key = re.sub(
            r"bert.encoder.layers.(\d+).norm1.(weight|bias)",
            r"bert.encoder.layers.\1.attention.output.LayerNorm.\2",
            key,
        )
        key = re.sub(
            r"bert.encoder.layers.(\d+).norm2.(weight|bias)",
            r"bert.encoder.layers.\1.output.LayerNorm.\2",
            key,
        )
        key = re.sub(
            r"cls.predictions.transform.layer_norm.(weight|bias)",
            r"cls.predictions.transform.LayerNorm.\1",
            key,
        )
        return key

    def inv_key_mapping_layers(key):
        return re.sub(r"bert.encoder.layers.", "bert.encoder.layer.", key)

    def inv_key_mapping_mlp(key):
        key = re.sub(
            r"bert.encoder.layer.(\d+).mlp.fc1.(weight|bias)",
            r"bert.encoder.layer.\1.intermediate.dense.\2",
            key,
        )
        key = re.sub(
            r"bert.encoder.layer.(\d+).mlp.fc2.(weight|bias)",
            r"bert.encoder.layer.\1.output.dense.\2",
            key,
        )
        return key

    def inv_key_mapping_attn(key):
        return re.sub(
            r"bert.encoder.layer.(\d+).attn.out_proj.(weight|bias)",
            r"bert.encoder.layer.\1.attention.output.dense.\2",
            key,
        )

    state_dict = OrderedDict((inv_key_mapping_ln(key), value) for key, value in state_dict.items())

    state_dict = OrderedDict((inv_key_mapping_layers(key), value) for key, value in state_dict.items())
    state_dict = OrderedDict((inv_key_mapping_mlp(key), value) for key, value in state_dict.items())
    state_dict = OrderedDict((inv_key_mapping_attn(key), value) for key, value in state_dict.items())

    state_dict["cls.predictions.bias"] = state_dict["cls.predictions.decoder.bias"]

    return state_dict
