import re
from collections import OrderedDict

import pytest
import torch
from transformers import AutoConfig, BertConfig
from transformers.models.bert.modeling_bert import BertForPreTraining as BertForPreTrainingHF

from contrastors.models.encoder.bert import (
    bert_config_to_nomic_config,
    inv_remap_state_dict,
    nomic_config_to_bert_config,
    remap_bert_state_dict,
)
from contrastors.models.encoder.configuration_nomic_bert import NomicBertConfig
from contrastors.models.encoder.modeling_nomic_bert import NomicBertForPreTraining
from contrastors.models.huggingface.modeling_hf_nomic_bert import NomicBertForPreTraining as NomicBertForPreTrainingHF
from contrastors.models.model_utils import state_dict_from_pretrained


def get_hf_models(model_name, config, dtype):
    pretrained_state_dict = state_dict_from_pretrained(model_name)

    def key_mapping_ln_gamma_beta(key):
        key = re.sub(r"LayerNorm.gamma$", "LayerNorm.weight", key)
        key = re.sub(r"LayerNorm.beta$", "LayerNorm.bias", key)
        return key

    pretrained_state_dict = OrderedDict((key_mapping_ln_gamma_beta(k), v) for k, v in pretrained_state_dict.items())
    model_hf = BertForPreTrainingHF(config)
    # Missing key(s) in state_dict: "bert.embeddings.position_ids", "cls.predictions.decoder.bias"
    # position_ids is a buffer, and predictions.decoder.bias is tied to predictions.bias.
    model_hf.load_state_dict(pretrained_state_dict, strict=False)
    model_hf.cuda().to(dtype=dtype)
    return model_hf


@pytest.mark.parametrize("model_name", ["bert-base-uncased"])
def test_flash_bert(model_name):
    dtype = torch.bfloat16
    hf_config = BertConfig.from_pretrained(model_name)
    config = bert_config_to_nomic_config(hf_config)
    config.add_pooling_layer = True

    model = NomicBertForPreTraining.from_pretrained(model_name, config)
    model = model.cuda().to(dtype=dtype)

    model_ref = get_hf_models(model_name, hf_config, torch.float32)
    model_hf = get_hf_models(model_name, hf_config, dtype)

    model.eval()
    model_ref.eval()
    model_hf.eval()

    torch.manual_seed(0)
    batch_size = 4
    max_seqlen = 512
    seqlens = torch.randint(max_seqlen // 2, max_seqlen + 1, (batch_size,), device="cuda")
    attention_mask = torch.arange(max_seqlen, device="cuda")[None, :] < seqlens[:, None]
    input_ids = torch.randint(0, config.vocab_size, (batch_size, max_seqlen), dtype=torch.long, device="cuda")
    out = model.bert(input_ids, attention_mask=attention_mask)
    sequence_output, pooled_output = out.last_hidden_state, out.pooler_output
    out_hf = model_hf.bert(input_ids, attention_mask=attention_mask)
    sequence_output_hf, pooled_output_hf = out_hf.last_hidden_state, out_hf.pooler_output
    sequence_output_hf[~attention_mask, :] = 0.0
    out_ref = model_ref.bert(input_ids, attention_mask=attention_mask)
    sequence_output_ref, pooled_output_ref = out_ref.last_hidden_state, out_ref.pooler_output
    sequence_output_ref[~attention_mask, :] = 0.0

    print(f"Output max diff: {(sequence_output - sequence_output_ref).abs().max().item()}")
    print(f"Output mean diff: {(sequence_output - sequence_output_ref).abs().mean().item()}")
    print(f"HF fp16 max diff: {(sequence_output_hf - sequence_output_ref).abs().max().item()}")
    print(f"HF fp16 mean diff: {(sequence_output_hf - sequence_output_ref).abs().mean().item()}")
    assert (sequence_output - sequence_output_ref).abs().max().item() < 3 * (
        sequence_output_hf - sequence_output_ref
    ).abs().max().item()
    assert (pooled_output - pooled_output_ref).abs().max().item() < 3 * (
        pooled_output_hf - pooled_output_ref
    ).abs().max().item()


@pytest.mark.parametrize("model_name", ["bert-base-uncased"])
def test_bert_convert_to_hf(model_name):
    dtype = torch.bfloat16
    hf_config = BertConfig.from_pretrained(model_name)
    config = bert_config_to_nomic_config(hf_config)
    config.add_pooling_layer = True

    model = NomicBertForPreTraining.from_pretrained(model_name, config)
    model = model.cuda().to(dtype=dtype)
    model.eval()

    hf_model = BertForPreTrainingHF(hf_config).to(dtype=dtype)
    remapped_weights = inv_remap_state_dict(model.state_dict(), config)
    result = hf_model.load_state_dict(remapped_weights, strict=False)
    assert result.missing_keys == ['cls.seq_relationship.weight', 'cls.seq_relationship.bias'], result.missing_keys

    hf_model = hf_model.cuda().to(dtype=dtype)
    hf_model.eval()

    model_ref = get_hf_models(model_name, hf_config, torch.float32)
    model_ref.eval()

    torch.manual_seed(0)
    batch_size = 4
    max_seqlen = 512
    seqlens = torch.randint(max_seqlen // 2, max_seqlen + 1, (batch_size,), device="cuda")
    attention_mask = torch.arange(max_seqlen, device="cuda")[None, :] < seqlens[:, None]
    input_ids = torch.randint(0, config.vocab_size, (batch_size, max_seqlen), dtype=torch.long, device="cuda")
    out = model.bert(input_ids, attention_mask=attention_mask)
    sequence_output, pooled_output = out.last_hidden_state, out.pooler_output
    out_hf = hf_model.bert(input_ids, attention_mask=attention_mask)
    sequence_output_hf, pooled_output_hf = out_hf.last_hidden_state, out_hf.pooler_output
    sequence_output_hf[~attention_mask, :] = 0.0
    out_ref = model_ref.bert(input_ids, attention_mask=attention_mask)
    sequence_output_ref, pooled_output_ref = out_ref.last_hidden_state, out_ref.pooler_output
    sequence_output_ref[~attention_mask, :] = 0.0

    print(f"Output max diff: {(sequence_output - sequence_output_ref).abs().max().item()}")
    print(f"Output mean diff: {(sequence_output - sequence_output_ref).abs().mean().item()}")
    print(f"HF fp16 max diff: {(sequence_output_hf - sequence_output_ref).abs().max().item()}")
    print(f"HF fp16 mean diff: {(sequence_output_hf - sequence_output_ref).abs().mean().item()}")
    assert (sequence_output - sequence_output_ref).abs().max().item() < 3 * (
        sequence_output_hf - sequence_output_ref
    ).abs().max().item()
    assert (pooled_output - pooled_output_ref).abs().max().item() < 3 * (
        pooled_output_hf - pooled_output_ref
    ).abs().max().item()


@pytest.mark.parametrize("model_name", ["bert-base-uncased"])
def test_inv_remap_state_dict(model_name: str):
    """
    Verify that we can convert a HF BERT model to flash_attn and back.
    """

    state_dict = BertForPreTrainingHF.from_pretrained(model_name).state_dict()
    config = BertConfig.from_pretrained(model_name)

    flash_state_dict = remap_bert_state_dict(state_dict, config, add_pooling_layer=True)
    recovered_state_dict = inv_remap_state_dict(flash_state_dict, config)

    assert (set(state_dict.keys()) - set(['cls.seq_relationship.bias', 'cls.seq_relationship.weight'])) == set(
        recovered_state_dict.keys()
    )

    for k in state_dict.keys():
        if k in ['cls.seq_relationship.bias', 'cls.seq_relationship.weight']:
            continue
        assert state_dict[k].shape == recovered_state_dict[k].shape
        torch.testing.assert_close(state_dict[k], recovered_state_dict[k], rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("model_name", ["bert-base-uncased"])
def test_nomic_bert_hf_comparison(model_name):
    dtype = torch.float32
    hf_config = BertConfig.from_pretrained(model_name)

    config = bert_config_to_nomic_config(hf_config)
    config.add_pooling_layer = True

    model = NomicBertForPreTrainingHF.from_pretrained(model_name, config)

    model = model.cuda().to(dtype=dtype)

    model_ref = get_hf_models(model_name, hf_config, dtype)

    model.eval()
    model_ref.eval()

    torch.manual_seed(0)
    batch_size = 4
    max_seqlen = 512
    seqlens = torch.randint(max_seqlen // 2, max_seqlen + 1, (batch_size,), device="cuda")
    attention_mask = torch.arange(max_seqlen, device="cuda")[None, :] < seqlens[:, None]
    input_ids = torch.randint(0, hf_config.vocab_size, (batch_size, max_seqlen), dtype=torch.long, device="cuda")
    out = model.bert(input_ids, attention_mask=attention_mask)
    sequence_output, pooled_output = out.last_hidden_state, out.pooler_output

    out_ref = model_ref.bert(input_ids, attention_mask=attention_mask)
    sequence_output_hf, pooled_output_hf = out_ref.last_hidden_state, out_ref.pooler_output

    assert torch.allclose(sequence_output, sequence_output_hf, atol=1e-6, rtol=1e-6)
    assert torch.allclose(pooled_output, pooled_output_hf, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("model_name", ["nomic-ai/nomic-bert-2048"])
def test_nomic_bert_hf(model_name):
    dtype = torch.bfloat16
    hf_config = NomicBertConfig.from_pretrained(model_name)
    config = nomic_config_to_bert_config(hf_config)
    config.add_pooling_layer = True

    model = NomicBertForPreTraining.from_pretrained(model_name)
    model = model.cuda().to(dtype=dtype)

    model_hf = NomicBertForPreTrainingHF.from_pretrained(model_name)
    model_hf = model_hf.cuda().to(dtype=dtype)

    model_hf_full_precision = NomicBertForPreTrainingHF.from_pretrained(model_name)
    model_hf_full_precision = model_hf_full_precision.cuda().to(dtype=torch.float32)

    model.eval()
    model_hf.eval()
    model_hf_full_precision.eval()

    for key in model.state_dict().keys():
        t1 = model.state_dict()[key]
        t2 = model_hf.state_dict()[key]
        assert t1.shape == t2.shape, f"{key}: {t1.shape} != {t2.shape}"
        assert torch.allclose(t1, t2, atol=1e-6, rtol=1e-6), f"{key}: {t1} != {t2}"

    torch.manual_seed(0)
    batch_size = 4
    max_seqlen = 512
    seqlens = torch.randint(max_seqlen // 2, max_seqlen + 1, (batch_size,), device="cuda")
    attention_mask = torch.arange(max_seqlen, device="cuda")[None, :] < seqlens[:, None]
    input_ids = torch.randint(0, config.vocab_size, (batch_size, max_seqlen), dtype=torch.long, device="cuda")
    out = model.bert(input_ids, attention_mask=attention_mask)
    sequence_output, _ = out.last_hidden_state, out.pooler_output

    out_ref = model_hf.bert(input_ids, attention_mask=attention_mask)
    sequence_output_hf, _ = out_ref.last_hidden_state, out_ref.pooler_output
    sequence_output_hf[~attention_mask, :] = 0.0

    out_hf = model_hf_full_precision.bert(input_ids, attention_mask=attention_mask)
    sequence_output_ref, _ = out_hf.last_hidden_state, out_hf.pooler_output
    sequence_output_ref[~attention_mask, :] = 0.0

    print(f"Output max diff: {(sequence_output - sequence_output_ref).abs().max().item()}")
    print(f"Output mean diff: {(sequence_output - sequence_output_ref).abs().mean().item()}")
    print(f"HF bf16 max diff: {(sequence_output_hf - sequence_output_ref).abs().max().item()}")
    print(f"HF bf16 mean diff: {(sequence_output_hf - sequence_output_ref).abs().mean().item()}")
    assert (sequence_output - sequence_output_ref).abs().max().item() < 3 * (
        sequence_output_hf - sequence_output_ref
    ).abs().max().item()
