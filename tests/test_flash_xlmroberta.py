import re
from collections import OrderedDict

import pytest
import torch
from transformers import XLMRobertaConfig
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaModel as XLMRobertaModelHF

from contrastors.models.encoder.bert import bert_config_to_nomic_config
from contrastors.models.encoder.modeling_nomic_bert import NomicBertModel
from contrastors.models.model_utils import state_dict_from_pretrained


def get_hf_models(model_name, config, dtype):
    pretrained_state_dict = state_dict_from_pretrained(model_name)

    def key_mapping_ln_gamma_beta(key):
        key = re.sub(r"LayerNorm.gamma$", "LayerNorm.weight", key)
        key = re.sub(r"LayerNorm.beta$", "LayerNorm.bias", key)
        return key

    pretrained_state_dict = OrderedDict((key_mapping_ln_gamma_beta(k), v) for k, v in pretrained_state_dict.items())

    def remove_bert_prefix(key):
        key = re.sub(r"^roberta\.", "", key)
        return key

    pretrained_state_dict = OrderedDict((remove_bert_prefix(k), v) for k, v in pretrained_state_dict.items())

    model_hf = XLMRobertaModelHF(config, add_pooling_layer=True)
    # Missing key(s) in state_dict: "bert.embeddings.position_ids", "cls.predictions.decoder.bias"
    # position_ids is a buffer, and predictions.decoder.bias is tied to predictions.bias.
    model_hf.load_state_dict(pretrained_state_dict, strict=False)
    model_hf.cuda().to(dtype=dtype)
    return model_hf


@pytest.mark.parametrize("model_name", ["FacebookAI/xlm-roberta-base"])
def test_flash_bert(model_name):
    dtype = torch.bfloat16
    hf_config = XLMRobertaConfig.from_pretrained(model_name)
    config = bert_config_to_nomic_config(hf_config)
    config.add_pooling_layer = True

    model = NomicBertModel.from_pretrained(model_name, config, add_pooling_layer=True)
    print(model)
    model = model.cuda().to(dtype=dtype)

    model_ref = get_hf_models(model_name, hf_config, torch.float32)
    print(model_ref)
    model_hf = get_hf_models(model_name, hf_config, dtype)

    model.eval()
    model_ref.eval()
    model_hf.eval()
    assert sum(p.numel() for p in model.parameters()) == sum(
        p.numel() for p in model_ref.parameters()
    ), f"{sum(p.numel() for p in model.parameters())} != {sum(p.numel() for p in model_ref.parameters())}"

    torch.manual_seed(0)
    batch_size = 4
    max_seqlen = 512
    seqlens = torch.randint(max_seqlen // 2, max_seqlen + 1, (batch_size,), device="cuda")
    attention_mask = torch.arange(max_seqlen, device="cuda")[None, :] < seqlens[:, None]
    input_ids = torch.randint(0, config.vocab_size, (batch_size, max_seqlen), dtype=torch.long, device="cuda")
    # xlm-roberta offsets with position ids from pad_idx + 1
    position_ids = torch.arange(2, max_seqlen + 2, device="cuda")[None, :].expand(batch_size, -1)
    out = model(input_ids, attention_mask=attention_mask, position_ids=position_ids)
    sequence_output, pooled_output = out.last_hidden_state, out.pooler_output
    out_hf = model_hf(input_ids, attention_mask=attention_mask, position_ids=position_ids)
    sequence_output_hf, pooled_output_hf = (
        out_hf.last_hidden_state,
        out_hf.pooler_output,
    )
    sequence_output_hf[~attention_mask, :] = 0.0
    out_ref = model_ref(input_ids, attention_mask=attention_mask, position_ids=position_ids)
    sequence_output_ref, pooled_output_ref = (
        out_ref.last_hidden_state,
        out_ref.pooler_output,
    )
    sequence_output_ref[~attention_mask, :] = 0.0

    print(f"Max diff between flash and hf fp32: {(sequence_output - sequence_output_ref).abs().max().item()}")
    print(f"Mean diff between flash and hf {dtype}: {(sequence_output - sequence_output_ref).abs().mean().item()}")
    print(f"Max diff between HF fp32 and {dtype}: {(sequence_output_hf - sequence_output_ref).abs().max().item()}")
    print(f"Mean diff between HF fp32 and {dtype}: {(sequence_output_hf - sequence_output_ref).abs().mean().item()}")



    assert (sequence_output - sequence_output_ref).abs().max().item() < 3 * (
        sequence_output_hf - sequence_output_ref
    ).abs().max().item()
    assert (sequence_output - sequence_output_ref).abs().mean().item() < 3 * (
        sequence_output_hf - sequence_output_ref
    ).abs().mean().item()

    assert (pooled_output - pooled_output_ref).abs().max().item() < 3 * (
        pooled_output_hf - pooled_output_ref
    ).abs().max().item()
    assert (pooled_output - pooled_output_ref).abs().mean().item() < 3 * (
        pooled_output_hf - pooled_output_ref
    ).abs().mean().item()
