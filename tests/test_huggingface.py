import pytest
import torch

from contrastors.models.encoder.bert import nomic_config_to_bert_config
from contrastors.models.encoder.configuration_nomic_bert import NomicBertConfig
from contrastors.models.encoder.modeling_nomic_bert import NomicBertForPreTraining
from contrastors.models.huggingface.modeling_hf_nomic_bert import NomicBertForPreTraining as NomicBertForPreTrainingHF


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

    print(f"Max diff between contrastors and hf ref: {(sequence_output - sequence_output_ref).abs().max().item()}")
    print(f"Mean diff between contrastors and hf ref: {(sequence_output - sequence_output_ref).abs().mean().item()}")
    print(f"Max diff between hf and hf bf16: {(sequence_output_hf - sequence_output_ref).abs().max().item()}")
    print(f"Mean diff between hf and hf bf16: {(sequence_output_hf - sequence_output_ref).abs().mean().item()}")
    assert (sequence_output - sequence_output_ref).abs().max().item() < 3 * (
        sequence_output_hf - sequence_output_ref
    ).abs().max().item()
    assert (sequence_output - sequence_output_ref).abs().mean().item() < 3 * (
        sequence_output_hf - sequence_output_ref
    ).abs().mean().item()   
