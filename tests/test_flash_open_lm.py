import torch
from transformers import AutoConfig, AutoModelForCausalLM
from contrastors.models.decoder import DecoderModel
from contrastors.models.decoder.open_lm import open_lm_config_to_gpt2_config


def test_openlm_forward():
    config = AutoConfig.from_pretrained("nomic-ai/open_lm_1B", trust_remote_code=True)
    config = open_lm_config_to_gpt2_config(config)

    flash_model = (
        DecoderModel.from_pretrained("nomic-ai/open_lm_1B", config=config, safe_serialization=True)
        .to("cuda")
        .to(torch.bfloat16)
    )

    # inputs = {"input_ids": torch.randint(0, config.vocab_size, (4, 1024)).to("cuda").long()}
    inputs = {"input_ids": torch.arange(0, 1024).unsqueeze(0).to("cuda").long()}
    inputs["attention_mask"] = torch.ones_like(inputs["input_ids"]).to("cuda").long()

    bf16_model = (
        AutoModelForCausalLM.from_pretrained("nomic-ai/open_lm_1B", trust_remote_code=True)
        .to("cuda")
        .to(torch.bfloat16)
    )

    assert sum(
        p.numel() for p in bf16_model.model.parameters() if p.requires_grad
    ) - bf16_model.model.output.weight.numel() == sum(p.numel() for p in flash_model.parameters() if p.requires_grad)
    flash_outputs = flash_model(**inputs)

    bf16_outputs = bf16_model(**inputs)
    bf16_hidden_states = bf16_outputs.hidden_states

    del flash_model
    del bf16_model

    model = (
        AutoModelForCausalLM.from_pretrained("nomic-ai/open_lm_1B", trust_remote_code=True).to("cuda").to(torch.float32)
    )

    fp32_outputs = model(**inputs)
    fp32_hidden_states = fp32_outputs.hidden_states

    # main error is due to numerical precision :/
    # https://github.com/Dao-AILab/flash-attention/issues/211
    # check is same as flash attention check, make sure flash error is less than 1.25 * |bf16 - fp32|
    print((flash_outputs.last_hidden_state - fp32_hidden_states).abs().max().item())
    print((flash_outputs.last_hidden_state - fp32_hidden_states).abs().mean().item())
    print((bf16_hidden_states - fp32_hidden_states).abs().max().item())
    print((bf16_hidden_states - fp32_hidden_states).abs().mean().item())

    assert (flash_outputs.last_hidden_state - fp32_hidden_states).abs().max().item() <= 1.25 * (
        bf16_hidden_states - fp32_hidden_states
    ).abs().max().item()
    assert (flash_outputs.last_hidden_state - fp32_hidden_states).abs().mean().item() <= 1.25 * (
        bf16_hidden_states - fp32_hidden_states
    ).abs().mean().item()
