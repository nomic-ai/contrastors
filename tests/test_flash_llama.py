import torch
from transformers import AutoConfig, AutoModel
from contrastors.models.decoder import DecoderModel
from contrastors.models.decoder.llama import llama_config_to_gpt2_config


def test_llama_forward():
    torch.manual_seed(0)

    config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-1B")
    config = llama_config_to_gpt2_config(config)

    flash_model = DecoderModel.from_pretrained("meta-llama/Llama-3.2-1B", config=config).to("cuda").to(torch.bfloat16)

    inputs = {"input_ids": torch.randint(0, config.vocab_size, (4, 1024)).to("cuda").long()}
    inputs["attention_mask"] = torch.ones_like(inputs["input_ids"]).to("cuda").long()

    bf16_model = AutoModel.from_pretrained("meta-llama/Llama-3.2-1B").to("cuda").to(torch.bfloat16)

    num_bf16_params = sum(p.numel() for p in bf16_model.parameters() if p.requires_grad)
    num_flash_params = sum(p.numel() for p in flash_model.parameters() if p.requires_grad)

    assert num_bf16_params == num_flash_params, f"{num_bf16_params=:,} != {num_flash_params=:,}"

    flash_outputs = flash_model(**inputs)

    bf16_outputs = bf16_model(**inputs)

    del flash_model
    del bf16_model

    model = AutoModel.from_pretrained("meta-llama/Llama-3.2-1B").to("cuda").to(torch.float32)

    fp32_outputs = model(**inputs)

    # main error is due to numerical precision :/
    # https://github.com/Dao-AILab/flash-attention/issues/211
    # check is same as flash attention check, make sure flash error is less than 1.25 * |bf16 - fp32|
    print(f"{(flash_outputs.last_hidden_state - fp32_outputs.last_hidden_state).abs().max().item()=}")
    print(f"{(flash_outputs.last_hidden_state - fp32_outputs.last_hidden_state).abs().mean().item()=}")
    print(f"{(bf16_outputs.last_hidden_state - fp32_outputs.last_hidden_state).abs().max().item()=}")
    print(f"{(bf16_outputs.last_hidden_state - fp32_outputs.last_hidden_state).abs().mean().item()=}")

    assert (flash_outputs.last_hidden_state - fp32_outputs.last_hidden_state).abs().max().item() <= 1.25 * (
        bf16_outputs.last_hidden_state - fp32_outputs.last_hidden_state
    ).abs().max().item()
    assert (flash_outputs.last_hidden_state - fp32_outputs.last_hidden_state).abs().mean().item() <= 1.25 * (
        bf16_outputs.last_hidden_state - fp32_outputs.last_hidden_state
    ).abs().mean().item()
