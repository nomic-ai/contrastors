import pytest
import torch
from transformers import AutoConfig, AutoImageProcessor, AutoModel

from contrastors.models.clip import CLIPConfig, CLIPModel


@pytest.mark.parametrize("model_name", ["openai/clip-vit-large-patch14", "nomic-ai/cc12m-vit-b-32-epoch-1"])
def test_openclip_forward(model_name):
    config = AutoConfig.from_pretrained(model_name)
    flash_config = CLIPConfig(text_name=model_name, vision_name=model_name)

    flash_model = CLIPModel(flash_config).to("cuda").to(torch.bfloat16)
    bf16_model = AutoModel.from_pretrained(model_name).to("cuda").to(torch.bfloat16)

    # ignore logit scale
    assert sum(p.numel() for p in bf16_model.parameters() if p.requires_grad) - 1 == sum(
        p.numel() for p in flash_model.parameters() if p.requires_grad
    )

    img_processor = AutoImageProcessor.from_pretrained(model_name)

    torch.manual_seed(0)
    random_image = torch.randint(0, 255, (1, 3, 224, 224)).to("cuda").to(torch.int32)
    processed = img_processor(random_image, return_tensors="pt").to("cuda")

    inputs = {"input_ids": torch.randint(0, config.text_config.vocab_size, (4, 77)).to("cuda").long()}
    inputs["input_ids"][:, -1] = config.text_config.eos_token_id
    inputs["attention_mask"] = torch.ones_like(inputs["input_ids"]).to("cuda").long()

    inputs = {
        "vision_inputs": {"input_ids": processed["pixel_values"]},
        "text_inputs": inputs,
    }

    flash_outputs = flash_model(**inputs)
    text_emb, vision_emb = (
        flash_outputs["text_embedding"],
        flash_outputs["vision_embedding"],
    )
    bf16_outputs = bf16_model(pixel_values=processed["pixel_values"].to(torch.bfloat16), **inputs["text_inputs"])
    bf16_text_emb, bf16_vision_emb = (
        bf16_outputs["text_embeds"],
        bf16_outputs["image_embeds"],
    )

    del flash_model
    del bf16_model

    model = AutoModel.from_pretrained(model_name).to("cuda").to(torch.float32)

    fp32_outputs = model(pixel_values=processed["pixel_values"].to(torch.float32), **inputs["text_inputs"])
    fp32_text_emb, fp32_vision_emb = (
        fp32_outputs["text_embeds"],
        fp32_outputs["image_embeds"],
    )

    # main error is due to numerical precision :/
    # https://github.com/Dao-AILab/flash-attention/issues/211
    # check is same as flash attention check, make sure flash error is less than 1.25 * |bf16 - fp32|
    print((text_emb - fp32_text_emb).abs().max().item())
    print((text_emb - fp32_text_emb).abs().mean().item())
    print((bf16_text_emb - fp32_text_emb).abs().max().item())
    print((bf16_text_emb - fp32_text_emb).abs().mean().item())

    print((vision_emb - fp32_vision_emb).abs().max().item())
    print((vision_emb - fp32_vision_emb).abs().mean().item())
    print((bf16_vision_emb - fp32_vision_emb).abs().max().item())
    print((bf16_vision_emb - fp32_vision_emb).abs().mean().item())

    assert (text_emb - fp32_text_emb).abs().max().item() <= 3.0 * (bf16_text_emb - fp32_text_emb).abs().max().item()
    assert (text_emb - fp32_text_emb).abs().mean().item() <= 3.0 * (bf16_text_emb - fp32_text_emb).abs().mean().item()
    assert (vision_emb - fp32_vision_emb).abs().max().item() <= 3.0 * (
        bf16_vision_emb - fp32_vision_emb
    ).abs().max().item()
    assert (vision_emb - fp32_vision_emb).abs().mean().item() <= 3.0 * (
        bf16_vision_emb - fp32_vision_emb
    ).abs().mean().item()
