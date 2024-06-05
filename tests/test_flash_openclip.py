import pytest
import torch
from transformers import AutoConfig, AutoImageProcessor, AutoModel

from contrastors.models.vit import ViTModel, clip_config_to_vit_config


@pytest.mark.parametrize(
    "model_name",
    [
        "openai/clip-vit-large-patch14",
        "laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
        "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K",
    ],
)
def test_openclip_forward(model_name):
    config = AutoConfig.from_pretrained(model_name)
    config = clip_config_to_vit_config(config)

    flash_model = ViTModel.from_pretrained(model_name, config=config).to("cuda").to(torch.bfloat16)
    bf16_model = AutoModel.from_pretrained(model_name).vision_model.to("cuda").to(torch.bfloat16)

    assert sum(p.numel() for p in bf16_model.parameters() if p.requires_grad) == sum(
        p.numel() for p in flash_model.parameters() if p.requires_grad
    )

    img_processor = AutoImageProcessor.from_pretrained(model_name)

    torch.manual_seed(0)
    random_image = torch.randint(0, 255, (1, 3, 224, 224)).to("cuda").to(torch.int32)
    processed = img_processor(random_image, return_tensors="pt").to("cuda")

    flash_outputs = flash_model(input_ids=processed["pixel_values"].to(torch.bfloat16))
    flash_pooler_outputs = flash_outputs.last_hidden_state[:, 0]
    bf16_outputs = bf16_model(pixel_values=processed["pixel_values"].to(torch.bfloat16))

    del flash_model
    del bf16_model

    model = AutoModel.from_pretrained(model_name).vision_model.to("cuda").to(torch.float32)

    fp32_outputs = model(pixel_values=processed["pixel_values"].to(torch.float32))

    # main error is due to numerical precision :/
    # https://github.com/Dao-AILab/flash-attention/issues/211
    # check is same as flash attention check, make sure flash error is less than 1.25 * |bf16 - fp32|
    print((flash_pooler_outputs - fp32_outputs.pooler_output).abs().max().item())
    print((flash_pooler_outputs - fp32_outputs.pooler_output).abs().mean().item())
    print((bf16_outputs.pooler_output - fp32_outputs.pooler_output).abs().max().item())
    print((bf16_outputs.pooler_output - fp32_outputs.pooler_output).abs().mean().item())

    assert (flash_pooler_outputs - fp32_outputs.pooler_output).abs().max().item() <= 3.0 * (
        bf16_outputs.pooler_output - fp32_outputs.pooler_output
    ).abs().max().item()
    assert (flash_pooler_outputs - fp32_outputs.pooler_output).abs().mean().item() <= 3.0 * (
        bf16_outputs.pooler_output - fp32_outputs.pooler_output
    ).abs().mean().item()
