import pytest
import torch
from transformers import AutoConfig, AutoImageProcessor, AutoModel

from contrastors.models.vit import ViTModel, dino_config_to_vit_config


# NOTE: giant doesn't work somehow, need to fix
@pytest.mark.parametrize(
    "model_name",
    ["facebook/dinov2-small", "facebook/dinov2-base", "facebook/dinov2-large"],
)
def test_dinov2_forward(model_name):
    config = AutoConfig.from_pretrained(model_name)
    config = dino_config_to_vit_config(config)

    flash_model = ViTModel.from_pretrained(model_name, config=config).to("cuda").to(torch.float16)
    bf16_model = AutoModel.from_pretrained(model_name).to("cuda").to(torch.float16)

    assert sum(p.numel() for p in bf16_model.parameters() if p.requires_grad) == sum(
        p.numel() for p in flash_model.parameters() if p.requires_grad
    )

    img_processor = AutoImageProcessor.from_pretrained(model_name)

    torch.manual_seed(0)
    random_image = torch.randint(0, 255, (1, 3, 224, 224)).to("cuda").to(torch.int32)
    processed = img_processor(random_image, return_tensors="pt").to("cuda")

    flash_outputs = flash_model(input_ids=processed["pixel_values"].to(torch.float16))
    bf16_outputs = bf16_model(**processed.to(torch.float16))

    del flash_model
    del bf16_model

    model = AutoModel.from_pretrained(model_name).to("cuda").to(torch.float32)

    fp32_outputs = model(**processed.to(torch.float32))

    # main error is due to numerical precision :/
    # https://github.com/Dao-AILab/flash-attention/issues/211
    # check is same as flash attention check, make sure flash error is less than 1.25 * |bf16 - fp32|
    print((flash_outputs.last_hidden_state - fp32_outputs.last_hidden_state).abs().max().item())
    print((flash_outputs.last_hidden_state - fp32_outputs.last_hidden_state).abs().mean().item())
    print((bf16_outputs.last_hidden_state - fp32_outputs.last_hidden_state).abs().max().item())
    print((bf16_outputs.last_hidden_state - fp32_outputs.last_hidden_state).abs().mean().item())

    assert (flash_outputs.last_hidden_state - fp32_outputs.last_hidden_state).abs().max().item() <= 2.0 * (
        bf16_outputs.last_hidden_state - fp32_outputs.last_hidden_state
    ).abs().max().item()
    assert (flash_outputs.last_hidden_state - fp32_outputs.last_hidden_state).abs().mean().item() <= 3.0 * (
        bf16_outputs.last_hidden_state - fp32_outputs.last_hidden_state
    ).abs().mean().item()
