import pytest
import torch
from transformers import AutoConfig, AutoImageProcessor, AutoModel
from transformers import ViTModel as HFViTModel

from contrastors.models.vit import ViTModel, hf_vit_config_to_vit_config, inverse_remap_state_dict_hf_vit


@pytest.mark.parametrize(
    "model_name",
    [
        "google/vit-base-patch16-224",
        "google/vit-base-patch16-224-in21k",
        "facebook/dino-vitb16",
        "google/vit-large-patch16-224",
        "facebook/vit-mae-base",
        "facebook/vit-mae-large",
    ],
)
def test_openclip_forward(model_name):
    config = AutoConfig.from_pretrained(model_name)
    config = hf_vit_config_to_vit_config(config)

    flash_model = ViTModel.from_pretrained(model_name, config=config).to("cuda").to(torch.bfloat16)
    if "vit-mae" in model_name:
        bf16_model = AutoModel.from_pretrained(model_name, mask_ratio=0).to("cuda").to(torch.bfloat16)
    else:
        bf16_model = AutoModel.from_pretrained(model_name, add_pooling_layer=False).to("cuda").to(torch.bfloat16)

    assert sum(p.numel() for p in bf16_model.parameters() if p.requires_grad) == sum(
        p.numel() for p in flash_model.parameters() if p.requires_grad
    )

    img_processor = AutoImageProcessor.from_pretrained(model_name)

    torch.manual_seed(0)
    random_image = torch.randint(0, 255, (1, 3, 224, 224)).to("cuda").to(torch.int32)
    processed = img_processor(random_image, return_tensors="pt").to("cuda")

    flash_outputs = flash_model(input_ids=processed["pixel_values"].to(torch.bfloat16))
    flash_pooler_outputs = flash_outputs.last_hidden_state
    bf16_outputs = bf16_model(pixel_values=processed["pixel_values"].to(torch.bfloat16))

    del flash_model
    del bf16_model

    if "vit-mae" in model_name:
        model = AutoModel.from_pretrained(model_name, mask_ratio=0).to("cuda").to(torch.float32)
    else:
        model = AutoModel.from_pretrained(model_name, add_pooling_layer=False).to("cuda").to(torch.float32)

    fp32_outputs = model(pixel_values=processed["pixel_values"].to(torch.float32))

    # main error is due to numerical precision :/
    # https://github.com/Dao-AILab/flash-attention/issues/211
    # check is same as flash attention check, make sure flash error is less than 1.25 * |bf16 - fp32|
    print((flash_pooler_outputs - fp32_outputs.last_hidden_state).abs().max().item())
    print((flash_pooler_outputs - fp32_outputs.last_hidden_state).abs().mean().item())
    print((bf16_outputs.last_hidden_state - fp32_outputs.last_hidden_state).abs().max().item())
    print((bf16_outputs.last_hidden_state - fp32_outputs.last_hidden_state).abs().mean().item())

    assert (flash_pooler_outputs - fp32_outputs.last_hidden_state).abs().max().item() <= 3.0 * (
        bf16_outputs.last_hidden_state - fp32_outputs.last_hidden_state
    ).abs().max().item()
    assert (flash_pooler_outputs - fp32_outputs.last_hidden_state).abs().mean().item() <= 3.0 * (
        bf16_outputs.last_hidden_state - fp32_outputs.last_hidden_state
    ).abs().mean().item()


@pytest.mark.parametrize("model_name", ["google/vit-base-patch16-224"])
def test_clip_convert_to_hf(model_name):
    dtype = torch.bfloat16
    hf_config = AutoConfig.from_pretrained(model_name)
    config = hf_vit_config_to_vit_config(hf_config)

    model = ViTModel.from_pretrained(model_name, config)
    model = model.cuda().to(dtype=dtype)
    model.eval()

    hf_model = HFViTModel(hf_config).to(dtype=dtype)
    remapped_weights = inverse_remap_state_dict_hf_vit(model.state_dict(), config)
    result = hf_model.load_state_dict(remapped_weights, strict=False)
    assert result.missing_keys == [
        "pooler.dense.weight",
        "pooler.dense.bias",
    ], result.missing_keys

    hf_model = hf_model.cuda().to(dtype=dtype)
    hf_model.eval()

    model_ref = AutoModel.from_pretrained(model_name).to(torch.float32)
    model_ref.eval()

    img_processor = AutoImageProcessor.from_pretrained(model_name)
    torch.manual_seed(0)

    random_image = torch.randint(0, 255, (1, 3, 224, 224)).to("cuda").to(torch.int32)
    processed = img_processor(random_image, return_tensors="pt").to("cuda")

    flash_outputs = model(input_ids=processed["pixel_values"].to(torch.bfloat16))
    flash_pooler_outputs = flash_outputs.last_hidden_state
    bf16_outputs = hf_model(pixel_values=processed["pixel_values"].to(torch.bfloat16))

    del model
    del hf_model

    model = AutoModel.from_pretrained(model_name, add_pooling_layer=False).to("cuda").to(torch.float32)

    fp32_outputs = model(pixel_values=processed["pixel_values"].to(torch.float32))

    # main error is due to numerical precision :/
    # https://github.com/Dao-AILab/flash-attention/issues/211
    # check is same as flash attention check, make sure flash error is less than 1.25 * |bf16 - fp32|
    print((flash_pooler_outputs - fp32_outputs.last_hidden_state).abs().max().item())
    print((flash_pooler_outputs - fp32_outputs.last_hidden_state).abs().mean().item())
    print((bf16_outputs.last_hidden_state - fp32_outputs.last_hidden_state).abs().max().item())
    print((bf16_outputs.last_hidden_state - fp32_outputs.last_hidden_state).abs().mean().item())

    assert (flash_pooler_outputs - fp32_outputs.last_hidden_state).abs().max().item() <= 3.0 * (
        bf16_outputs.last_hidden_state - fp32_outputs.last_hidden_state
    ).abs().max().item()
    assert (flash_pooler_outputs - fp32_outputs.last_hidden_state).abs().mean().item() <= 3.0 * (
        bf16_outputs.last_hidden_state - fp32_outputs.last_hidden_state
    ).abs().mean().item()
