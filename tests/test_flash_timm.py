import pytest
import timm
import torch
from transformers import AutoProcessor

from contrastors.models.vit import ViTModel
from contrastors.models.vit.timm_vit import timm_name_to_vit_config


@pytest.mark.parametrize(
    "name, dtype",
    [
        ("timm/vit_base_patch16_224.dino", torch.float16),
        ("timm/vit_base_patch16_224.dino", torch.bfloat16),
        ("timm/vit_base_patch16_224.augreg2_in21k_ft_in1k", torch.float16),
        ("timm/vit_base_patch16_224.augreg2_in21k_ft_in1k", torch.bfloat16),
        ("timm/vit_base_patch16_224.augreg_in21k", torch.float16),
        ("timm/vit_base_patch16_224.augreg_in21k", torch.bfloat16),
    ],
)
def test_vit_base16_timm(name, dtype):
    timm_model = timm.create_model(name, pretrained=True).to(device="cuda", dtype=dtype)
    timm_ref = timm.create_model(name, pretrained=True).to(device="cuda", dtype=torch.float32)

    c_config = timm_name_to_vit_config(name)
    flash_model = ViTModel.from_pretrained(name, config=c_config).to(device="cuda", dtype=dtype)

    # just use any random image processor that maps to 224x224
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch16")

    torch.manual_seed(0)
    random_image = torch.randint(0, 255, (1, 3, 224, 224)).to("cuda").to(torch.int32)

    processed = processor(images=random_image, return_tensors="pt").to("cuda")

    flash_outputs = flash_model(input_ids=processed["pixel_values"].to(dtype))
    flash_pooler_outputs = flash_outputs.last_hidden_state
    timm_outputs = timm_model.forward_features(processed["pixel_values"].to(dtype))

    timm_ref_outputs = timm_ref.forward_features(processed["pixel_values"].to(torch.float32))

    print(f"Max error between contrastors and timm ref: {(flash_pooler_outputs - timm_ref_outputs).abs().max().item()}")
    print(
        f"Mean error between contrastors and timm ref: {(flash_pooler_outputs - timm_ref_outputs).abs().mean().item()}"
    )
    print(f"Max error between timm ref and timm_{dtype} {(timm_ref_outputs - timm_outputs).abs().max().item()}")
    print(f"Mean error between timm ref and timm_{dtype} {(timm_ref_outputs - timm_outputs).abs().mean().item()}")

    assert (flash_pooler_outputs - timm_ref_outputs).abs().max().item() <= 2 * (
        timm_outputs - timm_ref_outputs
    ).abs().max().item()
    assert (flash_pooler_outputs - timm_ref_outputs).abs().mean().item() <= 2 * (
        timm_outputs - timm_ref_outputs
    ).abs().mean().item()


@pytest.mark.parametrize(
    "name, dtype",
    [
        ("timm/vit_base_patch14_reg4_dinov2.lvd142m", torch.float16),
        ("timm/vit_base_patch14_reg4_dinov2.lvd142m", torch.bfloat16),
    ],
)
def test_vit_w_registers(name, dtype):
    timm_model = timm.create_model(name, pretrained=True).to(device="cuda", dtype=dtype)
    timm_ref = timm.create_model(name, pretrained=True).to(device="cuda", dtype=torch.float32)

    c_config = timm_name_to_vit_config(name)
    flash_model = ViTModel.from_pretrained(name, config=c_config).to(device="cuda", dtype=dtype)

    # TODO get this working with a < 518 image
    processor = AutoProcessor.from_pretrained(
        "facebook/dinov2-base",
        crop_size={"height": 518, "width": 518},
        size={"shortest_edge": 518},
    )

    torch.manual_seed(0)
    random_image = torch.randint(0, 255, (1, 3, 518, 518)).to("cuda").to(torch.int32)

    processed = processor(images=random_image, return_tensors="pt").to("cuda")

    flash_outputs = flash_model(input_ids=processed["pixel_values"].to(dtype))
    flash_pooler_outputs = flash_outputs.last_hidden_state
    timm_outputs = timm_model.forward_features(processed["pixel_values"].to(dtype))

    timm_ref_outputs = timm_ref.forward_features(processed["pixel_values"].to(torch.float32))

    print(f"Max error between contrastors and timm ref: {(flash_pooler_outputs - timm_ref_outputs).abs().max().item()}")
    print(
        f"Mean error between contrastors and timm ref: {(flash_pooler_outputs - timm_ref_outputs).abs().mean().item()}"
    )
    print(f"Max error between timm ref and timm_{dtype} {(timm_ref_outputs - timm_outputs).abs().max().item()}")
    print(f"Mean error between timm ref and timm_{dtype} {(timm_ref_outputs - timm_outputs).abs().mean().item()}")

    assert (flash_pooler_outputs - timm_ref_outputs).abs().max().item() <= 2 * (
        timm_outputs - timm_ref_outputs
    ).abs().max().item()
    assert (flash_pooler_outputs - timm_ref_outputs).abs().mean().item() <= 2 * (
        timm_outputs - timm_ref_outputs
    ).abs().mean().item()


@pytest.mark.parametrize(
    "name, dtype",
    [
        ("timm/vit_base_patch16_rope_reg1_gap_256.sbb_in1k", torch.float16),
        ("timm/vit_base_patch16_rope_reg1_gap_256.sbb_in1k", torch.bfloat16),
    ],
)
def test_vit_w_rope(name, dtype):
    timm_model = timm.create_model(name, pretrained=True, img_size=224, ref_feat_shape=(14, 14), num_classes=0).to(
        device="cuda", dtype=dtype
    )
    timm_ref = timm.create_model(name, pretrained=True, img_size=224, ref_feat_shape=(14, 14), num_classes=0).to(
        device="cuda", dtype=torch.float32
    )

    c_config = timm_name_to_vit_config(name)
    flash_model = ViTModel.from_pretrained(name, config=c_config).to(device="cuda", dtype=dtype)

    processor = AutoProcessor.from_pretrained("google/vit-base-patch16-224")

    torch.manual_seed(0)
    random_image = torch.randint(0, 255, (1, 3, 256, 256)).to("cuda").to(torch.int32)

    processed = processor(images=random_image, return_tensors="pt").to("cuda")
    timm_outputs = timm_model.forward_features(processed["pixel_values"].to(dtype))
    # timm_outputs = timm_model.forward_head(timm_outputs, pre_logits=True)

    flash_outputs = flash_model(input_ids=processed["pixel_values"].to(dtype))
    flash_pooler_outputs = flash_outputs.last_hidden_state

    timm_ref_outputs = timm_ref.forward_features(processed["pixel_values"].to(torch.float32))
    # timm_ref_outputs = timm_ref.forward_head(timm_ref_outputs, pre_logits=True)

    print(f"Max error between contrastors and timm ref: {(flash_pooler_outputs - timm_ref_outputs).abs().max().item()}")
    print(
        f"Mean error between contrastors and timm ref: {(flash_pooler_outputs - timm_ref_outputs).abs().mean().item()}"
    )
    print(f"Max error between timm ref and timm_{dtype} {(timm_ref_outputs - timm_outputs).abs().max().item()}")
    print(f"Mean error between timm ref and timm_{dtype} {(timm_ref_outputs - timm_outputs).abs().mean().item()}")

    assert (flash_pooler_outputs - timm_ref_outputs).abs().max().item() <= 2.5 * (
        timm_outputs - timm_ref_outputs
    ).abs().max().item()
    assert (flash_pooler_outputs - timm_ref_outputs).abs().mean().item() <= 2.5 * (
        timm_outputs - timm_ref_outputs
    ).abs().mean().item()


@pytest.mark.parametrize(
    "name, dtype",
    [
        ("hf-hub:nomic-ai/vit_eva02_base_patch16_224.mim_in22k", torch.float16),
        ("hf-hub:nomic-ai/vit_eva02_base_patch16_224.mim_in22k", torch.bfloat16),
    ],
)
def test_vit_eva_base_patch16(name, dtype):
    timm_model = timm.create_model(
        name,
        pretrained=True,
        img_size=224,
        ref_feat_shape=(14, 14),
        patch_size=16,
        num_classes=0,
    ).to(device="cuda", dtype=dtype)
    timm_ref = timm.create_model(
        name,
        pretrained=True,
        img_size=224,
        ref_feat_shape=(14, 14),
        patch_size=16,
        num_classes=0,
    ).to(device="cuda", dtype=torch.float32)

    c_config = timm_name_to_vit_config(name)
    flash_model = ViTModel.from_pretrained(name.replace("hf-hub:", ""), config=c_config).to(device="cuda", dtype=dtype)

    processor = AutoProcessor.from_pretrained("google/vit-base-patch16-224")

    torch.manual_seed(0)
    random_image = torch.randint(0, 255, (1, 3, 224, 224)).to("cuda").to(torch.int32)

    processed = processor(images=random_image, return_tensors="pt").to("cuda")
    timm_outputs = timm_model.forward_features(processed["pixel_values"].to(dtype))
    # timm_outputs = timm_model.forward_head(timm_outputs, pre_logits=True)

    flash_outputs = flash_model(input_ids=processed["pixel_values"].to(dtype))
    flash_pooler_outputs = flash_outputs.last_hidden_state

    timm_ref_outputs = timm_ref.forward_features(processed["pixel_values"].to(torch.float32))
    # timm_ref_outputs = timm_ref.forward_head(timm_ref_outputs, pre_logits=True)

    print(f"Max error between contrastors and timm ref: {(flash_pooler_outputs - timm_ref_outputs).abs().max().item()}")
    print(
        f"Mean error between contrastors and timm ref: {(flash_pooler_outputs - timm_ref_outputs).abs().mean().item()}"
    )
    print(f"Max error between timm ref and timm_{dtype} {(timm_ref_outputs - timm_outputs).abs().max().item()}")
    print(f"Mean error between timm ref and timm_{dtype} {(timm_ref_outputs - timm_outputs).abs().mean().item()}")

    # eva doesn't seem to like fp16
    multiplier = 2 if dtype == torch.bfloat16 else 4
    assert (flash_pooler_outputs - timm_ref_outputs).abs().max().item() <= multiplier * (
        timm_outputs - timm_ref_outputs
    ).abs().max().item()
    assert (flash_pooler_outputs - timm_ref_outputs).abs().mean().item() <= multiplier * (
        timm_outputs - timm_ref_outputs
    ).abs().mean().item()
