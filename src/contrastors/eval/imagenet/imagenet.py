from dataclasses import dataclass
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from nomic import atlas
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPModel

from contrastors.dataset.constants import (
    IMAGENET_CLASSES,
    IMAGENET_FOLDER_TO_CLASS,
    OPENAI_IMAGE_DATASET_MEAN,
    OPENAI_IMAGE_DATASET_STD,
)
from contrastors.dataset.image_text_loader import get_imagenet
from contrastors.dataset.transform import image_transform
from contrastors.eval.metrics import accuracy
from contrastors.eval.zero_shot import zeroshot_classifier_weights
from contrastors.models.dual_encoder import DualEncoder, DualEncoderConfig


def evaluate_imagenet(
    dataloader, tokenizer, text=None, vision=None, model=None, dtype=torch.float32, return_embeddings=False, prefix=None
):
    if model is not None:
        assert text is None and vision is None
        device = next(model.parameters()).device
        text = partial(model.get_text_embedding)
        vision = partial(model.get_vision_embedding)
    else:
        assert text is not None and vision is not None
        device = next(text.parameters()).device
        text.eval()
        vision.eval()

    zs_classifier, text_embeddings = zeroshot_classifier_weights(
        text,
        tokenizer,
        device,
        IMAGENET_CLASSES,
        templates=None,
        dtype=dtype,
        return_text_embeddings=return_embeddings,
        add_eos=False,
        prefix=prefix,
    )

    n = 0
    top1 = 0
    top5 = 0

    embeddings = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            path, img, target = batch
            img, target = img.to(device, dtype=dtype), target.to(device)

            with torch.no_grad():
                image_embeddings = vision(input_ids=img)["embedding"]

            logits = 100.0 * torch.matmul(image_embeddings.to(dtype), zs_classifier)

            if return_embeddings:
                # e.g. ILSVRC2012_val_00000293_n01440764.JPEG will be in path static.nomic.ai/imagenet/data/n01440764/ILSVRC2012_val_00000293_n01440764.JPEG
                for i in range(img.shape[0]):
                    img_embedding = image_embeddings[i]

                    curr_path = path[i]
                    curr_img = curr_path.split("/")[-1]
                    derived_folder = curr_img.split("_")[-1].split(".")[0]
                    curr_class = IMAGENET_FOLDER_TO_CLASS[derived_folder]

                    static_path = f"https://static.nomic.ai/imagenet/data/{derived_folder}/{curr_img}"
                    pred = logits.topk(max((1, 5)), 1, True, True)[1]
                    predicted_class = IMAGENET_CLASSES[pred[i, 0]]
                    pred = pred.t()
                    correct = pred.eq(target.view(1, -1).expand_as(pred)).detach().cpu().numpy()
                    feats = {
                        # numpy doesn't support bf16
                        "embedding": img_embedding.to(torch.float16).cpu().detach().numpy(),
                        "class": curr_class,
                        "input": static_path,
                        "url": static_path,
                        "modality": "vision",
                        # NOTE: there are classname collisions in imagenet prompt but for sake of equal comparison we're keeping them as is
                        # e.g. there are two `missiles` classes
                        "top_1_correct": int(correct[:1, i].sum() > 0),
                        "top_5_correct": int(correct[:5, i].sum() > 0),
                        "predicted_class": predicted_class,
                    }
                    embeddings.append(feats)

            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += img.size(0)

    return torch.tensor([top1 / n]).to(device), torch.tensor([top5 / n]).to(device), embeddings, text_embeddings


@dataclass
class DataConfig:
    imagenet_val_path: str = "/home/paperspace/multimodal-embed/imagenet/"
    batch_size: int = 64
    workers: int = 0


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = CLIPModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")

    def forward(self, inputs):
        outputs = {}
        if "vision" in inputs:
            outputs["vision"] = self.model.get_image_features(**inputs["vision"])

        if "text" in inputs:
            outputs["text"] = self.model.get_text_features(**inputs["text"])

        return outputs


def main():
    val_transforms = image_transform(
        image_size=224, mean=OPENAI_IMAGE_DATASET_MEAN, std=OPENAI_IMAGE_DATASET_STD, is_train=False
    )
    dataloader = get_imagenet(DataConfig(), val_transforms)
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1")

    path = "/home/paperspace/contrastors-dev/src/contrastors/ckpts/lit-text-frozen-text-vit-b-77-dfn-mean-lin-proj-16k-deepspeed-1b/epoch_2_model"
    config = DualEncoderConfig.from_pretrained(path)
    model = DualEncoder.from_pretrained(path, config=config).to(device=device, dtype=torch.bfloat16)

    text = model.text
    vision = model.vision

    prefix = "search_query"
    top1, top5, metadata, _ = evaluate_imagenet(
        text=text,
        vision=vision,
        tokenizer=tokenizer,
        dataloader=dataloader.dataloader,
        prefix=prefix,
        return_embeddings=True,
    )

    embeddings = []
    for data in metadata:
        embeddings.append(data.pop("embedding"))

    atlas.map_data(
        metadata, embeddings=np.array(embeddings), identifier=f"imagenet-{prefix}-embeddings-with-predicted-classes"
    )

    # gather tensors from all process, average them, and print result
    all_top1 = [top1 for _ in range(dist.get_world_size())]
    dist.all_gather(all_top1, top1)
    top1 = torch.cat(all_top1).mean()
    all_top5 = [top5 for _ in range(dist.get_world_size())]
    dist.all_gather(all_top5, top5)
    top5 = torch.cat(all_top5).mean()
    print(f"top1: {top1} top5: {top5}")


if __name__ == "__main__":
    dist.init_process_group(backend='nccl', init_method='env://')
    main()
