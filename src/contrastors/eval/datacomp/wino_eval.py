"""Evaluate on WinoGAViL dataset."""

import os
from functools import partial

import datasets
import numpy as np
import torch

# from collections import Counter
from sklearn.metrics import jaccard_score
from tqdm import tqdm
from transformers import AutoTokenizer

from .wds_eval import create_model

# from transformers import CLIPModel, CLIPProcessor


class WinoDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None, text_transform=None):
        super().__init__()
        self._dataset = hf_dataset
        self.transform = (lambda x: x) if transform is None else transform
        self.text_transform = (lambda x: x) if text_transform is None else text_transform

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index: int):
        example = self._dataset[index]
        return (
            self.transform(example["candidate_images"]),
            self.text_transform(example["cue"]),
            np.isin(example["candidates"], example["associations"]),
        )


def evaluate_winogavil_dataset(
    model_arch,
    model_path,
    data_root=None,
    num_workers=4,
    batch_size=None,
    prefix=None,
):
    model, transform, device = create_model(model_arch, model_path)
    tokenizer = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1")
    tokenizer.model_max_length = 77
    tokenizer = partial(tokenizer, return_tensors="pt", truncation=True, padding="max_length")

    # Load data
    dataset = WinoDataset(
        datasets.load_dataset(
            "nlphuji/winogavil",
            split="test",
            # cache_dir=os.path.join(data_root, "hf_cache")
            # if data_root is not None
            # else None,
        ),
        transform=lambda imgs: torch.stack([transform(img) for img in imgs]),
        text_transform=lambda text: tokenizer([get_clip_prompt(text, prefix=prefix)]),
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda batch: batch[0],
    )

    all_groups = []
    all_scores = []

    # Iterate WinoGAViL Instances
    for idx, (images, text, y_true) in enumerate(tqdm(dataloader)):
        # Get example
        n_images = len(images)
        n_assoc = y_true.sum()
        # Featurize
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            image_features = model.encode_image(images.to(device), normalize=True)
            text_features = model.encode_text(text.to(device), normalize=True)
            # Compute similarities
            image_logits = (text_features @ image_features.T).squeeze(0).cpu().to(torch.float32).numpy()
        # Select topk
        topk_indices = np.argsort(image_logits)[-n_assoc:]
        y_pred = np.isin(np.arange(n_images), topk_indices)

        # Evaluate with Jaccard
        score = jaccard_score(y_true, y_pred)
        all_scores.append(score)
        all_groups.append(n_images)

        if idx > 0 and idx % 100 == 0:
            print(f"idx: {idx}, current Jaccard index average: {np.mean(all_scores)}")

    all_groups = np.array(all_groups)
    all_scores = np.array(all_scores)
    return {
        "avg_jaccard_score": all_scores.mean(),
        "jaccard_score_5": all_scores[all_groups == 5].mean(),
        "jaccard_score_6": all_scores[all_groups == 6].mean(),
        "jaccard_score_10": all_scores[all_groups == 10].mean(),
        "jaccard_score_12": all_scores[all_groups == 12].mean(),
        "jaccard_score_5-6": all_scores[all_groups <= 6].mean(),
        "jaccard_score_10-12": all_scores[all_groups >= 10].mean(),
    }


# def solve_winogavil_instance(clip_model, clip_processor, cue, num_associations, candidates, candidates_images):
#     clip_text = get_clip_txt(cue)

#     sim_for_image = {}
#     for img_name, img in zip(candidates, candidates_images):
#         processed_cue_img = clip_processor(text=[clip_text], images=img, return_tensors="pt")
#         output_cue_img = clip_model(**processed_cue_img).logits_per_image.item()
#         sim_for_image[img_name] = output_cue_img

#     sorted_sim_for_image = Counter(sim_for_image).most_common()[:num_associations]
#     clip_predictions = [x[0] for x in sorted_sim_for_image]
#     return clip_predictions


def get_clip_prompt(item, prefix=None):
    item = item.lower()
    vowels = ["a", "e", "i", "o", "u"]
    if item[0] in vowels:
        clip_txt = f"An {item}"
    else:
        clip_txt = f"A {item}"

    if prefix is not None:
        clip_txt = f"{prefix}: {clip_txt}"
    return clip_txt


# def get_vectors_similarity(v1, v2):
#     similarity = v1.detach().numpy() @ v2.detach().numpy().T
#     similarity_item = similarity.item()
#     return similarity_item


# def get_jaccard(s1, s2):
#     s1 = set(s1)
#     s2 = set(s2)
#     jaccard = int(len(s1.intersection(s2)) / len(s1.union(s2)) * 100)
#     return jaccard
