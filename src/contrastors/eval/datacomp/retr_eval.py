"""Evaluate on image-text retrieval datasets."""

import os
from functools import partial

import datasets
import torch
from clip_benchmark.datasets.builder import image_captions_collate_fn
from clip_benchmark.metrics import zeroshot_retrieval as zsr
from transformers import AutoTokenizer

from .wds_eval import create_model


class RetrievalDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        super().__init__()
        self._dataset = hf_dataset
        self.transform = (lambda x: x) if transform is None else transform

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index: int):
        return (
            self.transform(self._dataset[index]["image"]),
            self._dataset[index]["caption"],
        )


def image_captions_collage_fn_prefix(batch, prefix=None):
    imgs, texts = image_captions_collate_fn(batch)
    if prefix:
        for text_list in texts:
            for i, text in enumerate(text_list):
                text_list[i] = f"{prefix}: {text}"
    return imgs, texts


def evaluate_retrieval_dataset(task, model_arch, model_path, data_root=None, batch_size=64, num_workers=4, prefix=None):
    """Evaluate CLIP model on retrieval task."""

    model, transform, device = create_model(model_arch, model_path)
    tokenizer = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1")
    tokenizer.model_max_length = 77
    tokenizer = partial(tokenizer, return_tensors="pt", truncation=True, padding="max_length")

    dataset = RetrievalDataset(
        datasets.load_dataset(
            f"nlphuji/{task.replace('retrieval/', '')}",
            split="test",
            cache_dir=os.path.join(data_root, "hf_cache") if data_root is not None else None,
        ),
        transform=transform,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda x: image_captions_collage_fn_prefix(x, prefix=prefix),
    )

    metrics = zsr.evaluate(model, dataloader, tokenizer, recall_k_list=[1, 5, 10], device=device, amp=False)
    metrics["mean_recall@1"] = 0.5 * (metrics["text_retrieval_recall@1"] + metrics["image_retrieval_recall@1"])
    return metrics
