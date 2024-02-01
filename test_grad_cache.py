import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from grad_cache import GradCache
from transformers import AutoModel, AutoTokenizer

from contrastors.loss import clip_loss
from contrastors.loss import grad_cache_loss_biencoder as grad_cache_loss

# NOTE this requires you to pip install grad cache: https://github.com/luyug/GradCache/tree/main
# TODO: loss is the same but gradients aren't -> they are like 2x the other loss
# Run with `torchrun --nproc-per-node=2 test_grad_cache.py`


class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def __call__(self, input_ids, attention_mask=None, token_type_ids=None):
        return {
            "embedding": self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[
                "pooler_output"
            ]
        }


class LogitScale(nn.Module):
    def __init__(self, scale=0):
        super().__init__()
        self.scale = nn.Parameter(torch.ones([]) * np.log(scale))

    def forward(self, x):
        return x * self.scale.exp()


def print_rank0(*args, **kwargs):
    if dist.get_rank() == 0:
        print(*args, **kwargs)


if __name__ == "__main__":
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    print_rank0(f"Number of processes: {dist.get_world_size()}")

    query = ['this is an apple', 'steak should be cooked medium rare', 'cmu is pittsburgh', 'apple sells laptop']
    document = ['fruit', 'meat', 'school', 'company']

    device = torch.device(f'cuda:{rank}')
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    encoder = AutoModel.from_pretrained("bert-base-uncased").to(device)
    model = ModelWrapper(encoder)
    first = nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank)
    scale = nn.parallel.DistributedDataParallel(LogitScale(1 / 0.07).to(device))

    # NOTE! If using grad cache and > 1 gpu, you need to do the gather for the clip loss function yourself
    gc = GradCache(models=[first, first], chunk_sizes=2, loss_fn=clip_loss, get_rep_fn=lambda v: v["embedding"])
    xx = tokenizer(query, return_tensors='pt', padding=True).to(device)
    yy = tokenizer(document, return_tensors='pt', padding=True).to(device)

    loss = gc(xx, yy, logit_scale=scale, no_sync_except_last=True, gather_enabled=True)

    print_rank0("GradCache")
    print_rank0(f"Loss: {loss}")
    print_rank0(f"Sum of gradients: {sum([torch.sum(x.grad) for x in encoder.parameters()])}\n")
    first.zero_grad()
    del model, first, gc, xx, yy, loss

    xx = tokenizer(query, return_tensors='pt', padding=True).to(device)
    yy = tokenizer(document, return_tensors='pt', padding=True).to(device)

    inputs = {
        "query_input_ids": xx["input_ids"],
        "query_attention_mask": xx["attention_mask"],
        "query_token_type_ids": xx["token_type_ids"],
        "document_input_ids": yy["input_ids"],
        "document_attention_mask": yy["attention_mask"],
        "document_token_type_ids": yy["token_type_ids"],
    }

    encoder = AutoModel.from_pretrained("bert-base-uncased").to(device)
    model = nn.parallel.DistributedDataParallel(
        ModelWrapper(encoder), device_ids=[rank], output_device=rank, broadcast_buffers=False
    )
    scale = nn.parallel.DistributedDataParallel(LogitScale(1 / 0.07).to(device))
    our_loss = grad_cache_loss(model, inputs, chunk_size=2, logit_scale=scale)

    print_rank0("Our GradCache loss")
    print_rank0(f"Loss: {our_loss}")
    print_rank0(f"Sum of gradients: {sum([torch.sum(x.grad) for x in model.parameters()])}\n")

    model.zero_grad()
    del model, encoder, scale, our_loss, xx, yy

    xx = tokenizer(query, return_tensors='pt', padding=True).to(device)
    yy = tokenizer(document, return_tensors='pt', padding=True).to(device)

    encoder = AutoModel.from_pretrained("bert-base-uncased").to(device)
    model = nn.parallel.DistributedDataParallel(
        ModelWrapper(encoder), device_ids=[rank], output_device=rank, broadcast_buffers=False
    )
    scale = nn.parallel.DistributedDataParallel(LogitScale(1 / 0.07).to(device))

    query_emb = model(**xx)["embedding"]
    doc_emb = model(**yy)["embedding"]

    loss = clip_loss(query_emb, doc_emb, logit_scale=scale, gather_enabled=True)
    loss.backward()
    print_rank0("Our CLIP loss")
    print_rank0(f"Loss: {loss}")
    print_rank0(f"Sum of gradients: {sum([torch.sum(x.grad) for x in model.parameters()])}")
