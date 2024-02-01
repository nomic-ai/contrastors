import gzip
import json
import math
import os
from argparse import ArgumentParser
from datetime import timedelta
from pathlib import Path

import faiss
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from beir.datasets.data_loader import GenericDataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--beir_path", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--query_key", default="query")
    parser.add_argument("--document_key", default="document")
    parser.add_argument("--negatives_key", default="negatives")

    return parser.parse_args()


def load_beir(beir_path):
    corpus, queries, _ = GenericDataLoader(beir_path).load(split="train")
    all_queries = []
    for _, query in queries.items():
        all_queries.append(query)

    all_documents = []
    for _, doc in corpus.items():
        all_documents.append(doc.get("title") + " " + doc.get("text"))

    return all_queries, all_documents


def load_dataset(path):
    all_data = []
    path = Path(path)
    if path.is_dir():
        files = sorted(path.glob("shard-*.jsonl.gz"))
    else:
        files = [path]

    for file in tqdm(files, desc="Loading shards"):
        with gzip.open(file, "rt") as f:
            for line in f:
                data = json.loads(line)

                all_data.append(data)

    return all_data


def print_rank0(*args, **kwargs):
    if dist.get_rank() == 0:
        print(*args, **kwargs)


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def embed(model, tokenizer, dataset, batch_size):
    doc2emb = {}
    with torch.no_grad():
        for batch_start in tqdm(range(0, len(dataset), batch_size), desc=f"Embedding"):
            batch_end = min(batch_start + batch_size, len(dataset))
            batch = dataset[batch_start:batch_end]

            tokenized = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(model.device)

            model_output = model(**tokenized)
            pooled = mean_pooling(model_output, tokenized["attention_mask"])
            normalized_emb = F.normalize(pooled, p=2, dim=1)
            doc2emb.update({doc: emb for doc, emb in zip(batch, normalized_emb.cpu().numpy())})

    return doc2emb


def knn_neighbors(query2embed, index, batch_size, k):
    query2score, query2indices = {}, {}
    id2query = {i: query for i, query in enumerate(sorted(query2embed.keys()))}

    for i in tqdm(range(0, len(id2query), batch_size), disable=dist.get_rank() != 0):
        queries = [id2query[j] for j in range(i, min(i + batch_size, len(id2query)))]
        query_embs = [query2embed[query] for query in queries]

        top_k_scores, top_k_indices = index.search(np.array(query_embs).astype(np.float32), k)

        query2score.update({query: score for query, score in zip(queries, top_k_scores)})
        query2indices.update({query: indices for query, indices in zip(queries, top_k_indices)})

    assert len(query2score) == len(query2indices)
    assert len(query2embed) == len(query2score)
    return query2score, query2indices


if __name__ == "__main__":
    dist.init_process_group(timeout=timedelta(minutes=60))
    torch.cuda.set_device(dist.get_rank())
    args = parse_args()

    output_dir = Path(args.output_dir)
    if dist.get_rank() == 0:
        if not output_dir.exists():
            output_dir.mkdir(parents=True)

    model_name = "thenlper/gte-base"
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16).to(f"cuda:{dist.get_rank()}")

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.model_max_length = 512

    # initialize once in case we have more than one iteration
    queries, documents = load_beir(args.beir_path)
    dataset = load_dataset(args.dataset)

    query2embed = embed(model, tokenizer, queries, args.batch_size)
    doc2embed = embed(model, tokenizer, documents, args.batch_size)
    id2doc = {i: doc for i, doc in enumerate(sorted(doc2embed.keys()))}

    d_embed = []
    for i in range(len(doc2embed)):
        d_embed.append(doc2embed[id2doc[i]])

    del model
    torch.cuda.empty_cache()

    index = faiss.IndexFlatIP(query2embed[list(query2embed.keys())[0]].shape[0])
    co = faiss.GpuMultipleClonerOptions()
    co.shard = True
    co.useFloat16 = True
    index = faiss.index_cpu_to_all_gpus(index, co=co)
    index.add(np.array(d_embed).astype(np.float32))

    scores, indices = knn_neighbors(query2embed, index, args.batch_size, args.k)

    for i, data in enumerate(tqdm(dataset)):
        query = data[args.query_key]
        inxs = indices[query]
        filtered_inx = []
        for inx in inxs:
            if inx == -1:
                break
            selected_doc = id2doc[inx]
            if selected_doc != data[args.document_key] and selected_doc != query:
                filtered_inx.append(inx)

        data[args.negatives_key] = [id2doc[inx] for inx in filtered_inx]
        if len(data[args.negatives_key]) < args.k:
            remaining = args.k - len(data[args.negatives_key])
            while True:
                random_idx = np.random.randint(0, len(id2doc), size=remaining).tolist()
                kept_idxs = []
                for random in random_idx:
                    if random == -1:
                        continue

                    if id2doc[random] != data[args.document_key] and id2doc[random] != query:
                        kept_idxs.append(id2doc[random])

                if len(kept_idxs) == remaining:
                    break

            data[args.negatives_key].extend(kept_idxs)

    metadata = {
        "objective": {"self": [], "paired": [], "triplet": [[args.query_key, args.document_key, args.negatives_key]]}
    }
    shard_size = 100_000
    for shard_start in tqdm(range(0, len(dataset), shard_size), desc="Writing shards"):
        dataset_slice = dataset[shard_start : shard_start + shard_size]
        for record in dataset_slice:
            record["metadata"] = metadata
        shard_num = shard_start // shard_size
        with gzip.open(output_dir / f"shard-{shard_num:05d}.jsonl.gz", "wt") as f:
            for data in tqdm(dataset_slice):
                f.write(json.dumps(data) + "\n")
