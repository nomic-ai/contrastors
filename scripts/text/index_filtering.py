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
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from contrastors.models.encoder import BertConfig, BertModel, bert_config_to_gpt2_config


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--query_key", required=True)
    parser.add_argument("--document_key", required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--index_size", type=int, default=1_000_000)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--max_files", type=int, default=100)
    parser.add_argument("--file_start", type=int, default=0)

    return parser.parse_args()


def send_dict_to_rank0(tensor_dict):
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Separate keys and tensors, ensuring that the keys are sorted to maintain order
    keys = sorted(tensor_dict.keys())
    tensors = [tensor_dict[k] for k in keys]
    queries = [tensor[0] for tensor in tensors]
    documents = [tensor[1] for tensor in tensors]

    # On rank 0, prepare lists to gather tensors
    if rank == 0:
        gathered_queries = [[] for _ in range(len(queries))]
        gathered_documents = [[] for _ in range(len(documents))]
    else:
        gathered_queries = None
        gathered_documents = None

    # Gather tensors on rank 0
    for i, (query, document) in enumerate(zip(queries, documents)):
        # On rank 0, prepare a list to store gathered tensors from all ranks for the current tensor
        if rank == 0:
            gathered_q = [torch.empty_like(query) for _ in range(world_size)]
            gathered_queries[i] = gathered_q

            gathered_d = [torch.empty_like(document) for _ in range(world_size)]
            gathered_documents[i] = gathered_d

        else:
            gathered_q = None
            gathered_d = None

        # Gather the current tensor across all ranks
        dist.gather(query, gather_list=gathered_q, dst=0)
        dist.gather(document, gather_list=gathered_d, dst=0)

    if rank == 0:
        gathered_keys = [[] for _ in range(world_size)]
    else:
        gathered_keys = None

    dist.gather_object(keys, object_gather_list=gathered_keys, dst=0)

    if rank == 0:
        # Flatten the lists of keys and tensors
        flat_keys = [item for sublist in gathered_keys for item in sublist]
        flat_queries = [item for sublist in zip(*gathered_queries) for item in sublist]
        flat_documents = [item for sublist in zip(*gathered_documents) for item in sublist]

        # Rebuild the dictionary
        rebuilt_dict = {k: (q, d) for k, q, d in zip(flat_keys, flat_queries, flat_documents)}
        return rebuilt_dict
    else:
        return None


def print_rank0(*args, **kwargs):
    if dist.get_rank() == 0:
        print(*args, **kwargs)


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def load_dataset(path, query_key, document_key, file_start=0, max_files=100):
    dataset = []
    path = Path(path)
    if path.is_dir():
        files = sorted(path.glob("shard-*.jsonl"))[file_start : file_start + max_files]
    else:
        files = [path]

    i = 0
    for file in tqdm(files, desc="Loading shards", disable=dist.get_rank() != 0):
        filehandler = gzip.open(file, "rt") if file.suffix == ".gz" else open(file, "r")
        with filehandler as f:
            for line in f:
                data = json.loads(line)
                record = {query_key: data[query_key], document_key: data[document_key], "id": i}

                dataset.append(record)
                i += 1

    return dataset


def get_num_lines(dataset):
    num_lines = 0
    total_bytes = os.path.getsize(dataset)
    progbar = tqdm(total=total_bytes, unit="B", unit_scale=True, disable=dist.get_rank() != 0)
    if dataset.is_dir():
        files = sorted(dataset.glob("shard-*.jsonl.gz"))
    else:
        files = [dataset]
    for file in tqdm(files):
        filehandler = gzip.open(file, "rt") if file.endswith(".gz") else open(file, "r")
        with filehandler as f:
            for _ in f:
                num_lines += 1
                progbar.update(f.buffer.fileobj.tell() - progbar.n)

    return num_lines


def dict_collator(records, tokenizer, query_key, document_key, per_device_batch_size):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    batch = {"query": [], "document": [], "id": []}

    for i, record in enumerate(records):
        if i % world_size != rank:
            continue

        batch["query"].append(record[query_key])
        batch["document"].append(record[document_key])
        batch["id"].append(record["id"])

        if len(batch["query"]) == per_device_batch_size:
            tokenized_query = tokenizer(batch["query"], padding=True, truncation=True, return_tensors="pt")
            tokenized_document = tokenizer(batch["document"], padding=True, truncation=True, return_tensors="pt")
            yield {"query": tokenized_query, "document": tokenized_document, "id": batch["id"]}
            batch = {"query": [], "document": [], "id": []}

    # if we have a partial batch, yield it
    if len(batch["query"]) > 0:
        tokenized_query = tokenizer(batch["query"], padding=True, truncation=True, return_tensors="pt")
        tokenized_document = tokenizer(batch["document"], padding=True, truncation=True, return_tensors="pt")
        yield {"query": tokenized_query, "document": tokenized_document, "id": batch["id"]}


def jsonl_collator(path, tokenizer, query_key, document_key, per_device_batch_size):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    batch = {"query": [], "document": [], "id": []}

    filehandler = gzip.open(path, "rt") if path.endswith(".gz") else open(path, "r")
    with filehandler as f:
        for i, line in enumerate(f):
            if i % world_size != rank:
                continue

            data = json.loads(line)
            if isinstance(data, list):
                batch["query"].append(data[0])
                batch["document"].append(data[1])
                batch["id"].append(i)
            else:
                batch["query"].append(data[query_key])
                if isinstance(data[document_key], list):
                    # take first since it's easy to do and we don't have to find before
                    batch["document"].append(data[document_key][0])
                else:
                    batch["document"].append(data[document_key])

                batch["id"].append(i)

            if len(batch["query"]) == per_device_batch_size:
                tokenized_query = tokenizer(batch["query"], padding=True, truncation=True, return_tensors="pt")
                tokenized_document = tokenizer(batch["document"], padding=True, truncation=True, return_tensors="pt")
                yield {"query": tokenized_query, "document": tokenized_document, "id": batch["id"]}
                batch = {"query": [], "document": [], "id": []}

        # if we have a partial batch, yield it
        if len(batch["query"]) > 0:
            tokenized_query = tokenizer(batch["query"], padding=True, truncation=True, return_tensors="pt")
            tokenized_document = tokenizer(batch["document"], padding=True, truncation=True, return_tensors="pt")
            yield {"query": tokenized_query, "document": tokenized_document, "id": batch["id"]}


def embed(model, dataloader, batch_size, max_samples):
    id2embedding = {}
    examples_seen = 0
    progbar = tqdm(total=max_samples // batch_size + 1, disable=dist.get_rank() != 0)
    with torch.no_grad():
        for batch in dataloader:
            ids = batch.pop("id")
            query_inputs = {k: v.to(f"cuda:{dist.get_rank()}") for k, v in batch["query"].items()}
            query = model(**query_inputs)

            query = mean_pooling(query, query_inputs["attention_mask"])
            normalized_query = F.normalize(query, p=2, dim=1)

            answer_inputs = {k: v.to(f"cuda:{dist.get_rank()}") for k, v in batch["document"].items()}
            answer = model(**answer_inputs)

            answer = mean_pooling(answer, answer_inputs["attention_mask"])
            normlized_answer = F.normalize(answer, p=2, dim=1)

            id2embedding.update(
                {id: (query.cpu(), answer.cpu()) for id, query, answer in zip(ids, normalized_query, normlized_answer)}
            )

            progbar.update(1)
            examples_seen += batch_size
            if examples_seen >= max_samples:
                break

    return id2embedding


def filter_points(id2embeddings, batch_size=256):
    index = faiss.IndexFlatIP(len(id2embeddings[list(id2embeddings.keys())[0]][0]))
    co = faiss.GpuMultipleClonerOptions()
    co.shard = True
    co.useFloat16 = True
    print("building index")
    index = faiss.index_cpu_to_all_gpus(index, co=co)
    print("index built")

    id2doc_emb = {k: v[1] for k, v in id2embeddings.items()}
    id2_query_emb = {k: v[0] for k, v in id2embeddings.items()}
    range2id = {i: id for i, id in enumerate(sorted(id2doc_emb.keys()))}
    doc_emb = [id2doc_emb[range2id[i]] for i in range(len(range2id))]

    index.add(np.array(doc_emb).astype(np.float32))

    ids2keep = []
    for i in tqdm(range(0, len(range2id), batch_size), disable=dist.get_rank() != 0):
        atlas_ids = [range2id[j] for j in range(i, min(i + batch_size, len(range2id)))]
        query_embs = [id2_query_emb[atlas_id] for atlas_id in atlas_ids]
        _, top_k_indices = index.search(np.array(query_embs).astype(np.float32), 2)
        valid_pairs = (
            np.equal(top_k_indices, np.arange(i, min(i + batch_size, len(range2id)))[:, None]).sum(axis=1).tolist()
        )
        for j, is_valid in enumerate(valid_pairs):
            if is_valid:
                ids2keep.append(atlas_ids[j])

    return ids2keep


if __name__ == "__main__":
    dist.init_process_group(timeout=timedelta(minutes=60))
    torch.cuda.set_device(dist.get_rank())
    args = parse_args()

    output_dir = Path(args.output_dir)
    if dist.get_rank() == 0:
        if not output_dir.exists():
            output_dir.mkdir(parents=True)

    dataset = Path(args.dataset)
    if dataset.is_dir():
        records = load_dataset(dataset, args.query_key, args.document_key, args.file_start, args.max_files)
        num_lines = len(records)
    else:
        num_lines = get_num_lines(dataset)

    print_rank0(f"num lines: {num_lines}")

    num_examples_per_rank = math.ceil(num_lines / dist.get_world_size())
    print_rank0(f"num examples per rank: {num_examples_per_rank}")
    num_iterations = num_lines // args.index_size
    print_rank0(f"num iterations: {num_iterations}")

    per_device_max_samples = min(args.index_size // dist.get_world_size(), num_examples_per_rank)
    print_rank0(f"Total examples per device: {per_device_max_samples}")
    num_batches_per_device = per_device_max_samples // args.batch_size
    if per_device_max_samples % args.batch_size != 0:
        num_batches_per_device += 1
    print_rank0(f"Num batchers per device: {num_batches_per_device}")

    model_name = "thenlper/gte-base"
    hf_config = BertConfig.from_pretrained(model_name)
    config = bert_config_to_gpt2_config(hf_config)
    model = (
        BertModel.from_pretrained(model_name, config=config, add_pooling_layer=False)
        .to(f"cuda:{dist.get_rank()}")
        .to(dtype=torch.float16)
    )

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.model_max_length = 512

    # initialize once in case we have more than one iteration
    if dataset.is_dir():
        dataloader = dict_collator(records, tokenizer, args.query_key, args.document_key, args.batch_size)
    else:
        dataloader = jsonl_collator(dataset, tokenizer, args.query_key, args.document_key, args.batch_size)

    total_samples = 0
    total_kept = 0
    if num_iterations == 0:
        num_iterations = 1
    for i in tqdm(range(num_iterations), disable=dist.get_rank() != 0):
        # if we're on the last iteration and it's not divisible by batch_size * world_size, round down
        if i == num_iterations - 1:
            total_seen = i * num_batches_per_device * args.batch_size * dist.get_world_size()
            remaining = num_lines - total_seen
            per_device_max_samples = remaining - (remaining % (args.batch_size * dist.get_world_size()))
            per_device_max_samples = (per_device_max_samples // dist.get_world_size()) - 1

        print(f"rank {dist.get_rank()} embedding {per_device_max_samples} samples")
        embeddings = embed(model, dataloader, args.batch_size, per_device_max_samples)
        print(f"rank {dist.get_rank()} finished embedding {len(embeddings)} samples")

        dist.barrier()
        all_embeddings = send_dict_to_rank0(embeddings)

        if dist.get_rank() == 0:
            torch.cuda.empty_cache()
            all_embeddings = {k: (q.numpy(), d.numpy()) for k, (q, d) in all_embeddings.items()}
            ids_to_keep = filter_points(all_embeddings)
            print(f"keeping {len(ids_to_keep)} out of {len(all_embeddings)}")
            total_samples += len(all_embeddings)
            total_kept += len(ids_to_keep)
            existing_shards = output_dir.glob(f"ids_to_keep_*.json")
            current_shard = len(list(existing_shards))
            with open(output_dir / f"ids_to_keep_{current_shard}.json", "w") as f:
                json.dump(ids_to_keep, f)

        dist.barrier()

    print_rank0(f"Kept {total_kept} out of {total_samples}")
