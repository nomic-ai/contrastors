import gzip
import json
import math
import os
from argparse import ArgumentParser
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoConfig, AutoTokenizer
import concurrent.futures
from torch.utils.data import DataLoader, IterableDataset

import faiss
from contrastors.models.encoder import BertConfig, NomicBertModel, bert_config_to_nomic_config


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
    parser.add_argument("is_hf_dataset", action="store_true")

    return parser.parse_args()


def send_dict_to_rank0(tensor_dict):
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Separate keys and tensors, ensuring that the keys are sorted to maintain order
    keys = sorted(tensor_dict.keys())
    with open(f"keys_rank_{rank}.json", "w") as f:
        json.dump(keys, f)
            
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
    for i, (query, document) in enumerate(tqdm(zip(queries, documents), total=len(queries), disable=rank != 0)):
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

    print_rank0("gathered queries and documents")

    if rank == 0:
        gathered_keys = [[] for _ in range(world_size)]
        for this_rank in range(world_size):
            with open(f"keys_rank_{this_rank}.json", "r") as f:
                keys = json.load(f)
            gathered_keys[this_rank] = keys

            # delete the temporary files
            os.remove(f"keys_rank_{this_rank}.json")
    else:
        gathered_keys = None

    print_rank0("Gathered keys")

    if rank == 0:
        # Flatten the lists of keys and tensors
        flat_keys = [item for sublist in gathered_keys for item in sublist]
        print(f"{len(flat_keys)=}")
        flat_queries = [item for sublist in zip(*gathered_queries) for item in sublist]
        print(f"{len(flat_queries)=}")
        flat_documents = [item for sublist in zip(*gathered_documents) for item in sublist]
        print(f"{len(flat_documents)=}")

        # Rebuild the dictionary
        rebuilt_dict = {k: (q, d) for k, q, d in zip(flat_keys, flat_queries, flat_documents)}
        print(f"{len(rebuilt_dict)=}")
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


def count_lines_in_file(file_info):
    file, start_pos = file_info
    num_lines = 0
    filehandler = gzip.open(file, "rt") if str(file).endswith(".gz") else open(file, "r")
    
    with filehandler as f:
        for _ in f:
            num_lines += 1
        current_pos = f.buffer.fileobj.tell()
    
    return num_lines, current_pos - start_pos

def get_num_lines(dataset):
    if isinstance(dataset, str):
        dataset = Path(dataset)
    
    # Get total size and setup progress bar
    total_bytes = 0
    if dataset.is_dir():
        files = sorted(dataset.glob("shard-*.jsonl.gz"))
        total_bytes = sum(os.path.getsize(f) for f in files)
    else:
        files = [dataset]
        total_bytes = os.path.getsize(dataset)
    
    # Get current rank and world size
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Distribute files across ranks
    files_for_this_rank = [f for i, f in enumerate(files) if i % world_size == rank]
    
    progbar = tqdm(total=total_bytes // world_size, unit="B", unit_scale=True, 
                  disable=rank != 0)
    
    # Prepare file information for workers for this rank's files
    file_infos = [(f, 0) for f in files_for_this_rank]
    
    # Use process pool for true parallelism
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Submit this rank's files for processing
        future_to_file = {
            executor.submit(count_lines_in_file, file_info): file_info 
            for file_info in file_infos
        }
        
        total_lines = 0
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_file):
            try:
                num_lines, bytes_processed = future.result()
                total_lines += num_lines
                progbar.update(bytes_processed)
            except Exception as e:
                print(f'Rank {rank}: Generated an exception: {e}')
    
    # Gather results from all ranks
    all_lines = torch.tensor([total_lines], device=f"cuda:{rank}")
    dist.all_reduce(all_lines, op=dist.ReduceOp.SUM)
    
    progbar.close()
    return all_lines.item()


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

    if path.is_dir():
        path = sorted(path.glob("shard-*.jsonl.gz"))
    else:
        path = [dataset]

    worker_info = torch.utils.data.get_worker_info()
    num_workers = worker_info.num_workers if worker_info else 1
    worker_id = worker_info.id if worker_info else 0
    total_parallel = num_workers * world_size
    global_worker_rank = (rank * num_workers) + worker_id
    print(f"rank: {rank}, worker_id: {worker_id}, global_worker_rank: {global_worker_rank}, total_parallel: {total_parallel}")
    
    
    total_rows = 0  # Keep track of total rows seen by this worker
    total_yielded = 0  # Keep track of rows actually yielded
    
    ids = 0
    for file in path:
        filehandler = gzip.open(file, "rt") if str(file).endswith(".gz") else open(file, "r")
        with filehandler as f:
            for line in f:
                # get row for current rank + worker
                if ids % total_parallel == global_worker_rank:
                    total_rows += 1
                    data = json.loads(line)
                    if isinstance(data, list):
                        batch["query"].append(f"query: {data[0]}")
                        batch["document"].append(f"passage: {data[1]}")
                        batch["id"].append(ids)
                    else:
                        batch["query"].append(f"query: {data[query_key]}")
                        if isinstance(data[document_key], list):
                            batch["document"].append(f"passage: {data[document_key][0]}")
                        else:
                            batch["document"].append(f"passage: {data[document_key]}")
                        batch["id"].append(ids)

                    if len(batch["query"]) >= per_device_batch_size:
                        yield batch
                        batch = {"query": [], "document": [], "id": []}

                ids += 1

    # if we have a partial batch, yield it
    # if len(batch["query"]) > 0:
    #     tokenized_query = tokenizer(batch["query"], padding=True, truncation=True, return_tensors="pt")
    #     tokenized_document = tokenizer(batch["document"], padding=True, truncation=True, return_tensors="pt")
    #     yield {"query": tokenized_query, "document": tokenized_document, "id": batch["id"]}

class JSONLDataset(IterableDataset):
    def __init__(self, path, tokenizer, query_key, document_key, per_device_batch_size):
        self.path = path
        self.tokenizer = tokenizer
        self.query_key = query_key
        self.document_key = document_key
        self.per_device_batch_size = per_device_batch_size
    
    def __iter__(self):
        return jsonl_collator(
            self.path, 
            self.tokenizer,
            self.query_key,
            self.document_key,
            per_device_batch_size=self.per_device_batch_size,
        )

        
def collate_fn(batches):
    """
    Collate function to combine batches from multiple workers
    """
    
    # Tokenize the combined batch
    tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small")
    tokenized_query = tokenizer(
        batches["query"], 
        padding=True, 
        truncation=True, 
        return_tensors="pt"
    )
    tokenized_document = tokenizer(
        batches["document"], 
        padding=True, 
        truncation=True, 
        return_tensors="pt"
    )
    
    return {
        "query": tokenized_query,
        "document": tokenized_document,
        "id": batches["id"]
    }


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
    print_rank0(f"dataset: {dataset}, {dataset.is_dir()=}")
    if False:
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

    model_name = "intfloat/multilingual-e5-small"
    hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config = bert_config_to_nomic_config(hf_config)
    model = (
        NomicBertModel.from_pretrained(model_name, config=config, add_pooling_layer=False)
        .to(f"cuda:{dist.get_rank()}")
        .to(dtype=torch.float16)
    )

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.model_max_length = 512

    # initialize once in case we have more than one iteration
    if False:
        dataloader = dict_collator(records, tokenizer, args.query_key, args.document_key, args.batch_size)
    else:
        dataloader = JSONLDataset(
            dataset, tokenizer, args.query_key, args.document_key, args.batch_size)

        dataloader = DataLoader(dataloader, batch_size=None, num_workers=8, collate_fn=collate_fn, prefetch_factor=4)


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
        print(f"{type(embeddings)=}")
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

            print(f"rank {dist.get_rank()} finished writing {len(ids_to_keep)} samples")

        dist.barrier()

    print_rank0(f"Kept {total_kept:,} out of {total_samples:,}")
