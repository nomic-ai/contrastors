import gzip
import json
import logging
import os
import random
from pathlib import Path

import fsspec
import torch
import torch.distributed as dist
import webdataset as wds
import yaml
from pyarrow.json import read_json
from torch.utils.data import DataLoader, Dataset, DistributedSampler, IterableDataset
from tqdm import tqdm
from webdataset.tariterators import base_plus_ext

from contrastors.distributed import print_in_order, print_rank_zero

MAPPED_NAMES = {"paired": ["query", "document"], "self": ["query"], "triplet": ["query", "document", "negative"]}
KEY2PREFIX = {"query": "query", "document": "passage", "negative": "passage"}
S3_COMMAND = "pipe: aws s3 cp --endpoint-url https://9fa58365a1a3d032127970d0bd9a1290.r2.cloudflarestorage.com/ --cli-read-timeout=300 {s3_uri} -"
DEFAULT_COL_TO_MAX_TOKENS = {"query": 32, "document": 256, "negative": 256}

import mmap
import json
import os
from pathlib import Path
import struct

class MemoryMappedDict:
    def __init__(self, filename):
        self.filename = filename
        self.file_obj = None
        self.mmap_obj = None
        self.index = {}  # Stores {key: (offset, length)} pairs
        
    def save_dict(self, data):
        """Serialize dictionary with indexed access."""
        # First, write the index information
        current_offset = 0
        
        # Create temporary storage for serialized values
        serialized_values = []
        
        # Process each key-value pair
        for key, value in data.items():
            # Serialize the value
            value_json = json.dumps(value)
            value_bytes = value_json.encode('utf-8')
            
            # Store the offset and length
            self.index[key] = (current_offset, len(value_bytes))
            
            # Update offset for next value
            current_offset += len(value_bytes)
            
            # Store serialized value
            serialized_values.append(value_bytes)

        # Serialize the index itself
        index_json = json.dumps(self.index)
        index_bytes = index_json.encode('utf-8')
        
        # Write index size as 8-byte integer at start of file
        with open(self.filename, 'wb') as f:
            f.write(struct.pack('Q', len(index_bytes)))
            f.write(index_bytes)
            # Write all values
            for value_bytes in serialized_values:
                f.write(value_bytes)
        
        # Open file and create memory mapping
        self.file_obj = open(self.filename, 'r+b')
        self.mmap_obj = mmap.mmap(self.file_obj.fileno(), 0)
        
    def _load_index(self):
        """Load just the index portion of the file."""
        if not self.mmap_obj:
            self.file_obj = open(self.filename, 'r+b')
            self.mmap_obj = mmap.mmap(self.file_obj.fileno(), 0)
        
        # Read index size (first 8 bytes)
        index_size = struct.unpack('Q', self.mmap_obj[:8])[0]
        
        # Read and parse index
        index_bytes = self.mmap_obj[8:8 + index_size]
        self.index = json.loads(index_bytes.decode('utf-8'))
    
    def get(self, key):
        """Get a single value from the memory mapped file."""
        if not self.index:
            self._load_index()
            
        if key not in self.index:
            raise KeyError(f"Key '{key}' not found")
            
        offset, length = self.index[key]
        # Account for index size and its 8-byte length prefix
        index_size = struct.unpack('Q', self.mmap_obj[:8])[0]
        real_offset = 8 + index_size + offset
        
        # Read just this value's bytes
        value_bytes = self.mmap_obj[real_offset:real_offset + length]
        return json.loads(value_bytes.decode('utf-8'))
    
    def set(self, key, value):
        """This is a simplified set - it requires rewriting the entire file."""
        current_data = self.get_all()
        current_data[key] = value
        self.save_dict(current_data)
    
    def get_all(self):
        """Get all key-value pairs (loads everything into RAM)."""
        if not self.index:
            self._load_index()
            
        result = {}
        for key in self.index:
            result[key] = self.get(key)
        return result
    
    def keys(self):
        """Get list of keys without loading values."""
        if not self.index:
            self._load_index()
        return list(self.index.keys())
    
    def close(self):
        """Close memory mapping and file."""
        if self.mmap_obj is not None:
            self.mmap_obj.close()
        if self.file_obj is not None:
            self.file_obj.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True


def collate_fn(batch):
    return batch[0]


class StreamingShardDataset(IterableDataset):
    def __init__(
        self,
        ds_spec,
        global_batch_size,
        tokenizer,
        seed,
        add_eos=True,
        add_prefix=False,
        num_negatives=-1,
        download_locally=False,
        process_one_shard=False,
        weighted_sampling=False,
        verbose=True,
        infinite=False,
        sample_negatives=False,
        run_name=None,
        query_max_length=None,
        document_max_length=None,
    ):
        self.num_samples_per_shard = {}
        self.max_per_shard = {}
        self.total_samples = 0
        self.path2objective = {}
        self.path2offsets = {}
        self.max_per_ds = {}
        self.kd_loss = {}
        self.path2stream = {}
        self.path2prefix = {}
        self.query_only = set()
        self.global_batch_size = global_batch_size
        self.rng = random.Random(seed)
        self.add_eos = add_eos
        self.add_prefix = add_prefix
        self.num_negatives = num_negatives
        self.download_locally = download_locally
        self.process_one_shard = process_one_shard
        self.current_shard = None
        self.weighted_sampling = weighted_sampling
        self.verbose = verbose
        self.infinite = infinite
        self.sample_negatives = sample_negatives
        self.run_name = run_name 

        if query_max_length is not None and document_max_length is not None:
            self.col_max_length = {"query": query_max_length, "document": document_max_length, "negative": document_max_length} 
        else:
            self.col_max_length = DEFAULT_COL_TO_MAX_TOKENS

        if dist.is_initialized():
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

        self.rank_batch_size = self.global_batch_size // self.world_size

        # defaults to s3
        # when `parse_spec` is called, if this is not s3, it will be updated
        self.filesystem = "s3"
        self.fs = fsspec.filesystem(self.filesystem, config_kwargs={"connect_timeout": 600, "read_timeout": 600})
        self.ds_paths = self.parse_spec(ds_spec)
        self.current_paths = [path for path in self.ds_paths]

        print_rank_zero(f"Total samples: {self.total_samples:,}")
        print_rank_zero("Number of samples per dataset")
        print_rank_zero(json.dumps(self.max_per_ds, indent=3))
        print_rank_zero("Number of examples per shards")
        print_rank_zero(json.dumps(self.num_samples_per_shard, indent=3))
        print_rank_zero("Max samples per shard per rank")
        print_rank_zero(json.dumps(self.max_per_shard, indent=3))
        self.tokenizer = tokenizer

        path = ds_spec.replace(".yaml", "")
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        self.path = f"{path}/rank_{self.rank}_processed_{self.run_name}.json"
        with open(self.path, "w") as f:
            json.dump({path: 0 for path in self.ds_paths}, f, indent=3)

        if self.weighted_sampling:
            self.weights = self.calculate_weights()
            print_rank_zero("Weighting per file")
            print_rank_zero(json.dumps(self.weights, indent=3))

        self.path2stream = {}

    def normalize_url(self, urls):
        norm_urls = []
        for url in urls:
            # only keep the last two parts of the path
            # this assumes the path is in the form s3://bucket/dataset/file.jsonl.gz
            # UGH
            split = url.split("/")
            if len(split) >= 6:
                normed = "/".join(split[-4:])
            else:
                normed = "/".join(split[-3:])
            norm_urls.append(normed)

        return norm_urls

    def parse_spec(self, fname):
        with open(fname) as stream:
            spec = yaml.safe_load(stream)

        paths = []
        for ds in spec["datasets"]:
            assert set(ds.keys()).issubset(
                set("name bucket objective weight kd_loss query_only query_prefix document_prefix".split())
            ), list(ds.keys())
            bucket = ds["bucket"]
            # TODO: we can probably remove the webdataset dependency
            urls = wds.shardlists.expand_urls(bucket)

            # we don't normalize the urls since we need the full path to stream from s3
            paths.extend(urls)
            if self.filesystem == "s3":
                if not all(url.startswith("s3://") for url in urls):
                    self.filesystem = "file"
                    self.fs = fsspec.filesystem(self.filesystem)

            bucket = "/".join(ds["bucket"].split("/")[:-1])
            with self.fs.open(f"{bucket}/counts.json", "r") as stream:
                counts_per_file = json.load(stream)

            # annoying backwards compatability
            if "count_per_file" in counts_per_file:
                counts_per_file = counts_per_file["count_per_file"]

            # normalize urls for counts 
            counts_per_file = {url.replace("s3://", ""): count for url, count in counts_per_file.items()}
            with self.fs.open(f"{bucket}/offsets.json.gz", "rb", compression="gzip") as stream:
                offsets = json.load(stream)
            offsets = {url.replace("s3://", ""): offset for url, offset in offsets.items()}

            tmp_dir = Path(f"/tmp/{bucket.replace('s3://', '')}")
            if not tmp_dir.exists():
                tmp_dir.mkdir(parents=True, exist_ok=True)

            memmap = MemoryMappedDict(str(tmp_dir / f"offsets_{self.rank}_{self.run_name}.mmap"))
            memmap.save_dict(offsets)
            bucket2offsets = {bucket.replace("s3://", ""): memmap}

            self.path2offsets = {**self.path2offsets, **bucket2offsets}

            # edge case where we don't use all files in the bucket
            normalized_urls = self.normalize_url(urls)
            self.path2objective.update({url: ds["objective"] for url in normalized_urls})

            present_files = {url: counts_per_file.get(url.replace("s3://", ""), 0) for url in normalized_urls}
            max_per_file = {
                url: (
                    int(counts_per_file.get(url.replace("s3://", ""), 0) / self.world_size / (self.rank_batch_size))
                    * self.rank_batch_size
                )
                for url in normalized_urls
            }

            to_remove = []
            for file, count in max_per_file.items():
                if count == 0:
                    to_remove.append(file)

            for file in to_remove:
                del max_per_file[file]
                del present_files[file]
                paths.remove(f"s3://{file}")

            if sum(max_per_file.values()) == 0:
                print_rank_zero(f"WARNING!!!: No data for {ds['name']} with {ds['bucket']} and {ds['objective']}")

            self.max_per_ds[ds["name"]] = sum(max_per_file.values()) * self.world_size
            self.total_samples += sum(max_per_file.values()) * self.world_size
            self.num_samples_per_shard = {**self.num_samples_per_shard, **present_files}
            self.max_per_shard = {**self.max_per_shard, **max_per_file}
            self.kd_loss.update({url: ds.get("kd_loss", False) for url in urls})

            query_only = ds.get("query_only", False)
            if query_only:
                ds_name = Path(ds["bucket"]).parent.name
                self.query_only.add(ds_name)

            if ds.get("query_prefix", None):
                ds_name = Path(ds["bucket"]).parent.name
                path2prefix = {
                    ds_name: {"query": ds["query_prefix"], "document": ds.get("document_prefix", ds["query_prefix"])}
                }
                if self.num_negatives > 0:
                    path2prefix[ds_name]["negative"] = ds.get("document_prefix", ds["query_prefix"])
                self.path2prefix.update(path2prefix)

        return paths

    def load_state(self, path):
        """
        1. Read from saved files
        2. rewrite current_processed to `self.path
        3. remove all that have been processed from self.ds_paths
        """
        with open(f"{path}/rank_{self.rank}_processed.json") as f:
            processed = json.load(f)

        # overwrite the current processed
        with open(self.path, "w") as f:
            json.dump(processed, f, indent=3)

        current_paths = []
        # current paths is what we sample from so we need to remove all files that we have exceeded
        # BUT don't update ds_paths as it stores the original full list of paths
        for path in self.ds_paths:
            if processed[path] >= self.max_per_shard[path.replace("s3://", "")]:
                print(
                    f"Rank: {self.rank} has already processed {processed[path]} samples from {path}, removing from paths"
                )
            else:
                current_paths.append(path)

        self.current_paths = current_paths

    def __len__(self):
        return self.total_samples

    def __iter__(self):
        while True:
            while self.current_paths:
                stream, path = self.get_next_stream()
                normalized_path = self.normalize_url([path])[0]
                if stream is None:
                    break

                batch = []
                # Fetching batch_size lines to form a batch
                for item in stream:
                    batch.append(item)
                    if len(batch) >= self.rank_batch_size:
                        break

                # rough approximation, this won't work if the shard runs out of data
                current_processed = json.load(open(self.path))
                current_processed[path] += len(batch)
                with open(self.path, "w") as f:
                    json.dump(current_processed, f, indent=3)

                if current_processed[path] >= self.max_per_shard[normalized_path]:
                    if self.verbose:
                        print_rank_zero(f"Finished processing {path}")
                    self.current_paths.remove(path)
                    del self.path2stream[path]
                    # if we only process one shard at a time
                    if self.process_one_shard:
                        self.current_shard = None

                # after the rounding down above, this should never happen
                if len(batch) < self.rank_batch_size:
                    raise ValueError(
                        f"Batch size {len(batch)} is too small, something went wrong on rank {self.rank} for path {path}"
                    )

                batch = self.tokenize_pairs(batch, self.path2objective[normalized_path])

                yield batch

                if self.weighted_sampling:
                    self.weights = self.calculate_weights()
                    if self.verbose:
                        print_rank_zero("Weighting per file")
                        print_rank_zero(json.dumps(self.weights, indent=3))

            if not self.infinite:
                break

            if self.verbose:
                print_rank_zero("Finished all shards, resetting")

            self.current_paths = [path for path in self.ds_paths]
            with open(self.path, "w") as f:
                json.dump({path: 0 for path in self.current_paths}, f, indent=3)

    def calculate_weights(self):
        total_size = sum(self.num_samples_per_shard.values())

        already_processed = json.load(open(self.path))

        weights = {}
        for file, size in self.num_samples_per_shard.items():
            remaining_size = size - already_processed[f"s3://{file}"] * self.world_size
            weights[f"s3://{file}"] = remaining_size / total_size

        return weights

    def get_next_stream(self):
        if not self.current_paths:
            return None, None

        if self.process_one_shard:
            if self.current_shard is None:
                self.current_shard = self.rng.choice(self.current_paths)
            next_uri = self.current_shard
        else:
            # Randomly select the next S3 URI
            if self.weighted_sampling:
                current_weights = [self.weights[path] for path in self.current_paths]
                next_uri = self.rng.choices(self.current_paths, weights=current_weights, k=1)[0]
            else:
                next_uri = self.rng.choice(self.current_paths)

        current_processed = json.load(open(self.path))
        # total number of samples processed by all ranks, this marks the start of the next rank's batch
        num_processed = current_processed[next_uri] * self.world_size

        if self.verbose:
            print_in_order(f"Rank: {self.rank} is processing {next_uri}")
        return self.create_stream(next_uri, num_processed), next_uri

    def create_stream(self, path, num_processed):
        # set cache to not read full file into memory
        # hopefully this doesn't timeout or raise an error like before
        if self.download_locally:
            local_path = self.download_rank_zero(path)
            if path not in self.path2stream:
                self.path2stream[path] = gzip.open(local_path, "rb")
            stream = self.path2stream[path]
        else:
            stream = self.fs.open(path, "rb", compression="gzip", cache_type="background", block_size=2**20)

        # get offset for the rank since we read in `rank_batch_size` chunks
        rank_processed = num_processed + self.rank * self.rank_batch_size

        normalized_path = self.normalize_url([path])[0]
        bucket = "/".join(path.split("/")[:-1]).replace("s3://", "")

        memmap = self.path2offsets[bucket]
        offsets = memmap.get(normalized_path)
        seek_to = offsets[str(rank_processed)][0]

        # seek to current offset
        if stream.tell() != seek_to:
            print_rank_zero(f"Seeking to offset {seek_to}, at {stream.tell()}, {num_processed=}")
            stream.seek(seek_to)

        objective = self.path2objective[normalized_path]

        return self.jsonl_iterator(stream, objective, path, rank_processed, offsets)

    def download_rank_zero(self, s3_path):
        # only download if we are rank 0 otherwise wait for rank 0 to download
        # TODO: fix for multinode, need to update the if statement
        local_path = Path(f"/tmp/{self.normalize_url([s3_path])[0]}")
        if self.local_rank == 0:
            if not local_path.parent.exists():
                local_path.parent.mkdir(parents=True, exist_ok=True)
            if not local_path.exists():
                self.fs.get(s3_path, str(local_path))

        dist.barrier()

        return local_path

    def jsonl_iterator(self, fileobj, objective, path, num_processed, offsets, handler=log_and_continue):
        # returns generator of jsonl lines
        stream = fileobj
        # offset num_processed by local_rank so we read different parts of the file
        try:
            for i in range(num_processed, len(offsets)):
                fname = path
                start, end = offsets[str(i)]
                obj = stream.read(end - start)
                data = json.loads(obj.decode())
                scores = self.extract_kd_scores(data, path)
                data = self.group_by_keys_nothrow(data, fname)
                data = self.extract_pair(data, objective)
                if scores:
                    data["kd_scores"] = scores
                yield data
                stream.members = []
        except Exception as e:
            print("Error in jsonl_iterator for fileobj: ", fname)
            print(obj)
            print(f"start: {start}, end: {end}")
            print(e)
        del stream

    def group_by_keys_nothrow(self, filesample, fname, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
        """Return function over iterator that groups key, value pairs into samples.

        :param keys: function that splits the key into key and extension (base_plus_ext)
        :param lcase: convert suffixes to lower case (Default value = True)
        """
        current_sample = None
        assert isinstance(filesample, dict)
        value = filesample
        prefix, suffix = keys(fname)
        if lcase:
            suffix = suffix.lower()

        current_sample = dict(__key__=prefix, __url__=fname, file_type=suffix)
        current_sample[suffix] = value

        return current_sample

    def extract_pair(self, sample, objective_dict):
        """Extracts the relevant pairs from a sample using the metadata provided

        yields:
            dict of sample with the data mapped to `query` and `document`
        """
        data = sample[sample["file_type"]]
        metadata = data["metadata"]["objective"]
        contrastive_type = objective_dict["type"]
        columns = objective_dict["columns"]

        valid_columns = metadata[contrastive_type]
        assert (
            columns in valid_columns
        ), f"Invalid columns {columns} for contrastive type {contrastive_type}. Valid columns are {valid_columns}"

        paired_data = {}
        # pop negatives into document and sample N
        for mapped_name, col in zip(MAPPED_NAMES[contrastive_type], columns):
            if mapped_name == "negative":
                if len(data[col]) <= self.num_negatives:
                    negatives = data[col]
                else:
                    if self.sample_negatives:
                        negatives = random.sample(data[col], self.num_negatives)
                    else:
                        negatives = data[col][: self.num_negatives]
                paired_data["document"] = [paired_data["document"]] + negatives
            else:
                paired_data[mapped_name] = data[col]

        paired_data["__key__"] = sample["__key__"] + "." + sample["file_type"]
        return paired_data

    def extract_kd_scores(self, data, path):
        if self.kd_loss[path] is False:
            return None

        pos_score = data["document_score"]
        negative_scores = data["negatives_scores"]
        scores = [pos_score] + negative_scores[: self.num_negatives]

        return scores

    def tokenize_pairs(self, samples, objective):
        tokenized_inputs = {}
        contrastive_type = objective["type"]
        s3_path = samples[0]["__key__"]
        dataset_name = s3_path.split("/")[-2]
        if "mc4" in s3_path:
            dataset_name = f"mc4_{dataset_name}"
        elif "multilingual-cc-news" in s3_path:
            dataset_name = f"cc_news_{dataset_name}"

        tokenized_inputs["dataset_name"] = dataset_name
        for col in MAPPED_NAMES[contrastive_type]:
            # this is stored in `document` now
            if col == "negative":
                continue

            if self.add_eos:
                if isinstance(samples[0][col], list):
                    for sample in samples:
                        sample[col] = [text + self.tokenizer.eos_token for text in sample[col]]
                    collected = [sample[col] for sample in samples]

                else:
                    collected = [sample[col] + self.tokenizer.eos_token for sample in samples]
            else:
                collected = [sample[col] for sample in samples]

            # check if list, if so, flatten
            if isinstance(collected[0], list):
                collected = sum(collected, [])

            if self.add_prefix:
                # add either "query: " or "document: " to the beginning of the text
                if dataset_name in self.query_only and col != "query":
                    collected = [text for text in collected]
                else:
                    if dataset_name in self.path2prefix:
                        prefix = self.path2prefix[dataset_name][col]
                    elif dataset_name in self.query_only:
                        prefix = "query"
                    else:
                        prefix = KEY2PREFIX[col]

                    collected = [f"{prefix}: {text}" for text in collected]

            tokenized = self.tokenizer(collected, padding="max_length", truncation=True, return_tensors="pt", max_length=self.col_max_length[col])
            # if text gets truncated, we want to make sure the last token is the eos token
            # attention mask will already be full of 1s so we don't need to update
            # if text doesn't get truncated, the eos token will be the last token
            if self.add_eos:
                tokenized["input_ids"][:, -1] = self.tokenizer.eos_token_id
            tokenized = {f"{col}_{k}": v for k, v in tokenized.items()}
            tokenized_inputs = {**tokenized_inputs, **tokenized}

        if "kd_scores" in samples[0]:
            tokenized_inputs["kd_scores"] = torch.tensor(
                [sample["kd_scores"] for sample in samples], dtype=torch.float32
            )

        return tokenized_inputs


class LocalShardDataset(Dataset):
    def __init__(self, ds_spec, num_negatives=0, seed=42):
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

        self.fs = fsspec.filesystem("s3", config_kwargs={"connect_timeout": 600, "read_timeout": 600})
        self.path2objective = {}
        self.path2prefix = {}
        self.query_only = set()
        self.num_negatives = num_negatives
        self.rng = random.Random(seed)

        paths = self.parse_spec(ds_spec)
        # probably could parallelize this
        self.local_paths = []
        for path in paths:
            self.local_paths.append(self.download_rank_zero(path))

        self.examples = self.load_examples(self.local_paths)

    def parse_spec(self, fname):
        with open(fname) as stream:
            spec = yaml.safe_load(stream)

        paths = []
        for ds in spec["datasets"]:
            assert set(ds.keys()).issubset(
                set("name bucket objective weight kd_loss query_only query_prefix document_prefix".split())
            ), list(ds.keys())
            bucket = ds["bucket"]
            urls = wds.shardlists.expand_urls(bucket)

            paths.extend(urls)
            self.path2objective.update({url.replace("s3://", ""): ds["objective"] for url in urls})
            query_only = ds.get("query_only", False)
            if query_only:
                ds_name = Path(ds["bucket"]).parent.name
                self.query_only.add(ds_name)

            if ds.get("query_prefix", None):
                ds_name = Path(ds["bucket"]).parent.name
                path2prefix = {
                    ds_name: {"query": ds["query_prefix"], "document": ds.get("document_prefix", ds["query_prefix"])}
                }
                if self.num_negatives > 0:
                    path2prefix[ds_name]["negative"] = ds.get("document_prefix", ds["query_prefix"])
                self.path2prefix.update(path2prefix)

        return paths

    def download_rank_zero(self, s3_path):
        # only download if we are rank 0 otherwise wait for rank 0 to download
        # TODO: fix for multinode, need to update the if statement
        local_path = Path(f"/tmp/{s3_path.replace('s3://', '')}")
        if self.rank == 0:
            if not local_path.parent.exists():
                local_path.parent.mkdir(parents=True, exist_ok=True)
            if not local_path.exists():
                self.fs.get(s3_path, str(local_path))

        dist.barrier()

        return local_path

    def load_examples(self, paths):
        examples = []
        for path in tqdm(paths, desc="Loading examples", disable=self.rank > 0):
            ds_name = Path(path).parent.name
            contrastive_type_metadata = self.path2objective[str(path).replace("/tmp/", "")]
            contrastive_type = contrastive_type_metadata["type"]
            columns = contrastive_type_metadata["columns"]
            table = read_json(path)
            for batch in table.to_batches():
                for line in batch.to_pylist():
                    metadata = line["metadata"]["objective"]
                    assert (
                        columns in metadata[contrastive_type]
                    ), f"Invalid columns {columns} for contrastive type {contrastive_type}. Valid columns are {metadata[contrastive_type]}"

                    mapped_data = {}
                    for mapped_name, col in zip(MAPPED_NAMES[contrastive_type], columns):
                        mapped_data[mapped_name] = line[col]

                    mapped_data["dataset_name"] = ds_name
                    examples.append(mapped_data)

        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        data = self.examples[index]
        if "negative" in data:
            negatives = data.pop("negative")
            data["document"] = [data["document"]] + random.sample(negatives, self.num_negatives)
        return data


def collate_local_ds(batch, tokenizer, add_prefix=False, query_only=None, path2prefix=None):
    # assume all records have same column names
    # assume dataset_name is in there
    ds_names = [sample.pop("dataset_name") for sample in batch]
    keys = batch[0].keys()
    tokenized_inputs = {}
    for col in keys:
        collected = [sample[col] for sample in batch]
        if isinstance(collected[0], list):
            collected = sum(collected, [])
            # in case we have multiple negatives, we need to repeat the dataset name N negative times
            ds_names = sum([[ds_name] * len(sample[col]) for ds_name, sample in zip(ds_names, batch)], [])

        if add_prefix:
            # add prefix to beginning of column
            if path2prefix:
                # path2prefix is a dict of {ds_name: {col: prefix}}
                prefixes = [path2prefix[ds_name][col] for ds_name in ds_names]
            else:
                prefix = KEY2PREFIX[col]
                if query_only and col != "query":
                    prefixes = []
                    for ds_name in ds_names:
                        if ds_name in query_only:
                            prefixes.append("query")
                        else:
                            prefixes.append(prefix)
                else:
                    prefixes = [prefix] * len(collected)

            collected = [f"{prefix}: {text}" for prefix, text in zip(prefixes, collected)]

        tokenized = tokenizer(collected, padding="max_length", truncation=True, return_tensors="pt")
        tokenized = {f"{col}_{k}": v for k, v in tokenized.items()}
        tokenized_inputs = {**tokenized_inputs, **tokenized}

    return tokenized_inputs


def get_local_dataloader(ds_spec, batch_size, tokenizer, num_negatives, seed, add_prefix, num_workers=0, epoch=0):
    dataset = LocalShardDataset(ds_spec, num_negatives=num_negatives, seed=seed)
    if dist.is_initialized():
        sampler = DistributedSampler(
            dataset, shuffle=True, num_replicas=dist.get_world_size(), rank=dist.get_rank(), seed=seed
        )
        sampler.set_epoch(epoch)
    else:
        sampler = None

    collate_fn = lambda x: collate_local_ds(
        x, tokenizer, add_prefix=add_prefix, query_only=dataset.query_only, path2prefix=dataset.path2prefix
    )

    dataloader = DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn, drop_last=True, num_workers=num_workers
    )

    return dataloader
