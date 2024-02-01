import glob
import json
import logging
import math
import os
import random
import re
from dataclasses import dataclass, field
from functools import partial
from multiprocessing import Value
from typing import Dict, List

import braceexpand
import fsspec
import torch
import torch.distributed as dist
import torchvision.datasets as datasets
import webdataset as wds
import yaml
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler
from transformers import DefaultDataCollator
from webdataset.filters import _shuffle
from webdataset.handlers import reraise_exception
from webdataset.tariterators import base_plus_ext, tar_file_expander, tar_file_iterator, url_opener, valid_sample

_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000
# How many samples we processed
LAION_PROCESSED = 330_056_886

_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000
COMMAND = "pipe: aws s3 cp --endpoint-url https://9fa58365a1a3d032127970d0bd9a1290.r2.cloudflarestorage.com/ --cli-read-timeout=300 {s3_uri} -"
MAPPED_NAMES = {"paired": ["query", "document"], "self": ["query"]}


# thank you to open_clip for the great code :)
# a lot of this is adapted from https://github.com/mlfoundations/open_clip


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)

    def __len__(self):
        return self.dataloader.num_batches


@dataclass
class DataSource:
    """Class representing a data source."""

    name: str = ""
    resample: bool = False
    urls: List[str] = field(default_factory=list)
    weight: float = 1.0
    objective: Dict[str, List[str]] = None
    perepoch: int = 0


class MultiShardSample(IterableDataset):
    def __init__(self, fname, seed=None, deterministic=True, epoch=-1, state=None):
        """Construct a shardlist from multiple sources using a YAML spec."""
        self.epoch = -1
        self.rng = random.Random()
        self.deterministic = deterministic
        self.epoch = epoch
        self.total_samples = 0
        self.num_samples_per_shard = {}

        self.parse_spec(fname)
        self.memory = {}
        if state is not None:
            self.load_state(state)
        data_mix = self.num_samples_per_shard
        data_mix = {k: v - self.memory.get(k, 0) for k, v in data_mix.items()}
        print(f"Rank: {dist.get_rank()} Data Mix:\n{json.dumps(data_mix, indent=4)}")
        # set seed so ordering is the same across nodes/gpus
        self.set_epoch(seed)

    def parse_spec(self, fname):
        with open(fname) as stream:
            spec = yaml.safe_load(stream)

        self.sources = []
        fs = fsspec.filesystem("s3")
        for ds in spec["datasets"]:
            assert set(ds.keys()).issubset(set("name bucket objective weight".split())), list(ds.keys())
            bucket = ds["bucket"]
            urls = wds.shardlists.expand_urls(bucket)
            weight = ds.get("weight", 1.0)
            name = ds.get("name", "")
            objective = ds.get("objective", None)

            entry = DataSource(name=name, urls=urls, weight=weight, objective=objective)
            self.sources.append(entry)

            bucket = "/".join(ds["bucket"].split("/")[:-1])
            with fs.open(f"{bucket}/counts.json", "r") as stream:
                counts = json.load(stream)

            counts_per_file = counts["count_per_file"]
            # edge case where we don't use all files in the bucket
            present_files = {url: counts_per_file[url.strip("s3://")] for url in urls}

            self.total_samples += sum(present_files.values())
            self.num_samples_per_shard = {**self.num_samples_per_shard, **present_files}

    def load_state(self, checkpoint_dir):
        file = f"{checkpoint_dir}/rank_{dist.get_rank()}_processed.json"
        with open(file, "r") as f:
            self.memory = json.load(f)

        for dataset in self.sources:
            for url in dataset.urls:
                processed = self.memory.get(url, 0)
                total_in_shard = self.num_samples_per_shard[url]
                if processed >= total_in_shard:
                    print(f"Removing {url} from dataset")
                    dataset.urls.remove(url)
                    del self.num_samples_per_shard[url]

        all_ranks = glob.glob(f"{checkpoint_dir}/rank_*.json")
        for file in all_ranks:
            with open(file, "r") as f:
                memory = json.load(f)
            for _, count in memory.items():
                self.total_samples -= count

    def set_epoch(self, seed):
        """Set the current epoch (for consistent shard selection among nodes)."""
        # shuffle different only epochs
        seed += self.epoch.get_value()
        self.rng = random.Random(seed)

    def get_shards_for_epoch(self):
        result = []

        # TODO: how can we do this with a weighted choice for many shards?
        for source in self.sources:
            objective = source.objective
            if source.resample > 0:
                # sample with replacement
                l = self.rng.choices(source.urls, k=source.resample)
            elif source.perepoch > 0:
                # sample without replacement
                l = list(source.urls)
                self.rng.shuffle(l)
                l = l[: source.perepoch]
            else:
                l = list(source.urls)
            result.extend([(url, objective) for url in l])

        self.rng.shuffle(result)

        return result

    def __iter__(self):
        shards = self.get_shards_for_epoch()
        for shard, objective in shards:
            num_seen = self.memory.get(shard, 0)
            # skip shard if already processed
            if num_seen >= self.num_samples_per_shard[shard]:
                continue
            yield dict(url=shard, objective=objective, offset=num_seen)

        # once iterating through all shards, reset memory
        self.memory = {}


def expand_urls(urls, weights=None):
    if weights is None:
        expanded_urls = wds.shardlists.expand_urls(urls)
        return expanded_urls, None
    if isinstance(urls, str):
        urllist = urls.split("::")
        weights = weights.split('::')
        assert len(weights) == len(
            urllist
        ), f"Expected the number of data components ({len(urllist)}) and weights({len(weights)}) to match."
        weights = [float(weight) for weight in weights]
        all_urls, all_weights = [], []
        for url, weight in zip(urllist, weights):
            expanded_url = list(braceexpand.braceexpand(url))
            expanded_weights = [weight for _ in expanded_url]
            all_urls.extend(expanded_url)
            all_weights.extend(expanded_weights)
        return all_urls, all_weights
    else:
        all_urls = list(urls)
        return all_urls, weights


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True


def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"], file_type=filesample["file_type"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def buffer_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    # adapted from https://github.com/mlfoundations/open_clip/blob/f190703d847b7b234dbeb8265d7a69d7e7e4e996/src/training/data.py#L214
    # to allow reading of jsonl files
    streams = url_opener(src, handler=handler)
    files = file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def file_expander(data, handler=reraise_exception, select_files=None, rename_files=None):
    """For each sample, iterate over the opened stream and yield samples


    yields:
        dict of samples from the stream

    """
    for source in data:
        url = source["url"]
        file_type = source["file_type"]
        try:
            if file_type == "jsonl":
                objective = source["objective"]
                offset = source["offset"]
                for i, sample in enumerate(
                    jsonl_file_iterator(
                        source["stream"], handler=handler, select_files=select_files, rename_files=rename_files
                    )
                ):
                    # skip samples that have already been processed
                    if i < offset:
                        continue
                    sample["__url__"] = url
                    sample["data"]["curr_objective"] = objective
                    sample["file_type"] = file_type
                    yield sample
            elif file_type == "tar":
                for sample in tar_file_iterator(
                    source["stream"], handler=handler, select_files=select_files, rename_files=rename_files
                ):
                    sample["__url__"] = url
                    sample["file_type"] = file_type
                    yield sample
            else:
                raise ValueError(f"Unknown file type {file_type} for url {url}")
        except Exception as exn:
            exn.args = exn.args + (source.get("stream"), source.get("url"))
            if handler(exn):
                continue
            else:
                break


def jsonl_file_iterator(
    fileobj, handler=reraise_exception, skip_meta=r"__[^/]*__($|/)", select_files=None, rename_files=None
):

    stream = fileobj.stream
    for obj in stream:
        try:
            fname = re.search(r"s3:\/\/[^\/\s]+\/[^\/\s]+\/(shard)-\d{5}\.jsonl", fileobj.args[0][0].strip()).group(0)
            data = json.loads(obj.decode("utf-8"))
            result = dict(fname=fname, data=data)
            yield result
            stream.members = []
        except Exception as exn:
            if hasattr(exn, "args") and len(exn.args) > 0:
                exn.args = (exn.args[0] + " @ " + str(fileobj),) + exn.args[1:]
            if handler(exn):
                continue
            else:
                break
    del stream


def pytorch_worker_seed(increment=0):
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour using the seed already created for pytorch dataloader workers if it exists
        seed = worker_info.seed
        if increment:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment * max(1, worker_info.num_workers)
        return seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


def extract_pairs(samples):
    """Extracts the relevant pairs from a sample using the metadata provided

    yields:
        dict of sample with the data mapped to `query` and `document`
    """
    for sample in samples:
        data = sample[sample["file_type"]]
        objective = data["curr_objective"]
        assert len(objective) == 1, f"Expected only one objective, got {objective}"
        metadata = data["metadata"]["objective"]
        for contrastive_type, columns in objective.items():
            valid_columns = metadata[contrastive_type]
            assert (
                columns in valid_columns
            ), f"Invalid columns {columns} for contrastive type {contrastive_type}. Valid columns are {valid_columns}"

        paired_data = {}
        for mapped_name, col in zip(MAPPED_NAMES[contrastive_type], columns):
            paired_data[mapped_name] = data[col]

        paired_data["__key__"] = sample["__key__"] + "." + sample["file_type"]
        yield paired_data


def format_file_type(shards):
    # NOTE: assuming this is just a file path (s3 or local)
    for shard in shards:
        updated_shard = {k: v for k, v in shard.items()}
        url = shard["url"]
        if url.endswith(".tar"):
            updated_shard["file_type"] = "tar"
        elif url.endswith("jsonl"):
            updated_shard["file_type"] = "jsonl"
        else:
            raise ValueError(f"Unknown file type for url {url}")

        command = COMMAND.format(s3_uri=url)
        updated_shard["url"] = command

        yield updated_shard


def text_collate_fn(batches):
    """Collates batches of text where the inputs are query_input_ids, document_input_ids, query_attention_mask, document_attention_mask

    NOTE: this assumes the batches have all the same keys
    args:
        batches: List of dicts with tokenized inputs and attention masks

    returns:
        Dict of shape N x 2 x L where N is the number of batches and L is the length of the input
        with keys `input_ids` and `attention_mask`

    """

    text_inputs = {}
    for col in ["input_ids", "attention_mask"]:
        for prefix in ["query", "document"]:
            text_inputs[f"{prefix}_{col}"] = torch.stack([batch[f"{prefix}_{col}"].squeeze() for batch in batches])

    text_inputs["__key__"] = [batch["__key__"] for batch in batches]

    return text_inputs


def tokenize_pairs(samples, tokenizer=None):
    for sample in samples:
        for col in ["query", "document"]:
            if col in sample:
                tokenized = tokenizer(
                    sample[col] + tokenizer.eos_token, padding="max_length", truncation=True, return_tensors="pt"
                )
                # if text gets truncated, we want to make sure the last token is the eos token
                # attention mask will already be full of 1s so we don't need to update
                # if text doesn't get truncated, the eos token will be the last token
                tokenized["input_ids"][:, -1] = tokenizer.eos_token_id
                tokenized = {f"{col}_{k}": v for k, v in tokenized.items()}

                sample = {**sample, **tokenized}
        yield sample


def filter_missing_pair(sample):
    data = sample[sample["file_type"]]
    objective = data["curr_objective"]
    metadata = data["metadata"]["objective"]

    for contrastive_type, columns in objective.items():
        valid_columns = metadata[contrastive_type]
        if columns not in valid_columns:
            print(f"Filtering: {sample}")
            return False

        for col in columns:
            if col not in data:
                print(f"Filtering: {sample}")
                return False
            if data[col] is None or data[col] == "":
                print(f"Filtering: {sample}")
                return False

    return True


class ShardMemory:
    def __init__(self, num_samples_per_shard, path=None):
        self.num_samples_per_shard = num_samples_per_shard
        self.memory = {k: 0 for k in num_samples_per_shard.keys()}
        self.path = path

    def update(self, keys):
        counts = {}
        for key in keys:
            if key not in counts:
                counts[key] = 1
            else:
                counts[key] += 1

        for key, count in counts.items():
            self.memory[key] += count

        with open(self.path, "w") as f:
            json.dump(self.memory, f, indent=4)

    def save_state(self, path):
        # load currently processed data
        with open(self.path, "r") as f:
            memory = json.load(f)
        # save to new checkpoint path
        # this file will be loaded from MultiShard Dataloader
        # so we can filter out files we've already seen
        file = f"{path}/rank_{dist.get_rank()}_processed.json"
        with open(file, "w") as f:
            json.dump(memory, f, indent=4)


class MemoryDataPipeline(wds.DataPipeline):
    def __init__(self, *args, num_samples_per_shard, path=None, **kwargs):
        super().__init__(*args, **kwargs)
        path = path.replace(".yaml", "")
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        shard_mem = ShardMemory(num_samples_per_shard, path=f"{path}/rank_{dist.get_rank()}_processed.json")
        self.shard_memory = shard_mem

    def iterator(self):
        """Create an iterator through the entire dataset, using the given number of repetitions."""
        for _ in range(self.repetitions):
            for sample in self.iterator1():
                keys = sample.pop("__key__")
                self.shard_memory.update(keys)
                yield sample

    def save_state(self, path):
        self.shard_memory.save_state(path)


def get_wds_text_dataset(args, is_train, tokenizer, epoch=0, floor=False, state=None):
    input_shards = args.input_shards
    assert input_shards is not None
    resampled = getattr(args, 'dataset_resampled', False) and is_train

    shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc

    pipeline = [MultiShardSample(input_shards, epoch=shared_epoch, seed=args.seed, state=state)]
    pipeline.extend([format_file_type])

    # at this point we have an iterator over all the shards
    if is_train:
        if not resampled:
            pipeline.extend(
                [
                    wds.split_by_node,
                    wds.split_by_worker,
                ]
            )
        pipeline.extend(
            [
                # at this point, we have an iterator over the shards assigned to each worker at each node
                buffer_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
            ]
        )
        if args.shuffle:
            pipeline.extend(
                [
                    wds.shuffle(
                        bufsize=_SAMPLE_SHUFFLE_SIZE,
                        initial=_SAMPLE_SHUFFLE_INITIAL,
                    ),
                ]
            )
    else:
        pipeline.extend(
            [
                wds.split_by_worker,
                # at this point, we have an iterator over the shards assigned to each worker
                buffer_to_samples_nothrow,
            ]
        )
    pipeline.extend(
        [
            wds.select(filter_missing_pair),
            extract_pairs,
            partial(tokenize_pairs, tokenizer=tokenizer),
            # partial is a flag if we want to return partial batches
            wds.batched(args.batch_size, partial=not is_train, collation_fn=text_collate_fn),
        ]
    )

    # keep track of the number of samples seen per shard
    dataset = MemoryDataPipeline(
        *pipeline, num_samples_per_shard=pipeline[0].num_samples_per_shard, path=args.input_shards
    )

    if is_train:
        num_samples = pipeline[0].total_samples
        print(f"num_samples: {num_samples}")
        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if floor else math.ceil
        global_batch_size = args.batch_size * dist.get_world_size()
        num_batches = round_fn(num_samples / global_batch_size)
        print(f"num_batches: {num_batches}")
        num_workers = max(1, args.workers)
        num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
        print(f"rounded num_worker_batches: {num_worker_batches}")
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        print(f"rounded num_samples: {num_samples}")
        dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
    else:
        num_samples = None
        num_batches = None

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )

    # FIXME not clear which approach is better, with_epoch before vs after dataloader?
    # hoping to resolve via https://github.com/webdataset/webdataset/issues/169
    # if is_train:
    #     # roll over and repeat a few samples to get same number of full batches on each node
    #     global_batch_size = args.batch_size * args.world_size
    #     num_batches = math.ceil(num_samples / global_batch_size)
    #     num_workers = max(1, args.workers)
    #     num_batches = math.ceil(num_batches / num_workers) * num_workers
    #     num_samples = num_batches * global_batch_size
    #     dataloader = dataloader.with_epoch(num_batches)
    # else:
    #     # last batches are partial, eval is done on single (master) node
    #     num_batches = math.ceil(num_samples / args.batch_size)

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


def filter_no_caption_or_no_image(sample):
    has_caption = 'txt' in sample
    has_image = 'png' in sample or 'jpg' in sample or 'jpeg' in sample or 'webp' in sample
    empty_caption = has_caption and (
        len(sample['txt'].strip()) < 2 or sample['txt'].strip() == b'' or sample['txt'].strip() == b'"'
    )
    if empty_caption:
        print(sample["txt"])
    return has_caption and has_image and not empty_caption


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True


def pytorch_worker_seed(increment=0):
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour using the seed already created for pytorch dataloader workers if it exists
        seed = worker_info.seed
        if increment:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment * max(1, worker_info.num_workers)
        return seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


class detshuffle2(wds.PipelineStage):
    def __init__(
        self,
        bufsize=1000,
        initial=100,
        seed=0,
        epoch=-1,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            # If seed is negative, we use the worker's seed, this will be different across all nodes/workers
            seed = pytorch_worker_seed(epoch)
        else:
            # This seed to be deterministic AND the same across all nodes/workers in each epoch
            seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


def image_text_collate_fn(batch, return_metadata=False):
    collator = DefaultDataCollator()
    image_inputs = collator([{"input_ids": sample[0]} for sample in batch])
    text_samples = [sample[1] for sample in batch]
    text_inputs = {}
    for k in text_samples[0]:
        text_inputs[k] = torch.stack([sample[k].squeeze() for sample in text_samples])

    if len(batch[0]) > 2 and return_metadata:
        json = [sample[2] for sample in batch]

        return {"vision": image_inputs, "text": text_inputs, "metadata": json}
    else:
        return {"vision": image_inputs, "text": text_inputs}


def tokenize(text, tokenizer, add_eos=False, add_prefix=False):
    if add_eos:
        text = text + tokenizer.eos_token
    if add_prefix:
        text = f"image_search: {text}"
    tokenized = tokenizer(text, padding="max_length", truncation=True, return_tensors="pt")
    if add_eos:
        tokenized["input_ids"][:, -1] = tokenizer.eos_token_id

    return tokenized


def get_wds_image_text_dataset(
    args, transforms, is_train, tokenizer, epoch=0, floor=False, add_eos=False, add_prefix=False
):
    input_shards = args.image_text_shards
    assert input_shards is not None
    resampled = getattr(args, 'dataset_resampled', False) and is_train

    num_shards = None
    if is_train:
        if args.train_num_samples is not None:
            num_samples = args.train_num_samples
        else:
            num_samples, num_shards = LAION_PROCESSED, len(input_shards)
            if not num_samples:
                raise RuntimeError(
                    'Currently, the number of dataset samples must be specified for the training dataset. '
                    'Please specify it via `--train-num-samples` if no dataset length info is present.'
                )
    else:
        # Eval will just exhaust the iterator if the size is not specified.
        num_samples = args.val_num_samples or 0

    shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc

    pipeline = [wds.SimpleShardList(input_shards)]
    pipeline.extend([format_file_type])

    # at this point we have an iterator over all the shards
    if is_train:
        pipeline.extend(
            [
                detshuffle2(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=args.seed,
                    epoch=shared_epoch,
                ),
                wds.split_by_worker,
            ]
        )
        pipeline.extend(
            [
                # at this point, we have an iterator over the shards assigned to each worker at each node
                buffer_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
                wds.shuffle(
                    bufsize=_SAMPLE_SHUFFLE_SIZE,
                    initial=_SAMPLE_SHUFFLE_INITIAL,
                ),
            ]
        )
    else:
        pipeline.extend(
            [
                wds.split_by_worker,
                # at this point, we have an iterator over the shards assigned to each worker
                buffer_to_samples_nothrow,
            ]
        )
    pipeline.extend(
        [
            wds.select(filter_no_caption_or_no_image),
            # allow for non-bytes keys to be passed through
            wds.decode("pilrgb", handler=log_and_continue, partial=True),
            wds.rename(image="jpg;png;jpeg;webp", text="txt"),
            # TODO: if we ever use the imagebind tokenizer we need to change this
            # just using HF tokenizer for now
            wds.map_dict(
                image=transforms,
                text=partial(tokenize, tokenizer=tokenizer, add_eos=add_eos, add_prefix=add_prefix),
            ),
            wds.to_tuple("image", "text", "json"),
            # partial is a flag if we want to return partial batches
            wds.batched(args.batch_size, partial=not is_train, collation_fn=image_text_collate_fn),
        ]
    )

    dataset = wds.DataPipeline(*pipeline)

    if is_train:
        if not resampled:
            num_shards = num_shards or len(expand_urls(input_shards)[0])
            assert num_shards >= args.workers * dist.get_world_size(), 'number of shards must be >= total workers'
        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if floor else math.ceil
        global_batch_size = args.batch_size * dist.get_world_size()
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args.batch_size)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )

    # FIXME not clear which approach is better, with_epoch before vs after dataloader?
    # hoping to resolve via https://github.com/webdataset/webdataset/issues/169
    # if is_train:
    #     # roll over and repeat a few samples to get same number of full batches on each node
    #     global_batch_size = args.batch_size * args.world_size
    #     num_batches = math.ceil(num_samples / global_batch_size)
    #     num_workers = max(1, args.workers)
    #     num_batches = math.ceil(num_batches / num_workers) * num_workers
    #     num_samples = num_batches * global_batch_size
    #     dataloader = dataloader.with_epoch(num_batches)
    # else:
    #     # last batches are partial, eval is done on single (master) node
    #     num_batches = math.ceil(num_samples / args.batch_size)

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


class ImageNetDataset(datasets.ImageFolder):
    def __init__(self, data_path, transform):
        super().__init__(data_path, transform=transform)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return path, sample, target


def get_imagenet(args, transforms):
    data_path = args.imagenet_val_path
    assert data_path

    dataset = ImageNetDataset(data_path, transform=transforms)

    sampler = DistributedSampler(dataset, shuffle=False)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=sampler,
    )

    return DataInfo(dataloader=dataloader, sampler=sampler)
