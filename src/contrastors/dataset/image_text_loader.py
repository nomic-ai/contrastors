import logging
import math
import random
import sys
from dataclasses import dataclass, field
from functools import partial
from multiprocessing import Value
from typing import Dict, List

import braceexpand
import numpy as np
import torch
import torch.distributed as dist
import torchvision.datasets as datasets
import webdataset as wds
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler
from transformers import BatchEncoding, DataCollatorForLanguageModeling, DefaultDataCollator
from webdataset.filters import _shuffle
from webdataset.handlers import reraise_exception
from webdataset.tariterators import base_plus_ext, tar_file_iterator, url_opener, valid_sample

_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000
# How many samples we processed
LAION_PROCESSED = 330_056_886
DATACOMP_1B_PROCESSED = 1_173_491_100
DFN_2B_PROCESSED = 1_547_277_668


name2samples = {
    "laion400m": LAION_PROCESSED,
    "datacomp": DATACOMP_1B_PROCESSED,
    "dfn-datacomp-xlarge": DFN_2B_PROCESSED,
}

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
            for sample in tar_file_iterator(
                source["stream"], handler=handler, select_files=select_files, rename_files=rename_files
            ):
                sample["__url__"] = url
                sample["file_type"] = file_type
                yield sample
        except Exception as exn:
            exn.args = exn.args + (source.get("stream"), source.get("url"))
            if handler(exn):
                continue
            else:
                break


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


def filter_no_caption_or_no_image(sample):
    has_caption = 'txt' in sample
    has_image = 'png' in sample or 'jpg' in sample or 'jpeg' in sample or 'webp' in sample
    empty_caption = has_caption and (
        len(sample['txt'].strip()) < 2 or sample['txt'].strip() == b'' or sample['txt'].strip() == b'"'
    )
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


class ResampledShards2(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(
        self,
        urls,
        weights=None,
        nshards=sys.maxsize,
        worker_seed=None,
        deterministic=False,
        epoch=-1,
    ):
        """Sample shards from the shard list with replacement.

        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        urls, weights = expand_urls(urls, weights)
        self.urls = urls
        self.weights = weights
        if self.weights is not None:
            assert len(self.urls) == len(
                self.weights
            ), f"Number of urls {len(self.urls)} and weights {len(self.weights)} should match."
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.rng = random.Random()
        self.worker_seed = worker_seed
        self.deterministic = deterministic
        self.epoch = epoch

    def __iter__(self):
        """Return an iterator over the shards."""
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        if self.deterministic:
            # reset seed w/ epoch if deterministic
            if self.worker_seed is None:
                # pytorch worker seed should be deterministic due to being init by arg.seed + rank + worker id
                seed = pytorch_worker_seed(epoch)
            else:
                seed = self.worker_seed() + epoch
            self.rng.seed(seed)
        for _ in range(self.nshards):
            if self.weights is None:
                yield dict(url=self.rng.choice(self.urls))
            else:
                yield dict(url=self.rng.choices(self.urls, weights=self.weights, k=1)[0])


def image_text_collate_fn(batch, return_metadata=False, tokenizer=None, mlm_prob=0.3):
    collator = DefaultDataCollator()
    image_inputs = collator([{"input_ids": sample[0]} for sample in batch])
    text_samples = [sample[1] for sample in batch]

    if tokenizer:
        mlm_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=mlm_prob)
        for sample in text_samples:
            for k, v in sample.items():
                sample[k] = v.squeeze()

        text_inputs = mlm_collator(text_samples)
    else:
        if isinstance(text_samples[0], (dict, BatchEncoding)):
            text_inputs = {}
            for k in text_samples[0]:
                text_inputs[k] = torch.stack([sample[k].squeeze() for sample in text_samples])
        elif isinstance(text_samples[0], np.ndarray):
            text_inputs = {"text_embs": torch.from_numpy(np.array(text_samples))}
        else:
            raise ValueError(f"Unknown text input type: {type(text_samples[0])}")

    if len(batch[0]) > 2 and return_metadata:
        json = [sample[2] for sample in batch]

        return {"vision": image_inputs, "text": text_inputs, "metadata": json}
    else:
        return {"vision": image_inputs, "text": text_inputs}


def tokenize(text, tokenizer, add_eos=False, add_prefix=False):
    if add_eos:
        text = text + tokenizer.eos_token
    if add_prefix:
        text = f"search_query: {text}"
    tokenized = tokenizer(text, padding="max_length", truncation=True, return_tensors="pt")
    if add_eos:
        tokenized["input_ids"][:, -1] = tokenizer.eos_token_id

    return tokenized


def get_wds_image_text_dataset(
    args,
    transforms,
    is_train,
    tokenizer,
    epoch=0,
    floor=False,
    add_eos=False,
    add_prefix=False,
    precomputed_text=False,
):
    input_shards = args.image_text_shards
    assert input_shards is not None
    resampled = getattr(args, 'dataset_resampled', False) and is_train

    batch_size = int(args.batch_size // dist.get_world_size()) if dist.is_initialized() else args.batch_size

    num_shards = None
    if is_train:
        if args.train_num_samples is not None:
            num_samples = args.train_num_samples
        else:
            name = input_shards.split("/")[2]
            num_shards = len(expand_urls(input_shards)[0])

            num_samples = name2samples.get(name, LAION_PROCESSED)

            if not num_samples:
                raise RuntimeError(
                    'Currently, the number of dataset samples must be specified for the training dataset. '
                    'Please specify it via `--train-num-samples` if no dataset length info is present.'
                )
    else:
        # Eval will just exhaust the iterator if the size is not specified.
        num_samples = args.val_num_samples or 0

    shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc

    if resampled:
        pipeline = [
            ResampledShards2(
                input_shards,
                weights=getattr(args, "train-data_upsampling_factors", None),
                deterministic=True,
                epoch=shared_epoch,
                worker_seed=partial(pytorch_worker_seed, increment=dist.get_rank()),
            )
        ]
    else:
        pipeline = [wds.SimpleShardList(input_shards)]
    pipeline.extend([format_file_type])

    # at this point we have an iterator over all the shards
    # (TODO) zanussbaum: diverges with resampling for some reason
    # let's write some tests to figure out why
    if is_train:
        if not resampled:
            pipeline.extend(
                [
                    detshuffle2(
                        bufsize=_SHARD_SHUFFLE_SIZE,
                        initial=_SHARD_SHUFFLE_INITIAL,
                        seed=args.seed,
                        epoch=shared_epoch,
                    ),
                    wds.split_by_node,
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
                    rng=random.Random(args.seed + dist.get_rank()),
                ),
            ]
        )
    else:
        pipeline.extend(
            [
                wds.split_by_node,
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
            wds.to_tuple("image", "text", "json") if not precomputed_text else wds.to_tuple("image", "npy", "json"),
            # partial is a flag if we want to return partial batches
            wds.batched(
                batch_size,
                partial=not is_train,
                collation_fn=partial(
                    image_text_collate_fn,
                    return_metadata=not is_train,
                    tokenizer=tokenizer if args.mlm_prob else None,
                    mlm_prob=args.mlm_prob,
                ),
            ),
        ]
    )

    dataset = wds.DataPipeline(*pipeline)

    if is_train:
        if not resampled:
            num_shards = num_shards or len(expand_urls(input_shards)[0])
            assert (
                num_shards >= args.workers * dist.get_world_size()
            ), f'number of shards must be >= total workers: {num_shards} >= {args.workers * dist.get_world_size()}'
        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if floor else math.ceil
        global_batch_size = args.batch_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        # for future zanussbaum
        # this sets `nsamples` to `num_worker_batches` and updates `nrepititions`
        # turns an iterable dataset into a dataset with length!
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


class ImageFolderWithPathDataset(datasets.ImageFolder):
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

    dataset = ImageFolderWithPathDataset(data_path, transform=transforms)

    sampler = DistributedSampler(dataset, shuffle=False)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(args.batch_size // dist.get_world_size()),
        # sometimes this fails for > 4 workers
        num_workers=max(args.workers, 4),
        sampler=sampler,
    )

    return DataInfo(dataloader=dataloader, sampler=sampler)
