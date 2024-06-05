import concurrent.futures
import json
import multiprocessing as mp
from argparse import ArgumentParser
from pathlib import Path

import braceexpand
import fsspec
import pyarrow.parquet as pq
from tqdm import tqdm


def get_dataset_size(shard):
    fs = fsspec.filesystem('s3')
    try:
        with fs.open(shard.replace(".tar", "_stats.json"), "r") as f:
            stats = json.load(f)
            shard_size = int(stats["successes"])

    except Exception as e:
        print(f"Error reading {shard}: {e}")
        shard_size = 0

    return shard_size


if __name__ == "__main__":
    parser = ArgumentParser(description="Get the size of a dataset")
    parser.add_argument(
        "--shards",
        type=str,
        help="Path to the shards",
        default="s3://commonpool-medium/shards/{00000000..00012895}.tar",
    )
    parser.add_argument("--workers", type=int, help="Number of workers", default=mp.cpu_count())
    args = parser.parse_args()
    shards = args.shards

    shards_list = braceexpand.braceexpand(shards)
    shards_list = list(shards_list)

    num_shards = len(shards_list)
    print(num_shards)

    pbar = tqdm(total=num_shards)

    total_size = 0
    path2size = {}
    if args.workers == 1:
        for shard in shards_list:
            shard_size = get_dataset_size(shard)
            path2size[Path(shard).name] = shard_size
            total_size += shard_size
            pbar.update(1)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
            future2shard = {executor.submit(get_dataset_size, shard): shard for shard in shards_list}

            for future in concurrent.futures.as_completed(future2shard):
                shard = future2shard[future]
                try:
                    shard_size = future.result()
                    path2size[Path(shard).name] = shard_size
                    total_size += shard_size
                except Exception as e:
                    print(f"Shard {shard} generated an exception: {e}")

                pbar.update(1)

    print(f"Total size: {total_size:,}")
    # with open("shard2size.json", "w") as f:
    #     json.dump(path2size, f, indent=4)
