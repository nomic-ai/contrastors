import os
import json
import gzip
from argparse import ArgumentParser
import fsspec
import concurrent.futures

from tqdm import tqdm


def parser_args():
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)

    return parser.parse_args()

    
def read_jsonl(file, s3):
    idx2offset = {}
    count = 0
    with s3.open(file, "rt", compression="gzip") as f:
        previous = 0
        for i, line in enumerate(f):
            end = previous + len(line)
            idx2offset[i] = (previous, end)
            previous = end
            count += 1
            
    return idx2offset, count


args = parser_args()

s3 = fsspec.filesystem("s3")

data_dir = sorted(s3.ls(args.data_dir))

for path in tqdm(data_dir, desc="Processing languages"):
    lang = path.split(args.data_dir)[1].split("/")[1]
    if lang != "ca":
        continue


    counts = {"count_per_file": {}, "total_count": 0}
    offsets = {}

    files = s3.glob(f"{args.data_dir}/{lang}/shard-*.jsonl.gz")
    total_count = 0
    pbar = tqdm(total=len(files), desc=f"Processing {lang}") 

    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        future2file = {}
        for file in files:
            future = executor.submit(read_jsonl, file, s3)
            future.add_done_callback(lambda _: pbar.update(1))
            future2file[future] = file

        for future in concurrent.futures.as_completed(future2file):
            file_idx_offset, file_count = future.result()
            counts["count_per_file"][future2file[future]] = file_count
            offsets[future2file[future]] = file_idx_offset

    with s3.open(f"{path}/counts.json", "w") as f:
        counts["total_count"] = sum(counts["count_per_file"].values())
        json.dump(counts, f)

    with gzip.open(f"/tmp/{lang}_offsets.json.gz", "wt") as f:
        json.dump(offsets, f)

    s3.put(f"/tmp/{lang}_offsets.json.gz", f"{path}/offsets.json.gz")

    # Clean up
    os.remove(f"/tmp/{lang}_offsets.json.gz")
