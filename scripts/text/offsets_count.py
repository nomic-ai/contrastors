import gzip
import json
from argparse import ArgumentParser
from pathlib import Path

from tqdm import tqdm


def parser_args():
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)

    return parser.parse_args()


args = parser_args()

data_dir = Path(args.data_dir)

files = sorted(data_dir.glob("shard-*.jsonl.gz"))

counts = {"count_per_file": {}, "total_count": 0}
offsets = {}

total_count = 0
for file in tqdm(files):
    idx2offset = {}
    with gzip.open(file, "rt") as f:
        previous = 0
        curr_count = 0
        for i, line in enumerate(f):
            end = previous + len(line)
            idx2offset[i] = (previous, end)
            previous = end
            curr_count += 1

    counts["count_per_file"][f"contrastive/{data_dir.name}/{file.name}"] = curr_count
    offsets[f"contrastive/{data_dir.name}/{file.name}"] = idx2offset


with open(data_dir / "counts.json", "w") as f:
    counts["total_count"] = sum(counts["count_per_file"].values())
    json.dump(counts, f)

with gzip.open(data_dir / "offsets.json.gz", "wt") as f:
    json.dump(offsets, f)
