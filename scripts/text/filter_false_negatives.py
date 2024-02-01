import gzip
import json
import random
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from tqdm import tqdm


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--query_key", type=str, default="query")
    parser.add_argument("--document_key", type=str, default="document")
    parser.add_argument("--negatives_key", type=str, default="negatives")
    parser.add_argument("--min_negatives", type=int, default=30)

    return parser.parse_args()


args = parse_args()
query_key, document_key, negatives_key = args.query_key, args.document_key, args.negatives_key

files = list(Path(args.dataset).glob("*.jsonl.gz"))

all_documents = []
lines = []
for file in tqdm(files):
    with gzip.open(file, "rt") as f:
        for line in f:
            data = json.loads(line)
            all_documents.append(data[document_key])
            all_documents.extend(data[negatives_key])
            lines.append(data)

processed_records = []
for record in tqdm(lines):
    query = record[query_key]
    document = record[document_key]
    document_score = record["pos_score"]

    negative_scores = record["scores"]
    negatives = record[negatives_key]
    # sort by score
    filtered_negatives = []
    filtered_scores = []
    for i in np.argsort(negative_scores)[::-1]:
        score = negative_scores[i]
        if score < document_score:
            filtered_negatives.append(negatives[i])
            filtered_scores.append(score)

    if len(filtered_negatives) < args.min_negatives:
        while True:
            sampled = random.sample(all_documents, args.min_negatives - len(filtered_negatives))
            filtered_sampled = []
            for sample in sampled:
                if sample not in filtered_negatives and sample != document:
                    filtered_sampled.append(sample)

            filtered_negatives.extend(filtered_sampled)
            filtered_scores.extend([None] * len(filtered_sampled))
            if len(filtered_negatives) >= args.min_negatives:
                break

    new_record = {
        query_key: query,
        document_key: document,
        negatives_key: filtered_negatives,
        "pos_score": document_score,
        "scores": filtered_scores,
    }
    new_record["metadata"] = {"objective": {'self': [], "paired": [], "triplet": [["query", "document", "negatives"]]}}

    processed_records.append(new_record)

outdir = Path(args.outdir)
if not outdir.exists():
    outdir.mkdir()

shard_num = 0
shard_size = 100_000
counts = {"total_count": 0, "count_per_file": {}}
for shard_num in range(0, len(processed_records), shard_size):
    shard_name = shard_num // shard_size
    num_lines = min(shard_size, len(processed_records) - shard_num)
    with gzip.open(outdir / f"shard-{shard_name:05d}.jsonl.gz", "wt") as f:
        for record in tqdm(processed_records[shard_num : shard_num + shard_size]):
            f.write(json.dumps(record) + "\n")

    counts["count_per_file"][f"contrastive/{outdir.name}/shard-{shard_name:05d}.jsonl.gz"] = num_lines

idx2offset = {}
for file in tqdm(list(outdir.glob("*.jsonl.gz"))):
    offset = {}
    with gzip.open(file, "rt") as f:
        previous = 0
        for i, line in enumerate(f):
            end = previous + len(line)
            offset[i] = (previous, end)
            previous = end

    idx2offset[f"contrastive/{outdir.name}/{file.name}"] = offset


with gzip.open(outdir / "offsets.json.gz", "wt") as f:
    json.dump(idx2offset, f)

with open(outdir / "counts.json", "w") as f:
    json.dump(counts, f)
