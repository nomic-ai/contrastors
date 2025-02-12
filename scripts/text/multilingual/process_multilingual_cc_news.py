from datasets import load_dataset
import gzip
import json
import fsspec
import multiprocessing as mp
import concurrent.futures
from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm


LANGS = ['af', 'als', 'am', 'an', 'ar', 'arz', 'as', 'ast', 'av', 'az', 'azb', 'ba', 'bar', 'bcl', 'be', 'bg', 'bh', 'bn', 'bo', 'bpy', 'br', 'bs', 'bxr', 'ca', 'cbk', 'ce', 'ceb', 'ckb', 'co', 'cs', 'cv', 'cy', 'da', 'de', 'diq', 'dsb', 'dty', 'dv', 'el', 'eml', 'en', 'eo', 'es', 'et', 'eu', 'fa', 'fi', 'fr', 'fy', 'ga', 'gd', 'gl', 'gn', 'gom', 'gu', 'gv', 'he', 'hi', 'hif', 'hr', 'hsb', 'ht', 'hu', 'hy', 'ia', 'id', 'ie', 'ilo', 'io', 'is', 'it', 'ja', 'jbo', 'jv', 'ka', 'kk', 'km', 'kn', 'ko', 'krc', 'ku', 'kv', 'kw', 'ky', 'la', 'lb', 'lez', 'li', 'lmo', 'lo', 'lt', 'lv', 'mai', 'mg', 'mhr', 'min', 'mk', 'ml', 'mn', 'mr', 'mrj', 'ms', 'mt', 'mwl', 'my', 'myv', 'mzn', 'nah', 'nap', 'nds', 'ne', 'new', 'nl', 'nn', 'no', 'oc', 'or', 'os', 'pa', 'pam', 'pfl', 'pl', 'pms', 'pnb', 'ps', 'pt', 'qu', 'rm', 'ro', 'ru', 'sa', 'sah', 'sc', 'scn', 'sco', 'sd', 'sh', 'si', 'sk', 'sl', 'so', 'sq', 'sr', 'su', 'sv', 'sw', 'ta', 'te', 'tg', 'th', 'tk', 'tl', 'tr', 'tt', 'tyv', 'ug', 'uk', 'ur', 'uz', 'vec', 'vep', 'vi', 'vls', 'vo', 'wa', 'war', 'wuu', 'xal', 'xmf', 'yi', 'yo', 'yue', 'zh']
SHARD_SIZE = 100_000


def parse_args():
    parser =  ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("-j", "--workers", default=mp.cpu_count(), type=int)
    parser.add_argument("--query_col", default="title")
    parser.add_argument("--document_col", default="text")

    return parser.parse_args()

    
def process_shard(ds, shard_index, bucket, num_splits, query_col, document_col):
    ds_shard = ds.shard(num_shards=num_splits, index=shard_index)

    ds_shard = ds_shard.filter(lambda x: x[query_col] and x[document_col]).to_list()

    metadata = {
        "objective": {"self": [], "paired": [[query_col, document_col]], "triplet": []}
    }

    for item in ds_shard:
        item["metadata"] = metadata

    shard_name = f"{bucket}/shard-{shard_index:05d}.jsonl.gz"

    with gzip.open(f"/tmp/{shard_name}", "wt") as f:
        for record in ds_shard:
            f.write(json.dumps(record) + "\n")

    s3 = fsspec.filesystem("s3")

    s3.put(f"/tmp/{shard_name}", f"s3://{shard_name}")

    Path(f"/tmp/{shard_name}").unlink()

    return shard_name 
    
    
if __name__ == "__main__":
    args = parse_args() 

    for lang in tqdm(LANGS):
        # process each lang with `workers` using concurrent futures
        print(f"Processing {lang}")

        ds = load_dataset(args.dataset, lang, split="train", trust_remote_code=True, num_proc=8)

        num_shards = (len(ds) // SHARD_SIZE)
        if len(ds) % SHARD_SIZE != 0:
            num_shards += 1

        lang_bucket = f"{args.bucket}/{lang}"
        if not Path(f"/tmp/{lang_bucket}").exists():
            Path(f"/tmp/{lang_bucket}").mkdir(parents=True, exist_ok=True)

        pbar = tqdm(total=num_shards, desc=f"Processing {lang}", position=1, leave=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = []
            for i in range(num_shards):
                future = executor.submit(process_shard, ds, i, lang_bucket, num_shards, args.query_col, args.document_col)
                future.add_done_callback(lambda _: pbar.update(1))
                futures.append(future)

            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                print(result)

        pbar.close()
        
        # delete all files in /tmp/lang_bucket
        for file in Path(f"/tmp/{lang_bucket}").glob("*"):
            file.unlink()