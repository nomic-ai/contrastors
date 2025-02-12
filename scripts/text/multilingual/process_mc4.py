import subprocess
from datasets import load_dataset, disable_caching
import gzip
import json
import fsspec
import multiprocessing as mp
import concurrent.futures
from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm


# en-multi,es skipped
# LANGS = ['ar', 'az', 'be', 'bg', 'bg-Latn', 'bn', 'ca', 'ceb', 'co', 'cs', 'cy', 'da', 'de', 'el', 'el-Latn', 'en-multi', 'eo', 'es', 'et', 'eu', 'fa', 'fi', 'fil', 'fr', 'fy', 'ga', 'gd', 'gl', 'gu', 'ha', 'haw', 'hi', 'hi-Latn', 'hmn', 'ht', 'hu', 'hy', 'id', 'ig', 'is', 'it', 'iw', 'ja', 'ja-Latn', 'jv', 'ka', 'kk', 'km', 'kn', 'ko', 'ku', 'ky', 'la', 'lb', 'lo', 'lt', 'lv', 'mg', 'mi', 'mk', 'ml', 'mn', 'mr', 'ms', 'mt', 'my', 'ne', 'nl', 'no', 'ny', 'pa', 'pl', 'ps', 'pt', 'ro', 'ru', 'ru-Latn', 'sd', 'si', 'sk', 'sl', 'sm', 'sn', 'so', 'sq', 'sr', 'st', 'su', 'sv', 'sw', 'ta', 'te', 'tg', 'th', 'tr', 'uk', 'und', 'ur', 'uz', 'vi', 'xh', 'yi', 'yo', 'zh', 'zh-Latn', 'zu']
LANGS = ['fi', 'fil', 'fr', 'fy', 'ga', 'gd', 'gl', 'gu', 'ha', 'haw', 'hi', 'hi-Latn', 'hmn', 'ht', 'hu', 'hy', 'id', 'ig', 'is', 'it', 'iw', 'ja', 'ja-Latn', 'jv', 'ka', 'kk', 'km', 'kn', 'ko', 'ku', 'ky', 'la', 'lb', 'lo', 'lt', 'lv', 'mg', 'mi', 'mk', 'ml', 'mn', 'mr', 'ms', 'mt', 'my', 'ne', 'nl', 'no', 'ny', 'pa', 'pl', 'ps', 'pt', 'ro', 'ru', 'ru-Latn', 'sd', 'si', 'sk', 'sl', 'sm', 'sn', 'so', 'sq', 'sr', 'st', 'su', 'sv', 'sw', 'ta', 'te', 'tg', 'th', 'tr', 'uk', 'und', 'ur', 'uz', 'vi', 'xh', 'yi', 'yo', 'zh'][::-1]
SHARD_SIZE = 100_000

disable_caching()

def parse_args():
    parser =  ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("-j", "--workers", default=mp.cpu_count(), type=int)
    parser.add_argument("--query_col", default="title")
    parser.add_argument("--document_col", default="text")

    return parser.parse_args()


def split_title_text_batch(examples):
    titles = []
    texts = []
    for text in examples['text']:
        lines = text.split('\n', 1)
        titles.append(lines[0].strip() if lines else None)
        texts.append(lines[1].strip() if len(lines) > 1 else None)
    return {"title": titles, "text": texts}

    
def process_shard(ds, shard_index, bucket, num_splits, query_col, document_col):
    ds_shard = ds.shard(num_shards=num_splits, index=shard_index)

    metadata = {
        "objective": {"self": [], "paired": [[query_col, document_col]], "triplet": []}
    }

    # Assuming 'ds' is your existing dataset
    ds_shard = ds_shard.map(split_title_text_batch, remove_columns=['text', "timestamp"], batched=True)

    ds_shard = ds_shard.filter(lambda x: x['title'] and x['text']).to_list()

    for item in ds_shard:
        item["metadata"] = metadata

    shard_name = f"{bucket}/shard-{shard_index:05d}.jsonl.gz"

    with gzip.open(f"/tmp/{shard_name}", "wt") as f:
        for record in tqdm(ds_shard, desc="writing"):
            f.write(json.dumps(record) + "\n")

    s3 = fsspec.filesystem("s3")

    s3.put(f"/tmp/{shard_name}", f"s3://{shard_name}")

    Path(f"/tmp/{shard_name}").unlink()

    return shard_name 
    
    
if __name__ == "__main__":
    args = parse_args() 

    for lang in tqdm(LANGS):
        # process each lang with `workers` using concurrent futures
        print(f"Processing {lang}...")
        ds = load_dataset(args.dataset, lang, split="train", trust_remote_code=True, num_proc=8)

        num_shards = (len(ds) // SHARD_SIZE)
        if len(ds) % SHARD_SIZE != 0:
            num_shards += 1

        lang_bucket = f"{args.bucket}/{lang}"
        if not Path(f"/tmp/{lang_bucket}").exists():
            Path(f"/tmp/{lang_bucket}").mkdir(parents=True, exist_ok=True)

        pbar = tqdm(total=num_shards, desc=f"Processing {lang}", position=1, leave=True)
        if args.workers <= 1:
            for i in range(num_shards):
                shard = process_shard(ds, i, lang_bucket, num_shards, args.query_col, args.document_col)
                print(f"Finished {shard}")
                pbar.update(1)

        else:
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

        subprocess.run(["rm", "-rf", f"/root/.cache/huggingface/datasets/{args.dataset}"])
        subprocess.run(["rm", "-rf", f"/root/.cache/huggingface/datasets/downloads/"])