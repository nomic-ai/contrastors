import fsspec
import gzip
import json
from tqdm import tqdm
from pathlib import Path


bucket = "bge-m3-bge-m3"
buckets = list(Path(f"/tmp/{bucket}").glob("*"))

for key in buckets:
    if "miracl" not in str(key) and "mldr" not in str(key):
        continue 
    directory = Path(f"/tmp/{bucket}/{key.name}")
    out_dir = Path(f"/tmp/{bucket}-filtered/{key.name}")
    out_dir.mkdir(exist_ok=True, parents=True)
    files = list(directory.glob("shard-*.jsonl.gz"))
    nested = False
    if len(files) == 0:
        nested = True
        files = list(directory.glob("**/shard-*.jsonl.gz"))
    if len(files) == 0:
        print(f"No files found in {directory}")
        
    for file in tqdm(files):
        with gzip.open(file, "rt") as f:
            if nested:
                curr_dir = out_dir / file.parent.name
                curr_dir.mkdir(exist_ok=True, parents=True)
            else:
                curr_dir = out_dir
            print(f"Writing to {curr_dir / file.name}")
            with gzip.open(curr_dir / file.name, "wt") as f_out:
                num_lines = 0
                lt_neg = 0
                for line in f:
                    data = json.loads(line)
                    if len(data["neg"]) < 7:
                        lt_neg += 1
                    else:
                        f_out.write(json.dumps(data) + "\n")
                        
                    num_lines += 1
        print(f"{file}: {num_lines} lines, {lt_neg} < 7, {lt_neg / num_lines * 100:.2f}%")