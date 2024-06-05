import argparse
import os
import sys

import datasets
import yaml

VERBOSE = False


def main(args):
    global VERBOSE
    VERBOSE = args.verbose
    download_datasets(args.data_dir)


def wget(src, dst, verbose=False):
    vflag = "v" if VERBOSE or verbose else "nv"
    os.system(f"wget -{vflag} '{src}' -O '{dst}'")


def download_datasets(data_dir):
    local_urls = []
    # Get list of datasets
    with open("tasklist.yml") as f:
        tasks = yaml.safe_load(f)
    for task, task_info in tasks.items():
        task_name = task_info.get("name", task)
        if task.startswith("retrieval/") or task.startswith("misc/"):
            # Huggingface dataset loader, download those differently
            task = task.split("/", 1)[1]
            try:
                print()
                print(f"""{f" Download '{task_name}' ":=^40s}""")
                print()
                datasets.load_dataset(
                    f"nlphuji/{task}",
                    split="test",
                    ignore_verifications=False,
                    cache_dir=os.path.join(data_dir, "hf_cache"),
                )
            except Exception as e:
                print(
                    "Failed to download Huggingface dataset, check write permissions and Internet connection",
                    file=sys.stderr,
                )
                print(e)
            continue
        if task.startswith("fairness/"):
            task = task.split("/", 1)[1]
        # Download webdataset from Huggingface
        dir_name = f"wds_{task.replace('/', '-')}_test"
        source_url = f"https://huggingface.co/datasets/djghosh/{dir_name}"
        target_path = os.path.join(data_dir, dir_name)
        try:
            print()
            print(f"""{f" Download '{task_name}' ":=^40s}""")
            print()
            # Create directory
            os.makedirs(os.path.join(target_path, "test"), exist_ok=True)
            # Download metadata
            wget(
                os.path.join(source_url, "raw/main/classnames.txt"),
                os.path.join(target_path, "classnames.txt"),
            )
            wget(
                os.path.join(source_url, "raw/main/zeroshot_classification_templates.txt"),
                os.path.join(target_path, "zeroshot_classification_templates.txt"),
            )
            wget(
                os.path.join(source_url, "raw/main/test/nshards.txt"),
                os.path.join(target_path, "test/nshards.txt"),
            )
            # Get nshards
            with open(os.path.join(target_path, "test/nshards.txt")) as f:
                nshards = int(f.read())
            local_urls.append(os.path.join(target_path, f"test/{{0..{nshards-1}}}.tar"))
            # Check and optionally download TARs
            for index in range(nshards):
                local_tar_path = os.path.join(target_path, f"test/{index}.tar")
                if os.path.exists(local_tar_path):
                    # Check existing TAR
                    # Get expected size and checksum
                    with os.popen(f"curl -s '{os.path.join(source_url, f'raw/main/test/{index}.tar')}'") as tar_output:
                        tar_info = dict([line.split(maxsplit=1) for line in tar_output.read().splitlines()])
                        exp_checksum = tar_info["oid"].split(":")[1]
                        exp_size = int(tar_info["size"])
                    # Compute true size and checksum
                    with os.popen(f"sha256sum '{local_tar_path}'") as sha_output:
                        true_checksum = sha_output.read().split()[0]
                    true_size = os.path.getsize(local_tar_path)
                    # If equal, skip
                    if true_checksum == exp_checksum and true_size == exp_size:
                        print(f"Verified test/{index}.tar")
                        continue
                # TAR is corrupt or does not exist, download
                wget(
                    os.path.join(source_url, f"resolve/main/test/{index}.tar"),
                    local_tar_path,
                    verbose=True,
                )
            print("Successfully downloaded dataset")
        except Exception as e:
            print(
                "Failed to download dataset, check write permissions and Internet connection",
                file=sys.stderr,
            )
            print(e)
    print()
    # Print all local URLs
    print("Paths to all downloaded TAR files:")
    print(*local_urls, sep="\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download all data comp evaluation datasets")
    parser.add_argument("data_dir", help="Root directory into which all datasets will be downloaded")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print verbose download status")
    args = parser.parse_args()
    sys.exit(main(args))
