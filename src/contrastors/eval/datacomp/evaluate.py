import argparse
import copy
import json
import os
import pickle
import re
import shutil
import time
import warnings
from pathlib import Path
from time import gmtime, strftime

import numpy as np
import requests
import yaml
from cloudpathlib import CloudPath
from huggingface_hub import CommitOperationAdd, HfApi, Repository, dataset_info, delete_folder, upload_file
from requests.structures import CaseInsensitiveDict

from contrastors.eval.datacomp.eval_utils import evaluate_model

warnings.filterwarnings("ignore", message="Length of IterableDataset")


def path_or_cloudpath(s):
    if re.match(r"^\w+://", s):
        return CloudPath(s)
    return Path(s)


def submit_to_firebase(training_info, args, results):
    timestamp = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
    data = {
        "scale": training_info["scale"],
        "model": training_info["scale_config"]["model"],
        "dataset_size": args.dataset_size or training_info["scale_config"]["train_num_samples"],
        "checkpoint": str(training_info["checkpoint"]),
        "batch_size": training_info["scale_config"]["batch_size"],
        "learning_rate": training_info["scale_config"]["learning_rate"],
        "method_name": args.method_name,
        "author": args.author,
        "email": args.email,
        "hf_username": args.hf_username,
        "hf_repo_name": args.hf_repo_name,
        "timestamp": timestamp,
        "track": args.track,
        "writeup": args.writeup,
    }
    for dataset_name, dataset_results in results.items():
        if "main_metric" in dataset_results["metrics"]:
            metric = dataset_results["metrics"]["main_metric"]
            if metric is not None:
                data[dataset_name] = metric

    hf_hub_username = data["hf_username"]
    hf_hub_dirname = data["hf_repo_name"]

    key = f"{hf_hub_username}__{hf_hub_dirname}__{timestamp}".replace(".", "_")

    url = f"https://laion-tng-default-rtdb.firebaseio.com/{key}.json"

    headers = CaseInsensitiveDict()
    headers["Content-Type"] = "application/json"

    json_data = json.dumps(data)

    resp = requests.put(url, headers=headers, data=json_data)
    return resp


def submit_to_slack(train_info, args, results):
    scale = train_info.get("scale", "undefined")
    hf_hub_username = args.hf_username
    hf_hub_dirname = args.hf_repo_name
    hf_url = f"https://huggingface.co/{hf_hub_username}/{hf_hub_dirname}"

    avg_acc = np.mean(
        [val["metrics"]["main_metric"] for val in results.values() if val["metrics"]["main_metric"] is not None]
    )
    imagenet_acc = results["ImageNet 1k"]["metrics"]["acc1"]

    message = (
        f"New submission ({scale} scale, {args.track} track): {args.method_name}. "
        f"ImageNet accuracy: {imagenet_acc:.3f}. Average performance {avg_acc:.3f}. "
        f"From {args.author} ({args.email})."
    )
    if not args.skip_hf:
        message = message[:-1] + f", more details at {hf_url}."
    if args.writeup:
        message += f' Writeup: {args.writeup}'

    root = "hooks.slack.com"
    part1 = "T01AEJ66KHV"
    part2 = "B055EQE8U8N"
    part3 = "mgVJURCYuDirvkvyZ8wkuDwg"
    url = f"https://{root}/services/{part1}/{part2}/{part3}"

    headers = CaseInsensitiveDict()
    headers["Content-Type"] = "application/json"
    data = json.dumps({"text": message})
    resp = requests.put(url, headers=headers, data=data)

    return resp


def push_files_to_hub(train_info, args, results_filename):
    if "::" in str(args.samples):
        sample_files = [path_or_cloudpath(subdir) for subdir in str(args.samples).split("::")]
    else:
        sample_files = [args.samples]
    if len(sample_files) == 0:
        raise FileNotFoundError(f"Expected one or more files containing the sample ids but found none.")

    hf_api = HfApi()
    repo_id = args.hf_username + "/" + args.hf_repo_name
    print(f"Pushing files to HF Hub ({repo_id}). This may take a while.")
    results_filename = str(results_filename)
    scale = train_info["scale"]
    prefix = f"{scale}_scale"

    operations = [
        CommitOperationAdd(path_or_fileobj=results_filename, path_in_repo=f"{prefix}/results.jsonl"),
    ]

    if args.upload_checkpoint:
        model_checkpoint = str(train_info["checkpoint"])
        operations.append(CommitOperationAdd(path_or_fileobj=model_checkpoint, path_in_repo=f"{prefix}/checkpoint.pt"))

    for filename in sample_files:
        fileobj = filename.read_bytes()
        operations.append(
            CommitOperationAdd(
                path_or_fileobj=fileobj,
                path_in_repo=f"{prefix}/samples/{filename.name}",
            )
        )

    hf_api.create_commit(
        repo_id=repo_id,
        operations=operations,
        commit_message=f"Upload artifacts ({scale} scale)",
    )

    print("Done uploading files to HF Hub.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--track",
        type=str,
        required=False,
        choices=["filtering", "byod"],
        help="Competition track.",
    )
    parser.add_argument(
        "--train_output_dir",
        required=True,
        help="Path to output directory from training.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Path to output directory to use for evaluation. If nothing is passed, use the training output dir.",
    )
    parser.add_argument(
        "--data_dir",
        help="(Optional) Path to directory containing downloaded evaluation datasets.",
        default=None,
    )
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
    parser.add_argument("--prefix", type=str)

    # Submission flags
    parser_submit = parser.add_argument_group("submission")
    parser_submit.add_argument(
        "--submit",
        help="If true, submit the entry to the leaderboard.",
        action="store_true",
        default=False,
    )
    parser_submit.add_argument(
        "--method_name",
        type=str,
        help="Name of the method to be shown on the leaderboard. This *will* be shared publicly.",
        default=None,
    )
    parser_submit.add_argument(
        "--author",
        type=str,
        help="Name or names of the authors of this submission. This *will* be shared publicly.",
        default=None,
    )
    parser_submit.add_argument(
        "--writeup",
        type=str,
        help="Optional link for a paper or blog containing details for the method. This *will* be shared publicly if the flag is used.",
        default=None,
    )
    parser_submit.add_argument(
        "--email",
        type=str,
        help="Email for contact. This will *not* be shared publicly",
        default=None,
    )
    parser_submit.add_argument(
        "--hf_username",
        type=str,
        help="HuggingFace username. This will *not* be shared publicly",
        default=None,
    )
    parser_submit.add_argument(
        "--hf_repo_name",
        type=str,
        help="HuggingFace repository name. This will *not* be shared publicly",
        default=None,
    )
    parser_submit.add_argument("--dataset-size", type=str, default="", help="Optional size of the dataset.")
    parser_submit.add_argument(
        "--samples",
        type=path_or_cloudpath,
        help="Optional path to file(s) specifying the samples used for training. This must be specified.",
        default=None,
    )
    parser_submit.add_argument(
        "--upload_checkpoint",
        help="Whether or not to upload the checkpoint with the trained model",
        action="store_true",
        default=False,
    )

    # Debug-only flags. Using any of these might invalidate your submission.
    parser_debug = parser.add_argument_group("debug-only")
    parser_debug.add_argument(
        "--use_model",
        type=str,
        help='If set, manually specify a model architecture and checkpoint path ("model path")',
        default=None,
    )
    parser_debug.add_argument(
        "--skip_hf",
        help="If true,inodes skip uploading files to HF Hub",
        action="store_true",
        default=False,
    )
    parser_debug.add_argument(
        "--skip_db",
        help="If true, skip uploading information to databse",
        action="store_true",
        default=False,
    )
    parser_debug.add_argument(
        "--skip_notification",
        help="If true, skip notifying us from your submission",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()

    args.train_output_dir = Path(args.train_output_dir)
    if args.output_dir is None:
        args.output_dir = args.train_output_dir
    args.output_dir = Path(args.output_dir)

    if args.use_model is not None:
        args.train_output_dir = args.output_dir
        # Generate barebones info.pkl
        model_arch, model_checkpoint = args.use_model.split(maxsplit=1)
        Path.mkdir(args.output_dir, parents=True, exist_ok=True)
        with open(args.train_output_dir / "info.pkl", "wb") as f:
            pickle.dump(
                {"scale_config": {"model": model_arch}, "checkpoint": model_checkpoint},
                f,
            )

    if args.submit:
        assert (
            args.method_name is not None
        ), "Please specify your method name with --method_name for a valid submission."
        assert args.author is not None, "Please specify your author name with --author for a valid submission."
        assert args.email is not None, "Please specify your email with --email for a valid submission."
        assert (
            args.hf_username is not None
        ), "Please specify your huggingface username with --method_name for a valid submission."
        assert (
            args.hf_repo_name is not None
        ), "Please specify your huggingface repo name with --hf_repo_name for a valid submission."

    # Read training information
    train_info_filename = args.train_output_dir / "info.pkl"
    # train_info = pickle.load(open(train_info_filename, "rb"))
    train_info = {"checkpoint": args.train_output_dir, "scale_config": {"model": "large"}}

    results_filename = args.output_dir / "eval_results.jsonl"

    # Get list of datasets
    with open(os.path.join(os.path.dirname(__file__), "tasklist.yml")) as f:
        tasks = yaml.safe_load(f)

    # Check for cached results
    results = {}
    cached_train_info_filename = args.output_dir / "info.pkl"
    if args.output_dir.exists():
        # Read existing results
        if results_filename.exists():
            with open(results_filename, "r") as f:
                lines = [json.loads(s) for s in f.readlines()]
                for line in lines:
                    if line["key"] not in tasks:
                        continue
                    results[line["dataset"]] = line
            print(f"Found {len(results)} eval result(s) in {results_filename}.")
    else:
        Path.mkdir(args.output_dir, parents=True, exist_ok=True)
        # pickle.dump(train_info, open(cached_train_info_filename, "wb"))

    train_checkpoint = Path(train_info["checkpoint"])
    try:
        exists = Path(train_info["checkpoint"]).exists()
    except:
        exists = False
    if not exists and args.use_model is None:
        print(
            "Warning, did not find or could not read checkpoint at",
            train_info["checkpoint"],
        )
        default_checkpoint_name = args.train_output_dir / "checkpoints" / "epoch_latest.pt"
        print("Defaulting to", default_checkpoint_name)
        train_info["checkpoint"] = default_checkpoint_name

    print("Evaluating")

    starttime = int(time.time())

    for task_key in tasks:
        task_name = tasks[task_key].get("name", task_key)
        if task_name in results:
            print(f"Skipping {task_name} since results are already in {results_filename}")
        else:
            print(f"Evaluating on {task_name}")
            metrics = evaluate_model(
                task_key,
                train_info,
                args.data_dir,
                tasks[task_key].get("size"),
                batch_size=args.batch_size,
                prefix=args.prefix,
            )
            metrics["main_metric"] = metrics.get(tasks[task_key].get("main_metric", "acc1"))
            results[task_name] = {
                "key": task_key,
                "dataset": task_name,
                "metrics": metrics,
            }
            with open(results_filename, "a+") as f:
                f.write(json.dumps(results[task_name]) + "\n")

        if results[task_name]["metrics"]["main_metric"] is not None:
            print(f"Score: {results[task_name]['metrics']['main_metric']:.4f}")
        else:
            print(f"Score: No summary metric")

    elapsed = int(time.time()) - starttime
    print(f"Evaluation time: {elapsed // 3600} hour(s) {elapsed % 3600 // 60} minute(s) {elapsed % 60} second(s)")
    print()
    print("=== Final results ===")
    for line in results.values():
        print(f"{line['dataset']}: {line['metrics']['main_metric']}")
    if args.submit:
        print("=====================")
        average = np.mean(
            [val["metrics"]["main_metric"] for val in results.values() if val["metrics"]["main_metric"] is not None]
        )
        print(f"Average: {average}")

    if args.submit:
        print("Done with evaluations. Preparing your submission...")

        # Push models, results to HF Hub
        if not args.skip_hf:
            push_files_to_hub(train_info, args, results_filename)

        error_msg = """
            Error: something went wrong when submitting your results.
            Please check if your HF credentials are correct, and contact the team if errors persist.
        """
        error_msg = "=" * 100 + "\n" + error_msg + "\n" + "=" * 100

        # Submit jsonl to firebase
        if not args.skip_db:
            resp = submit_to_firebase(train_info, args, results)
            if resp.status_code != 200:
                print(error_msg)
                import sys

                sys.exit()

        # Slack notification
        if not args.skip_notification:
            resp = submit_to_slack(train_info, args, results)
            if resp.status_code != 200:
                print(error_msg)
                import sys

                sys.exit()

        print("Sucessfully submitted your results. Thanks for participating, and good luck!")
