import json
import logging
import os
import time
from argparse import ArgumentParser

from mteb import MTEB

from contrastors.eval.encoder import Encoder, OpenAI_Encoder, STransformer

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("main")

os.environ['OPENBLAS_NUM_THREADS'] = '16'


TASK_LIST_CLUSTERING = [
    "BigPatentClustering",
    "WikiCitiesClustering",
]


TASK_LIST_RETRIEVAL = ["NarrativeQARetrieval", "SciFact"]


TASK_LIST = TASK_LIST_CLUSTERING + TASK_LIST_RETRIEVAL


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument("--add_prefix", action="store_true")
    parser.add_argument("--tokenizer_name", type=str, default="bert-base-uncased")
    parser.add_argument("--seq_length", type=int, default=512)
    parser.add_argument("--rotary_scaling_factor", type=float, default=None)
    parser.add_argument("--openai_model", action="store_true")
    parser.add_argument("--batch_size", type=int, default=32)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_name = args.model_name
    tokenizer_name = args.tokenizer_name
    seq_length = args.seq_length
    if args.openai_model:
        model = OpenAI_Encoder(model_name, cutoff=seq_length)
    else:
        model = Encoder(
            model_name,
            seq_length=seq_length,
            rotary_scaling_factor=args.rotary_scaling_factor,
            tokenizer_name=tokenizer_name,
        )
        print(f"Add prefix: {args.add_prefix}")
        model = STransformer(model, add_prefix=args.add_prefix)

    task2prefix = {}

    for task in TASK_LIST_CLUSTERING:
        task2prefix[task] = {"query": "clustering", "document": "clustering"}

    for task in TASK_LIST_RETRIEVAL:
        task2prefix[task] = {"query": "search_query", "document": "search_document"}

    start = time.time()
    all_results = {}
    for task in TASK_LIST:
        logger.info(f"Running task: {task}")
        eval_splits = ["dev"] if task == "MSMARCO" else ["test"]
        evaluation = MTEB(tasks=[task], task_langs=["en"])  # Remove "en" for running all languages

        if args.add_prefix:
            prefixes = task2prefix[task]
            model.query_prefix = prefixes["query"]
            model.docoment_prefix = prefixes["document"]

        results = evaluation.run(
            model, output_folder=None, batch_size=args.batch_size, eval_splits=eval_splits, show_progress_bar=True
        )
        all_results[task] = results[task]['test']

    end = time.time()
    print(f"Time taken (mins): {(end-start)/60}")
    print(json.dumps(all_results, indent=3))
