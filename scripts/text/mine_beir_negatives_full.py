import numpy as np
import random
import gzip
import json
from argparse import ArgumentParser
from pathlib import Path
from beir.datasets.data_loader import GenericDataLoader

import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer



def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--min_negatives", type=int, default=10)
    parser.add_argument("--max_negatives", type=int, default=100)
    parser.add_argument("--k", type=int, default=75)
    parser.add_argument("--margin", type=float, default=0.95)
    parser.add_argument("--query_key", default="query")
    parser.add_argument("--document_key", default="pos")
    parser.add_argument("--negatives_key", default="neg")

    return parser.parse_args()

def load_beir(beir_path):
    corpus, queries, qrels = GenericDataLoader(beir_path).load(split="train")
    all_queries = []
    qid2index = {}
    for qid, query in queries.items():
        qid2index[qid] = len(all_queries)
        all_queries.append(query)

    all_documents = []
    docid2index = {}
    for doc_id, doc in corpus.items():
        docid2index[doc_id] = len(all_documents)
        all_documents.append((doc.get("title") + " " + doc.get("text")).strip())

    return corpus, queries, qrels, qid2index, docid2index, all_documents


if __name__ == "__main__":
    args = parse_args()
    if args.dataset is not None:
        names = [args.dataset] 
    else:
        names = ["beir/hotpotqa", "beir/fever", "beir/msmarco", "beir/nq"]

    for name in names:
        corpus, queries, qrels, qid2index, docid2index, documents = load_beir(name)

        print(f"{name=}, {len(corpus)=}, {len(queries)=}, {len(qrels)=}")

        output_dir = Path(args.output_dir) / (name.split("/")[-1])
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)

        model_name = "dunzhang/stella_en_1.5B_v5"
        query_prompt_name = "s2p_query"
        model = SentenceTransformer(model_name)
        dimension = model.get_sentence_embedding_dimension()

        if not Path(output_dir / f"{model_name.split('/')[-1]}_document_embeddings.npy").exists():
            pool = model.start_multi_process_pool()

            document_embeddings = model.encode_multi_process(documents,
                                            pool=pool,
                                            show_progress_bar=True, 
                                            batch_size=args.batch_size, 
                                            normalize_embeddings=True,
                                            )

            # save corpus embeddings since it's the most expensive part embeddings 
            np.save(output_dir / f"{model_name.split('/')[-1]}_document_embeddings.npy", document_embeddings)

            model.stop_multi_process_pool(pool)
        else:
            document_embeddings = np.load(output_dir / f"{model_name.split('/')[-1]}_document_embeddings.npy")

        min_negatives = args.k
        margin = args.margin
        mined_dataset = []
        lt_negatives = 0

        qids = list(qrels)
        mined_dataset = []
        batch_size = args.batch_size
        for batch_start in tqdm(range(0, len(qids), batch_size)):
            batch_end = min(batch_start + batch_size, len(qids))
            batch_qids = qids[batch_start:batch_end]
            
            # Batch encode queries
            batch_texts = [queries[qid] for qid in batch_qids]
            query_embs = model.encode(batch_texts, normalize_embeddings=True, prompt_name=query_prompt_name)

            # Batch compute similarities
            # Shape: [batch_size, num_documents]
            # in document idx order
            qd_sims = model.similarity(query_embs, document_embeddings)
            
            # Get top-k for all queries at once
            # scores shape: [batch_size, num_documents]
            # indices shape: [batch_size, num_documents]
            # indices[batch, i] gets you the document index
            scores, indices = qd_sims.topk(k=len(document_embeddings), dim=-1)

            # Process each query in the batch
            for batch_idx, query_id in enumerate(batch_qids):
                query_text = queries[query_id]
                pos_doc_ids = list(qrels[query_id])
                pos_doc_idxs = [docid2index[doc_id] for doc_id in pos_doc_ids]
                
                # Get this query's scores and indices
                query_scores = scores[batch_idx]
                query_indices = indices[batch_idx]

                for pos_doc_id, pos_doc_idx in zip(pos_doc_ids, pos_doc_idxs):
                    document = corpus[pos_doc_id]
                    document = (document.get("title") + " " + document.get("text")).strip()

                    qd_score = qd_sims[batch_idx, pos_doc_idx]
                    threshold = qd_score * margin

                    row = {args.query_key: query_text, args.document_key: document}
                    
                    neg_indices = query_indices[
                        (query_scores < threshold) & 
                        (~torch.isin(query_indices, torch.tensor(pos_doc_idxs)))
                    ][:args.max_negatives]

                    if len(neg_indices) < min_negatives:
                        lt_negatives += 1
                        continue

                    row[args.negatives_key] = [documents[j] for j in neg_indices]
                    mined_dataset.append(row)

        print(f"{lt_negatives=}")
        print(f"{len(mined_dataset)=}")

        # shuffle to get mix up duplicate qids
        random.shuffle(mined_dataset)

        metadata = {
            "objective": {"self": [], "paired": [], "triplet": [[args.query_key, args.document_key, args.negatives_key]]}
        }
        shard_size = 100_000
        for shard_start in tqdm(range(0, len(mined_dataset), shard_size), desc="Writing shards"):
            dataset_slice = mined_dataset[shard_start : shard_start + shard_size]
            for record in dataset_slice:
                record["metadata"] = metadata
            shard_num = shard_start // shard_size
            with gzip.open(output_dir / f"shard-{shard_num:05d}.jsonl.gz", "wt") as f:
                for data in tqdm(dataset_slice):
                    f.write(json.dumps(data) + "\n")
