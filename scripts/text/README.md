# Text Scripts for `nomic-embed-text-v1`

## Pretokenizing Data for Masked Language Modeling

To train `nomic-bert-2048`, we use [Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) and [Bookscorpus](https://huggingface.co/datasets/bookcorpus) which is the same training data as the original [BERT](https://arxiv.org/abs/1810.04805).

We pack the data into sentences of 2048 tokens using the `bert-base-uncased` tokenizer. If a sentence is shorter than 2048 tokens, we pack the sentence with the next sentence until we reach 2048 tokens. If a sentence is longer than 2048 tokens, we split the sentence into 2048 token chunks.

To generate the data run:

```bash
python pretokenize.py --tokenizer_name=bert-base-uncased --seq_len=2048 --hf_save_name<where in huggingface/locally you want to save to>
```

## Filtering Data For Contrastive Pretraining

`nomic-embed-text-v1` training data is generated using `gte-base` to filter out low-quality pairs of data. For each dataset, we sample `min(len(dataset), 1_000_000)` points, embed the queries and documents, add the documents to the index. For each original `(query_i, document_i)` pair, we get the k-top similar documents. If `document_i` is not in the top-k, we discard the point. Pseudocode is provided below:

```python

queries, documents = get_dataset()
k = 2

index = create_nn_index

q_embed = embed(queries)
d_embed = embed(documents)

index.add(d_embed)

filtered_dataset = []
for i, (q_i, d_i) in enumerate(zip(q_embed, d_embed)):
    neighbors = index.get_knn(q_i, k=k)
    if i in neighbors:
        filtered_dataset.add((q_i, d_i))
```

The dataset should be in the following jsonl format

```json
{"query": "Who won the World Series in 2016?", "document": "The Chicago Cubs won the World Series against the Cleveland Guardians."}
...
```

To filter an existing dataset, run

```bash
torchrun --nproc-per-node=<num_gpus> --dataset=<path_to_dataset_files_or_directory> --output_dir=<path_where_to_save_filtered_dataset> --query_key=<query_key_of_jsonl_file> --document_key=<document_of_key_jsonl_file> index_filtering.py
```

NOTE: You most likely we want to install `faiss-gpu`. To do so on a GPU with Cuda 12+, please follow [INSTALL_FAISS.md](INSTALL_FAISS.md).

## Mining Hard Negatives for Contrastive Finetuning

To mine negatives, we use `gte-base` to embed the queries and documents, add the documents to the index, and get the k-top similar documents. We then filter out the original document and any documents that are in the top-k.

To mine hard negatives, run

```bash
torchrun --nproc-per-node=1 --dataset=<path_to_dataset_files_or_directory> --output_dir=<path_where_to_save_filtered_dataset> --query_key=<query_key_of_jsonl_file> --document_key=<document_of_key_jsonl_file> --k=<number_of_hard_negatives_to_mine>
```
