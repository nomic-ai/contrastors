# Evaluating a Model

`contrastors` supports evaluation on 3 benchmarks: [MTEB](https://github.com/embeddings-benchmark/mteb), [LoCo](https://hazyresearch.stanford.edu/blog/2024-01-11-m2-bert-retrieval), and [Jina Long Context](https://arxiv.org/pdf/2310.19923.pdf).

You may need to install the "eval" dependencies to run the evaluation scripts. You can do this by running `pip install -e .[eval]`.

If you are evaluating Huggingface or OpneAI model that isn't natively supported by `contrastors`, you can pass
the flags `--hf_model` or `--openai_model` to the evaluation scripts respectively.

## Evaluting on MTEB

Evaluation of MTEB can take a long time so we implemented parallel evaluation. To run the evaluation, run

```bash
OPENBLAS_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=<GPUS> python eval/mteb_eval/eval_mteb.py --model_name=<model_name>
```

This will save all results in a folder `results/<model_name>`. The results are saved in `.json` files.
The `CQADupstack` results need to be merged into one result:

```bash
python eval/mteb_eval/merge_cqadupstack.py results/<model_name>
```

Then generate the metadata and scores:

```bash
python eval/mteb_eval/mteb_meta.py results/<model_name>
python eval/mteb_eval/score_mteb.py <last folder prefix>/mteb_metadata.md # e.g. if the folder is results/model/epoch_0, you should use epoch_0/mteb_metadata.md
```

## Evaluating on LoCo

To evaluate on LoCo, run

```bash
OPENBLAS_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=<GPU_IDS> python eval/eval_loco.py --model_name=<model_name>
```

## Evaluating on Jina Long Context

Jina Long Context tasks, as of Feb 1 2024, are not supported in the latest `MTEB` package release but are in `main`.
To run the evaluation, install `mteb` from source

```bash
git clone https://github.com/embeddings-benchmark/mteb
cd mteb
pip install -e .
```

and then evaluate the model

```bash
OPENBLAS_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=<GPU_IDS> python eval/eval_long_context.py --model_name=<model_name>
```
