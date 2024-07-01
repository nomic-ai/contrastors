# contrastors

`contrastors` is contrastive learning toolkit that enables researchers and engineers to train and evaluate contrastive models efficiently.


[![img](docs/atlas-nomic-embed.png)](https://atlas.nomic.ai/map/nomic-text-embed-v1-5m-sample)


## Features

- Built on top of [Flash Attention](https://github.com/Dao-AILab/flash-attention) for fast and efficient training
- Support for training on multiple GPUs
- [GradCache](https://github.com/luyug/GradCache) support for training with large batch sizes in constrained memory environments
- Huggingface Support for easy loading of common models (Pythia/GPTNeoX, BERT, etc.)
- Masked Language Modeling (MLM) Pretraining
- [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147) for flexible embedding sizes
- [CLIP](https://arxiv.org/abs/2103.00020) and [LiT](https://arxiv.org/abs/2111.07991) style contrastive learning
- Support for loading popular ViT (e.g. [timm](https://huggingface.co/timm)) models

## Research

* [Nomic Embed: Training a Reproducible Long Context Text Embedder](https://arxiv.org/abs/2402.01613) by Zach Nussbaum, Jack Morris, Andriy Mulyar, and Brandon Duderstadt
* [Nomic Embed Vision: Expanding the Latent Space](https://arxiv.org/abs/2406.18587) by Zach Nussbaum, Brandon Duderstadt, and Andriy Mulyar

## Getting Started and Requirements

The `contrastors` library relies on custom kernels from the [Flash Attention](https://github.com/Dao-AILab/flash-attention) repository. To setup your enviornment you will need to follow the steps below.

Make sure that you have Cuda 11.8+. You can check this by running `nvcc --version` or if you already have torch installed you can run `python -c "import torch; print(torch.version.cuda)"`

Create a python venv and activate it

```bash
python3 -m venv env
source env/bin/activate
```

Install [torch](https://pytorch.org/get-started/locally/). See the torch docs for specific instructions for your system (e.g. the default CUDA torch supports is 12.1 as of 12/12/2023).

```bash
pip3 install torch torchvision torchaudio
```

Install wheel, packaging, ninja for Flash Attention (so the builds don't take too long)

```bash
pip install wheel packaging ninja setuptools
```

Install Flash Attention and the custom kernels

```bash
pip install --no-cache-dir flash-attn --no-build-isolation git+https://github.com/HazyResearch/flash-attention.git#subdirectory=csrc/rotary git+https://github.com/HazyResearch/flash-attention.git#subdirectory=csrc/layer_norm git+https://github.com/HazyResearch/flash-attention.git#subdirectory=csrc/fused_dense_lib git+https://github.com/HazyResearch/flash-attention.git#subdirectory=csrc/xentropy
```

Install the rest of the requirements and the package

```bash
pip install -e . 
```

## Data Access

We provide access to the `nomic-embed-text-v1` dataset via the `nomic` package. To access the data, you will need to create an account and login to the `nomic` package. First create an account at [atlas.nomic.ai](https://atlas.nomic.ai), download the `nomic` Python client, and run the following commands:

```bash
pip install nomic
nomic login # follow prompts to login
python -c "from nomic import atlas; print(atlas._get_datastream_credentials(name='contrastors'))"
```

which will print out your access keys. You can then configure them by using `aws configure` or setting
the `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` environment variables.

If you do not have the AWS CLI installed, you can install it [here](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html).

To verify your access, you can run the following command to list the contents of the bucket:

```bash
aws s3 ls --endpoint-url=https://9fa58365a1a3d032127970d0bd9a1290.r2.cloudflarestorage.com/ s3://contrastive
aws s3 ls --endpoint-url=https://9fa58365a1a3d032127970d0bd9a1290.r2.cloudflarestorage.com/ s3://contrastive-index-filtered
```

You should be able to see the contents of the bucket and download the data.

If you intend to train using our data and the `contrastors` repo, you will need to setup `fsspec` support for Cloudflare R2. To do so,
create a file `~/.config/fsspec/s3.json` with the following contents:

```json
{
  "s3": {
    "client_kwargs": {
      "endpoint_url": "https://9fa58365a1a3d032127970d0bd9a1290.r2.cloudflarestorage.com/",
      "aws_access_key_id": <ACCESS_KEY_ID>,
      "aws_secret_access_key": <SECRET_KEY_ID>
    }
  }
}
```

### Nomic Data Format

Our text data is stored in gziped jsonl files with which we also store a `counts.json` file and `offsets.json.gzip`.

The `counts.json` file is a dictionary mapping the file name to the number of examples in the file. The `offsets.json.gz` file is a dictionary mapping the file name to a dictionary where each key is the index of the example and the value is a tuple of the start and end byte offset of the example in the file. We do this to allow for streaming of data in from R2, especially when the data is larger than the buffer size.

Here's a small example of what a dataset configuration might look like:

```yaml
datasets:
  - name: "paq"
    bucket: "s3://contrastive-index-filtered/paq_full/shard-{00000..00538}.jsonl.gz"
    query_prefix: "search_query"
    document_prefix: "search_document"
    objective: 
        type: "paired"
        columns: ["query", "document"]

```

`objective` defines if it's a paired or triplet objective. In both cases, the `columns` field defines the columns to use for each example.

## Training `nomic-embed-text-v1`

### Masked Language Modeling Pretraining

To train your own BERT from scratch (with all the optimizations) run

```bash
cd src/contrastors
deepspeed --num_gpus=8 train.py --config=configs/train/mlm.yaml --deepspeed_config=configs/deepspeed/ds_config.json --dtype=bf16
```

### Constrastive Pretraining and Finetuning

To launch an experiment run

```bash
cd src/contrastors
torchrun --nproc-per-node=8 train.py --config=configs/train/contrastive_pretrain.yaml --dtype=bf16
```

This will train a bert model on all ~200M examples. To change the dataset, you can modify `data_args.input_shards`.

To finetune `nomic-bert-embed-v1-unsupervised`, update the config to `configs/train/contrastive_finetune.yaml`.

### Generating Your Own Data

To generate your own data for any step of the pipeline, you can use the provided scripts in `scripts/text`. 

See the [README](scripts/text/README.md) in `scripts/text` for more information.



## Training `nomic-embed-vision-v1.5`

To align a vision model, you will need to curate a large image-text dataset. More details can be found [here](https://github.com/rom1504/img2dataset).

To align `nomic-embed-vision-v1.5` with `nomic-embed-text-v1.5`, you can run the following command:

```bash
deepspeed  train.py --deepspeed_config=configs/deepspeed/image_text.json --config=configs/train/nomic_embed_vision_v1.5.yaml --dtype=bf16
```

## Pretrained Models

We provide pretrained models for `Nomic Embed` at the following locations:

- [nomic-embed-text-v1](https://huggingface.co/nomic-ai/nomic-embed-text-v1)
- [nomic-embed-vision-v1](https://huggingface.co/nomic-ai/nomic-embed-vision-v1)
- [nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)
- [nomic-embed-vision-v1.5](https://huggingface.co/nomic-ai/nomic-embed-vision-v1.5)
- [nomic-embed-text-v1-ablated](https://huggingface.co/nomic-ai/nomic-embed-text-v1-ablated)
- [nomic-embed-text-v1-unsupervised](https://huggingface.co/nomic-ai/nomic-embed-text-v1-unsupervised)
- [nomic-bert-2048](https://huggingface.co/nomic-ai/nomic-bert-2048)

## Join the Nomic Community

- Nomic: [https://nomic.ai](https://nomic.ai)
- Discord: [https://discord.gg/myY5YDR8z8](https://discord.gg/myY5YDR8z8)
- Twitter: [https://twitter.com/nomic_ai](https://twitter.com/nomic_ai)

## License

This code is licensed under the [Apache 2.0 License](LICENSE). See the model cards for the individual license for each model. 

## Acknowledgements

We thank Tri Dao for his work on Flash Attention and the custom kernels that make this project possible, the [OpenCLIP](https://github.com/mlfoundations/open_clip) team for their
great repository with which much of this work is based on, and the Huggingface team for their great work on the transformers library.


## Citation

If you find the model, dataset, or training code useful, please cite our work

```bibtex
@misc{nussbaum2024nomic,
      title={Nomic Embed: Training a Reproducible Long Context Text Embedder}, 
      author={Zach Nussbaum and John X. Morris and Brandon Duderstadt and Andriy Mulyar},
      year={2024},
      eprint={2402.01613},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
@misc{nussbaum2024nomicembedvisionexpanding,
      title={Nomic Embed Vision: Expanding the Latent Space}, 
      author={Zach Nussbaum and Brandon Duderstadt and Andriy Mulyar},
      year={2024},
      eprint={2406.18587},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2406.18587}, 
}
```
