#!/bin/bash

# train and wait for it to finish
echo "Training model"

torchrun --nproc-per-node=8 train.py --config=configs/train/contrastive_pretrain_multilingual.yaml --dtype=bf16 --learning_rate=1.0e-4 --output_dir=ckpts/xlm-roberta-4k-base-1e4-pos-ids
OPENBLAS_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 python eval/mteb_eval/eval_mteb.py --model_name=ckpts/xlm-roberta-4k-base-1e4-pos-ids/epoch_0_model/ --add_prefix --no_normalize_classification --seq_len=128 --tokenizer_name=FacebookAI/xlm-roberta-base &

torchrun --nproc-per-node=8 train.py --config=configs/train/contrastive_pretrain_multilingual.yaml --dtype=bf16 --learning_rate=8.0e-5 --num_experts=8 --moe_top_k=1 --output_dir=ckpts/xlm-8eg1t1-4k-8e5-pos-ids --gradient_checkpointing
OPENBLAS_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 python eval/mteb_eval/eval_mteb.py --model_name=ckpts/xlm-8eg1t1-4k-8e5-pos-ids/epoch_0_model/ --add_prefix --no_normalize_classification --seq_len=128 --tokenizer_name=FacebookAI/xlm-roberta-base &

torchrun --nproc-per-node=8 train.py --config=configs/train/contrastive_pretrain_multilingual.yaml --dtype=bf16 --learning_rate=1.0e-4 --num_experts=8 --moe_top_k=1 --output_dir=ckpts/xlm-8eg1t1-4k-1e4-pos-ids --gradient_checkpointing
OPENBLAS_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 python eval/mteb_eval/eval_mteb.py --model_name=ckpts/xlm-8eg1t1-4k-1e4-pos-ids/epoch_0_model/ --add_prefix --no_normalize_classification --seq_len=128 --tokenizer_name=FacebookAI/xlm-roberta-base &

torchrun --nproc-per-node=8 train.py --config=configs/train/contrastive_pretrain_multilingual.yaml --dtype=bf16 --learning_rate=8.0e-5 --model_name=FacebookAI/xlm-roberta-large --tokenizer_name=FacebookAI/xlm-roberta-large --output_dir=ckpts/xlm-large-4k-8e5-pos-ids --gradient_checkpointing
OPENBLAS_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 python eval/mteb_eval/eval_mteb.py --model_name=ckpts/xlm-large-4k-8e5-pos-ids/epoch_0_model/ --add_prefix --no_normalize_classification --seq_len=128 --tokenizer_name=FacebookAI/xlm-roberta-large &
