#!/bin/bash

# Initialize variables
MODEL_PATH=""

# Function to show usage
usage() {
    echo "Usage: $0 <model_path> [--binarize]"
    exit 1
}

# Check for minimum number of arguments
if [ "$#" -lt 1 ]; then
    usage
fi

# Parse arguments
BINARIZE_MODE="off" # Default to not including the binarize flag
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --binarize) BINARIZE_MODE="on"; shift ;;
        *) MODEL_PATH="$1"; shift ;;
    esac
done

# Check if model path is not empty
if [ -z "$MODEL_PATH" ]; then
    usage
fi

# Array of dimensions
dims=(64 128 256 512 768)

# Iterate over the dimensions
for dim in "${dims[@]}"; do
    if [ "$BINARIZE_MODE" = "on" ]; then
        # If binarize mode is on, run both with and without the --binarize flag
        echo "Running evaluation for dimension: $dim with --binarize"
        OPENBLAS_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python eval/mteb_eval/eval_mteb.py \
            --model_name="${MODEL_PATH}" \
            --add_prefix \
            --no_normalize_classification \
            --matryoshka_dim="$dim" \
            --binarize &
        pid_binarize=$!
             
        
        echo "Running evaluation for dimension: $dim without --binarize"
        OPENBLAS_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python eval/mteb_eval/eval_mteb.py \
            --model_name="${MODEL_PATH}" \
            --add_prefix \
            --no_normalize_classification \
            --matryoshka_dim="$dim" &
        pid_no_binarize=$!

        wait $pid_binarize
        wait $pid_no_binarize
    else
        # If binarize mode is off, just run without the --binarize flag
        echo "Running evaluation for dimension: $dim"
        OPENBLAS_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python eval/mteb_eval/eval_mteb.py \
            --model_name="${MODEL_PATH}" \
            --add_prefix \
            --no_normalize_classification \
            --matryoshka_dim="$dim"
    fi
done