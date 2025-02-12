#!/bin/bash

# Define the values for hyperparameters
batch_sizes=(16 32)
learning_rates=(1.0e-5 2.0e-5 3.0e-5)
seeds=(42 19 17 717 10536)

# Calculate total number of jobs
total_jobs=$((${#batch_sizes[@]} * ${#learning_rates[@]} * ${#seeds[@]}))
num_gpus=8
current_gpu=0

echo "Total experiments to run: $total_jobs"
echo "Available GPUs: $num_gpus"

# Initialize array to track running jobs per GPU
declare -A gpu_jobs
for ((i=0; i<num_gpus; i++)); do
    gpu_jobs[$i]=0
done

# Counter for port offset to avoid conflicts
port_offset=0

# Run experiments
for batch_size in "${batch_sizes[@]}"; do
    for seed in "${seeds[@]}"; do
        for lr in "${learning_rates[@]}"; do
            # Get next GPU (doing the round-robin inline to avoid subshell issues)
            gpu_id=$current_gpu
            current_gpu=$(( (current_gpu + 1) % num_gpus ))
            
            echo "Using GPU: $gpu_id (next will be $current_gpu)"
            
            # Calculate unique port for this run
            port=$((12345 + port_offset))
            port_offset=$((port_offset + 1))
            
            echo "Starting job on GPU $gpu_id with learning rate: $lr, seed: $seed, batch size: $batch_size"
            
            CUDA_VISIBLE_DEVICES=$gpu_id torchrun \
                --nproc-per-node=1 \
                --master-port=$port \
                train.py \
                --dtype=bf16 \
                --config=configs/train/glue.yaml \
                --learning_rate=$lr \
                --seed=$seed \
                --batch_size=$batch_size \
                --output_dir="ckpts/glue-$lr-$seed-$batch_size-gpu${gpu_id}" \
                &
            
            # Increment job counter for this GPU
            gpu_jobs[$gpu_id]=$((gpu_jobs[$gpu_id] + 1))
            
            # Print current GPU allocation
            echo "Current GPU allocation:"
            for ((i=0; i<num_gpus; i++)); do
                echo "GPU $i: ${gpu_jobs[$i]} jobs"
            done
            echo "-------------------"
            
            # Small sleep to prevent race conditions
            sleep 10 
            
            # Wait if we've launched num_gpus jobs
            if [ $((port_offset % num_gpus)) -eq 0 ]; then
                echo "Waiting for current batch of jobs to complete..."
                wait
                # Reset job counters
                for ((i=0; i<num_gpus; i++)); do
                    gpu_jobs[$i]=0
                done
            fi
        done
    done
done

# Wait for any remaining jobs to complete
wait

echo "All training jobs completed!"