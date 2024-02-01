#!/bin/bash

#!/bin/bash

# Define the values for learning rate and seed
batch_sizes=(16 32)
learning_rates=(1.0e-5 2.0e-5 3.0e-5)
seeds=(42 19 17 717 10536) 

# Loop over each combination of learning rate and seed
for batch_size in "${batch_sizes[@]}"; do
    for lr in "${learning_rates[@]}"; do
        for ((i = 0; i < "${#seeds[@]}"; i++)); do
            seed="${seeds[$i]}"
            echo "Training model with learning rate: $lr, seed: $seed, batch size: $batch_size"
            CUDA_VISIBLE_DEVICES=$i accelerate launch --num_processes=1 --main_process_port=$((12345 + i)) --num_machines=1 --machine_rank=0  --mixed_precision=bf16 finetune_glue.py --config=configs/train/glue.yaml --learning_rate=$lr --seed=$seed --batch_size=$batch_size &
        done

        wait
    done
done