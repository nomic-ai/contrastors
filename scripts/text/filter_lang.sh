#!/bin/bash

# Directory containing MC4 language folders
MC4_DIR="mc4"

# Loop through all language directories in mc4
for lang_dir in "$MC4_DIR"/*/; do
    # Extract language code from directory path
    lang=$(basename "$lang_dir")
    
    # Skip if not a directory or if it ends with _filtered
    if [ ! -d "$lang_dir" ] || [[ "$lang" == *"_filtered"* ]] || [[ "$lang" == *"-Latn"* ]]; then
        continue
    fi
    
    filtered_dir="$MC4_DIR/${lang}_filtered"
    
    # Check if filtered directory doesn't exist or is empty
    if [ ! -d "$filtered_dir" ] || [ -z "$(ls -A "$filtered_dir" 2>/dev/null)" ]; then
        echo "Processing language: $lang"
        
        # Create filtered directory if it doesn't exist
        mkdir -p "$filtered_dir"
        
        # Run the filtering command
        torchrun --nproc-per-node=8 index_filtering.py \
            --dataset="$MC4_DIR/$lang" \
            --output_dir="$filtered_dir" \
            --query_key=title \
            --document_key=text \
            --batch_size=512 \
            --k=3
    else
        echo "Skipping $lang - filtered data already exists"
    fi
done