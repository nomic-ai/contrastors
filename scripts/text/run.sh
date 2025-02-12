#!/bin/bash

# Directory containing the arrow files
BASE_DIR="mc4"
STARTED=0

# Loop through all directories in BASE_DIR
for dir in $(ls "$BASE_DIR" | sort); do
    # Check if directory ends with _arrow_tokenized
    if [[ $dir == *"_arrow_tokenized" ]]; then
        # Extract language code
        lang=${dir%_arrow_tokenized}
        
        # Check if we've found fr (start marker)
        if [[ $lang == "fi" ]]; then
            STARTED=1
        fi
        
        # Only process if we've passed fr
        if [[ $STARTED -eq 1 ]]; then
            echo "Processing language: $lang"
            
            # Check if the tokenized directory exists
            if [ -d "$BASE_DIR/${lang}_arrow_tokenized" ]; then
                echo "Running processing for $lang..."
                
                torchrun --nproc-per-node=8 \
                    index_filtering_new.py \
                    --dataset="$BASE_DIR/${lang}_arrow_tokenized" \
                    --output_dir="$BASE_DIR/${lang}_arrow_filtered" \
                    --query_key=title \
                    --document_key=text \
                    --k=3 \
                    --batch_size=1024
                
                if [ $? -eq 0 ]; then
                    echo "Successfully processed $lang"
                else
                    echo "Error processing $lang"
                fi
            else
                echo "Skipping $lang - tokenized directory not found"
            fi
            echo "----------------------------------------"
        fi
    fi
done

echo "Processing complete"