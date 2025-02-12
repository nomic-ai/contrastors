import os
from datasets import Dataset
from tqdm import tqdm

# Directory containing the datasets
base_dir = "cc100"

# Dictionary to store dataset lengths
dataset_lengths = {}

# Iterate over all subdirectories in the base directory
for lang in tqdm(os.listdir(base_dir)):
    lang_path = os.path.join(base_dir, lang)
    
    # Check if it's a directory
    if os.path.isdir(lang_path):
        try:
            # Load the dataset
            dataset = Dataset.load_from_disk(lang_path)
            
            # Store the length in the dictionary
            dataset_lengths[lang] = len(dataset)
            
            print(f"Processed {lang}: {len(dataset)} samples")
        except Exception as e:
            print(f"Error processing {lang}: {str(e)}")

# Print the results
print("\nDataset Lengths:")
for lang, length in dataset_lengths.items():
    print(f"{lang}: {length}")

# Optionally, you can save the dictionary to a file
import json
with open("dataset_lengths.json", "w") as f:
    json.dump(dataset_lengths, f, indent=2)

print("\nResults saved to dataset_lengths.json")