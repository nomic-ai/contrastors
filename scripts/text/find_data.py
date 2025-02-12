import yaml
import fsspec
import json
from collections import defaultdict
import re
from tqdm import tqdm

def get_actual_shard_range(bucket_name, prefix):
    """
    List objects in S3 bucket to determine actual shard range
    """
    s3 = fsspec.filesystem('s3')
    all_shards = set()

    shards = sorted(s3.glob(f"s3://{bucket_name}/{prefix}/shard-*.jsonl.gz"))
    if shards:
        for shard in shards:
            shard_num = int(shard.split("-")[-1].split(".")[0])
            all_shards.add(shard_num)
    
    if all_shards:
        return min(all_shards), max(all_shards)
    return None, None

def update_bucket_path(bucket_path):
    """
    Update bucket path with actual shard range from S3
    """
    # Extract bucket name and prefix
    parts = bucket_path.split('/')
    bucket_name = parts[2]
    prefix = parts[3]
    
    # Check if path contains shard pattern
    shard_match = re.search(r'\{(\d+)\.\.(\d+)\}', bucket_path)
    if not shard_match:
        return bucket_path
    
    # Get actual shard range
    min_shard, max_shard = get_actual_shard_range(bucket_name, prefix)
    if min_shard is not None and max_shard is not None:
        # Format shard numbers with leading zeros
        min_str = str(min_shard).zfill(5)
        max_str = str(max_shard).zfill(5)
        # Replace the shard range in the path
        updated_path = re.sub(
            r'\{\d+\.\.\d+\}',
            f'{{{min_str}..{max_str}}}',
            bucket_path
        )
        return updated_path
    
    return bucket_path

def process_yaml():
    # Read the original YAML
    with open('/home/ubuntu/contrastors-dev/src/contrastors/configs/data/contrastive_pretrain_multilingual.yaml', 'r') as f:
        data = yaml.safe_load(f)
    
    # Process datasets to collect language info
    language_files = defaultdict(int)
    updated_datasets = []
    
    for dataset in tqdm(data['datasets']):
        bucket = dataset['bucket']
        
        # Extract language code if present
        lang_match = re.search(r'/([a-z]{2})/', bucket)
        if lang_match:
            continue
        else:
            lang = bucket.split("/")[3]
        
        # Update shard range based on actual S3 contents
        bucket = update_bucket_path(bucket)
        dataset['bucket'] = bucket
        
        # Count number of files based on shard pattern
        shard_match = re.search(r'\{(\d+)\.\.(\d+)\}', bucket)
        if shard_match:
            start, end = map(int, shard_match.groups())
            num_files = end - start + 1
        else:
            num_files = 1
            
        language_files[lang] += num_files

        updated_datasets.append(dataset)

    # add all missing datasets from multilingual data
    s3 = fsspec.filesystem("s3")
    multilingual_datasets = []
    mc4_columns = ["title", "text"]
    cc_news_columns = ["title", "maintext"]
    for multilingual_bucket in ["mc4-filtered", "multilingual-cc-news-filtered"]:
        metadata = {"objective": {"type": "paired", "columns": mc4_columns if multilingual_bucket == "mc4-filtered" else cc_news_columns},
                    "query_prefix": "search_query",
                    "document_prefix": "search_document"
                    }
        
        languages = sorted(s3.ls(f"s3://{multilingual_bucket}/"))

        for lang in tqdm(languages, desc=f"Processing {multilingual_bucket}"):
            if f"{multilingual_bucket}_{lang}" not in language_files:
                new_dataset = {"name": f"{lang.replace('/', '_')}", **metadata}
                bucket = update_bucket_path(f"s3://{lang}/shard-{{00000..1}}jsonl.gz")
                new_dataset["bucket"] = bucket
                multilingual_datasets.append(new_dataset)

                shard_match = re.search(r'\{(\d+)\.\.(\d+)\}', bucket)
                if shard_match:
                    start, end = map(int, shard_match.groups())
                    num_files = end - start + 1
                else:
                    num_files = 1

                language_files[f"{multilingual_bucket}_{lang}"] = num_files

    updated_datasets.extend(multilingual_datasets)
    
    # Write updated YAML
    with open('datasets_updated.yaml', 'w') as f:
        yaml.dump({'datasets': updated_datasets}, f, sort_keys=False)
    
    return language_files
    

def get_s3_counts():
    s3 = fsspec.filesystem('s3')
    total_counts = defaultdict(int)
    
    with open('datasets_updated.yaml', 'r') as f:
        data = yaml.safe_load(f)
    
    for dataset in tqdm(data['datasets']):
        parts = dataset['bucket'].split('/')
        bucket_name = parts[2]
        prefix = parts[3]
        
        try:
            with s3.open(f"s3://{bucket_name}/{prefix}/counts.json") as f:
                counts_data = json.load(f)
            total_counts[dataset['name']] = counts_data['total_count']
        except Exception as e:
            rows = 0
            for shard, num_rows in counts_data.items():
                if "shard-" not in shard:
                    continue
                rows += num_rows
            total_counts[dataset['name']] = rows
        
    
    return total_counts

def main():
    # Process YAML and print language statistics
    print("Updating YAML file with actual shard ranges...")
    language_files = process_yaml()
    
    print("\nNumber of files per language:")
    for lang, count in sorted(language_files.items(), key=lambda x: x[1], reverse=True):
        print(f"{lang}: {count} files")
    
    # Get and print counts from S3
    print("\nRetrieving sample counts from S3...")
    counts = get_s3_counts()
    total_samples = sum(counts.values())
    
    print("\nSample counts per dataset:")
    for dataset, count in sorted(counts.items()):
        if count == 0:
            continue
        print(f"{dataset}: {count:,} samples")
    print(f"\nTotal samples across all datasets: {total_samples:,}")

if __name__ == "__main__":
    main()