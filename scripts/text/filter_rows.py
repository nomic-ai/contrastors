import json
import gzip
from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm
import multiprocessing as mp
import concurrent.futures
import math

def parse_args():
    parser = ArgumentParser(description="Filter dataset based on row numbers to keep")
    parser.add_argument("--input_dir", required=True, help="Directory containing the original dataset files")
    parser.add_argument("--output_dir", required=True, help="Directory to save filtered dataset files")
    parser.add_argument("--ids_dir", required=True, help="Directory containing the row numbers to keep JSON files")
    parser.add_argument("--file_pattern", default="shard-*.jsonl.gz", help="Pattern to match input files")
    parser.add_argument("--num_processes", type=int, default=None, 
                       help="Number of processes to use. Defaults to CPU count - 1")
    return parser.parse_args()

def load_rows_to_keep(ids_dir):
    rows_to_keep = set()
    for json_file in Path(ids_dir).glob("ids_to_keep_*.json"):
        with open(json_file, 'r') as f:
            rows_to_keep.update(json.load(f))
    return rows_to_keep

def filter_single_file(args):
    input_file, output_dir, start_id, rows_to_keep, total_file_size = args
    num_kept = 0
    current_id = start_id

    pbar = tqdm(total=total_file_size, desc=f"Processing {input_file}")
    try:
        with gzip.open(input_file, 'rt') as in_f, gzip.open(output_dir / input_file.name, 'wt') as out_f:
            for line in in_f:
                if current_id in rows_to_keep:
                    out_f.write(line)
                    num_kept += 1
                current_id += 1
                pbar.update(1)
        pbar.close()
        return num_kept, 0  # Return (kept_rows, error_count)
    except Exception as e:
        print(f"Error processing {input_file}: {str(e)}")
        return 0, 1  # Return (kept_rows, error_count)

def calculate_start_ids(input_dir):
    """Calculate starting ID for each file based on line counts."""
    start_ids = {}
    current_id = 0
    
    # load counts.json from input_files directory
    # and count offsets by sorting the keys in numerical format
    with open(input_dir / 'counts.json', 'r') as f:
        counts = json.load(f)['count_per_file']

    for file_name in sorted(counts.keys()):
        start_ids[file_name] = current_id
        current_id += counts[file_name]

    return start_ids, counts

def filter_files_parallel(input_files, output_dir, rows_to_keep, num_processes=None):
    if num_processes is None:
        num_processes = max(1, mp.cpu_count() - 1)
    
    # Calculate starting IDs for each file
    input_dir = input_files[0].parent
    start_ids, counts = calculate_start_ids(input_dir)

    # Prepare arguments for parallel processing
    process_args = [
        (input_file, output_dir, start_ids[str("mc4" / input_file)], rows_to_keep, counts[str("mc4" / input_file)])
        for input_file in input_files
    ]
    
    total_kept = 0
    total_errors = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Use tqdm to show progress
        results = list(tqdm(
            executor.map(filter_single_file, process_args),
            total=len(process_args),
            desc="Filtering files"
        ))
        
        # Sum up results
        for kept, errors in results:
            total_kept += kept
            total_errors += errors
    
    return total_kept, total_errors

def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_to_keep = load_rows_to_keep(args.ids_dir)
    print(f"Loaded {len(rows_to_keep):,} row numbers to keep")

    input_files = sorted(input_dir.glob(args.file_pattern))
    print(f"Found {len(input_files):,} files to process")
    
    total_kept_rows, total_errors = filter_files_parallel(
        input_files, 
        output_dir, 
        rows_to_keep,
        num_processes=args.num_processes,
    )

    print(f"Filtered dataset saved to {output_dir}")
    print(f"Kept {total_kept_rows} rows")
    print(f"Missing {len(rows_to_keep) - total_kept_rows} rows")
    if total_errors > 0:
        print(f"Encountered errors while processing {total_errors} files")

if __name__ == "__main__":
    main()