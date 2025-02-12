import concurrent.futures
import json
import gzip
import os
from pathlib import Path
import pyarrow as pa
from typing import Union, Dict, Any, List
from tqdm import tqdm

def convert_single_file(args: tuple) -> Path:
    """
    Convert a single JSONL.GZ file to Arrow format.
    
    Args:
        args: Tuple containing (input_file, output_file, batch_size)
    
    Returns:
        Path to created Arrow file or None if failed
    """
    input_file, output_file, batch_size, query_key, document_key = args
    
    current_batch: list[Dict[str, Any]] = []
    schema = None
    writer = None

    pbar = tqdm(total=os.path.getsize(input_file), unit="B", desc=f"Processing {input_file.name}")
    
    try:
        with gzip.open(input_file, 'rt') as f:
            for line in f:
                record = json.loads(line)
                current_batch.append(record)
                
                if len(current_batch) >= batch_size:
                    if schema is None:
                        table = pa.Table.from_pylist(current_batch)
                        schema = table.schema
                        writer = pa.ipc.new_file(output_file, schema)
                    else:
                        table = pa.Table.from_pylist(current_batch, schema=schema)
                    
                    writer.write_table(table)
                    current_batch = []

                pbar.update(f.buffer.fileobj.tell() - pbar.n)
            
            if current_batch:
                if schema is None:
                    table = pa.Table.from_pylist(current_batch)
                    schema = table.schema
                    writer = pa.ipc.new_file(output_file, schema)
                else:
                    table = pa.Table.from_pylist(current_batch, schema=schema)
                writer.write_table(table)
        
        if writer:
            print(f"Finished processing {input_file}")
            writer.close()
            return output_file
            
    except Exception as e:
        print(f"Error processing {input_file}: {str(e)}")
        if writer:
            writer.close()
        if output_file.exists():
            output_file.unlink()  # Remove partial file on error
    
    return None

def convert_jsonl_directory_to_arrow(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    batch_size: int = 10000,
    file_pattern: str = "*.jsonl.gz",
    max_workers: int = None
) -> List[Path]:
    """
    Convert all JSONL.GZ files in a directory to Arrow format using multiple processes.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    conversion_args = []
    for input_file in input_dir.glob(file_pattern):
        output_file = output_dir / f"{input_file.stem.replace('.jsonl', '')}.arrow"
        if output_file.exists():
            continue
        conversion_args.append((input_file, output_file, batch_size, "title", "text"))
    
    created_files = []
    pbar = tqdm(total=len(conversion_args), desc="Converting files")
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(convert_single_file, args): args[0] 
                         for args in conversion_args}
        
        for future in concurrent.futures.as_completed(future_to_file):
            input_file = future_to_file[future]
            try:
                output_file = future.result()
                if output_file:
                    created_files.append(output_file)
                    pbar.update(1)
            except Exception as e:
                print(f"Failed to convert {input_file}: {str(e)}")
    
    return created_files

def process_mc4_languages(
    mc4_base_dir: Union[str, Path],
    output_base_dir: Union[str, Path],
    start_after_lang: str = "fr",
    batch_size: int = 10240,
    max_workers: int = 32
) -> None:
    """
    Process multiple language directories in MC4 dataset, starting after specified language.
    
    Args:
        mc4_base_dir: Base directory containing MC4 language folders
        output_base_dir: Base directory where Arrow files will be saved
        start_after_lang: Start processing after this language code
        batch_size: Number of records to process at once
        max_workers: Maximum number of worker processes per language
    """
    mc4_base_dir = Path(mc4_base_dir)
    output_base_dir = Path(output_base_dir)
    
    # Get all language directories
    lang_dirs = sorted([d for d in mc4_base_dir.iterdir() if d.is_dir() and not d.name.endswith('_arrow') and not d.name.endswith('_tokenized') and "arrow" not in d.name and "rows" not in d.name and "filtered" not in d.name])
    
    # Find the index of the language to start after
    start_idx = 0
    for idx, lang_dir in enumerate(lang_dirs):
        if lang_dir.name == start_after_lang:
            start_idx = idx + 1
            break

    # Process each language directory after the starting language
    for lang_dir in lang_dirs[0: 1]:
        lang_code = lang_dir.name
        print(f"\nProcessing language: {lang_code}")
        
        output_dir = output_base_dir / f"{lang_code}_arrow"
        
        # Convert files for this language
        arrow_files = convert_jsonl_directory_to_arrow(
            lang_dir,
            output_dir,
            batch_size=batch_size,
            max_workers=max_workers
        )
        
        print(f"Completed {lang_code}: Created {len(arrow_files)} Arrow files")

if __name__ == "__main__":
    mc4_dir = "/home/ubuntu/contrastors-dev/scripts/text/mc4"
    output_dir = "/home/ubuntu/contrastors-dev/scripts/text/mc4"
    
    process_mc4_languages(
        mc4_base_dir=mc4_dir,
        output_base_dir=output_dir,
        start_after_lang="en",
        batch_size=10240,
        max_workers=32
    )