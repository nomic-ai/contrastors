from datasets import Dataset
from pathlib import Path
from transformers import AutoTokenizer
from tqdm import tqdm
import pyarrow as pa
from concurrent.futures import ProcessPoolExecutor


def tokenizer_query_and_document(query, document, tokenizer):
    """Tokenize query and document pairs."""
    query_tokens = tokenizer([f"query: {q}" for q in query], truncation=True, max_length=512)
    query_tokens = {f"query_{k}": v for k, v in query_tokens.items()}
    document_tokens = tokenizer([f"passage: {d}" for d in document], truncation=True, max_length=512)
    document_tokens = {f"document_{k}": v for k, v in document_tokens.items()}

    result = {**query_tokens, **document_tokens}
    return result


def load_and_tokenize_file(args):
    """Load and tokenize a single Arrow file."""
    file, tokenizer, output_dir = args
    try:
        memmap = pa.memory_map(str(file))
        reader = pa.ipc.open_file(memmap)
        batches = []
        for batch_idx in range(reader.num_record_batches):
            batch = reader.get_batch(batch_idx)
            batches.append(batch)

        ds = Dataset(pa.Table.from_batches(batches))
        ds = ds.map(
            lambda x: tokenizer_query_and_document(x["title"], x["text"], tokenizer),
            batched=True,
            batch_size=10000
        )

        output_file = Path(output_dir) / file.name
        with pa.ipc.new_stream(str(output_file), schema=ds.data.table.schema) as writer:
            writer.write_table(ds.data.table, max_chunksize=100_000)
        return str(output_file)
    except Exception as e:
        print(f"Error processing {file}: {str(e)}")
        return None


def process_language_directory(
    input_dir: Path,
    output_dir: Path,
    tokenizer,
    max_workers: int = 64
) -> list:
    """Process all Arrow files in a language directory."""
    arrow_files = sorted(input_dir.glob("*.arrow"))
    if not arrow_files:
        print(f"No arrow files found in {input_dir}")
        return []

    if output_dir.exists():
        output_files = sorted(output_dir.glob("*.arrow"))
        if len(output_files) == len(arrow_files):
            print(f"All files already processed in {output_dir}")
            return []

    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Skip files that are already processed
    arrow_files = [f for f in arrow_files if not (output_dir / f.name).exists()]
    if not arrow_files:
        print(f"All files already processed in {input_dir}")
        return []

    processed_files = []
    pbar = tqdm(total=len(arrow_files), desc=f"Processing {input_dir.name}")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for file in arrow_files:
            future = executor.submit(load_and_tokenize_file, (file, tokenizer, output_dir))
            future.add_done_callback(lambda _: pbar.update(1))
            futures.append(future)

        for future in futures:
            result = future.result()
            if result:
                processed_files.append(result)

    pbar.close()
    return processed_files


def process_mc4_languages(
    mc4_base_dir: Path,
    start_after_lang: str = "eo",
    max_workers: int = 64
) -> None:
    """Process multiple language directories in MC4 dataset, starting after specified language."""
    mc4_base_dir = Path(mc4_base_dir)
    
    # Get all arrow language directories
    lang_dirs = sorted([
        d for d in mc4_base_dir.iterdir() 
        if d.is_dir() and d.name.endswith('_arrow') and not d.name.endswith('_tokenized')
    ])
    
    # Find the index of the language to start after
    start_idx = 0
    for idx, lang_dir in enumerate(lang_dirs):
        lang_code = lang_dir.name.replace('_arrow', '')
        if lang_code == start_after_lang:
            start_idx = idx + 1
            break

    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small")
    tokenizer.model_max_length = 512
    
    # Process each language directory after the starting language
    for lang_dir in lang_dirs[0:1]:
        lang_code = lang_dir.name.replace('_arrow', '')
        print(f"\nProcessing language: {lang_code}")
        
        output_dir = mc4_base_dir / f"{lang_code}_arrow_tokenized"
        
        processed_files = process_language_directory(
            input_dir=lang_dir,
            output_dir=output_dir,
            tokenizer=tokenizer,
            max_workers=max_workers
        )
        
        print(f"Completed {lang_code}: Processed {len(processed_files)} files")


if __name__ == "__main__":
    mc4_dir = Path("/home/ubuntu/contrastors-dev/scripts/text/mc4")
    
    process_mc4_languages(
        mc4_base_dir=mc4_dir,
        start_after_lang="hy",
        max_workers=32
    )