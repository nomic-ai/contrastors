import torch
import random
from typing import List, Dict, Optional
from torch.utils.data import IterableDataset
import torch.distributed as dist
from pathlib import Path
import pyarrow as pa
from contrastors.dataset.constants import MULTILINGUAL_LENGTHS
from contrastors.distributed import print_in_order
import numpy as np


class ArrowReader:
    def __init__(self, file_path: str | Path, infinite: bool = True) -> None:
        """
        Initialize the infinite reader for a memory mapped Arrow IPC stream file.
        
        Args:
            file_path: Path to the Arrow IPC stream file
        """
        self.file_path = Path(file_path)
        self.source: Optional[pa.MemoryMappedFile] = None
        self.reader: Optional[pa.ipc.RecordBatchStreamReader] = None
        self._initialize_reader()
        self.infinite = infinite

    def _initialize_reader(self) -> None:
        """Initialize or reinitialize the memory mapped file and stream reader."""
        # Close existing resources if they exist
        if self.reader is not None:
            self.reader.close()
        if self.source is not None:
            self.source.close()
        
        # Create new memory mapped file and reader
        self.source = pa.memory_map(str(self.file_path))
        self.reader = pa.ipc.open_stream(self.source)

    def read_next_batch(self) -> pa.RecordBatch:
        """
        Read the next batch, recreating the reader when reaching the end.
        
        Returns:
            pyarrow.RecordBatch: The next batch of data
        
        Raises:
            FileNotFoundError: If the source file doesn't exist
            pa.ArrowInvalid: If the file is not a valid Arrow IPC stream
        """
        try:
            if self.reader is None:
                self._initialize_reader()
            return self.reader.read_next_batch()
        except StopIteration as e:
            if not self.infinite:
                raise e
            print(f"{self.file_path} reached end of stream, re-initializing reader")
            self._initialize_reader()
            return self.reader.read_next_batch()

    def close(self) -> None:
        """Clean up resources."""
        if self.reader is not None:
            self.reader.close()
        if self.source is not None:
            self.source.close()

class BatchedArrowFileReader:
    def __init__(self, path: Path, infinite: bool = True):
        self.path = path

        self.stream = ArrowReader(path, infinite=infinite)
        self.row_overflow = None

        
    @property
    def schema(self):
        return self.stream.reader.schema
        
    def read_lines(self, num_lines: int):
        if self.row_overflow is not None:
            batch = self.row_overflow.slice(offset=0, length=num_lines)
            if len(self.row_overflow) - num_lines > 0:
                self.row_overflow = self.row_overflow.slice(offset=num_lines)
            else:
                self.row_overflow = None
        else:
            batch = pa.Table.from_batches([self.stream.read_next_batch()])
            # first get remaining rows in overflow
            if self.row_overflow is not None:
                batch = pa.concat_tables([batch, self.row_overflow])
                self.row_overflow = None

        while len(batch) < num_lines:
            next_batch = pa.Table.from_batches([self.stream.read_next_batch()])
            batch = pa.concat_tables([batch, next_batch])

        if len(batch) > num_lines:
            overflow = batch.slice(offset=num_lines) 
            batch = batch.slice(offset=0, length=num_lines)
            self.row_overflow = overflow

        return batch

    def close(self) -> None:
        """Clean up resources."""
        self.stream.close()


class DistributedIterableMLMDataset(IterableDataset):
    def __init__(
        self,
        dataset_name: str,
        languages: Optional[List[str]] = None,
        mlm_probability: float = 0.30,
        max_length: int = 2048,
        seed: int = 42,
        global_batch_size: int = 32,
    ):
        super().__init__()
        self.base_path = Path(dataset_name)
        self.mlm_probability = mlm_probability
        self.max_length = max_length
        self.seed = seed
        self.global_batch_size = global_batch_size
        
        # Setup distributed training info
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

        assert self.global_batch_size % self.world_size == 0
        self.batch_size = self.global_batch_size // self.world_size
            
        # Find all arrow files for each language
        self.language_files: Dict[str, List[Path]] = {}
        if languages is None:
            # Auto-detect languages from folders
            languages = [d.name for d in self.base_path.iterdir() if d.is_dir()]
        
        for lang in languages:
            lang_path = self.base_path / lang
            if not lang_path.exists():
                continue
            arrow_files = sorted(lang_path.glob("*.arrow"))
            if arrow_files:
                if lang == "en":
                    # save last file for eval
                    arrow_files = arrow_files[:-1]

                self.language_files[lang] = arrow_files
                
        if not self.language_files:
            raise ValueError(f"No arrow files found in {self.base_path}")
        
        # Calculate sizes and sampling weights
        self.sizes = MULTILINGUAL_LENGTHS
        self.weights = self._calculate_sampling_weights()
        
        print(f"Found {len(self.language_files)} languages: {list(self.language_files.keys())}")
        print(f"Total files: {sum(len(files) for files in self.language_files.values())}")
    
    def _calculate_sampling_weights(self, alpha: float = 0.3) -> np.ndarray:
        """Calculate sampling weights using exponential smoothing"""
        total_rows = sum(self.sizes.values())
        p_i = np.array([self.sizes[lang] / total_rows for lang in sorted(self.language_files.keys())])
        q_i = p_i ** alpha
        weights = q_i / q_i.sum()
        return weights

    def __iter__(self):
        # Setup RNG for language sampling
        rng = random.Random(self.seed)
        languages = sorted(self.language_files.keys())
        
        # Create arrow readers for each file
        arrow_streams = {
            lang: [BatchedArrowFileReader(file) for file in files]
            for lang, files in self.language_files.items()
        }
        try:
            while True:
                # Sample a language
                lang = rng.choices(languages, weights=self.weights, k=1)[0]
                if not arrow_streams[lang]:  # If no more files for this language
                    # this shouldn't happen since we have an infinite generator
                    if all(not r for r in arrow_streams.values()):  # If no more files at all
                        raise ValueError(f"No files left for {lang}. Something is wrong!")
                    continue
                
                # Get a random reader for this language
                stream_idx = rng.randrange(len(arrow_streams[lang]))
                arrow_stream = arrow_streams[lang][stream_idx]
                
                try:
                    schema = arrow_stream.schema
                    include_indices = [field.name for field in schema if field.name != "id"]
                    global_batch = arrow_stream.read_lines(self.global_batch_size)
                    global_batch = global_batch.select(include_indices)
                    local_batch_offset = self.batch_size * self.rank
                    local_batch = global_batch.slice(offset=local_batch_offset, length=self.batch_size)
                    # add column with language name
                    language = [lang] * self.batch_size
                    local_batch = local_batch.append_column("lang", pa.array(language, type=pa.string()))
                    print_in_order(f"{lang=}, {self.rank=}, {local_batch_offset=}, {local_batch.num_rows=}")

                    yield local_batch.to_pylist()
                            
                except Exception as e:
                    print(f"Error reading from {lang} file: {e}")
                    arrow_stream.close()
                    arrow_streams[lang].pop(stream_idx)
                    continue
                    
        finally:
            # Cleanup
            for lang_readers in arrow_streams.values():
                for reader in lang_readers:
                    reader.close()

                    
class EvalDistributedIterableMLMDataset(IterableDataset):
    def __init__(
        self,
        dataset_name: str,
        languages: Optional[List[str]] = None,
        mlm_probability: float = 0.30,
        max_length: int = 2048,
        seed: int = 42,
        global_batch_size: int = 32,
    ):
        super().__init__()
        self.base_path = Path(dataset_name)
        self.mlm_probability = mlm_probability
        self.max_length = max_length
        self.seed = seed
        self.global_batch_size = global_batch_size
        
        # Setup distributed training info
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

        assert self.global_batch_size % self.world_size == 0
        self.batch_size = self.global_batch_size // self.world_size
            
        # Find all arrow files for each language
        self.language_files: Dict[str, List[Path]] = {}
        assert languages == ["en"], languages
        if languages is None:
            # Auto-detect languages from folders
            languages = [d.name for d in self.base_path.iterdir() if d.is_dir()]

        lang_path = self.base_path / "en"
        arrow_files = sorted(lang_path.glob("*.arrow"))
        self.eval_file = arrow_files[-1]
        memmap = pa.memory_map(str(self.eval_file))
        stream = pa.ipc.open_stream(memmap)
        num_rows = 0
        for batch in stream:
            num_rows += len(batch)

        self.num_rows = num_rows
        print(f"{self.num_rows=}, {self.batch_size=}, {self.global_batch_size=}")
        self.num_batches = self.num_rows // self.global_batch_size

    def __iter__(self):
        arrow_stream = BatchedArrowFileReader(self.eval_file, infinite=False)

        while True:
            try:
                schema = arrow_stream.schema
                include_indices = [field.name for field in schema if field.name != "id"]
                global_batch = arrow_stream.read_lines(self.global_batch_size)
                global_batch = global_batch.select(include_indices)
                local_batch_offset = self.batch_size * self.rank
                local_batch = global_batch.slice(offset=local_batch_offset, length=self.batch_size)
                # add column with language name

                yield local_batch.to_pylist()

            except StopIteration:
                arrow_stream.close()
                break