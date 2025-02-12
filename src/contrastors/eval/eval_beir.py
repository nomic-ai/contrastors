import os
import tqdm
import torch.multiprocessing as mp
# Set start method to spawn
mp.set_start_method('spawn', force=True)

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import argparse
import logging
from datasets import Dataset
from typing import List, Dict
from functools import partial
from transformers import AutoTokenizer, PreTrainedTokenizerFast, BatchEncoding, DataCollatorWithPadding
from contrastors import BiEncoder, BiEncoderConfig
from torch.utils.data import DataLoader
from mteb import MTEB
from queue import Empty


def _setup_logger():
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    return logger


logger = _setup_logger()


parser = argparse.ArgumentParser(description='evaluation for BEIR benchmark')
parser.add_argument('--model-name-or-path', default='bert-base-uncased',
                    type=str, metavar='N', help='which model to use')
parser.add_argument('--tokenizer-name', default='FacebookAI/xlm-roberta-base',)
parser.add_argument('--output-dir', default='tmp-outputs/',
                    type=str, metavar='N', help='output directory')
parser.add_argument('--doc-as-query', action='store_true', help='use query prefix for passages')
parser.add_argument('--batch-size', default=1024, help='batch size', type=int)
parser.add_argument('--master-port', default=12355, help='master port', type=int)


args = parser.parse_args()
assert args.output_dir, 'output_dir should be set'
os.makedirs(args.output_dir, exist_ok=True)


def _transform_func(tokenizer: PreTrainedTokenizerFast,
                    examples: Dict[str, List]) -> BatchEncoding:
    return tokenizer(examples['contents'],
                     max_length=512,
                     padding=True,
                     truncation=True)


# Triton is not thread safe AFAICT so using naive DataParallel fails
class EncoderWorker(mp.Process):
    def __init__(self, rank, world_size, input_queue, output_queue, model_name, tokenizer_name, batch_size, master_port):
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.batch_size = batch_size
        self.master_port = master_port

    def run(self):
        # Initialize process group
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(self.master_port)
        os.environ['WORLD_SIZE'] = str(self.world_size)
        os.environ['RANK'] = str(self.rank)
        os.environ['LOCAL_RANK'] = str(self.rank)

        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(self.rank)

        # Initialize model
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        config = BiEncoderConfig.from_pretrained(self.model_name)
        encoder = BiEncoder.from_pretrained(self.model_name, config=config)
        encoder = encoder.to(self.rank)
        encoder = DistributedDataParallel(encoder, device_ids=[self.rank])
        encoder.eval()

        while True:
            try:
                # Get input texts from queue
                input_texts = self.input_queue.get(timeout=60)
                if input_texts is None:  # Poison pill
                    break

                # Process the batch
                dataset = Dataset.from_dict({'contents': input_texts})
                dataset.set_transform(partial(_transform_func, tokenizer))

                # Calculate actual number of samples for this worker
                total_size = len(dataset)
                per_worker = (total_size + self.world_size - 1) // self.world_size
                worker_start = self.rank * per_worker
                worker_end = min(worker_start + per_worker, total_size)
                actual_samples = worker_end - worker_start
                if actual_samples == 0:
                    # create fake work
                    worker_start -= per_worker

                print(f"Rank {self.rank} - Total size: {total_size}, Start: {worker_start}, End: {worker_end}, Actual samples: {actual_samples}")

                # Create indices for this worker
                indices = list(range(worker_start, worker_end))
                
                # Create a subset of the dataset
                subset = torch.utils.data.Subset(dataset, indices)

                data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
                loader = DataLoader(
                    subset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    collate_fn=data_collator,
                    num_workers=0,
                    pin_memory=True
                )

                local_embeds = []
                with torch.no_grad():
                    for batch_dict in tqdm.tqdm(loader, desc=f"Rank {self.rank}"):
                        batch_dict = {k: v.cuda(self.rank) for k, v in batch_dict.items()}
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            outputs = encoder(**batch_dict)
                            local_embeds.append(outputs["embedding"].cpu())

                local_embeds = torch.cat(local_embeds, dim=0)
                
                # Gather embeddings
                # Use actual_samples instead of embedding size for gathering
                print(f"Rank {self.rank} - Actual samples: {actual_samples}")
                local_size = torch.tensor([actual_samples], device=self.rank)
                print(f"Rank {self.rank} - Local size: {local_size}")
                all_sizes = [torch.zeros_like(local_size) for _ in range(self.world_size)]
                dist.all_gather(all_sizes, local_size)
                all_sizes = [size.item() for size in all_sizes]
                
                print(f"Rank {self.rank} max_size: {max(all_sizes)}, all_sizes: {all_sizes}")

                max_size = max(all_sizes)
                padded_embeds = torch.zeros(
                    max_size, local_embeds.shape[1],
                    dtype=local_embeds.dtype, device=self.rank
                )
                padded_embeds[:local_embeds.shape[0]] = local_embeds.cuda(self.rank)

                all_embeds = [torch.zeros_like(padded_embeds) for _ in range(self.world_size)]
                dist.all_gather(all_embeds, padded_embeds)

                if self.rank == 0:  # Only rank 0 returns results
                    result = []
                    for size, embeds in zip(all_sizes, all_embeds):
                        result.append(embeds[:size].cpu().numpy())
                    self.output_queue.put(np.concatenate(result, axis=0))
                
            except Empty:
                continue
            except Exception as e:
                print(f"Worker {self.rank} encountered error: {e}")
                if self.rank == 0:
                    self.output_queue.put(e)
                break

        dist.destroy_process_group()


class RetrievalModel:
    def __init__(self, **kwargs):
        self.world_size = torch.cuda.device_count()
        self.input_queue = mp.Queue()
        self.output_queue = mp.Queue()
        print(f"MASTER PORT: {args.master_port}")
        
        # Start worker processes
        self.workers = []
        for rank in range(self.world_size):
            worker = EncoderWorker(
                rank=rank,
                world_size=self.world_size,
                input_queue=self.input_queue,
                output_queue=self.output_queue,
                model_name=args.model_name_or_path,
                tokenizer_name=args.tokenizer_name,
                batch_size=args.batch_size,
                master_port=args.master_port
            )
            worker.start()
            self.workers.append(worker)

    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        input_texts = ['search_query: {}'.format(q) for q in queries]
        return self._do_encode(input_texts)

    def encode_corpus(self, corpus: List[Dict[str, str]], **kwargs) -> np.ndarray:
        if args.doc_as_query:
            return self.encode_queries([d['text'] for d in corpus], **kwargs)

        input_texts = ['{} {}'.format(doc.get('title', ''), doc['text']).strip() for doc in corpus]
        input_texts = ['search_document: {}'.format(t) for t in input_texts]
        return self._do_encode(input_texts)

    def _do_encode(self, input_texts: List[str]) -> np.ndarray:
        # Send input to all workers otherwise first worker will take all work
        for _ in range(self.world_size):
            self.input_queue.put(input_texts)
        
        # Get result from rank 0
        result = self.output_queue.get()
        
        # Check if result is an exception
        if isinstance(result, Exception):
            raise result
            
        print(f"Got result from rank 0: {len(result)}\t{result[:5]}")
        return result

    def __del__(self):
        # Send poison pills to workers
        for _ in range(self.world_size):
            self.input_queue.put(None)
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join()


def main():
    model = RetrievalModel()
    task_names = ['ArguAna', 'ClimateFEVER', 'CQADupstackAndroidRetrieval', 
                  'CQADupstackEnglishRetrieval', 'CQADupstackGamingRetrieval', 
                  'CQADupstackGisRetrieval', 'CQADupstackMathematicaRetrieval', 
                  'CQADupstackPhysicsRetrieval', 'CQADupstackProgrammersRetrieval', 
                  'CQADupstackStatsRetrieval', 'CQADupstackTexRetrieval', 
                  'CQADupstackUnixRetrieval', 'CQADupstackWebmastersRetrieval', 
                  'CQADupstackWordpressRetrieval', 'DBPedia', 'FEVER', 'FiQA2018', 
                  'HotpotQA', 'MSMARCO', 'NFCorpus', 'NQ', 'QuoraRetrieval', 
                  'SCIDOCS', 'SciFact', 'Touche2020', 'TRECCOVID']
    
    logger.info('Tasks: {}'.format(task_names))

    for task in task_names:
        logger.info('Processing task: {}'.format(task))

        args.doc_as_query = task in ['QuoraRetrieval']

        evaluation = MTEB(tasks=[task], task_langs=['en'])
        evaluation.run(model, eval_splits=["test" if task not in ['MSMARCO'] else 'dev'],
                       output_folder=args.output_dir,
                       overwrite_results=False)


if __name__ == '__main__':
    main()