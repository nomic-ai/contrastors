import logging
import math
import multiprocessing as mp
import os
import queue
import time
from typing import Dict, List

import numpy as np
import tiktoken
import torch
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from contrastors import BiEncoder, BiEncoderConfig

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("main")


def cutoff_long_text_for_embedding_generation(text, encoding, cutoff=8192):
    encoded_text = encoding.encode(text)[:cutoff]
    decoded_text = encoding.decode(encoded_text)
    return decoded_text


def split_list(input_list, chunk_size):
    for i in range(0, len(input_list), chunk_size):
        yield input_list[i : i + chunk_size]


class OpenAI_Encoder:
    def __init__(self, embedding_model="text-embedding-ada-002", batch_size=32, cutoff=8192, **kwargs):
        self.client = OpenAI()
        self.embedding_model = embedding_model
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.encoder_batch_size = batch_size
        self.cutoff = cutoff

    def encode(self, sentences: List[str], batch_size: int, **kwargs) -> np.ndarray:
        sentences = [
            cutoff_long_text_for_embedding_generation(sentence, self.encoding, cutoff=self.cutoff)
            for sentence in sentences
        ]
        total_encoded_sentences = []
        for sentence_chunks in tqdm(split_list(sentences, self.encoder_batch_size)):
            try:
                encoded_sentences = self.client.embeddings.create(input=sentence_chunks, model=self.embedding_model)
            except:
                time.sleep(30)
                encoded_sentences = self.client.embeddings.create(input=sentence_chunks, model=self.embedding_model)

            encoded_sentences = [sentence_encoding.embedding for sentence_encoding in encoded_sentences.data]
            total_encoded_sentences += encoded_sentences
        return np.array(total_encoded_sentences)

    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:

        queries = [
            cutoff_long_text_for_embedding_generation(query, self.encoding, cutoff=self.cutoff) for query in queries
        ]
        total_encoded_queries = []
        for query_chunks in tqdm(split_list(queries, self.encoder_batch_size)):
            try:
                encoded_queries = self.client.embeddings.create(input=query_chunks, model=self.embedding_model)
            except:
                time.sleep(30)
                encoded_queries = self.client.embeddings.Embedding.create(
                    input=query_chunks, model=self.embedding_model
                )

            encoded_queries = [query_encoding.embedding for query_encoding in encoded_queries.data]
            total_encoded_queries += encoded_queries
        return np.array(total_encoded_queries)

    # Write your own encoding corpus function (Returns: Document embeddings as numpy array)
    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs) -> np.ndarray:
        if isinstance(corpus[0], dict):
            passages = ['{} {}'.format(doc.get('title', ''), doc['text']).strip() for doc in corpus]
        else:
            passages = corpus
        passages = [
            cutoff_long_text_for_embedding_generation(passage, self.encoding, cutoff=self.cutoff)
            for passage in passages
        ]
        total_encoded_passages = []
        for passage_chunks in tqdm(split_list(passages, self.encoder_batch_size)):
            try:
                encoded_passages = self.client.embeddings.create(input=passage_chunks, model=self.embedding_model)
            except:
                time.sleep(30)
                encoded_passages = self.client.embeddings.create(input=passage_chunks, model=self.embedding_model)

            encoded_passages = [passage_encoding.embedding for passage_encoding in encoded_passages.data]
            total_encoded_passages += encoded_passages
        return np.array(total_encoded_passages)


class STransformer:
    def __init__(
        self,
        model,
        add_prefix=False,
        query_prefix="search_query",
        document_prefix="search_document",
        normalize=True,
        binarize=False,
    ):
        self.model = model
        self.gpu_pool = self.model.start_multi_process_pool()
        self.add_prefix = add_prefix
        self.doc_as_query = False
        self.query_prefix = query_prefix
        self.docoment_prefix = document_prefix
        self.normalize = normalize
        self.binarize = binarize

    def set_normalize(self, normalize):
        self.normalize = normalize

    def encode(self, sentences, **kwargs):
        if self.add_prefix:
            print(f"Adding prefix: {self.query_prefix}")
            sentences = [f"{self.query_prefix}: {sent}" for sent in sentences]
        kwargs["normalize"] = self.normalize
        kwargs["binarize"] = self.binarize
        return self.model.encode_multi_process(sentences, self.gpu_pool, **kwargs)

    def encode_queries(self, queries, **kwargs) -> np.ndarray:
        if self.add_prefix:
            input_texts = [f'{self.query_prefix}: {q}' for q in queries]
        else:
            input_texts = queries

        kwargs["normalize"] = self.normalize
        kwargs["binarize"] = self.binarize
        return self.model.encode_multi_process(input_texts, self.gpu_pool, **kwargs)

    def encode_corpus(self, corpus, **kwargs) -> np.ndarray:
        if isinstance(corpus[0], dict):
            input_texts = ['{} {}'.format(doc.get('title', ''), doc['text']).strip() for doc in corpus]
        else:
            input_texts = corpus
        if self.add_prefix:
            if self.doc_as_query:
                input_texts = [f'{self.query_prefix}: {t}' for t in input_texts]
            else:
                input_texts = [f'{self.docoment_prefix}: {t}' for t in input_texts]

        kwargs["normalize"] = self.normalize
        kwargs["binarize"] = self.binarize
        return self.model.encode_multi_process(input_texts, self.gpu_pool, **kwargs)


class Encoder:
    def __init__(
        self,
        model_name,
        tokenizer_name="bert-base-uncased",
        seq_length=512,
        rotary_scaling_factor=None,
        matryoshka_dim=None,
    ):

        if os.path.exists(model_name):
            config = BiEncoderConfig.from_pretrained(model_name)
            if rotary_scaling_factor is not None:
                config.rotary_scaling_factor = rotary_scaling_factor
            self.model = BiEncoder.from_pretrained(model_name, config=config).to(torch.bfloat16)
        else:
            config = BiEncoderConfig(model_name=model_name, encoder=True, pooling="mean")
            if rotary_scaling_factor is not None:
                config.rotary_scaling_factor = rotary_scaling_factor
            self.model = BiEncoder(config).to(torch.bfloat16)

        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.model_max_length = seq_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.matryoshka_dim = matryoshka_dim

    def encode(self, sentences, batch_size=256, **kwargs):
        embeddings = []

        device = kwargs.get("device", self.device)
        normalize = kwargs.get("normalize", True)
        binarize = kwargs.get("binarize", False)
        self.model.to(device)

        with torch.no_grad():
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i : i + batch_size]
                encoded = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
                outputs = self.model(**encoded.to(device), normalize=normalize, binarize=binarize)
                embs = outputs["embedding"].cpu().float().numpy()
                if self.matryoshka_dim:
                    embs = embs[:, : self.matryoshka_dim]
                    if normalize:
                        embs = embs / np.expand_dims(np.linalg.norm(embs, axis=-1), axis=1)
                embeddings.extend(embs)

        return embeddings

    def start_multi_process_pool(self, target_devices=None):
        """
        Starts multi process to process the encoding with several, independent processes.
        This method is recommended if you want to encode on multiple GPUs. It is advised
        to start only one process per GPU. This method works together with encode_multi_process

        :param target_devices: PyTorch target devices, e.g. cuda:0, cuda:1... If None, all available CUDA devices will be used
        :return: Returns a dict with the target processes, an input queue and and output queue.
        """
        if target_devices is None:
            if torch.cuda.is_available():
                target_devices = ['cuda:{}'.format(i) for i in range(torch.cuda.device_count())]
            else:
                logger.info("CUDA is not available. Start 4 CPU worker")
                target_devices = ['cpu'] * 4

        logger.info("Start multi-process pool on devices: {}".format(', '.join(map(str, target_devices))))

        ctx = mp.get_context('spawn')
        input_queue = ctx.Queue()
        output_queue = ctx.Queue()
        processes = []

        for cuda_id in target_devices:
            p = ctx.Process(
                target=Encoder._encode_multi_process_worker,
                args=(cuda_id, self, input_queue, output_queue),
                daemon=True,
            )
            p.start()
            processes.append(p)

        return {'input': input_queue, 'output': output_queue, 'processes': processes}

    @staticmethod
    def _encode_multi_process_worker(target_device: str, model, input_queue, results_queue):
        """
        Internal working process to encode sentences in multi-process setup
        """
        while True:
            try:
                id, batch_size, sentences, normalize, binarize = input_queue.get()
                embeddings = model.encode(
                    sentences,
                    device=target_device,
                    normalize=normalize,
                    binarize=binarize,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    batch_size=batch_size,
                )
                results_queue.put([id, embeddings])
            except queue.Empty:
                break

    @staticmethod
    def stop_multi_process_pool(pool):
        """
        Stops all processes started with start_multi_process_pool
        """
        for p in pool['processes']:
            p.terminate()

        for p in pool['processes']:
            p.join()
            p.close()

        pool['input'].close()
        pool['output'].close()

    def encode_multi_process(
        self,
        sentences,
        pool,
        batch_size=128,
        chunk_size=None,
        show_progress_bar=False,
        convert_to_numpy=None,
        convert_to_tensor=None,
        normalize=True,
        binarize=False,
    ):
        """
        This method allows to run encode() on multiple GPUs. The sentences are chunked into smaller packages
        and sent to individual processes, which encode these on the different GPUs. This method is only suitable
        for encoding large sets of sentences

        :param sentences: List of sentences
        :param pool: A pool of workers started with SentenceTransformer.start_multi_process_pool
        :param batch_size: Encode sentences with batch size
        :param chunk_size: Sentences are chunked and sent to the individual processes. If none, it determine a sensible size.
        :return: Numpy matrix with all embeddings
        """

        if chunk_size is None:
            chunk_size = min(math.ceil(len(sentences) / len(pool["processes"]) / 10), 5000)

        logger.debug(f"Chunk data into {math.ceil(len(sentences) / chunk_size)} packages of size {chunk_size}")

        input_queue = pool['input']
        last_chunk_id = 0
        chunk = []

        for sentence in sentences:
            chunk.append(sentence)
            if len(chunk) >= chunk_size:
                input_queue.put([last_chunk_id, batch_size, chunk, normalize, binarize])
                last_chunk_id += 1
                chunk = []

        if len(chunk) > 0:
            input_queue.put([last_chunk_id, batch_size, chunk, normalize, binarize])
            last_chunk_id += 1

        output_queue = pool['output']
        results_list = sorted([output_queue.get() for _ in range(last_chunk_id)], key=lambda x: x[0])
        embeddings = np.concatenate([np.array(t[1]) for t in results_list])
        return embeddings


class HFEncoder(Encoder):
    def __init__(self, model_name, seq_length):
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model.eval()
        self.tokenizer.model_max_length = seq_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode(self, sentences, batch_size=256, **kwargs):
        embeddings = []

        device = kwargs.get("device", self.device)
        self.model.to(device)

        with torch.no_grad():
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i : i + batch_size]
                encoded = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
                outputs = self.model(**encoded.to(device))
                pooled = self.mean_pooling(outputs, encoded["attention_mask"].to(device))
                embeddings.extend(pooled.cpu().float().numpy())

        return embeddings
