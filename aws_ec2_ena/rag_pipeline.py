#!/usr/bin/env python3
"""
RAG Pipeline Script with BM25S Retriever and vLLM Generator

Supports two modes:
1. Simple RAG: Single retrieval + generation
2. Fix-sentence RAG: Retrieval after each generated sentence

Usage:
    python rag_pipeline.py --mode simple
    python rag_pipeline.py --mode fix-sentence --corpus BeIR/hotpotqa --model meta-llama/Llama-3.2-1B
"""

import argparse
import logging
import pickle
import socket
import struct
import time
from pathlib import Path
from typing import List, Tuple

from vllm import LLM, SamplingParams
import bm25s
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def recv_exact(conn: socket.socket, nbytes: int) -> bytes:
    """Read exactly nbytes from a TCP connection (handles partial reads)."""
    chunks = []
    received = 0
    while received < nbytes:
        chunk = conn.recv(nbytes - received)
        if not chunk:
            raise ConnectionError(
                f"Connection closed after {received}/{nbytes} bytes"
            )
        chunks.append(chunk)
        received += len(chunk)
    return b"".join(chunks)


class LatencyTracker:
    """Track and report latency metrics."""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
    
    def start(self, name: str):
        """Start timing a metric."""
        self.start_times[name] = time.perf_counter()
    
    def stop(self, name: str) -> float:
        """Stop timing and record the metric."""
        if name not in self.start_times:
            logger.warning(f"Timer '{name}' was never started")
            return 0.0
        elapsed = time.perf_counter() - self.start_times[name]
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(elapsed)
        del self.start_times[name]
        return elapsed
    
    def get_summary(self) -> dict:
        """Get summary statistics for all metrics."""
        summary = {}
        for name, times in self.metrics.items():
            summary[name] = {
                'count': len(times),
                'total_ms': sum(times) * 1000,
                'avg_ms': (sum(times) / len(times)) * 1000 if times else 0,
                'min_ms': min(times) * 1000 if times else 0,
                'max_ms': max(times) * 1000 if times else 0,
            }
        return summary
    
    def print_summary(self):
        """Print formatted summary of all metrics."""
        summary = self.get_summary()
        logger.info("=" * 60)
        logger.info("LATENCY METRICS SUMMARY")
        logger.info("=" * 60)
        for name, stats in summary.items():
            logger.info(f"{name}:")
            logger.info(f"  Count: {stats['count']}")
            logger.info(f"  Total: {stats['total_ms']:.2f} ms")
            logger.info(f"  Avg: {stats['avg_ms']:.2f} ms")
            logger.info(f"  Min: {stats['min_ms']:.2f} ms")
            logger.info(f"  Max: {stats['max_ms']:.2f} ms")
        logger.info("=" * 60)


class BM25Retriever:
    """BM25S-based document retriever with custom tokenizer."""

    NUM_REMOTE_INDICES = 64
    REMOTE_TIMESTAMP_BYTES = 8
    REMOTE_INT_BYTES = NUM_REMOTE_INDICES * 4
    REMOTE_PAYLOAD_BYTES = REMOTE_TIMESTAMP_BYTES + REMOTE_INT_BYTES
    
    def __init__(self, corpus_name: str, cache_dir: str = "./cache",
                 retriever_tokenizer_name: str = "EleutherAI/gpt-j-6b",
                 use_remote_retrieval: bool = False,
                 remote_host: str = "0.0.0.0",
                 remote_port: int = 9000):
        self.corpus_name = corpus_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.retriever = None
        self.corpus_texts = None
        self.corpus_ids = None
        self.latency = LatencyTracker()
        self.retriever_tokenizer_name = retriever_tokenizer_name
        self.hf_tokenizer = None  # HuggingFace tokenizer for BM25
        self.use_remote_retrieval = use_remote_retrieval
        self.remote_host = remote_host
        self.remote_port = remote_port
        self.remote_server = None
        self.remote_conn = None
        self.remote_peer = None

    def _initialize_remote_receiver(self):
        """Initialize TCP server socket for remote index retrieval."""
        if self.remote_server is not None:
            return

        self.remote_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.remote_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.remote_server.bind((self.remote_host, self.remote_port))
        self.remote_server.listen(1)
        logger.info(
            "Remote retrieval enabled: listening on %s:%d",
            self.remote_host,
            self.remote_port,
        )

    def _ensure_remote_connection(self):
        """Accept a remote sender connection if no active connection exists."""
        if self.remote_conn is not None:
            return

        self._initialize_remote_receiver()
        self.remote_conn, self.remote_peer = self.remote_server.accept()
        logger.info(
            "Remote retrieval connection established from %s:%s",
            self.remote_peer[0],
            self.remote_peer[1],
        )

    def _receive_remote_indices(self) -> Tuple[List[int], float]:
        """Receive timestamp + 64 int32 indices over TCP and compute latency in ms."""
        while True:
            self._ensure_remote_connection()
            try:
                raw = recv_exact(self.remote_conn, self.REMOTE_PAYLOAD_BYTES)
                recv_time_ns = time.time_ns()

                sender_timestamp_ns = struct.unpack(
                    "!Q", raw[:self.REMOTE_TIMESTAMP_BYTES]
                )[0]
                fmt = f"!{self.NUM_REMOTE_INDICES}i"
                indices = list(struct.unpack(fmt, raw[self.REMOTE_TIMESTAMP_BYTES:]))

                delta_ns = recv_time_ns - sender_timestamp_ns
                logger.info(
                    "Remote retrieval timing sender_ns=%d recv_ns=%d delta_ns=%d",
                    sender_timestamp_ns,
                    recv_time_ns,
                    delta_ns,
                )
                return indices, delta_ns / 1_000_000.0
            except (ConnectionError, OSError) as e:
                logger.warning("Remote sender disconnected, waiting for reconnect: %s", e)
                try:
                    if self.remote_conn is not None:
                        self.remote_conn.close()
                finally:
                    self.remote_conn = None
                    self.remote_peer = None

    def _map_indices_to_documents(self, doc_indices: List[int]) -> List[Tuple[str, float]]:
        """Map received document indices to corpus texts with safety checks."""
        if self.corpus_texts is None:
            raise RuntimeError("Corpus texts not loaded. Call initialize() first.")

        retrieved_docs = []
        for idx in doc_indices:
            if 0 <= idx < len(self.corpus_texts):
                retrieved_docs.append((self.corpus_texts[idx], 0.0))
            else:
                logger.warning("Invalid remote document index %d, skipping", idx)
        return retrieved_docs
    
    def _get_cache_path(self) -> Path:
        """Get path for cached tokenized corpus."""
        safe_name = self.corpus_name.replace("/", "_").replace("\\", "_")
        tokenizer_suffix = self.retriever_tokenizer_name.replace("/", "_").replace("\\", "_")
        return self.cache_dir / f"{safe_name}_bm25s_{tokenizer_suffix}"
    
    def _load_corpus(self) -> Tuple[List[str], List[str]]:
        """Load corpus from HuggingFace datasets."""
        logger.info(f"Loading corpus: {self.corpus_name}")
        self.latency.start("corpus_loading")
        
        try:
            # Load BeIR dataset with corpus subset
            if "BeIR" in self.corpus_name or "beir" in self.corpus_name.lower():
                dataset_name = self.corpus_name
                dataset = load_dataset(dataset_name, "corpus", trust_remote_code=True)
                
                # BeIR datasets typically have 'corpus' split
                if "corpus" in dataset:
                    corpus_data = dataset["corpus"]
                elif "train" in dataset:
                    corpus_data = dataset["train"]
                else:
                    # Use first available split
                    split_name = list(dataset.keys())[0]
                    corpus_data = dataset[split_name]
                
                # Extract text and IDs
                texts = []
                ids = []
                for item in corpus_data:
                    # Combine title and text if available
                    text_parts = []
                    if "title" in item and item["title"]:
                        text_parts.append(item["title"])
                    if "text" in item and item["text"]:
                        text_parts.append(item["text"])
                    
                    if text_parts:
                        texts.append(" ".join(text_parts))
                        ids.append(item.get("_id", str(len(ids))))
            else:
                # Generic dataset loading
                dataset = load_dataset(self.corpus_name, trust_remote_code=True)
                split_name = list(dataset.keys())[0]
                corpus_data = dataset[split_name]
                
                texts = []
                ids = []
                for i, item in enumerate(corpus_data):
                    # Try common text field names
                    text = item.get("text") or item.get("content") or item.get("document") or str(item)
                    texts.append(text)
                    ids.append(item.get("id", str(i)))
            
            elapsed = self.latency.stop("corpus_loading")
            logger.info(f"Loaded {len(texts)} documents in {elapsed*1000:.2f} ms")
            
            return texts, ids
            
        except Exception as e:
            self.latency.stop("corpus_loading")
            logger.error(f"Failed to load corpus: {e}")
            raise
    
    def _tokenize_with_hf(self, texts: List[str], batch_size: int = 1000) -> List[List[str]]:
        """Tokenize texts using the HuggingFace tokenizer (GPT-J by default).
        
        Uses batch tokenization for efficiency. Returns token IDs as strings,
        which BM25S treats as vocabulary terms. This is much faster than
        decoding each token back to its string form.
        
        Args:
            texts: List of texts to tokenize
            batch_size: Number of texts to process in each batch
            
        Returns:
            List of token lists (token IDs as strings) for BM25S compatibility.
        """
        if self.hf_tokenizer is None:
            logger.info(f"Loading retriever tokenizer: {self.retriever_tokenizer_name}")
            self.hf_tokenizer = AutoTokenizer.from_pretrained(
                self.retriever_tokenizer_name,
                trust_remote_code=True
            )
            # Ensure tokenizer has padding token for batch processing
            if self.hf_tokenizer.pad_token is None:
                self.hf_tokenizer.pad_token = self.hf_tokenizer.eos_token
        
        tokenized = []
        num_texts = len(texts)
        
        # Process in batches for efficiency
        for start_idx in range(0, num_texts, batch_size):
            end_idx = min(start_idx + batch_size, num_texts)
            batch_texts = texts[start_idx:end_idx]
            
            # Batch tokenization - much faster than one-by-one
            batch_result = self.hf_tokenizer(
                batch_texts,
                add_special_tokens=False,
                return_attention_mask=False,
                padding=False,  # Don't pad, we want variable length
                truncation=False,
            )
            
            # Convert token IDs to strings for BM25S
            # BM25S treats each unique string as a vocabulary term
            for token_ids in batch_result['input_ids']:
                tokens = [str(tid) for tid in token_ids]
                tokenized.append(tokens)
            
            if end_idx % 10000 == 0 or end_idx == num_texts:
                logger.info(f"Tokenized {end_idx}/{num_texts} documents")
        
        return tokenized
    
    def initialize(self):
        """Initialize the retriever, loading from cache if available."""
        if self.use_remote_retrieval:
            self._initialize_remote_receiver()

        cache_path = self._get_cache_path()
        
        if cache_path.exists():
            logger.info(f"Loading tokenized corpus from cache: {cache_path}")
            self.latency.start("cache_loading")
            
            try:
                # Load BM25 index WITHOUT corpus - we manage corpus_texts separately
                # This prevents bm25s from trying to return documents during retrieve()
                self.retriever = bm25s.BM25.load(cache_path, load_corpus=False)
                
                # Load corpus texts and IDs from our own metadata file
                corpus_meta_path = cache_path / "corpus_meta.pkl"
                if corpus_meta_path.exists():
                    with open(corpus_meta_path, "rb") as f:
                        meta = pickle.load(f)
                        self.corpus_texts = meta["texts"]
                        self.corpus_ids = meta["ids"]
                else:
                    raise FileNotFoundError("corpus_meta.pkl not found in cache")
                
                # Validate that corpus_texts was loaded properly
                if self.corpus_texts is None or len(self.corpus_texts) == 0:
                    raise ValueError("Corpus texts are empty after loading from cache")
                
                elapsed = self.latency.stop("cache_loading")
                logger.info(f"Loaded cached retriever in {elapsed*1000:.2f} ms")
                logger.info(f"Corpus size: {len(self.corpus_texts)} documents")
                return
                
            except Exception as e:
                self.latency.stop("cache_loading")
                logger.warning(f"Failed to load cache, rebuilding: {e}")
        
        # Load and tokenize corpus
        self.corpus_texts, self.corpus_ids = self._load_corpus()
        
        logger.info(f"Tokenizing corpus with {self.retriever_tokenizer_name} tokenizer...")
        self.latency.start("tokenization")
        
        # Tokenize corpus using HuggingFace tokenizer (GPT-J)
        corpus_tokens = self._tokenize_with_hf(self.corpus_texts)
        
        elapsed = self.latency.stop("tokenization")
        logger.info(f"Tokenization completed in {elapsed*1000:.2f} ms")
        
        # Create BM25 retriever
        logger.info("Building BM25 index...")
        self.latency.start("indexing")
        
        self.retriever = bm25s.BM25()
        self.retriever.index(corpus_tokens)
        
        elapsed = self.latency.stop("indexing")
        logger.info(f"Indexing completed in {elapsed*1000:.2f} ms")
        
        # Save to cache
        logger.info(f"Saving tokenized corpus to cache: {cache_path}")
        self.latency.start("cache_saving")
        
        # Save BM25 index without corpus - we manage corpus separately in corpus_meta.pkl
        self.retriever.save(cache_path)
        
        # Save corpus metadata (original texts and IDs)
        corpus_meta_path = cache_path / "corpus_meta.pkl"
        with open(corpus_meta_path, "wb") as f:
            pickle.dump({"texts": self.corpus_texts, "ids": self.corpus_ids}, f)
        
        elapsed = self.latency.stop("cache_saving")
        logger.info(f"Cache saved in {elapsed*1000:.2f} ms")
    
    def get_documents(self, doc_ids: List[int]) -> List[Tuple[str, float]]:
        """Get document texts by their indices."""
        if self.corpus_texts is None:
            raise RuntimeError("Corpus texts not loaded. Call initialize() first.")
        
        docs = []
        for idx in doc_ids:
            if 0 <= idx < len(self.corpus_texts):
                docs.append((self.corpus_texts[idx], 0))
            else:
                logger.warning(f"Invalid document index requested: {idx}")
                docs.append(("", 0))
        return docs
    
    def retrieve(self, query: str, k: int = 64) -> Tuple[List[Tuple[str, float]], float]:
        """Retrieve top-k documents for a query."""
        if self.retriever is None:
            raise RuntimeError("Retriever not initialized. Call initialize() first.")
        
        if self.corpus_texts is None or len(self.corpus_texts) == 0:
            raise RuntimeError("Corpus texts not loaded. Re-initialize the retriever.")

        if self.use_remote_retrieval:
            logger.info("Retrieving indices from remote Ethernet sender")
            remote_indices, remote_latency_ms = self._receive_remote_indices()

            effective_indices = remote_indices[:k] if k > 0 else []
            retrieved_docs = self._map_indices_to_documents(effective_indices)

            logger.info(
                "Remote retrieval received %d indices, mapped %d documents",
                len(effective_indices),
                len(retrieved_docs),
            )
            logger.info("Remote retrieval latency: %.2f ms", remote_latency_ms)
            return retrieved_docs, remote_latency_ms
        
        # Handle empty or very short queries
        if not query or len(query.strip()) == 0:
            logger.warning("Empty query provided, returning empty results")
            return [], 0.0
        
        query_preview = query[:100] if len(query) > 100 else query
        logger.info(f"Retrieving top-{k} documents for query: '{query_preview}...'")
        
        # Tokenize query using HuggingFace tokenizer (GPT-J)
        query_tokens = self._tokenize_with_hf([query])
        logger.info(f"Tokenized query to {len(query_tokens[0])} tokens")
        logger.info(f"query tokens ids: {query_tokens[0]}")

        self.latency.start("retrieval")

        # Adjust k if corpus is smaller
        effective_k = min(k, len(self.corpus_texts))
        if effective_k < k:
            logger.warning(f"Corpus size ({len(self.corpus_texts)}) is smaller than k ({k}), using k={effective_k}")

        if effective_k == 0:
            logger.warning("Effective k is 0, returning empty results")
            self.latency.stop("retrieval")
            return [], 0.0

        # Retrieve - returns numpy arrays of shape (n_queries, k)
        # Note: We don't pass corpus here; bm25s returns indices which we map to our corpus_texts
        results, scores = self.retriever.retrieve(query_tokens, k=effective_k)

        elapsed = self.latency.stop("retrieval") * 1000
        logger.info(f"Retrieval completed in {elapsed:.2f} ms")
        logger.info(f"top indices: {results[0]}")

        # Get document texts with safety checks
        # results and scores are numpy arrays of shape (n_queries, k)
        retrieved_docs = []
        try:
            # results[0] gives the indices for the first (and only) query
            for i in range(len(results[0])):
                idx = int(results[0][i])
                score = float(scores[0][i])
                if 0 <= idx < len(self.corpus_texts):
                    retrieved_docs.append((self.corpus_texts[idx], score))
                else:
                    logger.warning(f"Invalid document index {idx}, skipping")
        except Exception as e:
            logger.error(f"Error processing retrieval results: {e}")
        
        logger.info(f"Retrieved {len(retrieved_docs)} documents")
        return retrieved_docs, elapsed


class RAGGenerator:
    """LLM-based generator with RAG support using vLLM."""
    
    def __init__(self, model_name: str, device: str = None, tensor_parallel_size: int = 1):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tensor_parallel_size = tensor_parallel_size
        self.llm = None
        self._sampling_params_cls = None
        self.tokenizer = None
        self.latency = LatencyTracker()
        
        # Sentence-ending tokens for fix-sentence mode
        self.sentence_end_chars = {'.', '!', '?', '\n'}
    
    def initialize(self):
        """Load the model using vLLM."""
        logger.info(f"Loading model with vLLM: {self.model_name}")
        logger.info(f"Device: {self.device}, Tensor Parallel Size: {self.tensor_parallel_size}")
        self.latency.start("model_loading")
        
        try:

            # Load tokenizer for chat template support
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Set pad token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Initialize vLLM engine
            self.llm = LLM(
                model=self.model_name,
                tensor_parallel_size=self.tensor_parallel_size,
                trust_remote_code=True,
                dtype="half" if self.device == "cuda" else "float32",
                max_model_len=16384,
            )
            self._sampling_params_cls = SamplingParams
            
            elapsed = self.latency.stop("model_loading")
            logger.info(f"vLLM model loaded in {elapsed*1000:.2f} ms")
            
        except Exception as e:
            self.latency.stop("model_loading")
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _has_chat_template(self) -> bool:
        """Check if the tokenizer has a chat template."""
        if self.tokenizer is None:
            return False
        # Check if chat_template attribute exists and is not None/empty
        return (hasattr(self.tokenizer, 'chat_template') and 
                self.tokenizer.chat_template is not None and
                len(self.tokenizer.chat_template) > 0)
    
    def _build_context_string(self, documents: List[Tuple[str, float]], 
                               context_prefix: str = "") -> str:
        """Build context string from retrieved documents."""
        context_parts = []
        for i, (doc, score) in enumerate(documents, 1):
            # Truncate long documents
            doc_text = doc[:1000] if len(doc) > 1000 else doc
            context_parts.append(f"[Document {i}] {doc_text}")
        
        context = "\n\n".join(context_parts)
        return f"{context_prefix}{context}" if context_prefix else context
    
    def _build_rag_prompt(self, query: str, documents: List[Tuple[str, float]], 
                          context_prefix: str = "") -> str:
        """Build a RAG prompt with retrieved documents, using chat template if available."""
        context = self._build_context_string(documents, context_prefix)
        
        # Build the user message content
        user_content = f"""Based on the following context, answer the question.

Context:
{context}

Question: {query}"""

        # Check if model has a chat template
        if self._has_chat_template():
            logger.info("Using model's chat template for prompt formatting")
            messages = [
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context. Be concise and accurate."},
                {"role": "user", "content": user_content}
            ]
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True,
                    enable_thinking=False
                )
                return prompt
            except Exception as e:
                logger.warning(f"Failed to apply chat template: {e}. Using default format.")
        
        # Fallback to default format (no chat template)
        logger.info("Using default prompt format (no chat template available)")
        prompt = f"""{user_content}

Answer:"""
        
        return prompt
    
    def _build_continuation_prompt(self, original_prompt: str, generated_text: str,
                                    new_docs: List[Tuple[str, float]]) -> str:
        """Build a continuation prompt for fix-sentence RAG with new retrieved documents."""
        # Build context insert from new documents
        context_parts = []
        for i, (doc, score) in enumerate(new_docs, 1):
            doc_text = doc[:500] if len(doc) > 500 else doc
            context_parts.append(f"[Doc {i}] {doc_text}")
        
        context_insert = "\n\n[Additional Context from Retrieval]\n" + "\n".join(context_parts) + "\n\n[Continue Answer]\n"
        
        # For chat template models, we need to handle this differently
        if self._has_chat_template():
            # Append the generated text and new context, then continue
            return original_prompt + generated_text + context_insert
        else:
            return original_prompt + generated_text + context_insert
    
    def generate_simple(self, query: str, documents: List[Tuple[str, float]], 
                        max_new_tokens: int = 256) -> str:
        """Generate a response using simple RAG with vLLM."""
        if self.llm is None:
            raise RuntimeError("Generator not initialized. Call initialize() first.")
        
        prompt = self._build_rag_prompt(query, documents)
        
        logger.info("Generating response (simple RAG) with vLLM...")
        
        # Count input tokens for logging
        input_tokens = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=16384)
        logger.info(f"device: {self.device}, input tokens: {input_tokens['input_ids'].shape[1]}")
        
        # Set up sampling parameters
        sampling_params = self._sampling_params_cls(
            max_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
        )
        
        self.latency.start("generation")
        
        # Generate with vLLM
        outputs = self.llm.generate([prompt], sampling_params)
        
        elapsed = self.latency.stop("generation")
        
        # Extract the generated text
        response = outputs[0].outputs[0].text
        num_tokens = len(outputs[0].outputs[0].token_ids)
    
        logger.info(f"Generation completed in {elapsed*1000:.2f} ms")
        logger.info(f"Generated {num_tokens} tokens")
        
        return response.strip()
    
    def generate_fix_sentence(self, query: str, retriever: BM25Retriever,
                              initial_docs: List[Tuple[str, float]],
                              max_new_tokens: int = 256,
                              retrieval_k: int = 3) -> str:
        """Generate response with fix-sentence RAG (retrieval after each sentence) using vLLM."""
        if self.llm is None:
            raise RuntimeError("Generator not initialized. Call initialize() first.")
        
        logger.info("Generating response (fix-sentence RAG) with vLLM...")
        logger.info(f"Retrieval k for subsequent sentences: {retrieval_k}")
        
        # Initial prompt with retrieved documents
        current_docs = initial_docs
        generated_text = ""
        current_sentence = ""
        sentence_count = 0
        total_tokens_generated = 0
        
        self.latency.start("generation_total")
        
        # Build initial prompt
        prompt = self._build_rag_prompt(query, current_docs)
        current_prompt = prompt
        
        # Sampling params for single token generation
        single_token_params = self._sampling_params_cls(
            max_tokens=1,
            temperature=0.7,
            top_p=0.9,
        )
        
        while total_tokens_generated < max_new_tokens:
            self.latency.start("token_generation")
            
            # Generate one token at a time for sentence detection
            outputs = self.llm.generate([current_prompt + generated_text], single_token_params)
            
            self.latency.stop("token_generation")
            
            # Get the generated token
            if not outputs[0].outputs[0].token_ids:
                logger.info("No token generated, stopping")
                break
            
            new_token_id = outputs[0].outputs[0].token_ids[0]
            new_token = outputs[0].outputs[0].text
            
            # Check for EOS
            if new_token_id == self.tokenizer.eos_token_id:
                logger.info("EOS token generated, stopping")
                break
            
            current_sentence += new_token
            generated_text += new_token
            total_tokens_generated += 1
            
            # Check for sentence end
            if any(char in new_token for char in self.sentence_end_chars):
                sentence_count += 1
                sentence_text = current_sentence.strip()
                
                if sentence_text:
                    logger.info(f"Sentence {sentence_count} completed: '{sentence_text[:100]}...'")
                    
                    # Retrieve new documents using the generated sentence
                    logger.info(f"Retrieving documents for sentence {sentence_count}")
                    new_docs, _ = retriever.retrieve(sentence_text, k=retrieval_k)
                    
                    if new_docs:
                        # Build continuation prompt with new context
                        current_prompt = self._build_continuation_prompt(prompt, generated_text, new_docs)
                        # Reset generated_text since it's now part of current_prompt
                        generated_text = ""
                        
                        logger.info(f"Appended {len(new_docs)} new documents to context")
                    
                    current_sentence = ""
        
        elapsed = self.latency.stop("generation_total")
        logger.info(f"Fix-sentence generation completed in {elapsed*1000:.2f} ms")
        logger.info(f"Generated {total_tokens_generated} tokens, {sentence_count} sentences")
        
        return generated_text.strip()


class RAGPipeline:
    """Main RAG pipeline combining retriever and generator."""
    
    def __init__(self, corpus_name: str, model_name: str, 
                 initial_k: int = 64, sentence_k: int = 3,
                 cache_dir: str = "./cache",
                 retriever_tokenizer: str = "EleutherAI/gpt-j-6b",
                 use_remote_retrieval: bool = False,
                 remote_host: str = "0.0.0.0",
                 remote_port: int = 9000):
        self.retriever = BM25Retriever(
            corpus_name,
            cache_dir,
            retriever_tokenizer,
            use_remote_retrieval,
            remote_host,
            remote_port,
        )
        self.generator = RAGGenerator(model_name)
        self.initial_k = initial_k
        self.sentence_k = sentence_k
        self.latency = LatencyTracker()
    
    def initialize(self):
        """Initialize both retriever and generator."""
        logger.info("=" * 60)
        logger.info("INITIALIZING RAG PIPELINE")
        logger.info("=" * 60)
        
        self.latency.start("pipeline_init")
        
        self.retriever.initialize()
        self.generator.initialize()
        
        elapsed = self.latency.stop("pipeline_init")
        logger.info(f"Pipeline initialization completed in {elapsed*1000:.2f} ms")
    
    def query(self, question: str, mode: str = "simple", 
              max_new_tokens: int = 256) -> str:
        """Process a query and generate a response."""
        logger.info("=" * 60)
        logger.info(f"PROCESSING QUERY (mode: {mode})")
        logger.info(f"Question: {question}")
        logger.info("=" * 60)
        
        # Retrieve documents
        documents, kernel_time = self.retriever.retrieve(question, k=self.initial_k)
        
        self.latency.start("total_query")
        # Generate response based on mode
        if mode == "simple":
            response = self.generator.generate_simple(question, documents, max_new_tokens)
        elif mode == "fix-sentence":
            response = self.generator.generate_fix_sentence(
                question, self.retriever, documents, 
                max_new_tokens, self.sentence_k
            )
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'simple' or 'fix-sentence'")
        
        elapsed = self.latency.stop("total_query")
        print(f"\nTotal query time: {elapsed*1000 + kernel_time:.2f} ms")
        
        return response
    
    def gen_warmup(self):
        logger.info("Starting generation warmup...")
        documents = self.retriever.get_documents([
            235906, 1329624, 3853890, 4033148, 1143300, 170388, 193570, 2129634, 1792798, 2421336, 1182126, 519672, 3936786, 409458, 4824994, 4138176, 2568248, 743084, 4896700, 3919388, 4168079, 1127326, 843730, 4681695, 3822656, 1176055, 2127298, 4108943, 2393310, 388004, 434690, 3988418, 1166386, 3151314, 4986968, 3019791, 363164, 1463090, 5012462, 993906, 2289340, 4755356, 1009676, 3827140, 921151, 2278300, 2909804, 1372754, 4568258, 3660158, 3430802, 3295214, 4920243, 3055680, 4726658, 446370, 1757627, 515163, 651794, 4716290, 950258
        ])
        for i in range(6):
            response = self.generator.generate_simple(
                """
                United States v. Cecil Price, et al., also known as the Mississippi Burning trial, was a criminal trial where the United States charged a group of 18 men with conspiring in a Ku Klux Klan plot to murder three young civil rights workers (Michael Schwerner, James Chaney, and Andrew Goodman) in Philadelphia, Mississippi on which date, the murders of Chaney, Goodman, and Schwerner, also known as the Freedom Summer murders, involved three activists that were abducted and murdered in Neshoba County, Mississippi, during the Civil Rights Movement? Answer in detail.
                """,
                documents,
                64
            )
            logger.info(response)

    
    def print_metrics(self):
        """Print all latency metrics."""
        logger.info("\n" + "=" * 60)
        logger.info("COMBINED METRICS")
        logger.info("=" * 60)
        
        self.latency.print_summary()
        
        logger.info("\nRETRIEVER METRICS:")
        self.retriever.latency.print_summary()
        
        logger.info("\nGENERATOR METRICS:")
        self.generator.latency.print_summary()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="RAG Pipeline with BM25S and HuggingFace",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--corpus", "-c",
        type=str,
        default="BeIR/hotpotqa",
        help="HuggingFace dataset name for corpus"
    )

    parser.add_argument(
        "--model", "-m",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="HuggingFace model name for generation"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["simple", "fix-sentence"],
        default="simple",
        help="RAG mode: 'simple' for single retrieval, 'fix-sentence' for per-sentence retrieval"
    )
    
    parser.add_argument(
        "--initial-k", "-k",
        type=int,
        default=64,
        help="Number of documents to retrieve for initial query"
    )
    
    parser.add_argument(
        "--sentence-k",
        type=int,
        default=3,
        help="Number of documents to retrieve for each sentence (fix-sentence mode)"
    )
    
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum new tokens to generate"
    )
    
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./cache",
        help="Directory to cache tokenized corpus"
    )
    
    parser.add_argument(
        "--retriever-tokenizer",
        type=str,
        default="EleutherAI/gpt-j-6b",
        help="Tokenizer to use for BM25 retrieval (default: GPT-J-6B)"
    )

    parser.add_argument(
        "--remote-retrieval",
        action="store_true",
        help="Receive RAG indices remotely over Ethernet instead of local BM25 retrieval"
    )

    parser.add_argument(
        "--remote-host",
        type=str,
        default="0.0.0.0",
        help="Local interface to bind for remote retrieval mode"
    )

    parser.add_argument(
        "--remote-port",
        type=int,
        default=9000,
        help="TCP port to listen on for remote retrieval mode"
    )
    
    parser.add_argument(
        "--question", "-q",
        type=str,
        default=None,
        help="Question to ask (if not provided, enters interactive mode)"
    )
    
    return parser.parse_args()
    
def main():
    """Main entry point."""
    args = parse_args()
    
    logger.info("=" * 60)
    logger.info("RAG PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Corpus: {args.corpus}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Retriever Tokenizer: {args.retriever_tokenizer}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Remote retrieval: {args.remote_retrieval}")
    if args.remote_retrieval:
        logger.info(f"Remote bind: {args.remote_host}:{args.remote_port}")
    logger.info(f"Initial K: {args.initial_k}")
    logger.info(f"Sentence K: {args.sentence_k}")
    logger.info(f"Max tokens: {args.max_tokens}")
    logger.info(f"Cache dir: {args.cache_dir}")
    logger.info("=" * 60)
    
    # Initialize pipeline
    pipeline = RAGPipeline(
        corpus_name=args.corpus,
        model_name=args.model,
        initial_k=args.initial_k,
        sentence_k=args.sentence_k,
        cache_dir=args.cache_dir,
        retriever_tokenizer=args.retriever_tokenizer,
        use_remote_retrieval=args.remote_retrieval,
        remote_host=args.remote_host,
        remote_port=args.remote_port,
    )
    
    pipeline.initialize()

    #pipeline.gen_warmup()
    
    if args.question:
        # Single question mode
        response = pipeline.query(args.question, mode=args.mode, 
                                  max_new_tokens=args.max_tokens)
        
        logger.info("=" * 60)
        logger.info("RESPONSE")
        logger.info("=" * 60)
        print(f"\n{response}\n")
        
        pipeline.print_metrics()
    else:
        # Interactive mode
        print("\nEntering interactive mode. Type 'quit' or 'exit' to stop.")
        print("Type 'metrics' to show latency metrics.")
        print("Type 'warmup' to run generation warmup.")
        print("Type 'mode simple' or 'mode fix-sentence' to change mode.")
        print("-" * 60)
        
        current_mode = args.mode
        
        while True:
            try:
                print(f"\n[{current_mode}] Enter your question: ", end="")
                user_input = input().strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ["quit", "exit"]:
                    logger.info("Exiting...")
                    break
                
                if user_input.lower() == "metrics":
                    pipeline.print_metrics()
                    continue
                
                if user_input.lower() == "warmup":
                    pipeline.gen_warmup()
                    continue
                
                if user_input.lower().startswith("mode "):
                    new_mode = user_input[5:].strip()
                    if new_mode in ["simple", "fix-sentence"]:
                        current_mode = new_mode
                        logger.info(f"Mode changed to: {current_mode}")
                    else:
                        logger.warning("Invalid mode. Use 'simple' or 'fix-sentence'")
                    continue
                
                # Process the question
                response = pipeline.query(user_input, mode=current_mode,
                                          max_new_tokens=args.max_tokens)
                
                print("\n" + "=" * 60)
                print("RESPONSE")
                print("=" * 60)
                print(f"\n{response}\n")
                
            except KeyboardInterrupt:
                print()
                logger.info("Interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                continue
        
        # Print final metrics
        pipeline.print_metrics()


if __name__ == "__main__":
    main()