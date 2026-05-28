#!/usr/bin/env python3
"""
Export BM25S Retriever Data for C++ Loading

This script parses the cached BM25S retriever and exports:
1. Document frequency dictionary: token_id -> number of documents containing this token
2. Per-document term frequency: For each document, a sparse mapping of token_id -> term count

The data is exported in binary format optimized for fast C++ loading.

Binary Format:
--------------
1. doc_freq.bin:
   - Header: [uint32: num_vocab]
   - Data: [uint32 doc_freq] * num_vocab (indexed by vocab_idx)

2. vocab_mapping.bin:
   - Header: [uint32: num_vocab]
   - For each vocab entry: [uint32 token_id, uint32 vocab_idx]
   
3. term_freq.bin:
   - Header: [uint32: num_docs, uint64: total_entries]
   - Offsets: [uint64 offset] * (num_docs + 1)  # byte offset to each doc's data
   - For each document:
     - [uint32: num_terms]
     - [uint32 vocab_idx, uint32 count] * num_terms

Usage:
    python export_bm25_data.py --cache-dir ./cache/BeIR_hotpotqa_bm25s_EleutherAI_gpt-j-6b --output-dir ./export
"""

import argparse
import json
import logging
import os
import pickle
import struct
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_bm25_cache(cache_dir: Path) -> Tuple[Dict, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Load BM25S cache components.
    
    Returns:
        vocab_dict: token_str -> vocab_idx
        data: BM25 scores (float32)
        indices: document indices (int32)
        indptr: column pointers (int32)
        corpus_texts: list of document texts
    """
    logger.info(f"Loading BM25 cache from {cache_dir}")
    
    # Load vocab
    with open(cache_dir / "vocab.index.json", "r") as f:
        vocab_dict = json.load(f)
    logger.info(f"Loaded vocab with {len(vocab_dict)} terms")
    
    # Load sparse matrix components
    data = np.load(cache_dir / "data.csc.index.npy")
    indices = np.load(cache_dir / "indices.csc.index.npy")
    indptr = np.load(cache_dir / "indptr.csc.index.npy")
    logger.info(f"Loaded CSC matrix: {len(data)} entries, {len(indptr)-1} terms")
    
    # Load corpus
    with open(cache_dir / "corpus_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    corpus_texts = meta["texts"]
    logger.info(f"Loaded {len(corpus_texts)} documents")
    
    return vocab_dict, data, indices, indptr, corpus_texts


def compute_document_frequency(indptr: np.ndarray) -> np.ndarray:
    """Compute document frequency for each term from CSC indptr.
    
    In CSC format, indptr[i+1] - indptr[i] = number of non-zero entries in column i
    which equals the number of documents containing term i.
    """
    doc_freq = np.diff(indptr).astype(np.uint32)
    return doc_freq


def tokenize_corpus_for_tf(corpus_texts: List[str], vocab_dict: Dict[str, int],
                           tokenizer_name: str = "EleutherAI/gpt-j-6b",
                           batch_size: int = 1000) -> List[Dict[int, int]]:
    """Tokenize corpus and compute term frequencies per document.
    
    Args:
        corpus_texts: List of document texts
        vocab_dict: token_str -> vocab_idx mapping
        tokenizer_name: HuggingFace tokenizer name
        batch_size: Batch size for tokenization
        
    Returns:
        List of dicts, each mapping vocab_idx -> term_count for a document
    """
    logger.info(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    doc_term_freqs = []
    num_docs = len(corpus_texts)
    
    logger.info(f"Tokenizing {num_docs} documents...")
    start_time = time.perf_counter()
    
    for start_idx in range(0, num_docs, batch_size):
        end_idx = min(start_idx + batch_size, num_docs)
        batch_texts = corpus_texts[start_idx:end_idx]
        
        # Batch tokenization
        batch_result = tokenizer(
            batch_texts,
            add_special_tokens=False,
            return_attention_mask=False,
            padding=False,
            truncation=False,
        )
        
        # Count term frequencies for each document
        for token_ids in batch_result['input_ids']:
            tf_dict = defaultdict(int)
            for tid in token_ids:
                token_str = str(tid)
                if token_str in vocab_dict:
                    vocab_idx = vocab_dict[token_str]
                    tf_dict[vocab_idx] += 1
            doc_term_freqs.append(dict(tf_dict))
        
        if end_idx % 100000 == 0 or end_idx == num_docs:
            elapsed = time.perf_counter() - start_time
            rate = end_idx / elapsed if elapsed > 0 else 0
            logger.info(f"Tokenized {end_idx}/{num_docs} documents ({rate:.0f} docs/sec)")
    
    elapsed = time.perf_counter() - start_time
    logger.info(f"Tokenization completed in {elapsed:.2f}s")
    
    return doc_term_freqs


def export_document_frequency(doc_freq: np.ndarray, output_path: Path):
    """Export document frequency array in binary format.
    
    Format:
        [uint32: num_vocab]
        [uint32: doc_freq] * num_vocab
    """
    logger.info(f"Exporting document frequency to {output_path}")
    
    with open(output_path, "wb") as f:
        # Header
        f.write(struct.pack("<I", len(doc_freq)))
        # Data - direct numpy array write
        doc_freq.astype(np.uint32).tofile(f)
    
    file_size = output_path.stat().st_size
    logger.info(f"Exported doc_freq: {len(doc_freq)} entries, {file_size/1024/1024:.2f} MB")


def export_vocab_mapping(vocab_dict: Dict[str, int], output_path: Path):
    """Export vocabulary mapping in binary format.
    
    Format:
        [uint32: num_vocab]
        [uint32: token_id, uint32: vocab_idx] * num_vocab
        
    Note: Special tokens (empty string, etc.) are mapped to UINT32_MAX (4294967295)
    """
    logger.info(f"Exporting vocab mapping to {output_path}")
    
    # Filter and process vocab items
    valid_items = []
    special_items = []
    
    for token_str, vocab_idx in vocab_dict.items():
        if token_str == '' or not token_str.lstrip('-').isdigit():
            # Handle empty string or non-numeric tokens
            # Use UINT32_MAX as a sentinel value
            special_items.append((token_str, vocab_idx))
            valid_items.append((0xFFFFFFFF, vocab_idx))  # UINT32_MAX
        else:
            token_id = int(token_str)
            valid_items.append((token_id, vocab_idx))
    
    if special_items:
        logger.warning(f"Found {len(special_items)} special tokens (mapped to UINT32_MAX): {special_items[:5]}")
    
    with open(output_path, "wb") as f:
        # Header
        f.write(struct.pack("<I", len(valid_items)))
        
        # Data - sorted by vocab_idx for efficient lookup
        sorted_items = sorted(valid_items, key=lambda x: x[1])
        for token_id, vocab_idx in sorted_items:
            f.write(struct.pack("<II", token_id, vocab_idx))
    
    file_size = output_path.stat().st_size
    logger.info(f"Exported vocab_mapping: {len(vocab_dict)} entries, {file_size/1024/1024:.2f} MB")


def export_term_frequencies(doc_term_freqs: List[Dict[int, int]], output_path: Path):
    """Export per-document term frequencies in binary format.
    
    Format:
        Header:
            [uint32: num_docs]
            [uint64: total_entries]
        Offset table:
            [uint64: offset] * (num_docs + 1)  # byte offset to each doc's data
        Document data (for each document):
            [uint32: num_terms]
            [uint32: vocab_idx, uint32: count] * num_terms
    """
    logger.info(f"Exporting term frequencies to {output_path}")
    
    num_docs = len(doc_term_freqs)
    total_entries = sum(len(tf) for tf in doc_term_freqs)
    
    # Pre-compute offsets
    # Each document: 4 bytes (num_terms) + 8 bytes (vocab_idx + count) * num_terms
    header_size = 4 + 8  # num_docs (4) + total_entries (8)
    offset_table_size = 8 * (num_docs + 1)
    data_start = header_size + offset_table_size
    
    offsets = []
    current_offset = data_start
    for tf in doc_term_freqs:
        offsets.append(current_offset)
        current_offset += 4 + 8 * len(tf)  # num_terms + (vocab_idx, count) pairs
    offsets.append(current_offset)  # End offset
    
    with open(output_path, "wb") as f:
        # Header
        f.write(struct.pack("<I", num_docs))
        f.write(struct.pack("<Q", total_entries))
        
        # Offset table
        for offset in offsets:
            f.write(struct.pack("<Q", offset))
        
        # Document data
        for doc_idx, tf in enumerate(doc_term_freqs):
            f.write(struct.pack("<I", len(tf)))
            # Sort by vocab_idx for cache-friendly access
            for vocab_idx, count in sorted(tf.items()):
                f.write(struct.pack("<II", vocab_idx, count))
            
            if (doc_idx + 1) % 500000 == 0:
                logger.info(f"Exported {doc_idx + 1}/{num_docs} documents")
    
    file_size = output_path.stat().st_size
    logger.info(f"Exported term_freq: {num_docs} docs, {total_entries} entries, {file_size/1024/1024:.2f} MB")


def export_inverted_index(indices: np.ndarray, indptr: np.ndarray, 
                          doc_term_freqs: List[Dict[int, int]], output_path: Path):
    """Export inverted index (term -> list of (doc_id, tf)) in binary format.
    
    This is useful for fast term-centric retrieval.
    
    Format:
        Header:
            [uint32: num_vocab]
            [uint64: total_postings]
        Offset table:
            [uint64: offset] * (num_vocab + 1)
        Postings (for each term):
            [uint32: num_docs]
            [uint32: doc_id, uint32: term_freq] * num_docs
    """
    logger.info(f"Exporting inverted index to {output_path}")
    
    num_vocab = len(indptr) - 1
    total_postings = len(indices)
    
    # Build term frequency lookup: (vocab_idx, doc_id) -> tf
    # This is memory intensive but allows fast export
    logger.info("Building term frequency lookup...")
    
    # Pre-compute header and offset sizes
    header_size = 4 + 8  # num_vocab + total_postings
    offset_table_size = 8 * (num_vocab + 1)
    data_start = header_size + offset_table_size
    
    # Compute offsets
    offsets = []
    current_offset = data_start
    for vocab_idx in range(num_vocab):
        offsets.append(current_offset)
        num_docs_for_term = indptr[vocab_idx + 1] - indptr[vocab_idx]
        current_offset += 4 + 8 * num_docs_for_term
    offsets.append(current_offset)
    
    with open(output_path, "wb") as f:
        # Header
        f.write(struct.pack("<I", num_vocab))
        f.write(struct.pack("<Q", total_postings))
        
        # Offset table
        for offset in offsets:
            f.write(struct.pack("<Q", offset))
        
        # Postings for each term
        for vocab_idx in range(num_vocab):
            start = indptr[vocab_idx]
            end = indptr[vocab_idx + 1]
            doc_ids = indices[start:end]
            num_docs = len(doc_ids)
            
            f.write(struct.pack("<I", num_docs))
            for doc_id in doc_ids:
                # Look up term frequency
                tf = doc_term_freqs[doc_id].get(vocab_idx, 1)  # Default to 1 if not found
                f.write(struct.pack("<II", doc_id, tf))
            
            if (vocab_idx + 1) % 10000 == 0:
                logger.info(f"Exported {vocab_idx + 1}/{num_vocab} terms")
    
    file_size = output_path.stat().st_size
    logger.info(f"Exported inverted_index: {num_vocab} terms, {total_postings} postings, {file_size/1024/1024:.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Export BM25S cache data for C++ loading",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--cache-dir", "-c",
        type=str,
        default="./cache/BeIR_hotpotqa_bm25s_EleutherAI_gpt-j-6b",
        help="Path to BM25S cache directory"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./export",
        help="Output directory for exported files"
    )
    
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="EleutherAI/gpt-j-6b",
        help="HuggingFace tokenizer name"
    )
    
    parser.add_argument(
        "--skip-tf",
        action="store_true",
        help="Skip term frequency computation (use only CSC data)"
    )
    
    parser.add_argument(
        "--export-inverted",
        action="store_true",
        help="Also export inverted index"
    )
    
    args = parser.parse_args()
    
    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("BM25 DATA EXPORT")
    logger.info("=" * 60)
    logger.info(f"Cache dir: {cache_dir}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Tokenizer: {args.tokenizer}")
    logger.info("=" * 60)
    
    total_start = time.perf_counter()
    
    # Load cache
    vocab_dict, data, indices, indptr, corpus_texts = load_bm25_cache(cache_dir)
    
    # Compute and export document frequency
    doc_freq = compute_document_frequency(indptr)
    export_document_frequency(doc_freq, output_dir / "doc_freq.bin")
    
    # Export vocab mapping
    export_vocab_mapping(vocab_dict, output_dir / "vocab_mapping.bin")
    
    # Compute and export term frequencies per document
    if not args.skip_tf:
        doc_term_freqs = tokenize_corpus_for_tf(
            corpus_texts, vocab_dict, args.tokenizer
        )
        export_term_frequencies(doc_term_freqs, output_dir / "term_freq.bin")
        
        # Optionally export inverted index
        if args.export_inverted:
            export_inverted_index(indices, indptr, doc_term_freqs, 
                                  output_dir / "inverted_index.bin")
    else:
        logger.info("Skipping term frequency computation")
    
    # Export metadata
    metadata = {
        "num_vocab": len(vocab_dict),
        "num_docs": len(corpus_texts),
        "total_postings": len(indices),
        "tokenizer": args.tokenizer,
        "cache_dir": str(cache_dir),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    total_elapsed = time.perf_counter() - total_start
    logger.info("=" * 60)
    logger.info(f"Export completed in {total_elapsed:.2f}s")
    logger.info(f"Output files in: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
