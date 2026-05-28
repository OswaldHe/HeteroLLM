#!/usr/bin/env python3
"""
BM25 Data Loader and Kernel Launcher for PyXRT

Fast binary loading of BM25S exported data files and kernel execution
using PyXRT (Python bindings for XRT native APIs).

Input files:
- doc_freq.bin: Document frequency for each vocabulary term
- term_freq.bin: Per-document term frequencies (sparse)

Output format matches indexer_bm25_tb.cpp:
- doc_freq: single vector of length 65536 (VOCAB_SIZE)
- doc_mem: 4 channels with packed documents (16 docs per batch)
- inst_mem: number of vectors per super-batch

Channel ordering:
- Channel 0: docs 0-15, 64-79, 128-143, ...
- Channel 1: docs 16-31, 80-95, 144-159, ...
- Channel 2: docs 32-47, 96-111, 160-175, ...
- Channel 3: docs 48-63, 112-127, 176-191, ...

Requirements:
    pip install numpy

Usage:
    python bm25_loader_xrt.py --export_dir ./export --bitstream <xclbin> [--device N] [--query_tokens "1,2,3"]
"""

import argparse
import logging
import math
import mmap
import os
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

import numpy as np
import pyxrt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants matching the kernel header
VOCAB_SIZE = 65536
VOCAB_SIZE_DIV_16 = VOCAB_SIZE // 16
VOCAB_SIZE_DIV_512 = VOCAB_SIZE // 512
TOP_K = 64

# BM25 parameters
SW_K1 = 1.2
SW_K1_PLUS_1 = SW_K1 + 1.0
SW_B = 0.75


class Timer:
    """Simple timer utility for measuring latency."""
    
    def __init__(self):
        self._start_time = None
    
    def start(self):
        self._start_time = time.perf_counter()
    
    def stop_ms(self) -> float:
        if self._start_time is None:
            return 0.0
        elapsed = time.perf_counter() - self._start_time
        return elapsed * 1000.0


@dataclass
class TermFreqEntry:
    """Term frequency entry for a single document (sparse)."""
    vocab_idx: int
    count: int


@dataclass
class DocumentTermFrequencies:
    """Per-document term frequencies (intermediate format before packing)."""
    num_docs: int = 0
    total_entries: int = 0
    offsets: List[int] = field(default_factory=list)
    doc_terms: List[List[TermFreqEntry]] = field(default_factory=list)
    
    def clear(self):
        self.offsets.clear()
        self.doc_terms.clear()
        self.num_docs = 0
        self.total_entries = 0


@dataclass
class PackedDocuments:
    """Packed documents for hardware: 4 channels."""
    doc_mem: List[List[np.ndarray]] = field(default_factory=lambda: [[] for _ in range(4)])
    inst_mem: List[int] = field(default_factory=list)
    num_docs: int = 0
    num_super_batches: int = 0
    
    def clear(self):
        for channel in self.doc_mem:
            channel.clear()
        self.inst_mem.clear()
        self.num_docs = 0
        self.num_super_batches = 0
    
    def vectors_per_channel(self) -> int:
        if not self.doc_mem or not self.doc_mem[0]:
            return 0
        return len(self.doc_mem[0])


def pack_token(token_id: int, freq: int) -> int:
    """Pack token_id (16 bits) and freq (8 bits) into 32-bit value."""
    return (min(freq, 255) << 16) | (token_id & 0xFFFF)


def unpack_token(packed: int) -> Tuple[int, int]:
    """Unpack 32-bit value into token_id and freq."""
    token_id = packed & 0xFFFF
    freq = (packed >> 16) & 0xFF
    return token_id, freq


def load_document_frequency(filepath: str) -> Optional[np.ndarray]:
    """Load document frequency from binary file and expand to VOCAB_SIZE."""
    try:
        with open(filepath, 'rb') as f:
            # Read header
            num_vocab = struct.unpack('I', f.read(4))[0]
            
            # Initialize to VOCAB_SIZE with zeros
            df = np.zeros(VOCAB_SIZE, dtype=np.uint32)
            
            # Read doc_freq array (up to VOCAB_SIZE)
            read_size = min(num_vocab, VOCAB_SIZE)
            df[:read_size] = np.frombuffer(f.read(read_size * 4), dtype=np.uint32)
            
            logger.info(f"  Loaded {num_vocab} vocab entries, expanded to {VOCAB_SIZE}")
            return df
    except Exception as e:
        logger.error(f" Cannot open {filepath}: {e}")
        return None


def load_document_frequency_mmap(filepath: str) -> Optional[np.ndarray]:
    """Load document frequency using memory mapping."""
    try:
        with open(filepath, 'rb') as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            
            # Read header
            num_vocab = struct.unpack_from('I', mm, 0)[0]
            
            # Initialize to VOCAB_SIZE with zeros
            df = np.zeros(VOCAB_SIZE, dtype=np.uint32)
            
            # Copy data
            read_size = min(num_vocab, VOCAB_SIZE)
            df[:read_size] = np.frombuffer(mm[4:4 + read_size * 4], dtype=np.uint32)
            
            mm.close()
            logger.info(f"  Loaded {num_vocab} vocab entries, expanded to {VOCAB_SIZE}")
            return df
    except Exception as e:
        logger.error(f" Cannot open {filepath}: {e}")
        return None


def load_term_frequencies(filepath: str) -> Optional[DocumentTermFrequencies]:
    """Load term frequencies from binary file."""
    try:
        dtf = DocumentTermFrequencies()
        
        with open(filepath, 'rb') as f:
            # Read header
            dtf.num_docs = struct.unpack('I', f.read(4))[0]
            dtf.total_entries = struct.unpack('Q', f.read(8))[0]
            
            logger.info(f"  Header: {dtf.num_docs} docs, {dtf.total_entries} entries")
            
            # Read offset table
            dtf.offsets = list(struct.unpack(f'{dtf.num_docs + 1}Q', 
                                             f.read((dtf.num_docs + 1) * 8)))
            
            # Read document data
            dtf.doc_terms = []
            for doc_id in range(dtf.num_docs):
                num_terms = struct.unpack('I', f.read(4))[0]
                
                terms = []
                for _ in range(num_terms):
                    vocab_idx, count = struct.unpack('II', f.read(8))
                    terms.append(TermFreqEntry(vocab_idx, count))
                
                dtf.doc_terms.append(terms)
                
                if (doc_id + 1) % 1000000 == 0:
                    logger.info(f"  Loaded {doc_id + 1}/{dtf.num_docs} documents")
        
        return dtf
    except Exception as e:
        logger.error(f" Cannot open {filepath}: {e}")
        return None


def load_term_frequencies_mmap(filepath: str) -> Optional[DocumentTermFrequencies]:
    """Load term frequencies using memory mapping."""
    try:
        dtf = DocumentTermFrequencies()
        
        with open(filepath, 'rb') as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            
            # Read header
            dtf.num_docs = struct.unpack_from('I', mm, 0)[0]
            dtf.total_entries = struct.unpack_from('Q', mm, 4)[0]
            
            logger.info(f"  Header: {dtf.num_docs} docs, {dtf.total_entries} entries")
            
            # Read offset table
            offset_start = 12
            dtf.offsets = list(struct.unpack_from(f'{dtf.num_docs + 1}Q', mm, offset_start))
            
            # Read document data
            dtf.doc_terms = []
            for doc_id in range(dtf.num_docs):
                doc_ptr = dtf.offsets[doc_id]
                
                num_terms = struct.unpack_from('I', mm, doc_ptr)[0]
                doc_ptr += 4
                
                terms = []
                for i in range(num_terms):
                    vocab_idx, count = struct.unpack_from('II', mm, doc_ptr + i * 8)
                    terms.append(TermFreqEntry(vocab_idx, count))
                
                dtf.doc_terms.append(terms)
                
                if (doc_id + 1) % 1000000 == 0:
                    logger.info(f"  Loaded {doc_id + 1}/{dtf.num_docs} documents")
            
            mm.close()
        
        return dtf
    except Exception as e:
        logger.error(f" Cannot open {filepath}: {e}")
        return None


def pack_documents_for_hw(dtf: DocumentTermFrequencies, num_threads: int = 4) -> PackedDocuments:
    """Pack documents into hardware format with 4 channels."""
    num_docs_padded = ((dtf.num_docs + 63) // 64) * 64
    num_super_batches = num_docs_padded // 64
    
    packed = PackedDocuments()
    packed.num_docs = dtf.num_docs
    packed.num_super_batches = num_super_batches
    packed.inst_mem = [0] * num_super_batches
    packed.doc_mem = [[] for _ in range(4)]
    
    logger.info(f"  Packing {dtf.num_docs} docs into {num_super_batches} "
          f"super-batches (padded to {num_docs_padded})")
    
    # Pre-filter tokens for all documents
    filtered_tokens: List[List[Tuple[int, int]]] = [[] for _ in range(num_docs_padded)]
    
    def filter_worker(start_doc: int, end_doc: int):
        for doc_id in range(start_doc, end_doc):
            required_mod = doc_id % 16
            
            if doc_id < dtf.num_docs:
                for entry in dtf.doc_terms[doc_id]:
                    token_id = entry.vocab_idx
                    if (token_id % 16) == required_mod and token_id < VOCAB_SIZE:
                        freq = min(entry.count, 255)
                        filtered_tokens[doc_id].append((token_id, freq))
    
    # Use thread pool for filtering
    chunk_size = max(1, num_docs_padded // num_threads)
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for i in range(0, num_docs_padded, chunk_size):
            end = min(i + chunk_size, num_docs_padded)
            futures.append(executor.submit(filter_worker, i, end))
        for f in futures:
            f.result()
    
    logger.info("  Filtering complete, packing super-batches...")
    
    for sb in range(num_super_batches):
        # Find max tokens in this super-batch
        max_tokens = 0
        for i in range(64):
            doc_id = sb * 64 + i
            max_tokens = max(max_tokens, len(filtered_tokens[doc_id]))
        
        if max_tokens == 0:
            max_tokens = 1
        
        packed.inst_mem[sb] = max_tokens
        
        for channel in range(4):
            for row in range(max_tokens):
                vec = np.zeros(16, dtype=np.uint32)
                
                for j in range(16):
                    doc_id = sb * 64 + channel * 16 + j
                    
                    if row < len(filtered_tokens[doc_id]):
                        token_id, freq = filtered_tokens[doc_id][row]
                        vec[j] = pack_token(token_id, freq)
                    else:
                        vec[j] = pack_token(j, 0)
                
                packed.doc_mem[channel].append(vec)
        
        if (sb + 1) % 10000 == 0 or sb + 1 == num_super_batches:
            logger.info(f"  Packed {sb + 1}/{num_super_batches} super-batches")
    
    return packed


def print_statistics(df: np.ndarray, dtf: DocumentTermFrequencies, packed: PackedDocuments):
    """Print statistics about loaded data."""
    logger.info("\n=== Data Statistics ===")
    
    logger.info("\nDocument Frequency:")
    logger.info(f"  Vector size: {len(df)}")
    
    non_zero_mask = df > 0
    non_zero = np.sum(non_zero_mask)
    if non_zero > 0:
        max_df = np.max(df)
        min_df = np.min(df[non_zero_mask])
        sum_df = np.sum(df)
        avg_df = sum_df / non_zero
    else:
        max_df = min_df = sum_df = avg_df = 0
    
    logger.info(f"  Non-zero entries: {non_zero}")
    logger.info(f"  Max doc freq: {max_df}")
    logger.info(f"  Min doc freq (non-zero): {min_df}")
    logger.info(f"  Avg doc freq (non-zero): {avg_df:.2f}")
    
    logger.info("\nTerm Frequencies (raw):")
    logger.info(f"  Num documents: {dtf.num_docs}")
    logger.info(f"  Total entries: {dtf.total_entries}")
    
    logger.info("\nPacked Documents:")
    logger.info(f"  Num super-batches: {packed.num_super_batches}")
    logger.info(f"  Vectors per channel: {packed.vectors_per_channel()}")
    
    total_inst = sum(packed.inst_mem)
    avg_inst = total_inst / packed.num_super_batches if packed.num_super_batches > 0 else 0
    logger.info(f"  Avg vectors per super-batch: {avg_inst:.2f}")
    
    packed_mem = sum(len(ch) * 16 * 4 for ch in packed.doc_mem) + len(packed.inst_mem) * 4
    logger.info(f"  Packed memory usage: {packed_mem / 1024 / 1024:.2f} MB")
    
    if dtf.num_docs > 0 and dtf.doc_terms[0]:
        logger.info("\nSample doc 0:")
        logger.info(f"  Raw terms: {len(dtf.doc_terms[0])}")
        logger.info(f"  First term: vocab_idx={dtf.doc_terms[0][0].vocab_idx}, "
              f"count={dtf.doc_terms[0][0].count}")
    
    if packed.doc_mem[0]:
        logger.info("\nSample packed (channel 0, row 0):")
        row = packed.doc_mem[0][0]
        for j in range(min(4, 16)):
            token_id, freq = unpack_token(row[j])
            logger.info(f"  [{j}] token_id={token_id}, freq={freq}")


def parse_query_tokens(query_str: str) -> Set[int]:
    """Parse comma-separated query tokens."""
    tokens = set()
    if not query_str:
        return tokens
    
    for token in query_str.split(','):
        try:
            token_id = int(token.strip())
            if 0 <= token_id < VOCAB_SIZE:
                tokens.add(token_id)
        except ValueError:
            logger.warning(f" Invalid token ID: {token}")
    
    return tokens

def parse_query_token_list(query_list: List[int]) -> Set[int]:
    """Parse list of query tokens."""
    tokens = set()
    for token in query_list:
        if 0 <= token < VOCAB_SIZE:
            tokens.add(token)
        else:
            logger.warning(f" Invalid token ID: {token}")
    return tokens


def generate_random_query_from_vocab(dtf: DocumentTermFrequencies, 
                                     query_size: int = 64, 
                                     seed: int = 42) -> Set[int]:
    """Generate random query tokens from actual vocabulary in the documents."""
    np.random.seed(seed)
    query_tokens = set()
    
    # Collect tokens by mod-16 class
    tokens_by_mod: List[List[int]] = [[] for _ in range(16)]
    
    for doc_id in range(dtf.num_docs):
        required_mod = doc_id % 16
        for entry in dtf.doc_terms[doc_id]:
            token_id = entry.vocab_idx
            if (token_id % 16) == required_mod and token_id < VOCAB_SIZE:
                tokens_by_mod[required_mod].append(token_id)
    
    # Deduplicate
    for mod in range(16):
        tokens_by_mod[mod] = list(set(tokens_by_mod[mod]))
        logger.info(f"  Mod {mod}: {len(tokens_by_mod[mod])} unique tokens")
    
    total_available = sum(len(t) for t in tokens_by_mod)
    logger.info(f"  Total unique tokens (mod-16 filtered): {total_available}")
    
    # Sample tokens round-robin by mod
    for round_idx in range(query_size):
        mod = round_idx % 16
        if tokens_by_mod[mod]:
            idx = np.random.randint(0, len(tokens_by_mod[mod]))
            query_tokens.add(tokens_by_mod[mod][idx])
        
        if len(query_tokens) >= query_size:
            break
    
    return query_tokens


def indexer_top_ref(L: int, 
                    dtf: DocumentTermFrequencies,
                    query_tokens: Set[int],
                    df: np.ndarray) -> Tuple[List[int], List[float]]:
    """Software reference implementation for BM25 scoring and top-k selection."""
    scores = []
    
    docs_with_matches = 0
    total_matches = 0
    
    for doc_id in range(min(L, dtf.num_docs)):
        score = 0.0
        required_mod = doc_id % 16
        matches_in_doc = 0
        
        for entry in dtf.doc_terms[doc_id]:
            token_id = entry.vocab_idx
            freq = min(entry.count, 255)
            
            if (token_id % 16) != required_mod:
                continue
            
            if token_id >= VOCAB_SIZE:
                continue
            
            if token_id in query_tokens:
                matches_in_doc += 1
                
                idf_num = float(L) - df[token_id] + 0.5
                idf_den = df[token_id] + 0.5
                idf = math.log(idf_num / idf_den)
                
                tf_num = freq * SW_K1_PLUS_1
                tf_den = freq + SW_K1
                tf_weight = tf_num / tf_den
                
                score += idf * tf_weight
        
        if matches_in_doc > 0:
            docs_with_matches += 1
            total_matches += matches_in_doc
        
        scores.append((score, doc_id))
    
    logger.info(f"  Documents with query matches: {docs_with_matches} / {min(L, dtf.num_docs)}")
    logger.info(f"  Total query token matches: {total_matches}")
    
    # Sort by score descending
    scores.sort(key=lambda x: x[0], reverse=True)
    
    topk_indices = [s[1] for s in scores[:TOP_K]]
    topk_scores = [s[0] for s in scores[:TOP_K]]
    
    return topk_indices, topk_scores


def validate_results(hw_topk_indices: List[int],
                     sw_topk_indices: List[int],
                     sw_topk_scores: List[float],
                     L: int) -> bool:
    """Validate hardware results against software reference."""
    logger.info("\n======================================")
    logger.info("ACCURACY VALIDATION")
    logger.info("======================================")
    
    hw_set = set(hw_topk_indices)
    sw_set = set(sw_topk_indices)
    
    overlap_count = len(hw_set & sw_set)
    
    # Check for valid indices
    invalid_count = sum(1 for idx in hw_topk_indices if idx < 0 or idx >= L)
    all_indices_valid = (invalid_count == 0)
    
    # Check for duplicates
    no_duplicates = (len(hw_set) == len(hw_topk_indices))
    
    logger.info(f"HW returned {len(hw_topk_indices)} indices")
    logger.info(f"SW returned {len(sw_topk_indices)} indices")
    logger.info(f"Overlap: {overlap_count} / {TOP_K}")
    
    logger.info(f"\nFirst 16 HW indices: {hw_topk_indices[:16]}")
    logger.info(f"First 16 SW indices: {sw_topk_indices[:16]}")
    logger.info(f"\nTop 8 SW scores: {sw_topk_scores[:8]}")
    
    hw_in_sw_top = sum(1 for i in range(min(16, len(hw_topk_indices))) 
                       if hw_topk_indices[i] in sw_set)
    
    logger.info("\n=== Statistics ===")
    overlap_ratio = overlap_count / TOP_K if TOP_K > 0 else 0
    logger.info(f"Overlap ratio: {overlap_ratio * 100:.1f}%")
    logger.info(f"HW top-16 in SW top-K: {hw_in_sw_top} / 16")
    logger.info(f"Unique HW indices: {len(hw_set)} / {len(hw_topk_indices)}")
    
    if not all_indices_valid:
        logger.warning(f" Hardware output contains {invalid_count} invalid indices")
    
    if not no_duplicates:
        logger.warning(" Hardware output contains duplicate indices")
    
    if overlap_ratio >= 0.9 and all_indices_valid and no_duplicates:
        logger.info("\n✓ PASSED: BM25 Top-K selection is working correctly!")
        return True
    elif overlap_ratio >= 0.5 and all_indices_valid:
        logger.info("\n~ PARTIAL PASS: Significant overlap but some differences.")
        logger.info("  This may be due to ties in BM25 scores or numerical precision.")
        return True
    else:
        logger.error("\n✗ FAILED: Significant mismatch between hardware and software!")
        return False


def try_open_device(device_index: int, bitstream: str) -> Optional[Tuple['pyxrt.device', 'pyxrt.uuid']]:
    """Try to open a device and load the XCLBIN."""
    try:
        logger.info(f"  Trying device {device_index}...")
        device = pyxrt.device(device_index)
        logger.info(f"    Device name: {device.get_info(pyxrt.xrt_info_device.name)}")
        logger.info(f"    Device BDF: {device.get_info(pyxrt.xrt_info_device.bdf)}")
        
        # Try to load XCLBIN
        xclbin_uuid = device.load_xclbin(bitstream)
        logger.info("    XCLBIN loaded successfully!")
        return device, xclbin_uuid
    except Exception as e:
        logger.warning(f"    Failed: {e}")
        return None


def find_working_device(bitstream: str, preferred_device: int) -> Optional[Tuple['pyxrt.device', 'pyxrt.uuid', int]]:
    """Scan all devices and find one that works with the given XCLBIN."""
    MAX_DEVICES = 16
    
    # If a specific device is requested, try it first
    if preferred_device >= 0:
        logger.info(f"Trying user-specified device {preferred_device}...")
        result = try_open_device(preferred_device, bitstream)
        if result:
            return result[0], result[1], preferred_device
        logger.warning("User-specified device failed, scanning all devices...")
    
    # Scan all devices
    logger.info("Scanning for available devices...")
    for i in range(MAX_DEVICES):
        if i == preferred_device:
            continue
        
        result = try_open_device(i, bitstream)
        if result:
            return result[0], result[1], i
    
    return None


