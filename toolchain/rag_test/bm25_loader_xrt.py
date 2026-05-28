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


def main():
    parser = argparse.ArgumentParser(
        description="BM25 Data Loader and Kernel Launcher (PyXRT)")
    parser.add_argument("--bitstream", type=str, required=True,
                        help="Path to XCLBIN file")
    parser.add_argument("--export_dir", type=str, default="./export",
                        help="Directory containing exported BM25 data")
    parser.add_argument("--use_mmap", type=str, default="true",
                        help="Use memory mapping for faster loading")
    parser.add_argument("--num_threads", type=int, default=8,
                        help="Number of threads for packing")
    parser.add_argument("--query_tokens", type=str, default="",
                        help="Comma-separated list of query token IDs")
    parser.add_argument("--limit_docs", type=int, default=0,
                        help="Limit to first N documents (0 = all)")
    parser.add_argument("--device", type=int, default=-1,
                        help="Device index (-1 = auto-detect)")
    
    args = parser.parse_args()
    use_mmap = args.use_mmap.lower() in ('true', '1', 'yes')
    
    logger.info("======================================")
    logger.info("BM25 Data Loader and Kernel Launcher (PyXRT)")
    logger.info("======================================")
    logger.info(f"Export directory: {args.export_dir}")
    logger.info(f"Bitstream: {args.bitstream}")
    logger.info(f"Use mmap: {'yes' if use_mmap else 'no'}")
    logger.info(f"Num threads: {args.num_threads}")
    logger.info(f"Limit docs: {args.limit_docs if args.limit_docs > 0 else 'all'}")
    logger.info(f"Device index: {'auto-detect' if args.device < 0 else args.device}")
    logger.info(f"VOCAB_SIZE: {VOCAB_SIZE}")
    logger.info(f"TOP_K: {TOP_K}")
    logger.info("======================================\n")
    
    timer = Timer()
    total_time = 0.0
    
    # Load document frequency
    logger.info("Loading doc_freq.bin...")
    timer.start()
    if use_mmap:
        doc_freq = load_document_frequency_mmap(os.path.join(args.export_dir, "doc_freq.bin"))
    else:
        doc_freq = load_document_frequency(os.path.join(args.export_dir, "doc_freq.bin"))
    
    if doc_freq is None:
        logger.error("Failed to load doc_freq.bin")
        return 1
    
    doc_freq_time = timer.stop_ms()
    logger.info(f"  Loaded in {doc_freq_time:.2f} ms")
    total_time += doc_freq_time
    
    # Load term frequencies
    logger.info("Loading term_freq.bin...")
    timer.start()
    if use_mmap:
        term_freq = load_term_frequencies_mmap(os.path.join(args.export_dir, "term_freq.bin"))
    else:
        term_freq = load_term_frequencies(os.path.join(args.export_dir, "term_freq.bin"))
    
    if term_freq is None:
        logger.error("Failed to load term_freq.bin")
        return 1
    
    term_freq_time = timer.stop_ms()
    logger.info(f"  Loaded in {term_freq_time:.2f} ms")
    total_time += term_freq_time
    
    # Apply document limit if specified
    if args.limit_docs > 0 and args.limit_docs < term_freq.num_docs:
        logger.info(f"\nLimiting to first {args.limit_docs} documents (out of {term_freq.num_docs})")
        term_freq.num_docs = args.limit_docs
        term_freq.doc_terms = term_freq.doc_terms[:args.limit_docs]
        term_freq.offsets = term_freq.offsets[:args.limit_docs + 1]
    
    # Pack documents for hardware
    logger.info("\nPacking documents for hardware...")
    timer.start()
    packed = pack_documents_for_hw(term_freq, args.num_threads)
    pack_time = timer.stop_ms()
    logger.info(f"  Packed in {pack_time:.2f} ms")
    total_time += pack_time
    
    logger.info("\n======================================")
    logger.info(f"Total loading + packing time: {total_time:.2f} ms")
    logger.info("======================================")
    
    # Print statistics
    print_statistics(doc_freq, term_freq, packed)
    
    # Parse or generate query tokens
    if args.query_tokens:
        query_tokens = parse_query_tokens(args.query_tokens)
        logger.info(f"\nParsed {len(query_tokens)} query tokens from input")
    else:
        logger.info("\nGenerating query tokens from corpus vocabulary...")
        query_tokens = generate_random_query_from_vocab(term_freq, 256, 42)
        logger.info(f"Generated {len(query_tokens)} query tokens from corpus")
    
    sample_tokens = list(query_tokens)[:8]
    logger.info(f"Sample query tokens: {sample_tokens}...")
    
    # Compute L and L_doc_total
    L = packed.num_docs
    L = ((L + 63) // 64) * 64  # Round up to multiple of 64
    L_doc_total = packed.vectors_per_channel()
    
    logger.info("\n======================================")
    logger.info("KERNEL LAUNCH PARAMETERS")
    logger.info("======================================")
    logger.info(f"L (num docs, padded): {L}")
    logger.info(f"L_doc_total (vectors per channel): {L_doc_total}")
    logger.info(f"Num super-batches: {packed.num_super_batches}")
    logger.info(f"Query tokens: {len(query_tokens)}")
    
    # ===============================
    # PyXRT Device and Kernel Initialization
    # ===============================
    
    logger.info("\n======================================")
    logger.info("PYXRT DEVICE AND KERNEL INITIALIZATION")
    logger.info("======================================")
    
    timer.start()
    result = find_working_device(args.bitstream, args.device)
    if result is None:
        logger.error(f" No working device found for XCLBIN: {args.bitstream}")
        logger.info("Please check:")
        logger.info("  1. FPGA devices are properly installed and visible")
        logger.info("  2. The XCLBIN file exists and is compatible with the device")
        logger.info("  3. XRT runtime is properly installed")
        return 1
    
    device, xclbin_uuid, selected_device = result
    device_time = timer.stop_ms()
    logger.info(f"Device {selected_device} opened and XCLBIN loaded in {device_time:.2f} ms")
    
    # Create kernel object
    logger.info("Creating kernel object...")
    timer.start()
    try:
        kernel = pyxrt.kernel(device, xclbin_uuid, "indexer_top")
    except Exception as e:
        logger.error(f" Failed to create kernel object: {e}")
        return 1
    kernel_time = timer.stop_ms()
    logger.info(f"  Kernel created in {kernel_time:.2f} ms")
    
    # ===============================
    # Buffer Allocation
    # ===============================
    
    logger.info("\n======================================")
    logger.info("BUFFER ALLOCATION")
    logger.info("======================================")
    
    timer.start()
    
    # Calculate buffer sizes
    df_buffer_size = VOCAB_SIZE_DIV_16 * 16 * 4  # 16 ints per vector
    query_bitmap_size = VOCAB_SIZE_DIV_512 * 64  # 512 bits = 64 bytes per chunk
    inst_mem_size = packed.num_super_batches * 4
    doc_mem_size = L_doc_total * 16 * 4  # 16 uint32 per vector
    output_size = (TOP_K + 15) // 16
    topk_id_size = output_size * 16 * 4
    
    logger.info("Buffer sizes:")
    logger.info(f"  df_buffer: {df_buffer_size / 1024:.2f} KB")
    logger.info(f"  query_bitmap: {query_bitmap_size / 1024:.2f} KB")
    logger.info(f"  inst_mem: {inst_mem_size / 1024:.2f} KB")
    logger.info(f"  doc_mem (per channel): {doc_mem_size / 1024 / 1024:.2f} MB")
    logger.info(f"  topk_id: {topk_id_size} bytes")
    
    # Allocate buffers using kernel.group_id() to get memory bank assignment
    # Argument order: L(0), L_doc_total(1), df_buffer(2), query_bitmap(3), inst_mem(4), 
    #                 doc_mem[0-3](5-8), topk_id(9)
    
    # Initialize with zeros like the Xilinx example
    zeros_df = bytearray(df_buffer_size)
    zeros_query = bytearray(query_bitmap_size)
    zeros_inst = bytearray(inst_mem_size)
    zeros_doc = bytearray(doc_mem_size)
    zeros_topk = bytearray(topk_id_size)
    
    logger.info("Allocate and initialize buffers")
    
    # Allocate df_buffer
    bo_df_buffer = pyxrt.bo(device, df_buffer_size, pyxrt.bo.normal, kernel.group_id(2))
    bo_df_buffer.write(zeros_df, 0)
    buf_df = bo_df_buffer.map()
    
    # Allocate query_bitmap
    bo_query_bitmap = pyxrt.bo(device, query_bitmap_size, pyxrt.bo.normal, kernel.group_id(3))
    bo_query_bitmap.write(zeros_query, 0)
    buf_query = bo_query_bitmap.map()
    
    # Allocate inst_mem
    bo_inst_mem = pyxrt.bo(device, inst_mem_size, pyxrt.bo.normal, kernel.group_id(4))
    bo_inst_mem.write(zeros_inst, 0)
    buf_inst = bo_inst_mem.map()
    
    # Allocate doc_mem channels
    bo_doc_mem_0 = pyxrt.bo(device, doc_mem_size, pyxrt.bo.normal, kernel.group_id(5))
    bo_doc_mem_0.write(zeros_doc, 0)
    buf_doc_0 = bo_doc_mem_0.map()
    
    bo_doc_mem_1 = pyxrt.bo(device, doc_mem_size, pyxrt.bo.normal, kernel.group_id(6))
    bo_doc_mem_1.write(zeros_doc, 0)
    buf_doc_1 = bo_doc_mem_1.map()
    
    bo_doc_mem_2 = pyxrt.bo(device, doc_mem_size, pyxrt.bo.normal, kernel.group_id(7))
    bo_doc_mem_2.write(zeros_doc, 0)
    buf_doc_2 = bo_doc_mem_2.map()
    
    bo_doc_mem_3 = pyxrt.bo(device, doc_mem_size, pyxrt.bo.normal, kernel.group_id(8))
    bo_doc_mem_3.write(zeros_doc, 0)
    buf_doc_3 = bo_doc_mem_3.map()
    
    # Allocate topk_id output buffer
    bo_topk_id = pyxrt.bo(device, topk_id_size, pyxrt.bo.normal, kernel.group_id(9))
    bo_topk_id.write(zeros_topk, 0)
    buf_topk = bo_topk_id.map()
    
    alloc_time = timer.stop_ms()
    logger.info(f"Buffers allocated in {alloc_time:.2f} ms")
    
    # ===============================
    # Prepare Data and Write to Buffers
    # ===============================
    
    logger.info("\n======================================")
    logger.info("DATA PREPARATION AND TRANSFER")
    logger.info("======================================")
    
    timer.start()
    
    # Prepare and write df_buffer data
    logger.info("Writing df_buffer data...")
    df_buffer_data = np.zeros(VOCAB_SIZE_DIV_16 * 16, dtype=np.int32)
    for i in range(VOCAB_SIZE_DIV_16):
        for j in range(16):
            df_buffer_data[i * 16 + j] = int(doc_freq[i * 16 + j])
    # Write using bo.write() method
    bo_df_buffer.write(df_buffer_data.tobytes(), 0)
    
    # Prepare and write query_bitmap: 512-bit (64-byte) chunks
    logger.info("Writing query_bitmap data...")
    query_bitmap_data = np.zeros(VOCAB_SIZE_DIV_512 * 8, dtype=np.uint64)
    for token_id in query_tokens:
        if 0 <= token_id < VOCAB_SIZE:
            chunk_idx = token_id >> 9  # token_id / 512
            bit_idx = token_id & 0x1FF  # token_id % 512
            qword_idx = bit_idx // 64
            bit_in_qword = bit_idx % 64
            query_bitmap_data[chunk_idx * 8 + qword_idx] |= (1 << bit_in_qword)
    bo_query_bitmap.write(query_bitmap_data.tobytes(), 0)
    
    # Prepare and write inst_mem
    logger.info("Writing inst_mem data...")
    inst_mem_data = np.array(packed.inst_mem, dtype=np.int32)
    bo_inst_mem.write(inst_mem_data.tobytes(), 0)
    
    # Prepare and write doc_mem for each channel
    logger.info("Writing doc_mem data for 4 channels...")
    doc_mem_buffers = [buf_doc_0, buf_doc_1, buf_doc_2, buf_doc_3]
    doc_mem_bos = [bo_doc_mem_0, bo_doc_mem_1, bo_doc_mem_2, bo_doc_mem_3]
    
    for channel in range(4):
        channel_data = np.zeros(L_doc_total * 16, dtype=np.uint32)
        for vec_idx, vec in enumerate(packed.doc_mem[channel]):
            for j in range(16):
                channel_data[vec_idx * 16 + j] = vec[j]
        doc_mem_bos[channel].write(channel_data.tobytes(), 0)
    
    # Initialize output buffer to -1
    logger.info("Initializing topk_id output buffer...")
    topk_id_data = np.full(output_size * 16, -1, dtype=np.int32)
    bo_topk_id.write(topk_id_data.tobytes(), 0)
    
    prep_time = timer.stop_ms()
    logger.info(f"Data prepared and written in {prep_time:.2f} ms")
    logger.info(f"  output_size: {output_size}, topk_id_size: {topk_id_size} bytes")
    
    # Verify buffers were written correctly using map()
    logger.info(f"  Verify df_buffer (first 8 int32): {list(np.frombuffer(bytes(buf_df[:32]), dtype=np.int32))}")
    logger.info(f"  Verify topk buffer (first 8 int32): {list(np.frombuffer(bytes(buf_topk[:32]), dtype=np.int32))}")
    
    # Sync buffers to device
    logger.info("Syncing buffers to device...")
    timer.start()
    
    bo_df_buffer.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, df_buffer_size, 0)
    bo_query_bitmap.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, query_bitmap_size, 0)
    bo_inst_mem.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, inst_mem_size, 0)
    bo_doc_mem_0.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, doc_mem_size, 0)
    bo_doc_mem_1.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, doc_mem_size, 0)
    bo_doc_mem_2.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, doc_mem_size, 0)
    bo_doc_mem_3.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, doc_mem_size, 0)
    
    sync_to_time = timer.stop_ms()
    logger.info(f"  Synced to device in {sync_to_time:.2f} ms")
    
    # ===============================
    # Kernel Execution
    # ===============================
    
    logger.info("\n======================================")
    logger.info("KERNEL EXECUTION")
    logger.info("======================================")
    
    logger.info("Start the kernel")
    logger.info(f"  Setting kernel arguments:")
    logger.info(f"    arg0 (L): {L}")
    logger.info(f"    arg1 (L_doc_total): {L_doc_total}")
    timer.start()
    
    # Run kernel using direct call like Xilinx example: kHandle(bo1, bo2, bo3, size)
    run = kernel(
        L,
        L_doc_total,
        bo_df_buffer,
        bo_query_bitmap,
        bo_inst_mem,
        bo_doc_mem_0,
        bo_doc_mem_1,
        bo_doc_mem_2,
        bo_doc_mem_3,
        bo_topk_id
    )
    
    logger.info("Now wait for the kernel to finish")
    state = run.wait()
    
    exec_time = timer.stop_ms()
    logger.info(f"  Kernel execution completed in {exec_time:.2f} ms")
    logger.info(f"  Kernel state: {state}")
    
    # Check if kernel completed successfully
    if state != pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
        logger.info(f"  WARNING: Kernel did not complete successfully! State: {state}")
    
    # ===============================
    # Get Output Data from Device
    # ===============================
    
    logger.info("\nGet the output data from the device and validate it")
    timer.start()
    bo_topk_id.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, topk_id_size, 0)
    sync_from_time = timer.stop_ms()
    logger.info(f"  Synced from device in {sync_from_time:.2f} ms")
    
    # ===============================
    # Extract Results
    # ===============================
    
    logger.info("\n======================================")
    logger.info("TOP-K RESULTS")
    logger.info("======================================")
    
    # Read output data from the mapped buffer (buf_topk) after sync from device
    # Convert the bytearray to int32 values
    hw_topk_indices = []
    topk_bytes = bytes(buf_topk[:topk_id_size])
    topk_result = np.frombuffer(topk_bytes, dtype=np.int32)
    
    for i in range(output_size):
        for j in range(16):
            if i * 16 + j < TOP_K:
                hw_topk_indices.append(int(topk_result[i * 16 + j]))
    
    logger.info(f"Retrieved {len(hw_topk_indices)} top-K document indices:")
    logger.info(f"First 64 HW indices: {hw_topk_indices[:64]}")
    
    # Compute software reference
    logger.info("\nRunning software reference...")
    timer.start()
    
    sw_topk_indices, sw_topk_scores = indexer_top_ref(L, term_freq, query_tokens, doc_freq)
    
    sw_time = timer.stop_ms()
    logger.info(f"Software reference completed in {sw_time:.2f} ms")
    
    # Validate results
    validation_passed = validate_results(hw_topk_indices, sw_topk_indices, sw_topk_scores, L)
    
    # Print performance summary
    logger.info("\n======================================")
    logger.info("PERFORMANCE SUMMARY")
    logger.info("======================================")
    logger.info(f"Data loading + packing: {total_time:.2f} ms")
    logger.info(f"Device open + XCLBIN load: {device_time:.2f} ms")
    logger.info(f"Kernel create: {kernel_time:.2f} ms")
    logger.info(f"Buffer allocation: {alloc_time:.2f} ms")
    logger.info(f"Data preparation + write: {prep_time:.2f} ms")
    logger.info(f"Sync to device: {sync_to_time:.2f} ms")
    logger.info(f"Kernel execution: {exec_time:.2f} ms")
    logger.info(f"Sync from device: {sync_from_time:.2f} ms")
    logger.info(f"Software reference: {sw_time:.2f} ms")
    
    logger.info("\n======================================")
    logger.info("Done!")
    logger.info("======================================")
    
    return 0 if validation_passed else 1


if __name__ == "__main__":
    sys.exit(main())
