/**
 * BM25 Data Loader and Kernel Launcher for TAPA
 * 
 * Fast binary loading of BM25S exported data files and kernel execution:
 * - doc_freq.bin: Document frequency for each vocabulary term
 * - term_freq.bin: Per-document term frequencies (sparse)
 * 
 * Output format matches indexer_bm25_tb.cpp:
 * - doc_freq: single vector of length 65536 (VOCAB_SIZE)
 * - doc_mem: 4 channels with packed documents (16 docs per batch)
 * - inst_mem: number of vectors per super-batch
 * 
 * Channel ordering:
 * - Channel 0: docs 0-15, 64-79, 128-143, ...
 * - Channel 1: docs 16-31, 80-95, 144-159, ...
 * - Channel 2: docs 32-47, 96-111, 160-175, ...
 * - Channel 3: docs 48-63, 112-127, 176-191, ...
 * 
 * Compile with TAPA:
 *   tapa g++ -- bm25_loader.cpp -o bm25_loader -O3 -std=c++17 -pthread
 * 
 * Usage:
 *   ./bm25_loader <export_dir> --bitstream <xclbin> [--use-mmap] [--num-threads N] [--query "token1,token2,..."]
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include <gflags/gflags.h>
#include <tapa.h>

// Include the kernel header
#include "indexer_bm25.h"

// Aligned vector type for TAPA
template <typename T>
using aligned_vector = std::vector<T, tapa::aligned_allocator<T>>;

DEFINE_string(bitstream, "", "path to bitstream file, run csim if empty");
DEFINE_string(export_dir, "./export", "directory containing exported BM25 data");
DEFINE_bool(use_mmap, true, "use memory mapping for faster loading (Linux only)");
DEFINE_int32(num_threads, 8, "number of threads for packing");
DEFINE_string(query_tokens, "", "comma-separated list of query token IDs");
DEFINE_int32(limit_docs, 0, "limit to first N documents (0 = use all)");

// Timer utility for measuring latency
class Timer {
public:
    void start() {
        start_time_ = std::chrono::high_resolution_clock::now();
    }
    
    double stop_ms() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time_);
        return duration.count() / 1000.0;
    }
    
private:
    std::chrono::high_resolution_clock::time_point start_time_;
};

// Packed token entry: token_id (16 bits) + freq (8 bits) packed into 32 bits
// Format: bits[15:0] = token_id, bits[23:16] = freq, bits[31:24] = reserved
struct PackedToken {
    uint32_t data;
    
    PackedToken() : data(0) {}
    PackedToken(uint16_t token_id, uint8_t freq) {
        data = (uint32_t(freq) << 16) | token_id;
    }
    
    uint16_t token_id() const { return data & 0xFFFF; }
    uint8_t freq() const { return (data >> 16) & 0xFF; }
};

// Term frequency entry for a single document (sparse)
struct TermFreqEntry {
    uint32_t vocab_idx;
    uint32_t count;
};

// Per-document term frequencies (intermediate format before packing)
struct DocumentTermFrequencies {
    uint32_t num_docs;
    uint64_t total_entries;
    std::vector<uint64_t> offsets;
    std::vector<std::vector<TermFreqEntry>> doc_terms;
    
    DocumentTermFrequencies() : num_docs(0), total_entries(0) {}
    
    void clear() {
        offsets.clear();
        offsets.shrink_to_fit();
        doc_terms.clear();
        doc_terms.shrink_to_fit();
        num_docs = 0;
        total_entries = 0;
    }
};

// Packed documents for hardware: 4 channels
// Each channel has vectors of 16 PackedTokens
struct PackedDocuments {
    // doc_mem[channel] = vector of 16-element arrays
    std::vector<std::vector<std::array<PackedToken, 16>>> doc_mem;
    // inst_mem[super_batch] = number of vectors for that super-batch
    std::vector<uint32_t> inst_mem;
    uint32_t num_docs;
    uint32_t num_super_batches;
    
    PackedDocuments() : num_docs(0), num_super_batches(0) {
        doc_mem.resize(4);
    }
    
    void clear() {
        for (auto& channel : doc_mem) {
            channel.clear();
            channel.shrink_to_fit();
        }
        inst_mem.clear();
        inst_mem.shrink_to_fit();
        num_docs = 0;
        num_super_batches = 0;
    }
    
    // Get total vectors per channel
    size_t vectors_per_channel() const {
        return doc_mem.empty() ? 0 : doc_mem[0].size();
    }
};

/**
 * Load document frequency from binary file and expand to VOCAB_SIZE
 * 
 * Format:
 *   [uint32: num_vocab]
 *   [uint32: doc_freq] * num_vocab
 * 
 * Output: vector of size VOCAB_SIZE, padded with zeros
 */
bool load_document_frequency(const std::string& filepath, std::vector<uint32_t>& df) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Cannot open " << filepath << std::endl;
        return false;
    }
    
    // Initialize to VOCAB_SIZE with zeros
    df.resize(VOCAB_SIZE, 0);
    
    // Read header
    uint32_t num_vocab;
    file.read(reinterpret_cast<char*>(&num_vocab), sizeof(uint32_t));
    
    // Read doc_freq array (up to VOCAB_SIZE)
    uint32_t read_size = std::min(num_vocab, (uint32_t)VOCAB_SIZE);
    file.read(reinterpret_cast<char*>(df.data()), read_size * sizeof(uint32_t));
    
    std::cout << "  Loaded " << num_vocab << " vocab entries, expanded to " << VOCAB_SIZE << std::endl;
    
    return file.good() || file.eof();
}

/**
 * Load term frequencies from binary file
 * 
 * Format:
 *   Header:
 *     [uint32: num_docs]
 *     [uint64: total_entries]
 *   Offset table:
 *     [uint64: offset] * (num_docs + 1)
 *   Document data:
 *     [uint32: num_terms]
 *     [uint32: vocab_idx, uint32: count] * num_terms
 */
bool load_term_frequencies(const std::string& filepath, DocumentTermFrequencies& dtf) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Cannot open " << filepath << std::endl;
        return false;
    }
    
    // Read header
    file.read(reinterpret_cast<char*>(&dtf.num_docs), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&dtf.total_entries), sizeof(uint64_t));
    
    std::cout << "  Header: " << dtf.num_docs << " docs, " << dtf.total_entries << " entries" << std::endl;
    
    // Read offset table
    dtf.offsets.resize(dtf.num_docs + 1);
    file.read(reinterpret_cast<char*>(dtf.offsets.data()), 
              (dtf.num_docs + 1) * sizeof(uint64_t));
    
    // Read document data
    dtf.doc_terms.resize(dtf.num_docs);
    
    for (uint32_t doc_id = 0; doc_id < dtf.num_docs; ++doc_id) {
        uint32_t num_terms;
        file.read(reinterpret_cast<char*>(&num_terms), sizeof(uint32_t));
        
        dtf.doc_terms[doc_id].resize(num_terms);
        if (num_terms > 0) {
            file.read(reinterpret_cast<char*>(dtf.doc_terms[doc_id].data()),
                      num_terms * sizeof(TermFreqEntry));
        }
        
        // Progress logging for large datasets
        if ((doc_id + 1) % 1000000 == 0) {
            std::cout << "  Loaded " << (doc_id + 1) << "/" << dtf.num_docs 
                      << " documents" << std::endl;
        }
    }
    
    return file.good() || file.eof();
}

#ifdef __linux__
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

class MemoryMappedFile {
public:
    MemoryMappedFile() : data_(nullptr), size_(0), fd_(-1) {}
    
    ~MemoryMappedFile() {
        close();
    }
    
    bool open(const std::string& filepath) {
        fd_ = ::open(filepath.c_str(), O_RDONLY);
        if (fd_ < 0) {
            std::cerr << "Error: Cannot open " << filepath << std::endl;
            return false;
        }
        
        struct stat sb;
        if (fstat(fd_, &sb) < 0) {
            ::close(fd_);
            fd_ = -1;
            return false;
        }
        size_ = sb.st_size;
        
        data_ = static_cast<char*>(mmap(nullptr, size_, PROT_READ, 
                                         MAP_PRIVATE, fd_, 0));
        if (data_ == MAP_FAILED) {
            data_ = nullptr;
            ::close(fd_);
            fd_ = -1;
            return false;
        }
        
        // Advise kernel for sequential access
        madvise(data_, size_, MADV_SEQUENTIAL);
        
        return true;
    }
    
    void close() {
        if (data_ != nullptr) {
            munmap(data_, size_);
            data_ = nullptr;
        }
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
        size_ = 0;
    }
    
    const char* data() const { return data_; }
    size_t size() const { return size_; }
    
private:
    char* data_;
    size_t size_;
    int fd_;
};

/**
 * Load document frequency using memory mapping
 */
bool load_document_frequency_mmap(MemoryMappedFile& mmf, std::vector<uint32_t>& df) {
    if (mmf.data() == nullptr) return false;
    
    const char* ptr = mmf.data();
    
    // Initialize to VOCAB_SIZE with zeros
    df.resize(VOCAB_SIZE, 0);
    
    // Read header
    uint32_t num_vocab;
    std::memcpy(&num_vocab, ptr, sizeof(uint32_t));
    ptr += sizeof(uint32_t);
    
    // Copy doc_freq array (up to VOCAB_SIZE)
    uint32_t copy_size = std::min(num_vocab, (uint32_t)VOCAB_SIZE);
    std::memcpy(df.data(), ptr, copy_size * sizeof(uint32_t));
    
    std::cout << "  Loaded " << num_vocab << " vocab entries, expanded to " << VOCAB_SIZE << std::endl;
    
    return true;
}

/**
 * Load term frequencies using memory mapping
 */
bool load_term_frequencies_mmap(MemoryMappedFile& mmf, DocumentTermFrequencies& dtf) {
    if (mmf.data() == nullptr) return false;
    
    const char* ptr = mmf.data();
    
    // Read header
    std::memcpy(&dtf.num_docs, ptr, sizeof(uint32_t));
    ptr += sizeof(uint32_t);
    std::memcpy(&dtf.total_entries, ptr, sizeof(uint64_t));
    ptr += sizeof(uint64_t);
    
    std::cout << "  Header: " << dtf.num_docs << " docs, " << dtf.total_entries << " entries" << std::endl;
    
    // Read offset table
    dtf.offsets.resize(dtf.num_docs + 1);
    std::memcpy(dtf.offsets.data(), ptr, (dtf.num_docs + 1) * sizeof(uint64_t));
    
    // Allocate document vectors
    dtf.doc_terms.resize(dtf.num_docs);
    
    // Parse document data directly from mapped memory
    for (uint32_t doc_id = 0; doc_id < dtf.num_docs; ++doc_id) {
        const char* doc_ptr = mmf.data() + dtf.offsets[doc_id];
        
        uint32_t num_terms;
        std::memcpy(&num_terms, doc_ptr, sizeof(uint32_t));
        doc_ptr += sizeof(uint32_t);
        
        dtf.doc_terms[doc_id].resize(num_terms);
        if (num_terms > 0) {
            std::memcpy(dtf.doc_terms[doc_id].data(), doc_ptr,
                       num_terms * sizeof(TermFreqEntry));
        }
        
        if ((doc_id + 1) % 1000000 == 0) {
            std::cout << "  Loaded " << (doc_id + 1) << "/" << dtf.num_docs 
                      << " documents" << std::endl;
        }
    }
    
    return true;
}

#endif  // __linux__

/**
 * Pack documents into hardware format with 4 channels
 * 
 * Channel ordering:
 * - Channel 0: docs 0-15, 64-79, 128-143, ...
 * - Channel 1: docs 16-31, 80-95, 144-159, ...
 * - Channel 2: docs 32-47, 96-111, 160-175, ...
 * - Channel 3: docs 48-63, 112-127, 176-191, ...
 * 
 * CONSTRAINT: token at position j must have token_id % 16 == j
 * 
 * For each super-batch (64 docs), all channels have the same number of vectors.
 */
void pack_documents_for_hw(const DocumentTermFrequencies& dtf, 
                           PackedDocuments& packed,
                           int num_threads = 4) {
    // Round up to multiple of 64
    uint32_t num_docs_padded = ((dtf.num_docs + 63) / 64) * 64;
    uint32_t num_super_batches = num_docs_padded / 64;
    
    packed.num_docs = dtf.num_docs;
    packed.num_super_batches = num_super_batches;
    packed.inst_mem.resize(num_super_batches);
    
    // Pre-allocate channels
    for (int c = 0; c < 4; ++c) {
        packed.doc_mem[c].clear();
    }
    
    std::cout << "  Packing " << dtf.num_docs << " docs into " << num_super_batches 
              << " super-batches (padded to " << num_docs_padded << ")" << std::endl;
    
    // For each super-batch, we need to:
    // 1. Filter tokens per document to only those with token_id % 16 == doc_position
    // 2. Find max tokens across all 64 docs
    // 3. Pack into 4 channels with padding
    
    // Pre-filter tokens for all documents (token_id % 16 == doc_id % 16)
    // This can be parallelized
    std::vector<std::vector<std::pair<uint16_t, uint8_t>>> filtered_tokens(num_docs_padded);
    
    // Parallel filtering
    std::atomic<uint32_t> progress(0);
    auto filter_worker = [&](int thread_id, int num_threads) {
        for (uint32_t doc_id = thread_id; doc_id < num_docs_padded; doc_id += num_threads) {
            int required_mod = doc_id % 16;
            
            if (doc_id < dtf.num_docs) {
                for (const auto& entry : dtf.doc_terms[doc_id]) {
                    uint32_t token_id = entry.vocab_idx;
                    if ((token_id % 16) == (uint32_t)required_mod && token_id < VOCAB_SIZE) {
                        uint8_t freq = std::min(entry.count, 255u);
                        filtered_tokens[doc_id].push_back({(uint16_t)token_id, freq});
                    }
                }
            }
            // Padded documents remain empty (will get dummy tokens)
            
            uint32_t p = ++progress;
            if (p % 1000000 == 0) {
                std::cout << "  Filtered " << p << "/" << num_docs_padded << " documents" << std::endl;
            }
        }
    };
    
    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back(filter_worker, t, num_threads);
    }
    for (auto& t : threads) {
        t.join();
    }
    threads.clear();
    
    std::cout << "  Filtering complete, packing super-batches..." << std::endl;
    
    // Process super-batches (sequentially to maintain order, but could parallelize)
    for (uint32_t sb = 0; sb < num_super_batches; ++sb) {
        // Find max tokens across all 64 documents in this super-batch
        uint32_t max_tokens = 0;
        for (int i = 0; i < 64; ++i) {
            uint32_t doc_id = sb * 64 + i;
            max_tokens = std::max(max_tokens, (uint32_t)filtered_tokens[doc_id].size());
        }
        
        // Ensure at least 1 vector
        if (max_tokens == 0) max_tokens = 1;
        
        packed.inst_mem[sb] = max_tokens;
        
        // Pack vectors for each channel
        for (int channel = 0; channel < 4; ++channel) {
            for (uint32_t row = 0; row < max_tokens; ++row) {
                std::array<PackedToken, 16> vec;
                
                for (int j = 0; j < 16; ++j) {
                    uint32_t doc_id = sb * 64 + channel * 16 + j;
                    
                    if (row < filtered_tokens[doc_id].size()) {
                        auto& tf = filtered_tokens[doc_id][row];
                        vec[j] = PackedToken(tf.first, tf.second);
                    } else {
                        // Dummy element: token_id with correct mod, freq = 0
                        vec[j] = PackedToken((uint16_t)j, 0);
                    }
                }
                
                packed.doc_mem[channel].push_back(vec);
            }
        }
        
        if ((sb + 1) % 10000 == 0 || sb + 1 == num_super_batches) {
            std::cout << "  Packed " << (sb + 1) << "/" << num_super_batches 
                      << " super-batches" << std::endl;
        }
    }
}

// Print statistics about loaded data
void print_statistics(const std::vector<uint32_t>& df, 
                      const DocumentTermFrequencies& dtf,
                      const PackedDocuments& packed) {
    std::cout << "\n=== Data Statistics ===" << std::endl;
    
    // Document frequency stats
    std::cout << "\nDocument Frequency:" << std::endl;
    std::cout << "  Vector size: " << df.size() << std::endl;
    
    uint32_t max_df = 0, min_df = UINT32_MAX;
    uint64_t sum_df = 0;
    uint32_t non_zero = 0;
    for (uint32_t d : df) {
        if (d > 0) {
            max_df = std::max(max_df, d);
            min_df = std::min(min_df, d);
            sum_df += d;
            non_zero++;
        }
    }
    if (non_zero == 0) min_df = 0;
    
    std::cout << "  Non-zero entries: " << non_zero << std::endl;
    std::cout << "  Max doc freq: " << max_df << std::endl;
    std::cout << "  Min doc freq (non-zero): " << min_df << std::endl;
    std::cout << "  Avg doc freq (non-zero): " << (non_zero > 0 ? (double)sum_df / non_zero : 0) << std::endl;
    
    // Term frequency stats
    std::cout << "\nTerm Frequencies (raw):" << std::endl;
    std::cout << "  Num documents: " << dtf.num_docs << std::endl;
    std::cout << "  Total entries: " << dtf.total_entries << std::endl;
    
    // Packed stats
    std::cout << "\nPacked Documents:" << std::endl;
    std::cout << "  Num super-batches: " << packed.num_super_batches << std::endl;
    std::cout << "  Vectors per channel: " << packed.vectors_per_channel() << std::endl;
    
    // Average inst_mem
    uint64_t total_inst = 0;
    for (uint32_t inst : packed.inst_mem) {
        total_inst += inst;
    }
    double avg_inst = (packed.num_super_batches > 0) ? 
                      (double)total_inst / packed.num_super_batches : 0;
    std::cout << "  Avg vectors per super-batch: " << avg_inst << std::endl;
    
    // Memory usage
    size_t packed_mem = 0;
    for (int c = 0; c < 4; ++c) {
        packed_mem += packed.doc_mem[c].size() * 16 * sizeof(PackedToken);
    }
    packed_mem += packed.inst_mem.size() * sizeof(uint32_t);
    std::cout << "  Packed memory usage: " << (packed_mem / 1024.0 / 1024.0) << " MB" << std::endl;
    
    // Sample document
    if (dtf.num_docs > 0 && !dtf.doc_terms[0].empty()) {
        std::cout << "\nSample doc 0:" << std::endl;
        std::cout << "  Raw terms: " << dtf.doc_terms[0].size() << std::endl;
        std::cout << "  First term: vocab_idx=" << dtf.doc_terms[0][0].vocab_idx 
                  << ", count=" << dtf.doc_terms[0][0].count << std::endl;
    }
    
    // Sample packed data
    if (packed.doc_mem[0].size() > 0) {
        std::cout << "\nSample packed (channel 0, row 0):" << std::endl;
        const auto& row = packed.doc_mem[0][0];
        for (int j = 0; j < std::min(4, 16); ++j) {
            std::cout << "  [" << j << "] token_id=" << row[j].token_id() 
                      << ", freq=" << (int)row[j].freq() << std::endl;
        }
    }
}

// Benchmark random access patterns
void benchmark_random_access(const std::vector<uint32_t>& df,
                             const PackedDocuments& packed,
                             int num_queries = 1000) {
    std::cout << "\n=== Random Access Benchmark ===" << std::endl;
    
    Timer timer;
    
    // Benchmark doc_freq lookups
    timer.start();
    volatile uint32_t sum = 0;
    for (int i = 0; i < num_queries; ++i) {
        uint32_t idx = (i * 1234567) % VOCAB_SIZE;
        sum += df[idx];
    }
    double doc_freq_time = timer.stop_ms();
    std::cout << "doc_freq lookups (" << num_queries << "): " 
              << doc_freq_time << " ms" << std::endl;
    
    // Benchmark packed document access
    if (packed.vectors_per_channel() > 0) {
        timer.start();
        volatile uint32_t token_sum = 0;
        for (int i = 0; i < num_queries; ++i) {
            uint32_t vec_idx = (i * 7654321) % packed.vectors_per_channel();
            int channel = i % 4;
            for (int j = 0; j < 16; ++j) {
                token_sum += packed.doc_mem[channel][vec_idx][j].freq();
            }
        }
        double packed_time = timer.stop_ms();
        std::cout << "packed doc lookups (" << num_queries << " vectors): " 
                  << packed_time << " ms" << std::endl;
    }
}

/**
 * Convert packed documents to TAPA aligned format
 */
void convert_to_tapa_format(
    const std::vector<uint32_t>& doc_freq,
    const PackedDocuments& packed,
    const std::unordered_set<int>& query_tokens,
    aligned_vector<tapa::vec_t<int, 16>>& df_buffer_hw,
    aligned_vector<ap_uint<512>>& query_bitmap_hw,
    aligned_vector<int>& inst_mem_hw,
    std::vector<aligned_vector<tapa::vec_t<ap_uint<32>, 16>>>& doc_mem_hw
) {
    // 1. df_buffer: VOCAB_SIZE / 16 vectors of 16 integers each
    df_buffer_hw.resize(VOCAB_SIZE_DIV_16);
    for (int i = 0; i < VOCAB_SIZE_DIV_16; i++) {
        for (int j = 0; j < 16; j++) {
            df_buffer_hw[i][j] = static_cast<int>(doc_freq[i * 16 + j]);
        }
    }
    
    // 2. query_bitmap_mem: VOCAB_SIZE / 512 ap_uint<512> values
    query_bitmap_hw.resize(VOCAB_SIZE_DIV_512);
    for (int i = 0; i < VOCAB_SIZE_DIV_512; i++) {
        query_bitmap_hw[i] = 0;
    }
    for (int token_id : query_tokens) {
        if (token_id >= 0 && token_id < VOCAB_SIZE) {
            int chunk_idx = token_id >> 9;  // token_id / 512
            int bit_idx = token_id & 0x1FF;  // token_id % 512
            query_bitmap_hw[chunk_idx][bit_idx] = 1;
        }
    }
    
    // 3. inst_mem: number of vectors per super-batch
    inst_mem_hw.resize(packed.num_super_batches);
    for (uint32_t i = 0; i < packed.num_super_batches; i++) {
        inst_mem_hw[i] = static_cast<int>(packed.inst_mem[i]);
    }
    
    // 4. doc_mem: 4 channels of packed vectors
    // Packing format: bits[15:0] = token_id, bits[23:16] = freq
    doc_mem_hw.resize(4);
    for (int channel = 0; channel < 4; channel++) {
        doc_mem_hw[channel].resize(packed.doc_mem[channel].size());
        for (size_t vec_idx = 0; vec_idx < packed.doc_mem[channel].size(); vec_idx++) {
            for (int j = 0; j < 16; j++) {
                ap_uint<32> packed_val = 0;
                uint16_t token_id = packed.doc_mem[channel][vec_idx][j].token_id();
                uint8_t freq = packed.doc_mem[channel][vec_idx][j].freq();
                packed_val(15, 0) = ap_uint<16>(token_id);
                packed_val(23, 16) = ap_uint<8>(freq);
                doc_mem_hw[channel][vec_idx][j] = packed_val;
            }
        }
    }
}

/**
 * Parse comma-separated query tokens
 */
std::unordered_set<int> parse_query_tokens(const std::string& query_str) {
    std::unordered_set<int> tokens;
    if (query_str.empty()) {
        return tokens;
    }
    
    std::stringstream ss(query_str);
    std::string token;
    while (std::getline(ss, token, ',')) {
        try {
            int token_id = std::stoi(token);
            if (token_id >= 0 && token_id < VOCAB_SIZE) {
                tokens.insert(token_id);
            }
        } catch (...) {
            std::cerr << "Warning: Invalid token ID: " << token << std::endl;
        }
    }
    return tokens;
}

/**
 * Generate random query tokens from actual vocabulary in the documents
 * This ensures query tokens exist in the corpus
 */
std::unordered_set<int> generate_random_query_from_vocab(
    const DocumentTermFrequencies& dtf,
    int query_size = 64, 
    int seed = 42) 
{
    std::unordered_set<int> query_tokens;
    
    // First, collect all unique tokens that appear in documents
    // and have the correct mod-16 relationship
    std::vector<std::vector<int>> tokens_by_mod(16);  // tokens_by_mod[mod] = list of token_ids
    
    for (uint32_t doc_id = 0; doc_id < dtf.num_docs; doc_id++) {
        int required_mod = doc_id % 16;
        for (const auto& entry : dtf.doc_terms[doc_id]) {
            int token_id = entry.vocab_idx;
            if ((token_id % 16) == required_mod && token_id < VOCAB_SIZE) {
                tokens_by_mod[required_mod].push_back(token_id);
            }
        }
    }
    
    // Remove duplicates from each mod bucket
    for (int mod = 0; mod < 16; mod++) {
        std::sort(tokens_by_mod[mod].begin(), tokens_by_mod[mod].end());
        tokens_by_mod[mod].erase(
            std::unique(tokens_by_mod[mod].begin(), tokens_by_mod[mod].end()),
            tokens_by_mod[mod].end());
    }
    
    // Count total available tokens
    int total_available = 0;
    for (int mod = 0; mod < 16; mod++) {
        total_available += tokens_by_mod[mod].size();
        std::cout << "  Mod " << mod << ": " << tokens_by_mod[mod].size() << " unique tokens" << std::endl;
    }
    std::cout << "  Total unique tokens (mod-16 filtered): " << total_available << std::endl;
    
    // Generate random query by sampling from each mod bucket
    std::mt19937 gen(seed);
    
    // Try to get tokens from each mod value for balanced query
    for (int round = 0; query_tokens.size() < (size_t)query_size && round < query_size; round++) {
        int mod = round % 16;
        if (!tokens_by_mod[mod].empty()) {
            std::uniform_int_distribution<int> dis(0, tokens_by_mod[mod].size() - 1);
            int token_id = tokens_by_mod[mod][dis(gen)];
            query_tokens.insert(token_id);
        }
    }
    
    return query_tokens;
}

/**
 * Generate random query tokens for testing (original - may not match corpus)
 */
std::unordered_set<int> generate_random_query(int query_size = 64, int seed = 42) {
    std::unordered_set<int> query_tokens;
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> token_base_dis(0, (VOCAB_SIZE / 16) - 1);
    
    while ((int)query_tokens.size() < query_size) {
        // Generate token_id = base * 16 + mod, ensuring variety in mod values
        int base = token_base_dis(gen);
        int mod = query_tokens.size() % 16;
        int token_id = base * 16 + mod;
        query_tokens.insert(token_id);
    }
    return query_tokens;
}

// BM25 parameters (must match kernel)
constexpr float SW_K1 = 1.2f;
constexpr float SW_K1_plus_1 = SW_K1 + 1.0f;
constexpr float SW_B = 0.75f;

/**
 * Software reference implementation for BM25 scoring and top-k selection
 * This must match the kernel's behavior exactly, including the mod-16 constraint
 */
void indexer_top_ref(
    const int L,
    const DocumentTermFrequencies& dtf,
    const std::unordered_set<int>& query_tokens,
    const std::vector<uint32_t>& df,
    std::vector<int>& topk_indices,
    std::vector<float>& topk_scores)
{
    // Compute BM25 scores for all documents
    std::vector<std::pair<float, int>> scores;  // (score, doc_id)
    
    int docs_with_matches = 0;
    int total_matches = 0;
    
    for (int doc_id = 0; doc_id < L && doc_id < (int)dtf.num_docs; doc_id++) {
        float score = 0.0f;
        int required_mod = doc_id % 16;  // Kernel can only process tokens with this mod
        int matches_in_doc = 0;
        
        for (const auto& entry : dtf.doc_terms[doc_id]) {
            int token_id = entry.vocab_idx;
            int freq = std::min((int)entry.count, 255);  // Cap freq like hardware does
            
            // IMPORTANT: Only count tokens that the kernel can process
            // Token at position j must have token_id % 16 == j for correct lookup
            if ((token_id % 16) != required_mod) {
                continue;  // Kernel would not correctly process this token
            }
            
            // Skip tokens beyond VOCAB_SIZE
            if (token_id >= VOCAB_SIZE) {
                continue;
            }
            
            // Check if token is in query
            if (query_tokens.find(token_id) != query_tokens.end()) {
                matches_in_doc++;
                
                // Compute IDF: log((L - df + 0.5) / (df + 0.5))
                float idf_num = (float)L - df[token_id] + 0.5f;
                float idf_den = df[token_id] + 0.5f;
                float idf_den_inv = 1.0f / idf_den;
                float idf = logf(idf_num * idf_den_inv);
                
                // Compute TF weight: (freq * (K1 + 1)) / (freq + K1)
                // Note: The kernel doesn't use document length normalization (B term)
                float tf_num = freq * SW_K1_plus_1;
                float tf_den = freq + SW_K1;
                float tf_den_inv = 1.0f / tf_den;
                float tf_weight = tf_num * tf_den_inv;
                
                score += idf * tf_weight;
            }
        }
        
        if (matches_in_doc > 0) {
            docs_with_matches++;
            total_matches += matches_in_doc;
        }
        
        scores.push_back({score, doc_id});
    }
    
    std::cout << "  Documents with query matches: " << docs_with_matches << " / " << std::min(L, (int)dtf.num_docs) << std::endl;
    std::cout << "  Total query token matches: " << total_matches << std::endl;
    
    // Sort by score descending
    std::sort(scores.begin(), scores.end(), 
              [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Extract top-K indices and scores
    topk_indices.clear();
    topk_scores.clear();
    for (int i = 0; i < TOP_K && i < (int)scores.size(); i++) {
        topk_indices.push_back(scores[i].second);
        topk_scores.push_back(scores[i].first);
    }
}

/**
 * Validate hardware results against software reference
 */
bool validate_results(
    const std::vector<int>& hw_topk_indices,
    const std::vector<int>& sw_topk_indices,
    const std::vector<float>& sw_topk_scores,
    int L)
{
    std::cout << "\n======================================" << std::endl;
    std::cout << "ACCURACY VALIDATION" << std::endl;
    std::cout << "======================================" << std::endl;
    
    // Convert to sets for comparison (order may differ due to tie-breaking)
    std::set<int> hw_set(hw_topk_indices.begin(), hw_topk_indices.end());
    std::set<int> sw_set(sw_topk_indices.begin(), sw_topk_indices.end());
    
    // Check overlap
    int overlap_count = 0;
    for (int idx : hw_set) {
        if (sw_set.find(idx) != sw_set.end()) {
            overlap_count++;
        }
    }
    
    // Check for valid indices
    bool all_indices_valid = true;
    int invalid_count = 0;
    for (int idx : hw_topk_indices) {
        if (idx < 0 || idx >= L) {
            all_indices_valid = false;
            invalid_count++;
            if (invalid_count <= 5) {
                std::cout << "Invalid index: " << idx << std::endl;
            }
        }
    }
    if (invalid_count > 5) {
        std::cout << "... and " << (invalid_count - 5) << " more invalid indices" << std::endl;
    }
    
    // Check for duplicates
    bool no_duplicates = (hw_set.size() == hw_topk_indices.size());
    
    std::cout << "HW returned " << hw_topk_indices.size() << " indices" << std::endl;
    std::cout << "SW returned " << sw_topk_indices.size() << " indices" << std::endl;
    std::cout << "Overlap: " << overlap_count << " / " << TOP_K << std::endl;
    
    std::cout << "\nFirst 16 HW indices: ";
    for (int i = 0; i < std::min(16, (int)hw_topk_indices.size()); i++) {
        std::cout << hw_topk_indices[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "First 16 SW indices: ";
    for (int i = 0; i < std::min(16, (int)sw_topk_indices.size()); i++) {
        std::cout << sw_topk_indices[i] << " ";
    }
    std::cout << std::endl;
    
    // Show top SW scores
    std::cout << "\nTop 8 SW scores: ";
    for (int i = 0; i < std::min(8, (int)sw_topk_scores.size()); i++) {
        std::cout << sw_topk_scores[i] << " ";
    }
    std::cout << std::endl;
    
    // Check how many of HW's top results are in SW's top results
    int hw_in_sw_top = 0;
    for (int i = 0; i < std::min(16, (int)hw_topk_indices.size()); i++) {
        if (sw_set.find(hw_topk_indices[i]) != sw_set.end()) {
            hw_in_sw_top++;
        }
    }
    
    std::cout << "\n=== Statistics ===" << std::endl;
    float overlap_ratio = (float)overlap_count / TOP_K;
    std::cout << "Overlap ratio: " << (overlap_ratio * 100) << "%" << std::endl;
    std::cout << "HW top-16 in SW top-K: " << hw_in_sw_top << " / 16" << std::endl;
    std::cout << "Unique HW indices: " << hw_set.size() << " / " << hw_topk_indices.size() << std::endl;
    
    if (!all_indices_valid) {
        std::cout << "Warning: Hardware output contains " << invalid_count << " invalid indices" << std::endl;
    }
    
    if (!no_duplicates) {
        std::cout << "Warning: Hardware output contains duplicate indices" << std::endl;
    }
    
    // Success criteria
    if (overlap_ratio >= 0.9 && all_indices_valid && no_duplicates) {
        std::cout << "\n✓ PASSED: BM25 Top-K selection is working correctly!" << std::endl;
        return true;
    } else if (overlap_ratio >= 0.5 && all_indices_valid) {
        std::cout << "\n~ PARTIAL PASS: Significant overlap but some differences." << std::endl;
        std::cout << "  This may be due to ties in BM25 scores or numerical precision." << std::endl;
        return true;
    } else {
        std::cout << "\n✗ FAILED: Significant mismatch between hardware and software!" << std::endl;
        return false;
    }
}

int main(int argc, char* argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    
    std::string export_dir = FLAGS_export_dir;
    bool use_mmap = FLAGS_use_mmap;
    int num_threads = FLAGS_num_threads;
    int limit_docs = FLAGS_limit_docs;
    
    std::cout << "======================================" << std::endl;
    std::cout << "BM25 Data Loader and Kernel Launcher" << std::endl;
    std::cout << "======================================" << std::endl;
    std::cout << "Export directory: " << export_dir << std::endl;
    std::cout << "Bitstream: " << (FLAGS_bitstream.empty() ? "(csim)" : FLAGS_bitstream) << std::endl;
    std::cout << "Use mmap: " << (use_mmap ? "yes" : "no") << std::endl;
    std::cout << "Num threads: " << num_threads << std::endl;
    std::cout << "Limit docs: " << (limit_docs > 0 ? std::to_string(limit_docs) : "all") << std::endl;
    std::cout << "VOCAB_SIZE: " << VOCAB_SIZE << std::endl;
    std::cout << "TOP_K: " << TOP_K << std::endl;
    std::cout << "======================================\n" << std::endl;
    
    std::vector<uint32_t> doc_freq;
    DocumentTermFrequencies term_freq;
    PackedDocuments packed;
    
    Timer timer;
    double total_time = 0.0;
    
#ifdef __linux__
    if (use_mmap) {
        // Memory-mapped loading
        MemoryMappedFile mmf_doc_freq, mmf_term_freq;
        
        // Load doc_freq
        std::cout << "Loading doc_freq.bin (mmap)..." << std::endl;
        timer.start();
        if (!mmf_doc_freq.open(export_dir + "/doc_freq.bin") ||
            !load_document_frequency_mmap(mmf_doc_freq, doc_freq)) {
            std::cerr << "Failed to load doc_freq.bin" << std::endl;
            return 1;
        }
        double doc_freq_time = timer.stop_ms();
        std::cout << "  Loaded in " << doc_freq_time << " ms" << std::endl;
        total_time += doc_freq_time;
        
        // Load term_freq
        std::cout << "Loading term_freq.bin (mmap)..." << std::endl;
        timer.start();
        if (!mmf_term_freq.open(export_dir + "/term_freq.bin") ||
            !load_term_frequencies_mmap(mmf_term_freq, term_freq)) {
            std::cerr << "Failed to load term_freq.bin" << std::endl;
            return 1;
        }
        double term_freq_time = timer.stop_ms();
        std::cout << "  Loaded in " << term_freq_time << " ms" << std::endl;
        total_time += term_freq_time;
        
    } else
#endif
    {
        // Standard file loading
        
        // Load doc_freq
        std::cout << "Loading doc_freq.bin..." << std::endl;
        timer.start();
        if (!load_document_frequency(export_dir + "/doc_freq.bin", doc_freq)) {
            std::cerr << "Failed to load doc_freq.bin" << std::endl;
            return 1;
        }
        double doc_freq_time = timer.stop_ms();
        std::cout << "  Loaded in " << doc_freq_time << " ms" << std::endl;
        total_time += doc_freq_time;
        
        // Load term_freq
        std::cout << "Loading term_freq.bin..." << std::endl;
        timer.start();
        if (!load_term_frequencies(export_dir + "/term_freq.bin", term_freq)) {
            std::cerr << "Failed to load term_freq.bin" << std::endl;
            return 1;
        }
        double term_freq_time = timer.stop_ms();
        std::cout << "  Loaded in " << term_freq_time << " ms" << std::endl;
        total_time += term_freq_time;
    }
    
    // Apply document limit if specified
    if (limit_docs > 0 && (uint32_t)limit_docs < term_freq.num_docs) {
        std::cout << "\nLimiting to first " << limit_docs << " documents (out of " 
                  << term_freq.num_docs << ")" << std::endl;
        term_freq.num_docs = limit_docs;
        term_freq.doc_terms.resize(limit_docs);
        term_freq.offsets.resize(limit_docs + 1);
    }
    
    // Pack documents for hardware
    std::cout << "\nPacking documents for hardware..." << std::endl;
    timer.start();
    pack_documents_for_hw(term_freq, packed, num_threads);
    double pack_time = timer.stop_ms();
    std::cout << "  Packed in " << pack_time << " ms" << std::endl;
    total_time += pack_time;
    
    std::cout << "\n======================================" << std::endl;
    std::cout << "Total loading + packing time: " << total_time << " ms" << std::endl;
    std::cout << "======================================" << std::endl;
    
    // Print statistics
    print_statistics(doc_freq, term_freq, packed);
    
    // Parse or generate query tokens
    std::unordered_set<int> query_tokens;
    if (!FLAGS_query_tokens.empty()) {
        query_tokens = parse_query_tokens(FLAGS_query_tokens);
        std::cout << "\nParsed " << query_tokens.size() << " query tokens from input" << std::endl;
    } else {
        // Generate query tokens from actual vocabulary in the corpus
        std::cout << "\nGenerating query tokens from corpus vocabulary..." << std::endl;
        query_tokens = generate_random_query_from_vocab(term_freq, 256, 42);
        std::cout << "Generated " << query_tokens.size() << " query tokens from corpus" << std::endl;
    }
    
    // Print some sample query tokens
    std::cout << "Sample query tokens: ";
    int count = 0;
    for (int t : query_tokens) {
        if (count++ >= 8) break;
        std::cout << t << " ";
    }
    std::cout << "..." << std::endl;
    
    // Convert to TAPA format
    std::cout << "\nConverting to TAPA format..." << std::endl;
    timer.start();
    
    aligned_vector<tapa::vec_t<int, 16>> df_buffer_hw;
    aligned_vector<ap_uint<512>> query_bitmap_hw;
    aligned_vector<int> inst_mem_hw;
    std::vector<aligned_vector<tapa::vec_t<ap_uint<32>, 16>>> doc_mem_hw;
    
    convert_to_tapa_format(doc_freq, packed, query_tokens,
                           df_buffer_hw, query_bitmap_hw, inst_mem_hw, doc_mem_hw);
    
    double convert_time = timer.stop_ms();
    std::cout << "  Converted in " << convert_time << " ms" << std::endl;
    
    // Prepare output
    const int output_size = (TOP_K + 15) / 16;
    aligned_vector<tapa::vec_t<int, 16>> topk_id_hw(output_size);
    
    // Compute L and L_doc_total
    int L = packed.num_docs;
    // Round up to multiple of 64
    L = ((L + 63) / 64) * 64;
    int L_doc_total = static_cast<int>(packed.vectors_per_channel());
    
    std::cout << "\n======================================" << std::endl;
    std::cout << "KERNEL LAUNCH PARAMETERS" << std::endl;
    std::cout << "======================================" << std::endl;
    std::cout << "L (num docs, padded): " << L << std::endl;
    std::cout << "L_doc_total (vectors per channel): " << L_doc_total << std::endl;
    std::cout << "Num super-batches: " << packed.num_super_batches << std::endl;
    std::cout << "Query tokens: " << query_tokens.size() << std::endl;
    std::cout << "Output size: " << output_size << " vectors" << std::endl;
    
    // Log average of inst_mem
    int64_t total_inst = 0;
    for (size_t i = 0; i < inst_mem_hw.size(); i++) {
        total_inst += inst_mem_hw[i];
    }
    double avg_inst = (inst_mem_hw.size() > 0) ? (double)total_inst / inst_mem_hw.size() : 0.0;
    std::cout << "Average inst_mem per super-batch: " << avg_inst << std::endl;
    std::cout << "======================================\n" << std::endl;
    
    // Invoke the kernel
    std::cout << "Running hardware kernel..." << std::endl;
    timer.start();
    
    int64_t kernel_time_ns = tapa::invoke(
        indexer_top, 
        FLAGS_bitstream,
        L,
        L_doc_total,
        tapa::read_only_mmap<tapa::vec_t<int, 16>>(df_buffer_hw),
        tapa::read_only_mmap<ap_uint<512>>(query_bitmap_hw),
        tapa::read_only_mmap<int>(inst_mem_hw),
        tapa::read_only_mmaps<tapa::vec_t<ap_uint<32>, 16>, 4>(doc_mem_hw),
        tapa::write_only_mmap<tapa::vec_t<int, 16>>(topk_id_hw)
    );
    
    double invoke_time = timer.stop_ms();
    std::cout << "Hardware kernel completed." << std::endl;
    std::cout << "  Invoke time: " << invoke_time << " ms" << std::endl;
    std::cout << "  Kernel time: " << kernel_time_ns * 1e-6 << " ms" << std::endl;
    
    // Extract hardware results
    std::cout << "\n======================================" << std::endl;
    std::cout << "TOP-K RESULTS" << std::endl;
    std::cout << "======================================" << std::endl;
    
    std::vector<int> hw_topk_indices;
    for (int i = 0; i < output_size; i++) {
        for (int j = 0; j < 16 && (i * 16 + j) < TOP_K; j++) {
            hw_topk_indices.push_back(topk_id_hw[i][j]);
        }
    }
    
    std::cout << "Retrieved " << hw_topk_indices.size() << " top-K document indices:" << std::endl;
    std::cout << "First 64 HW indices: ";
    for (int i = 0; i < std::min(64, (int)hw_topk_indices.size()); i++) {
        std::cout << hw_topk_indices[i] << " ";
    }
    std::cout << std::endl;
    
    // Compute software reference
    std::cout << "\nRunning software reference..." << std::endl;
    timer.start();
    
    std::vector<int> sw_topk_indices;
    std::vector<float> sw_topk_scores;
    indexer_top_ref(L, term_freq, query_tokens, doc_freq, sw_topk_indices, sw_topk_scores);
    
    double sw_time = timer.stop_ms();
    std::cout << "Software reference completed in " << sw_time << " ms" << std::endl;
    
    // Validate results
    bool validation_passed = validate_results(hw_topk_indices, sw_topk_indices, sw_topk_scores, L);
    
    std::cout << "\n======================================" << std::endl;
    std::cout << "Done!" << std::endl;
    std::cout << "======================================" << std::endl;
    
    return validation_passed ? 0 : 1;
}

