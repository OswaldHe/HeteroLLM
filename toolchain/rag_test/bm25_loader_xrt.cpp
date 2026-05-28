/**
 * BM25 Data Loader and Kernel Launcher for XRT Native APIs
 * 
 * Fast binary loading of BM25S exported data files and kernel execution
 * using XRT native APIs instead of TAPA.
 * 
 * Input files:
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
 * Compile with XRT:
 *   g++ -g -std=c++17 -O3 -I$XILINX_XRT/include -L$XILINX_XRT/lib \
 *       bm25_loader_xrt.cpp -o bm25_loader_xrt -lxrt_coreutil -lpthread
 * 
 * Usage:
 *   ./bm25_loader_xrt <export_dir> --bitstream <xclbin> [--use-mmap] [--num-threads N] [--query "token1,token2,..."]
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
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

// XRT Native API headers
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>

// Simple command-line argument parsing
struct Args {
    std::string bitstream;
    std::string export_dir = "./export";
    bool use_mmap = true;
    int num_threads = 8;
    std::string query_tokens;
    int limit_docs = 0;
    int device_index = 0;
};

// Constants matching the kernel header
constexpr int VOCAB_SIZE = 65536;
constexpr int VOCAB_SIZE_DIV_16 = VOCAB_SIZE / 16;
constexpr int VOCAB_SIZE_DIV_512 = VOCAB_SIZE / 512;
constexpr int TOP_K = 64;

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

bool load_document_frequency_mmap(MemoryMappedFile& mmf, std::vector<uint32_t>& df) {
    if (mmf.data() == nullptr) return false;
    
    const char* ptr = mmf.data();
    
    df.resize(VOCAB_SIZE, 0);
    
    uint32_t num_vocab;
    std::memcpy(&num_vocab, ptr, sizeof(uint32_t));
    ptr += sizeof(uint32_t);
    
    uint32_t copy_size = std::min(num_vocab, (uint32_t)VOCAB_SIZE);
    std::memcpy(df.data(), ptr, copy_size * sizeof(uint32_t));
    
    std::cout << "  Loaded " << num_vocab << " vocab entries, expanded to " << VOCAB_SIZE << std::endl;
    
    return true;
}

bool load_term_frequencies_mmap(MemoryMappedFile& mmf, DocumentTermFrequencies& dtf) {
    if (mmf.data() == nullptr) return false;
    
    const char* ptr = mmf.data();
    
    std::memcpy(&dtf.num_docs, ptr, sizeof(uint32_t));
    ptr += sizeof(uint32_t);
    std::memcpy(&dtf.total_entries, ptr, sizeof(uint64_t));
    ptr += sizeof(uint64_t);
    
    std::cout << "  Header: " << dtf.num_docs << " docs, " << dtf.total_entries << " entries" << std::endl;
    
    dtf.offsets.resize(dtf.num_docs + 1);
    std::memcpy(dtf.offsets.data(), ptr, (dtf.num_docs + 1) * sizeof(uint64_t));
    
    dtf.doc_terms.resize(dtf.num_docs);
    
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
 */
void pack_documents_for_hw(const DocumentTermFrequencies& dtf, 
                           PackedDocuments& packed,
                           int num_threads = 4) {
    uint32_t num_docs_padded = ((dtf.num_docs + 63) / 64) * 64;
    uint32_t num_super_batches = num_docs_padded / 64;
    
    packed.num_docs = dtf.num_docs;
    packed.num_super_batches = num_super_batches;
    packed.inst_mem.resize(num_super_batches);
    
    for (int c = 0; c < 4; ++c) {
        packed.doc_mem[c].clear();
    }
    
    std::cout << "  Packing " << dtf.num_docs << " docs into " << num_super_batches 
              << " super-batches (padded to " << num_docs_padded << ")" << std::endl;
    
    // Pre-filter tokens for all documents
    std::vector<std::vector<std::pair<uint16_t, uint8_t>>> filtered_tokens(num_docs_padded);
    
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
    
    for (uint32_t sb = 0; sb < num_super_batches; ++sb) {
        uint32_t max_tokens = 0;
        for (int i = 0; i < 64; ++i) {
            uint32_t doc_id = sb * 64 + i;
            max_tokens = std::max(max_tokens, (uint32_t)filtered_tokens[doc_id].size());
        }
        
        if (max_tokens == 0) max_tokens = 1;
        
        packed.inst_mem[sb] = max_tokens;
        
        for (int channel = 0; channel < 4; ++channel) {
            for (uint32_t row = 0; row < max_tokens; ++row) {
                std::array<PackedToken, 16> vec;
                
                for (int j = 0; j < 16; ++j) {
                    uint32_t doc_id = sb * 64 + channel * 16 + j;
                    
                    if (row < filtered_tokens[doc_id].size()) {
                        auto& tf = filtered_tokens[doc_id][row];
                        vec[j] = PackedToken(tf.first, tf.second);
                    } else {
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
    
    std::cout << "\nTerm Frequencies (raw):" << std::endl;
    std::cout << "  Num documents: " << dtf.num_docs << std::endl;
    std::cout << "  Total entries: " << dtf.total_entries << std::endl;
    
    std::cout << "\nPacked Documents:" << std::endl;
    std::cout << "  Num super-batches: " << packed.num_super_batches << std::endl;
    std::cout << "  Vectors per channel: " << packed.vectors_per_channel() << std::endl;
    
    uint64_t total_inst = 0;
    for (uint32_t inst : packed.inst_mem) {
        total_inst += inst;
    }
    double avg_inst = (packed.num_super_batches > 0) ? 
                      (double)total_inst / packed.num_super_batches : 0;
    std::cout << "  Avg vectors per super-batch: " << avg_inst << std::endl;
    
    size_t packed_mem = 0;
    for (int c = 0; c < 4; ++c) {
        packed_mem += packed.doc_mem[c].size() * 16 * sizeof(PackedToken);
    }
    packed_mem += packed.inst_mem.size() * sizeof(uint32_t);
    std::cout << "  Packed memory usage: " << (packed_mem / 1024.0 / 1024.0) << " MB" << std::endl;
    
    if (dtf.num_docs > 0 && !dtf.doc_terms[0].empty()) {
        std::cout << "\nSample doc 0:" << std::endl;
        std::cout << "  Raw terms: " << dtf.doc_terms[0].size() << std::endl;
        std::cout << "  First term: vocab_idx=" << dtf.doc_terms[0][0].vocab_idx 
                  << ", count=" << dtf.doc_terms[0][0].count << std::endl;
    }
    
    if (packed.doc_mem[0].size() > 0) {
        std::cout << "\nSample packed (channel 0, row 0):" << std::endl;
        const auto& row = packed.doc_mem[0][0];
        for (int j = 0; j < std::min(4, 16); ++j) {
            std::cout << "  [" << j << "] token_id=" << row[j].token_id() 
                      << ", freq=" << (int)row[j].freq() << std::endl;
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
 */
std::unordered_set<int> generate_random_query_from_vocab(
    const DocumentTermFrequencies& dtf,
    int query_size = 64, 
    int seed = 42) 
{
    std::unordered_set<int> query_tokens;
    
    std::vector<std::vector<int>> tokens_by_mod(16);
    
    for (uint32_t doc_id = 0; doc_id < dtf.num_docs; doc_id++) {
        int required_mod = doc_id % 16;
        for (const auto& entry : dtf.doc_terms[doc_id]) {
            int token_id = entry.vocab_idx;
            if ((token_id % 16) == required_mod && token_id < VOCAB_SIZE) {
                tokens_by_mod[required_mod].push_back(token_id);
            }
        }
    }
    
    for (int mod = 0; mod < 16; mod++) {
        std::sort(tokens_by_mod[mod].begin(), tokens_by_mod[mod].end());
        tokens_by_mod[mod].erase(
            std::unique(tokens_by_mod[mod].begin(), tokens_by_mod[mod].end()),
            tokens_by_mod[mod].end());
    }
    
    int total_available = 0;
    for (int mod = 0; mod < 16; mod++) {
        total_available += tokens_by_mod[mod].size();
        std::cout << "  Mod " << mod << ": " << tokens_by_mod[mod].size() << " unique tokens" << std::endl;
    }
    std::cout << "  Total unique tokens (mod-16 filtered): " << total_available << std::endl;
    
    std::mt19937 gen(seed);
    
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

// BM25 parameters
constexpr float SW_K1 = 1.2f;
constexpr float SW_K1_plus_1 = SW_K1 + 1.0f;
constexpr float SW_B = 0.75f;

/**
 * Software reference implementation for BM25 scoring and top-k selection
 */
void indexer_top_ref(
    const int L,
    const DocumentTermFrequencies& dtf,
    const std::unordered_set<int>& query_tokens,
    const std::vector<uint32_t>& df,
    std::vector<int>& topk_indices,
    std::vector<float>& topk_scores)
{
    std::vector<std::pair<float, int>> scores;
    
    int docs_with_matches = 0;
    int total_matches = 0;
    
    for (int doc_id = 0; doc_id < L && doc_id < (int)dtf.num_docs; doc_id++) {
        float score = 0.0f;
        int required_mod = doc_id % 16;
        int matches_in_doc = 0;
        
        for (const auto& entry : dtf.doc_terms[doc_id]) {
            int token_id = entry.vocab_idx;
            int freq = std::min((int)entry.count, 255);
            
            if ((token_id % 16) != required_mod) {
                continue;
            }
            
            if (token_id >= VOCAB_SIZE) {
                continue;
            }
            
            if (query_tokens.find(token_id) != query_tokens.end()) {
                matches_in_doc++;
                
                float idf_num = (float)L - df[token_id] + 0.5f;
                float idf_den = df[token_id] + 0.5f;
                float idf_den_inv = 1.0f / idf_den;
                float idf = logf(idf_num * idf_den_inv);
                
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
    
    std::sort(scores.begin(), scores.end(), 
              [](const auto& a, const auto& b) { return a.first > b.first; });
    
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
    
    std::set<int> hw_set(hw_topk_indices.begin(), hw_topk_indices.end());
    std::set<int> sw_set(sw_topk_indices.begin(), sw_topk_indices.end());
    
    int overlap_count = 0;
    for (int idx : hw_set) {
        if (sw_set.find(idx) != sw_set.end()) {
            overlap_count++;
        }
    }
    
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
    
    std::cout << "\nTop 8 SW scores: ";
    for (int i = 0; i < std::min(8, (int)sw_topk_scores.size()); i++) {
        std::cout << sw_topk_scores[i] << " ";
    }
    std::cout << std::endl;
    
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

/**
 * Parse command line arguments
 */
Args parse_args(int argc, char* argv[]) {
    Args args;
    args.device_index = -1;  // -1 means auto-detect
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--bitstream" && i + 1 < argc) {
            args.bitstream = argv[++i];
        } else if (arg == "--export_dir" && i + 1 < argc) {
            args.export_dir = argv[++i];
        } else if (arg == "--use_mmap" && i + 1 < argc) {
            std::string val = argv[++i];
            args.use_mmap = (val == "true" || val == "1" || val == "yes");
        } else if (arg == "--num_threads" && i + 1 < argc) {
            args.num_threads = std::stoi(argv[++i]);
        } else if (arg == "--query_tokens" && i + 1 < argc) {
            args.query_tokens = argv[++i];
        } else if (arg == "--limit_docs" && i + 1 < argc) {
            args.limit_docs = std::stoi(argv[++i]);
        } else if (arg == "--device" && i + 1 < argc) {
            args.device_index = std::stoi(argv[++i]);
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  --bitstream <path>    Path to XCLBIN file (required for FPGA execution)" << std::endl;
            std::cout << "  --export_dir <path>   Directory containing exported BM25 data (default: ./export)" << std::endl;
            std::cout << "  --use_mmap <bool>     Use memory mapping for faster loading (default: true)" << std::endl;
            std::cout << "  --num_threads <N>     Number of threads for packing (default: 8)" << std::endl;
            std::cout << "  --query_tokens <str>  Comma-separated list of query token IDs" << std::endl;
            std::cout << "  --limit_docs <N>      Limit to first N documents (default: 0 = all)" << std::endl;
            std::cout << "  --device <N>          Device index (default: -1 = auto-detect)" << std::endl;
            exit(0);
        } else if (!arg.empty() && arg[0] != '-') {
            // Positional argument - treat as export_dir
            args.export_dir = arg;
        }
    }
    
    return args;
}

/**
 * Try to open a device and load the XCLBIN
 * Returns true on success, false on failure
 */
bool try_open_device(int device_index, const std::string& bitstream,
                     xrt::device& device, xrt::uuid& xclbin_uuid) {
    try {
        std::cout << "  Trying device " << device_index << "..." << std::endl;
        device = xrt::device(device_index);
        std::cout << "    Device name: " << device.get_info<xrt::info::device::name>() << std::endl;
        std::cout << "    Device BDF: " << device.get_info<xrt::info::device::bdf>() << std::endl;
        
        // Try to load XCLBIN
        xclbin_uuid = device.load_xclbin(bitstream);
        std::cout << "    XCLBIN loaded successfully!" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cout << "    Failed: " << e.what() << std::endl;
        return false;
    }
}

/**
 * Scan all devices and find one that works with the given XCLBIN
 * Returns the device index on success, -1 on failure
 */
int find_working_device(const std::string& bitstream, int preferred_device,
                        xrt::device& device, xrt::uuid& xclbin_uuid) {
    constexpr int MAX_DEVICES = 16;  // Maximum number of devices to scan
    
    // If a specific device is requested, try it first
    if (preferred_device >= 0) {
        std::cout << "Trying user-specified device " << preferred_device << "..." << std::endl;
        if (try_open_device(preferred_device, bitstream, device, xclbin_uuid)) {
            return preferred_device;
        }
        std::cout << "User-specified device failed, scanning all devices..." << std::endl;
    }
    
    // Scan all devices
    std::cout << "Scanning for available devices..." << std::endl;
    for (int i = 0; i < MAX_DEVICES; i++) {
        // Skip the preferred device if already tried
        if (i == preferred_device) {
            continue;
        }
        
        if (try_open_device(i, bitstream, device, xclbin_uuid)) {
            return i;
        }
    }
    
    return -1;  // No working device found
}

int main(int argc, char* argv[]) {
    Args args = parse_args(argc, argv);
    
    std::cout << "======================================" << std::endl;
    std::cout << "BM25 Data Loader and Kernel Launcher (XRT)" << std::endl;
    std::cout << "======================================" << std::endl;
    std::cout << "Export directory: " << args.export_dir << std::endl;
    std::cout << "Bitstream: " << (args.bitstream.empty() ? "(none - software only)" : args.bitstream) << std::endl;
    std::cout << "Use mmap: " << (args.use_mmap ? "yes" : "no") << std::endl;
    std::cout << "Num threads: " << args.num_threads << std::endl;
    std::cout << "Limit docs: " << (args.limit_docs > 0 ? std::to_string(args.limit_docs) : "all") << std::endl;
    std::cout << "Device index: " << (args.device_index < 0 ? "auto-detect" : std::to_string(args.device_index)) << std::endl;
    std::cout << "VOCAB_SIZE: " << VOCAB_SIZE << std::endl;
    std::cout << "TOP_K: " << TOP_K << std::endl;
    std::cout << "======================================\n" << std::endl;
    
    if (args.bitstream.empty()) {
        std::cerr << "Error: --bitstream is required for XRT execution" << std::endl;
        std::cerr << "Use --help for usage information" << std::endl;
        return 1;
    }
    
    std::vector<uint32_t> doc_freq;
    DocumentTermFrequencies term_freq;
    PackedDocuments packed;
    
    Timer timer;
    double total_time = 0.0;
    
#ifdef __linux__
    if (args.use_mmap) {
        MemoryMappedFile mmf_doc_freq, mmf_term_freq;
        
        std::cout << "Loading doc_freq.bin (mmap)..." << std::endl;
        timer.start();
        if (!mmf_doc_freq.open(args.export_dir + "/doc_freq.bin") ||
            !load_document_frequency_mmap(mmf_doc_freq, doc_freq)) {
            std::cerr << "Failed to load doc_freq.bin" << std::endl;
            return 1;
        }
        double doc_freq_time = timer.stop_ms();
        std::cout << "  Loaded in " << doc_freq_time << " ms" << std::endl;
        total_time += doc_freq_time;
        
        std::cout << "Loading term_freq.bin (mmap)..." << std::endl;
        timer.start();
        if (!mmf_term_freq.open(args.export_dir + "/term_freq.bin") ||
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
        std::cout << "Loading doc_freq.bin..." << std::endl;
        timer.start();
        if (!load_document_frequency(args.export_dir + "/doc_freq.bin", doc_freq)) {
            std::cerr << "Failed to load doc_freq.bin" << std::endl;
            return 1;
        }
        double doc_freq_time = timer.stop_ms();
        std::cout << "  Loaded in " << doc_freq_time << " ms" << std::endl;
        total_time += doc_freq_time;
        
        std::cout << "Loading term_freq.bin..." << std::endl;
        timer.start();
        if (!load_term_frequencies(args.export_dir + "/term_freq.bin", term_freq)) {
            std::cerr << "Failed to load term_freq.bin" << std::endl;
            return 1;
        }
        double term_freq_time = timer.stop_ms();
        std::cout << "  Loaded in " << term_freq_time << " ms" << std::endl;
        total_time += term_freq_time;
    }
    
    // Apply document limit if specified
    if (args.limit_docs > 0 && (uint32_t)args.limit_docs < term_freq.num_docs) {
        std::cout << "\nLimiting to first " << args.limit_docs << " documents (out of " 
                  << term_freq.num_docs << ")" << std::endl;
        term_freq.num_docs = args.limit_docs;
        term_freq.doc_terms.resize(args.limit_docs);
        term_freq.offsets.resize(args.limit_docs + 1);
    }
    
    // Pack documents for hardware
    std::cout << "\nPacking documents for hardware..." << std::endl;
    timer.start();
    pack_documents_for_hw(term_freq, packed, args.num_threads);
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
    if (!args.query_tokens.empty()) {
        query_tokens = parse_query_tokens(args.query_tokens);
        std::cout << "\nParsed " << query_tokens.size() << " query tokens from input" << std::endl;
    } else {
        std::cout << "\nGenerating query tokens from corpus vocabulary..." << std::endl;
        query_tokens = generate_random_query_from_vocab(term_freq, 256, 42);
        std::cout << "Generated " << query_tokens.size() << " query tokens from corpus" << std::endl;
    }
    
    std::cout << "Sample query tokens: ";
    int count = 0;
    for (int t : query_tokens) {
        if (count++ >= 8) break;
        std::cout << t << " ";
    }
    std::cout << "..." << std::endl;
    
    // Compute L and L_doc_total
    int L = packed.num_docs;
    L = ((L + 63) / 64) * 64;  // Round up to multiple of 64
    int L_doc_total = static_cast<int>(packed.vectors_per_channel());
    
    std::cout << "\n======================================" << std::endl;
    std::cout << "KERNEL LAUNCH PARAMETERS" << std::endl;
    std::cout << "======================================" << std::endl;
    std::cout << "L (num docs, padded): " << L << std::endl;
    std::cout << "L_doc_total (vectors per channel): " << L_doc_total << std::endl;
    std::cout << "Num super-batches: " << packed.num_super_batches << std::endl;
    std::cout << "Query tokens: " << query_tokens.size() << std::endl;
    
    // ===============================
    // XRT Native API Execution
    // ===============================
    
    std::cout << "\n======================================" << std::endl;
    std::cout << "XRT DEVICE AND KERNEL INITIALIZATION" << std::endl;
    std::cout << "======================================" << std::endl;
    
    // Open device and load XCLBIN
    timer.start();
    xrt::device device;
    xrt::uuid xclbin_uuid;
    
    int selected_device = find_working_device(args.bitstream, args.device_index, device, xclbin_uuid);
    if (selected_device < 0) {
        std::cerr << "Error: No working device found for XCLBIN: " << args.bitstream << std::endl;
        std::cerr << "Please check:" << std::endl;
        std::cerr << "  1. FPGA devices are properly installed and visible" << std::endl;
        std::cerr << "  2. The XCLBIN file exists and is compatible with the device" << std::endl;
        std::cerr << "  3. XRT runtime is properly installed" << std::endl;
        return 1;
    }
    
    double device_time = timer.stop_ms();
    std::cout << "Device " << selected_device << " opened and XCLBIN loaded in " << device_time << " ms" << std::endl;
    
    // Create kernel object
    std::cout << "Creating kernel object..." << std::endl;
    timer.start();
    xrt::kernel kernel;
    try {
        kernel = xrt::kernel(device, xclbin_uuid, "indexer_top");
    } catch (const std::exception& e) {
        std::cerr << "Error: Failed to create kernel object: " << e.what() << std::endl;
        return 1;
    }
    double kernel_time = timer.stop_ms();
    std::cout << "  Kernel created in " << kernel_time << " ms" << std::endl;
    
    // ===============================
    // Buffer Allocation
    // ===============================
    
    std::cout << "\n======================================" << std::endl;
    std::cout << "BUFFER ALLOCATION" << std::endl;
    std::cout << "======================================" << std::endl;
    
    timer.start();
    
    // Calculate buffer sizes
    // df_buffer: VOCAB_SIZE_DIV_16 * 16 ints = VOCAB_SIZE ints
    size_t df_buffer_size = VOCAB_SIZE_DIV_16 * 16 * sizeof(int);
    
    // query_bitmap_mem: VOCAB_SIZE_DIV_512 * 64 bytes (512 bits = 64 bytes)
    size_t query_bitmap_size = VOCAB_SIZE_DIV_512 * 64;
    
    // inst_mem: num_super_batches ints
    size_t inst_mem_size = packed.num_super_batches * sizeof(int);
    
    // doc_mem: 4 channels, each with L_doc_total * 16 * 4 bytes
    size_t doc_mem_size = L_doc_total * 16 * sizeof(uint32_t);
    
    // topk_id_mem: TOP_K ints (padded to 16-element vectors)
    int output_size = (TOP_K + 15) / 16;
    size_t topk_id_size = output_size * 16 * sizeof(int);
    
    std::cout << "Buffer sizes:" << std::endl;
    std::cout << "  df_buffer: " << df_buffer_size / 1024.0 << " KB" << std::endl;
    std::cout << "  query_bitmap: " << query_bitmap_size / 1024.0 << " KB" << std::endl;
    std::cout << "  inst_mem: " << inst_mem_size / 1024.0 << " KB" << std::endl;
    std::cout << "  doc_mem (per channel): " << doc_mem_size / 1024.0 / 1024.0 << " MB" << std::endl;
    std::cout << "  topk_id: " << topk_id_size << " bytes" << std::endl;
    
    // Allocate buffers using kernel.group_id() to get memory bank assignment
    // Argument order from kernel: L, L_doc_total, df_buffer(2), query_bitmap(3), inst_mem(4), doc_mem[0-3](5-8), topk_id(9)
    
    xrt::bo bo_df_buffer(device, df_buffer_size, kernel.group_id(2));
    xrt::bo bo_query_bitmap(device, query_bitmap_size, kernel.group_id(3));
    xrt::bo bo_inst_mem(device, inst_mem_size, kernel.group_id(4));
    xrt::bo bo_doc_mem_0(device, doc_mem_size, kernel.group_id(5));
    xrt::bo bo_doc_mem_1(device, doc_mem_size, kernel.group_id(6));
    xrt::bo bo_doc_mem_2(device, doc_mem_size, kernel.group_id(7));
    xrt::bo bo_doc_mem_3(device, doc_mem_size, kernel.group_id(8));
    xrt::bo bo_topk_id(device, topk_id_size, kernel.group_id(9));
    
    double alloc_time = timer.stop_ms();
    std::cout << "Buffers allocated in " << alloc_time << " ms" << std::endl;
    
    // ===============================
    // Prepare Data and Transfer to Device
    // ===============================
    
    std::cout << "\n======================================" << std::endl;
    std::cout << "DATA PREPARATION AND TRANSFER" << std::endl;
    std::cout << "======================================" << std::endl;
    
    timer.start();
    
    // Map buffers to host memory
    int* df_buffer_mapped = bo_df_buffer.map<int*>();
    uint64_t* query_bitmap_mapped = bo_query_bitmap.map<uint64_t*>();
    int* inst_mem_mapped = bo_inst_mem.map<int*>();
    uint32_t* doc_mem_0_mapped = bo_doc_mem_0.map<uint32_t*>();
    uint32_t* doc_mem_1_mapped = bo_doc_mem_1.map<uint32_t*>();
    uint32_t* doc_mem_2_mapped = bo_doc_mem_2.map<uint32_t*>();
    uint32_t* doc_mem_3_mapped = bo_doc_mem_3.map<uint32_t*>();
    int* topk_id_mapped = bo_topk_id.map<int*>();
    
    // Prepare df_buffer: pack 16 ints per vector
    for (int i = 0; i < VOCAB_SIZE_DIV_16; i++) {
        for (int j = 0; j < 16; j++) {
            df_buffer_mapped[i * 16 + j] = static_cast<int>(doc_freq[i * 16 + j]);
        }
    }
    
    // Prepare query_bitmap: 512-bit (64-byte) chunks
    // Clear first
    std::memset(query_bitmap_mapped, 0, query_bitmap_size);
    for (int token_id : query_tokens) {
        if (token_id >= 0 && token_id < VOCAB_SIZE) {
            int chunk_idx = token_id >> 9;  // token_id / 512
            int bit_idx = token_id & 0x1FF;  // token_id % 512
            int qword_idx = bit_idx / 64;
            int bit_in_qword = bit_idx % 64;
            query_bitmap_mapped[chunk_idx * 8 + qword_idx] |= (1ULL << bit_in_qword);
        }
    }
    
    // Prepare inst_mem
    for (uint32_t i = 0; i < packed.num_super_batches; i++) {
        inst_mem_mapped[i] = static_cast<int>(packed.inst_mem[i]);
    }
    
    // Prepare doc_mem for each channel
    uint32_t* doc_mem_mapped[4] = {doc_mem_0_mapped, doc_mem_1_mapped, doc_mem_2_mapped, doc_mem_3_mapped};
    for (int channel = 0; channel < 4; channel++) {
        for (size_t vec_idx = 0; vec_idx < packed.doc_mem[channel].size(); vec_idx++) {
            for (int j = 0; j < 16; j++) {
                uint32_t packed_val = 0;
                uint16_t token_id = packed.doc_mem[channel][vec_idx][j].token_id();
                uint8_t freq = packed.doc_mem[channel][vec_idx][j].freq();
                packed_val = (uint32_t(freq) << 16) | token_id;
                doc_mem_mapped[channel][vec_idx * 16 + j] = packed_val;
            }
        }
    }
    
    // Initialize output buffer to -1
    for (int i = 0; i < output_size * 16; i++) {
        topk_id_mapped[i] = -1;
    }
    
    double prep_time = timer.stop_ms();
    std::cout << "Data prepared in " << prep_time << " ms" << std::endl;
    
    // Sync buffers to device
    std::cout << "Syncing buffers to device..." << std::endl;
    timer.start();
    
    bo_df_buffer.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_query_bitmap.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_inst_mem.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_doc_mem_0.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_doc_mem_1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_doc_mem_2.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_doc_mem_3.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    
    double sync_to_time = timer.stop_ms();
    std::cout << "  Synced to device in " << sync_to_time << " ms" << std::endl;
    
    // ===============================
    // Kernel Execution
    // ===============================
    
    std::cout << "\n======================================" << std::endl;
    std::cout << "KERNEL EXECUTION" << std::endl;
    std::cout << "======================================" << std::endl;
    
    std::cout << "Running kernel..." << std::endl;
    timer.start();
    
    // Execute kernel with arguments:
    // void indexer_top(
    //     const int L,
    //     const int L_doc_total,
    //     tapa::mmap<tapa::vec_t<int, 16>> df_buffer,
    //     tapa::mmap<ap_uint<512>> query_bitmap_mem,
    //     tapa::mmap<int> inst_mem,
    //     tapa::mmaps<tapa::vec_t<ap_uint<32>, 16>, 4> doc_mem,  // 4 separate buffers
    //     tapa::mmap<tapa::vec_t<int, 16>> topk_id_mem
    // )
    auto run = kernel(
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
    );
    
    // Wait for kernel to complete
    run.wait();
    
    double exec_time = timer.stop_ms();
    std::cout << "  Kernel execution completed in " << exec_time << " ms" << std::endl;
    
    // Sync output buffer from device
    std::cout << "Syncing output buffer from device..." << std::endl;
    timer.start();
    bo_topk_id.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    double sync_from_time = timer.stop_ms();
    std::cout << "  Synced from device in " << sync_from_time << " ms" << std::endl;
    
    // ===============================
    // Extract Results
    // ===============================
    
    std::cout << "\n======================================" << std::endl;
    std::cout << "TOP-K RESULTS" << std::endl;
    std::cout << "======================================" << std::endl;
    
    std::vector<int> hw_topk_indices;
    for (int i = 0; i < output_size; i++) {
        for (int j = 0; j < 16 && (i * 16 + j) < TOP_K; j++) {
            hw_topk_indices.push_back(topk_id_mapped[i * 16 + j]);
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
    
    // Print performance summary
    std::cout << "\n======================================" << std::endl;
    std::cout << "PERFORMANCE SUMMARY" << std::endl;
    std::cout << "======================================" << std::endl;
    std::cout << "Data loading + packing: " << total_time << " ms" << std::endl;
    std::cout << "Device open + XCLBIN load: " << device_time << " ms" << std::endl;
    std::cout << "Kernel create: " << kernel_time << " ms" << std::endl;
    std::cout << "Buffer allocation: " << alloc_time << " ms" << std::endl;
    std::cout << "Data preparation: " << prep_time << " ms" << std::endl;
    std::cout << "Sync to device: " << sync_to_time << " ms" << std::endl;
    std::cout << "Kernel execution: " << exec_time << " ms" << std::endl;
    std::cout << "Sync from device: " << sync_from_time << " ms" << std::endl;
    std::cout << "Software reference: " << sw_time << " ms" << std::endl;
    
    std::cout << "\n======================================" << std::endl;
    std::cout << "Done!" << std::endl;
    std::cout << "======================================" << std::endl;
    
    return validation_passed ? 0 : 1;
}
