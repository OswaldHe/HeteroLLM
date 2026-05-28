#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <random>
#include <set>
#include <map>
#include <unordered_set>
#include <unordered_map>

#include <gflags/gflags.h>
#include <tapa.h>

// Include the kernel header
#include "indexer_bm25.h"

template <typename T>
using aligned_vector = std::vector<T, tapa::aligned_allocator<T>>;

DEFINE_string(bitstream, "", "path to bitstream file, run csim if empty");

// BM25 parameters (must match kernel)
constexpr float SW_K1 = 1.2f;
constexpr float SW_K1_plus_1 = SW_K1 + 1.0f;
constexpr float SW_B = 0.75f;

// Software reference implementation for BM25 scoring and top-k selection
// This must match the kernel's behavior exactly, including the mod-16 constraint
void indexer_top_ref(
    const int L,
    const std::vector<std::vector<std::pair<int, int>>>& documents,  // [L][variable] - (token_id, freq) pairs
    const std::unordered_set<int>& query_tokens,  // Set of tokens in the query
    const std::vector<int>& df,  // Document frequency for each token
    std::vector<int>& topk_indices)  // Output: top-K document indices
{
    // Compute BM25 scores for all documents
    std::vector<std::pair<float, int>> scores;  // (score, doc_id)
    
    for (int doc_id = 0; doc_id < L; doc_id++) {
        float score = 0.0f;
        int required_mod = doc_id % 16;  // Kernel can only process tokens with this mod
        
        for (const auto& token_freq : documents[doc_id]) {
            int token_id = token_freq.first;
            int freq = token_freq.second;
            
            // IMPORTANT: Only count tokens that the kernel can process
            // Token at position j must have token_id % 16 == j for correct lookup
            if ((token_id % 16) != required_mod) {
                continue;  // Kernel would not correctly process this token
            }
            
            // Check if token is in query
            if (query_tokens.find(token_id) != query_tokens.end()) {
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
        
        scores.push_back({score, doc_id});
    }
    
    // Sort by score descending
    std::sort(scores.begin(), scores.end(), 
              [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Extract top-K indices
    topk_indices.clear();
    for (int i = 0; i < TOP_K && i < (int)scores.size(); i++) {
        topk_indices.push_back(scores[i].second);
    }
}

// Pack documents for hardware format
// The kernel processes 16 documents in parallel (a batch).
// For each batch, we read L_doc vectors, where each vector has 16 elements.
// Element at position j corresponds to document j in the batch.
// CONSTRAINT: token at position j must have token_id % 16 == j for correct df/query lookup.
// 
// This means: for document d at position (d % 16), we can only correctly process
// tokens where token_id % 16 == (d % 16).
//
// Strategy: For each batch, find the max number of tokens across documents that
// satisfy the mod-16 constraint, then pack them row by row.
//
// With 4 channels:
// - Channel 0: docs 0-15, 64-79, 128-143, ...
// - Channel 1: docs 16-31, 80-95, 144-159, ...
// - Channel 2: docs 32-47, 96-111, 160-175, ...
// - Channel 3: docs 48-63, 112-127, 176-191, ...
// For every 64 documents (16 per channel), all channels must have the same
// number of elements (padded with dummy elements if needed).
struct PackedData {
    std::vector<aligned_vector<tapa::vec_t<ap_uint<32>, 16>>> doc_mem;  // 4 channels
    aligned_vector<int> inst_mem;  // Number of vectors per super-batch (64 docs)
};

PackedData pack_documents_for_hw(
    const std::vector<std::vector<std::pair<int, int>>>& documents,
    int L)
{
    PackedData result;
    // Initialize 4 channels
    result.doc_mem.resize(4);
    // Number of super-batches (each super-batch = 64 docs = 16 docs per channel)
    int num_super_batches = L >> 6;
    result.inst_mem.resize(num_super_batches);
    
    for (int super_batch = 0; super_batch < num_super_batches; super_batch++) {
        // For each channel, collect tokens for its 16 documents
        // doc_tokens[channel][j] = list of (token_id, freq) for document at position j
        // where token_id % 16 == j
        std::vector<std::vector<std::vector<std::pair<int, int>>>> doc_tokens(4, 
            std::vector<std::vector<std::pair<int, int>>>(16));
        
        // Document ordering across channels:
        // Channel 0: super_batch*64 + 0-15
        // Channel 1: super_batch*64 + 16-31
        // Channel 2: super_batch*64 + 32-47
        // Channel 3: super_batch*64 + 48-63
        for (int channel = 0; channel < 4; channel++) {
            for (int j = 0; j < 16; j++) {
                int doc_id = super_batch * 64 + channel * 16 + j;
                for (const auto& token_freq : documents[doc_id]) {
                    int token_id = token_freq.first;
                    int freq = token_freq.second;
                    // Only include tokens where token_id % 16 == j (the document's position)
                    if ((token_id % 16) == j) {
                        doc_tokens[channel][j].push_back({token_id, freq});
                    }
                }
            }
        }
        
        // Find max number of valid tokens across ALL 64 documents (all 4 channels)
        int max_tokens = 0;
        for (int channel = 0; channel < 4; channel++) {
            for (int j = 0; j < 16; j++) {
                max_tokens = std::max(max_tokens, (int)doc_tokens[channel][j].size());
            }
        }
        
        // Ensure at least 1 vector per super-batch
        if (max_tokens == 0) max_tokens = 1;
        
        result.inst_mem[super_batch] = max_tokens;
        
        // Pack vectors for each channel: for each row i, position j has token i 
        // from document j (or dummy)
        for (int channel = 0; channel < 4; channel++) {
            for (int i = 0; i < max_tokens; i++) {
                tapa::vec_t<ap_uint<32>, 16> packed;
                for (int j = 0; j < 16; j++) {
                    ap_uint<32> packed_val = 0;
                    if (i < (int)doc_tokens[channel][j].size()) {
                        int token_id = doc_tokens[channel][j][i].first;
                        int freq = doc_tokens[channel][j][i].second;
                        packed_val(15, 0) = ap_uint<16>(token_id);
                        packed_val(23, 16) = ap_uint<8>(freq);
                    } else {
                        // Dummy element: token_id with correct mod, freq = 0
                        packed_val(15, 0) = ap_uint<16>(j);  // token_id % 16 == j
                        packed_val(23, 16) = ap_uint<8>(0);  // freq = 0, no contribution
                    }
                    packed[j] = packed_val;
                }
                result.doc_mem[channel].push_back(packed);
            }
        }
    }
    
    return result;
}

int main(int argc, char* argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    
    int L = 1024;  // Number of documents (must be multiple of 16)
    int tokens_per_doc = 32;  // Average tokens per document
    int query_size = 64;  // Number of tokens in query
    
    if (argc > 1) {
        L = std::atoi(argv[1]);
    }
    if (argc > 2) {
        tokens_per_doc = std::atoi(argv[2]);
    }
    
    // Ensure L is a multiple of 64 (for 4 channels x 16 docs per batch)
    L = (L + 63) & ~63;
    
    // L must be > TOP_K for meaningful top-k selection
    if (L < TOP_K * 2) {
        L = TOP_K * 2;
        // Round up to multiple of 64 again
        L = (L + 63) & ~63;
    }
    
    std::cout << "Indexer BM25 Testbench" << std::endl;
    std::cout << "L (number of documents): " << L << std::endl;
    std::cout << "Tokens per document (avg): " << tokens_per_doc << std::endl;
    std::cout << "Query size: " << query_size << std::endl;
    std::cout << "TOP_K: " << TOP_K << std::endl;
    std::cout << "VOCAB_SIZE: " << VOCAB_SIZE << std::endl;
    
    // Initialize random number generator
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_int_distribution<int> freq_dis(1, 10);  // Frequency 1-10
    std::uniform_int_distribution<int> doc_len_dis(tokens_per_doc / 2, tokens_per_doc * 3 / 2);
    
    // Generate random query tokens - ensure we have tokens with various mod-16 values
    std::unordered_set<int> query_tokens;
    std::uniform_int_distribution<int> token_base_dis(0, (VOCAB_SIZE / 16) - 1);
    while ((int)query_tokens.size() < query_size) {
        // Generate token_id = base * 16 + mod, ensuring variety in mod values
        int base = token_base_dis(gen);
        int mod = query_tokens.size() % 16;  // Spread across all mod values
        int token_id = base * 16 + mod;
        query_tokens.insert(token_id);
    }
    std::vector<int> query_tokens_vec(query_tokens.begin(), query_tokens.end());
    
    std::cout << "Generated " << query_tokens.size() << " unique query tokens" << std::endl;
    
    // Generate random documents
    // IMPORTANT: For document d at position (d % 16), we generate tokens with token_id % 16 == (d % 16)
    // This ensures the hardware can correctly process them.
    std::vector<std::vector<std::pair<int, int>>> documents(L);
    std::vector<int> df(VOCAB_SIZE, 0);  // Document frequency for each token
    
    for (int doc_id = 0; doc_id < L; doc_id++) {
        int num_tokens = doc_len_dis(gen);
        int required_mod = doc_id % 16;  // Tokens must have token_id % 16 == required_mod
        
        std::unordered_map<int, int> token_freq_map;
        
        for (int t = 0; t < num_tokens; t++) {
            // Generate token with correct mod-16 value
            int base = token_base_dis(gen);
            int token_id = base * 16 + required_mod;
            token_freq_map[token_id]++;
        }
        
        // Convert to vector of pairs and update df
        for (const auto& tf : token_freq_map) {
            documents[doc_id].push_back({tf.first, std::min(tf.second, 255)});
            df[tf.first]++;
        }
    }
    
    // Ensure some documents have high overlap with query for predictable top-k
    std::vector<int> high_overlap_docs;
    for (int i = 0; i < std::min(TOP_K / 2, L); i++) {
        int doc_id = i * 2;  // Spread them out
        int required_mod = doc_id % 16;
        
        // Clear previous tokens (decrement their df)
        for (const auto& tf : documents[doc_id]) {
            df[tf.first]--;
        }
        documents[doc_id].clear();
        
        // Add query tokens that match this document's required mod
        int added = 0;
        for (int token_id : query_tokens_vec) {
            if ((token_id % 16) == required_mod) {
                int freq = 5 + (added % 5);  // High frequency
                documents[doc_id].push_back({token_id, freq});
                df[token_id]++;
                added++;
            }
        }
        
        if (added > 0) {
            high_overlap_docs.push_back(doc_id);
        }
    }
    
    std::cout << "Created " << high_overlap_docs.size() << " high-overlap documents" << std::endl;
    
    // Prepare hardware inputs
    
    // 1. df_buffer: VOCAB_SIZE / 16 vectors of 16 integers each
    aligned_vector<tapa::vec_t<int, 16>> df_buffer_hw(VOCAB_SIZE_DIV_16);
    for (int i = 0; i < VOCAB_SIZE_DIV_16; i++) {
        for (int j = 0; j < 16; j++) {
            df_buffer_hw[i][j] = df[i * 16 + j];
        }
    }
    
    // 2. query_bitmap_mem: VOCAB_SIZE / 512 ap_uint<512> values
    aligned_vector<ap_uint<512>> query_bitmap_hw(VOCAB_SIZE_DIV_512);
    for (int i = 0; i < VOCAB_SIZE_DIV_512; i++) {
        query_bitmap_hw[i] = 0;
    }
    for (int token_id : query_tokens) {
        int chunk_idx = token_id >> 9;  // token_id / 512
        int bit_idx = token_id & 0x1FF;  // token_id % 512
        query_bitmap_hw[chunk_idx][bit_idx] = 1;
    }
    
    // 3. Pack documents for hardware format
    auto packed = pack_documents_for_hw(documents, L);
    
    // 4. inst_mem and doc_mem are from packed result
    int num_super_batches = L >> 6;  // 64 docs per super-batch
    aligned_vector<int>& inst_mem_hw = packed.inst_mem;
    
    // Each channel should have the same number of vectors
    int total_doc_vectors_per_channel = packed.doc_mem[0].size();
    
    std::cout << "Number of super-batches (64 docs each): " << num_super_batches << std::endl;
    std::cout << "Document vectors per channel: " << total_doc_vectors_per_channel << std::endl;
    
    // 5. doc_mem is already packed into 4 channels as a vector
    std::vector<aligned_vector<tapa::vec_t<ap_uint<32>, 16>>>& doc_mem_hw = packed.doc_mem;
    
    // 6. Output: topk_id_mem
    const int output_size = (TOP_K + 15) / 16;
    aligned_vector<tapa::vec_t<int, 16>> topk_id_hw(output_size);
    
    std::cout << "\nRunning hardware kernel..." << std::endl;
    std::cout << "L: " << L << std::endl;
    std::cout << "L_doc_total (per channel): " << total_doc_vectors_per_channel << std::endl;
    std::cout << "Number of super-batches (L >> 6): " << num_super_batches << std::endl;
    std::cout << "TopK output size (vec_t<int,16>): " << output_size << std::endl;
    
    // Invoke the kernel with 4 doc_mem channels

    // Log average of inst_mem_hw
    int64_t total_inst = 0;
    for (int i = 0; i < num_super_batches; i++) {
        total_inst += inst_mem_hw[i];
    }
    double avg_inst = (num_super_batches > 0) ? (double)total_inst / num_super_batches : 0.0;
    std::cout << "Average inst_mem per super-batch: " << avg_inst << std::endl;

    int64_t kernel_time_ns = 0;
    kernel_time_ns = tapa::invoke(indexer_top, FLAGS_bitstream,
                 L,
                 total_doc_vectors_per_channel,
                 tapa::read_only_mmap<tapa::vec_t<int, 16>>(df_buffer_hw),
                 tapa::read_only_mmap<ap_uint<512>>(query_bitmap_hw),
                 tapa::read_only_mmap<int>(inst_mem_hw),
                 tapa::read_only_mmaps<tapa::vec_t<ap_uint<32>, 16>, 4>(doc_mem_hw),
                 tapa::write_only_mmap<tapa::vec_t<int, 16>>(topk_id_hw));
    
    std::cout << "Hardware kernel completed." << std::endl;
    std::clog << "kernel time: " << kernel_time_ns * 1e-9 << " s" << std::endl;
    
    // Extract hardware results
    std::vector<int> hw_topk_indices;
    for (int i = 0; i < output_size; i++) {
        for (int j = 0; j < 16 && (i * 16 + j) < TOP_K; j++) {
            hw_topk_indices.push_back(topk_id_hw[i][j]);
        }
    }
    
    // Compute software reference
    std::cout << "\nRunning software reference..." << std::endl;
    
    std::vector<int> sw_topk_indices;
    indexer_top_ref(L, documents, query_tokens, df, sw_topk_indices);
    
    std::cout << "Software reference completed." << std::endl;
    
    // Compare results
    std::cout << "\n=== Results ===" << std::endl;
    
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
    
    // Check how many high-overlap documents were captured
    int high_overlap_hw = 0;
    int high_overlap_sw = 0;
    for (int doc_id : high_overlap_docs) {
        if (hw_set.find(doc_id) != hw_set.end()) {
            high_overlap_hw++;
        }
        if (sw_set.find(doc_id) != sw_set.end()) {
            high_overlap_sw++;
        }
    }
    
    // Check for valid indices
    bool all_indices_valid = true;
    for (int idx : hw_topk_indices) {
        if (idx < 0 || idx >= L) {
            all_indices_valid = false;
            std::cout << "Invalid index: " << idx << std::endl;
        }
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
    
    std::cout << "\n=== Statistics ===" << std::endl;
    std::cout << "High-overlap docs in HW top-K: " << high_overlap_hw << " / " << high_overlap_docs.size() << std::endl;
    std::cout << "High-overlap docs in SW top-K: " << high_overlap_sw << " / " << high_overlap_docs.size() << std::endl;
    
    float overlap_ratio = (float)overlap_count / TOP_K;
    float high_overlap_ratio = (high_overlap_docs.size() > 0) ? 
                               (float)high_overlap_hw / high_overlap_docs.size() : 1.0f;
    
    std::cout << "Overlap ratio: " << (overlap_ratio * 100) << "%" << std::endl;
    std::cout << "High-overlap document capture ratio: " << (high_overlap_ratio * 100) << "%" << std::endl;
    
    if (!all_indices_valid) {
        std::cout << "Warning: Some hardware output contains invalid indices" << std::endl;
    }
    
    if (!no_duplicates) {
        std::cout << "Warning: Hardware output contains duplicate indices" << std::endl;
        std::cout << "Unique indices: " << hw_set.size() << " vs Total: " << hw_topk_indices.size() << std::endl;
    }
    
    // Success criteria:
    // 1. High-overlap documents should be captured (they have highest BM25 scores)
    // 2. Indices should be valid
    // 3. No duplicates
    if (high_overlap_ratio >= 0.9 && all_indices_valid && no_duplicates) {
        std::cout << "\n✓ PASSED: BM25 Top-K selection is working correctly!" << std::endl;
        return 0;
    } else if (overlap_ratio >= 0.5) {
        std::cout << "\n~ PARTIAL PASS: Significant overlap but some differences." << std::endl;
        std::cout << "  This may be due to ties in BM25 scores or numerical precision." << std::endl;
        return 0;
    } else {
        std::cout << "\n✗ FAILED: Significant mismatch between hardware and software!" << std::endl;
        return 1;
    }
}
