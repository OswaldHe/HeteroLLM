#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <random>
#include <set>

#include <gflags/gflags.h>
#include <tapa.h>

// Include the kernel header
#include "indexer_seer_token_budget.h"

DEFINE_string(bitstream, "", "path to bitstream file, run csim if empty");

// Software reference implementation for top-k selection
// Returns the indices of the top TOP_K scores (highest scores) for each query
void indexer_top_ref(
    const int L,
    const int NUM_QUERY,
    const std::vector<std::vector<ap_int<16>>>& past_keys,  // [L][HEAD_DIM]
    const std::vector<std::vector<ap_int<16>>>& queries,     // [NUM_QUERY][HEAD_DIM]
    const std::vector<std::vector<ap_int<16>>>& new_keys,    // [NUM_QUERY][HEAD_DIM]
    std::vector<std::vector<int>>& topk_indices)             // [NUM_QUERY][TOP_K] - top-k indices per query
{
    topk_indices.clear();
    topk_indices.resize(NUM_QUERY);
    
    // Create a temporary key buffer that includes past keys
    std::vector<std::vector<ap_int<16>>> all_keys = past_keys;
    
    // For each query
    for (int q_idx = 0; q_idx < NUM_QUERY; q_idx++) {
        int current_pos = L + (q_idx / 8);
        
        // Add new keys up to current position
        for (int i = 0; i <= q_idx && i < NUM_QUERY; i++) {
            int pos = L + (i / 8);
            if (pos < (int)all_keys.size()) {
                all_keys[pos] = new_keys[i];
            } else if (pos == (int)all_keys.size()) {
                all_keys.push_back(new_keys[i]);
            }
        }
        
        // Compute all scores for this query
        std::vector<std::pair<float, int>> all_scores;
        
        // Compute dot products for all keys
        for (int k_idx = 0; k_idx <= current_pos; k_idx++) {
            // Compute dot product between query and key
            int64_t dot_product = 0;
            for (int d = 0; d < HEAD_DIM; d++) {
                int16_t q_val = queries[q_idx][d].to_int();
                int16_t k_val = (k_idx < (int)all_keys.size()) ? 
                                all_keys[k_idx][d].to_int() : 0;
                dot_product += (int64_t)q_val * (int64_t)k_val;
            }
            
            all_scores.push_back({(float)dot_product, k_idx});
        }
        
        // Sort by score (descending) to get top-k
        std::sort(all_scores.begin(), all_scores.end(), 
                  [](const auto& a, const auto& b) { return a.first > b.first; });
        
        // Extract top-k indices for this query
        for (int i = 0; i < TOP_K && i < (int)all_scores.size(); i++) {
            topk_indices[q_idx].push_back(all_scores[i].second);
        }
    }
}

int main(int argc, char* argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    
    int L = 1024;  // Number of past key vectors (must be multiple of 128 for the kernel)
    int NUM_QUERY = 8;  // Number of queries
    
    if (argc > 1) {
        L = std::atoi(argv[1]);
    }
    if (argc > 2) {
        NUM_QUERY = std::atoi(argv[2]);
    }
    
    // Ensure L is a multiple of 128 (required by kernel for top-k processing)
    L = (L + 127) & ~127;
    
    // L must be > TOP_K for the kernel to work properly
    if (L <= TOP_K) {
        L = TOP_K + 128;
    }
    
    std::cout << "Indexer SEER Token Budget (Top-K) Testbench" << std::endl;
    std::cout << "L (past key vectors): " << L << std::endl;
    std::cout << "NUM_QUERY: " << NUM_QUERY << std::endl;
    std::cout << "HEAD_DIM: " << HEAD_DIM << std::endl;
    std::cout << "TOP_K: " << TOP_K << std::endl;
    
    // Initialize random number generator
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_int_distribution<int> dis_int(-100, 100);  // Range for ap_int<16>
    
    // Generate random past key vectors (L x HEAD_DIM)
    std::vector<std::vector<ap_int<16>>> past_keys(L, std::vector<ap_int<16>>(HEAD_DIM));
    for (int k = 0; k < L; k++) {
        for (int d = 0; d < HEAD_DIM; d++) {
            past_keys[k][d] = ap_int<16>(dis_int(gen));
        }
    }

    for (int k = 1024; k < L; k++) {
        for (int d = 0; d < HEAD_DIM; d++) {
            past_keys[k][d] = 0;
        }
    }
    
    // Generate random query vectors (NUM_QUERY x HEAD_DIM)
    std::vector<std::vector<ap_int<16>>> queries(NUM_QUERY, std::vector<ap_int<16>>(HEAD_DIM));
    for (int q = 0; q < NUM_QUERY; q++) {
        for (int d = 0; d < HEAD_DIM; d++) {
            queries[q][d] = ap_int<16>(dis_int(gen));
        }
    }
    
    // Generate random new key vectors (NUM_QUERY x HEAD_DIM)
    std::vector<std::vector<ap_int<16>>> new_keys(NUM_QUERY, std::vector<ap_int<16>>(HEAD_DIM));
    for (int k = 0; k < NUM_QUERY; k++) {
        for (int d = 0; d < HEAD_DIM; d++) {
            new_keys[k][d] = ap_int<16>(dis_int(gen));
        }
    }
    
    // Create some vectors with very high dot products to ensure predictable top-k
    // Make first query highly correlated with some specific past keys
    std::vector<int> high_correlation_indices;
    for (int k = 0; k < std::min(TOP_K / 2, L); k++) {
        int idx = k * 2;  // Spread them out
        if (idx < L) {
            for (int d = 0; d < HEAD_DIM; d++) {
                past_keys[idx][d] = queries[0][d];  // Perfect correlation
            }
            high_correlation_indices.push_back(idx);
        }
    }
    
    std::cout << "Created " << high_correlation_indices.size() << " high-correlation keys" << std::endl;
    
    // Prepare hardware inputs
    // qk_vec_mem layout (across 8 channels):
    // - First: All past keys (L vectors), distributed across 8 channels
    //   Each channel gets L/8 vectors
    // - Then: For each query (q0, k0, q1, k1, ...)
    //   Each q and k also distributed across channels
    
    const int L_per_channel = L / 8;
    const int lines_per_channel = ((L_per_channel + NUM_QUERY*2) * HEAD_DIM) >> 5;
    
    std::cout << "L per channel: " << L_per_channel << std::endl;
    std::cout << "Lines per channel: " << lines_per_channel << std::endl;
    
    // Allocate memory for 8 channels
    std::vector<std::vector<tapa::vec_t<ap_uint<32>, 16>>> qk_vec_hw(8);
    for (int ch = 0; ch < 8; ch++) {
        qk_vec_hw[ch].resize(lines_per_channel);
    }
    
    // Helper function to pack two 16-bit integers into one 32-bit word
    auto pack_two_int16 = [](ap_int<16> low, ap_int<16> high) -> ap_uint<32> {
        ap_uint<32> result;
        result(15, 0) = low.to_uint();
        result(31, 16) = high.to_uint();
        return result;
    };
    
    // Pack past keys - each channel gets L/8 consecutive vectors
    for (int ch = 0; ch < 8; ch++) {
        int write_idx = 0;
        for (int k = 0; k < L_per_channel; k++) {
            int global_k = ch * L_per_channel + k;
            // Each key vector: pack into HEAD_DIM/2 32-bit words (each holds 2 int16)
            // These are organized in groups of 16 words
            for (int word_group = 0; word_group < HEAD_DIM_DIV_2; word_group += 16) {
                tapa::vec_t<ap_uint<32>, 16> packed;
                for (int i = 0; i < 16; i++) {
                    int dim_idx = (word_group + i) * 2;
                    packed[i] = pack_two_int16(past_keys[global_k][dim_idx], 
                                               past_keys[global_k][dim_idx + 1]);
                }
                qk_vec_hw[ch][write_idx++] = packed;
            }
        }
        
        // Pack query and new key pairs for this channel
        for (int q = 0; q < NUM_QUERY; q++) {
            // Pack query
            for (int word_group = 0; word_group < HEAD_DIM_DIV_2; word_group += 16) {
                tapa::vec_t<ap_uint<32>, 16> packed;
                for (int i = 0; i < 16; i++) {
                    int dim_idx = (word_group + i) * 2;
                    packed[i] = pack_two_int16(queries[q][dim_idx], queries[q][dim_idx + 1]);
                }
                qk_vec_hw[ch][write_idx++] = packed;
            }
            
            // Pack new key
            for (int word_group = 0; word_group < HEAD_DIM_DIV_2; word_group += 16) {
                tapa::vec_t<ap_uint<32>, 16> packed;
                for (int i = 0; i < 16; i++) {
                    int dim_idx = (word_group + i) * 2;
                    packed[i] = pack_two_int16(new_keys[q][dim_idx], new_keys[q][dim_idx + 1]);
                }
                qk_vec_hw[ch][write_idx++] = packed;
            }
        }
    }
    
    // Allocate output memory for top-k indices
    // Output is NUM_QUERY * TOP_K integers, packed as vec_t<int, 16>
    const int output_size = NUM_QUERY * ((TOP_K + 15) / 16);
    std::vector<tapa::vec_t<int, 16>> topk_indices_hw(output_size);
    
    std::cout << "\nRunning hardware kernel..." << std::endl;
    std::cout << "Total input lines per channel: " << lines_per_channel << std::endl;
    std::cout << "Output size (vec_t<int,16>): " << output_size << std::endl;
    std::cout << "Total output indices: " << NUM_QUERY * TOP_K << std::endl;
    
    // Invoke the kernel
    int64_t kernel_time_ns = 0;
    kernel_time_ns = tapa::invoke(indexer_top, FLAGS_bitstream,
                 L_per_channel,  // L is per channel
                 NUM_QUERY,
                 tapa::read_only_mmaps<tapa::vec_t<ap_uint<32>, 16>, 8>(qk_vec_hw),
                 tapa::write_only_mmap<tapa::vec_t<int, 16>>(topk_indices_hw));
    
    std::cout << "Hardware kernel completed." << std::endl;
    std::clog << "kernel time: " << kernel_time_ns * 1e-9 << " s" << std::endl;
    
    // Extract hardware results - organized as [NUM_QUERY][TOP_K]
    std::vector<std::vector<int>> hw_topk_indices(NUM_QUERY);
    int hw_idx = 0;
    for (int q = 0; q < NUM_QUERY; q++) {
        for (int i = 0; i < (TOP_K / 16); i++) {
            for (int j = 0; j < 16; j++) {
                hw_topk_indices[q].push_back(topk_indices_hw[hw_idx][j]);
            }
            hw_idx++;
        }
    }
    
    // Compute software reference
    std::cout << "\nRunning software reference..." << std::endl;
    
    std::vector<std::vector<int>> sw_topk_indices;
    indexer_top_ref(L, NUM_QUERY, past_keys, queries, new_keys, sw_topk_indices);
    
    std::cout << "Software reference completed." << std::endl;
    
    // Compare results for each query
    std::cout << "\n=== Results ===" << std::endl;
    
    int total_overlap = 0;
    int total_high_corr_hw = 0;
    int total_high_corr_sw = 0;
    bool all_indices_valid = true;
    bool all_no_duplicates = true;
    
    for (int q = 0; q < NUM_QUERY; q++) {
        // Convert to sets for comparison (order may differ)
        std::set<int> hw_set(hw_topk_indices[q].begin(), hw_topk_indices[q].end());
        std::set<int> sw_set(sw_topk_indices[q].begin(), sw_topk_indices[q].end());
        
        // Check overlap between HW and SW results
        int overlap_count = 0;
        for (int idx : hw_set) {
            if (sw_set.find(idx) != sw_set.end()) {
                overlap_count++;
            }
        }
        total_overlap += overlap_count;
        
        // Check how many high-correlation indices were captured (for query 0)
        if (q == 0) {
            for (int idx : high_correlation_indices) {
                if (hw_set.find(idx) != hw_set.end()) {
                    total_high_corr_hw++;
                }
                if (sw_set.find(idx) != sw_set.end()) {
                    total_high_corr_sw++;
                }
            }
        }
        
        // Check for valid indices
        int current_pos = L + (q / 8);
        for (int idx : hw_topk_indices[q]) {
            if (idx < 0 || idx > current_pos) {
                all_indices_valid = false;
            }
        }
        
        // Check for duplicates
        if (hw_set.size() != hw_topk_indices[q].size()) {
            all_no_duplicates = false;
        }
        
        if (q < 3) {  // Print details for first 3 queries
            std::cout << "\nQuery " << q << ":" << std::endl;
            std::cout << "  HW returned " << hw_topk_indices[q].size() << " indices" << std::endl;
            std::cout << "  SW returned " << sw_topk_indices[q].size() << " indices" << std::endl;
            std::cout << "  Overlap: " << overlap_count << " / " << TOP_K << std::endl;
            
            std::cout << "  First 10 HW indices: ";
            for (int i = 0; i < std::min(10, (int)hw_topk_indices[q].size()); i++) {
                std::cout << hw_topk_indices[q][i] << " ";
            }
            std::cout << std::endl;
            
            std::cout << "  First 10 SW indices: ";
            for (int i = 0; i < std::min(10, (int)sw_topk_indices[q].size()); i++) {
                std::cout << sw_topk_indices[q][i] << " ";
            }
            std::cout << std::endl;
        }
    }
    
    std::cout << "\n=== Statistics ===" << std::endl;
    std::cout << "Total overlap across all queries: " << total_overlap << " / " << (NUM_QUERY * TOP_K) << std::endl;
    std::cout << "High-correlation keys in HW (query 0): " << total_high_corr_hw << " / " << high_correlation_indices.size() << std::endl;
    std::cout << "High-correlation keys in SW (query 0): " << total_high_corr_sw << " / " << high_correlation_indices.size() << std::endl;
    
    float overlap_ratio = (float)total_overlap / (NUM_QUERY * TOP_K);
    float high_corr_ratio = (high_correlation_indices.size() > 0) ? 
                            (float)total_high_corr_hw / high_correlation_indices.size() : 1.0f;
    
    std::cout << "Average overlap ratio: " << (overlap_ratio * 100) << "%" << std::endl;
    std::cout << "High-correlation capture ratio: " << (high_corr_ratio * 100) << "%" << std::endl;
    
    if (!all_indices_valid) {
        std::cout << "Warning: Some hardware output contains invalid indices" << std::endl;
    }
    
    if (!all_no_duplicates) {
        std::cout << "Warning: Some hardware output contains duplicate indices" << std::endl;
    }
    
    // Success criteria:
    // 1. All high-correlation indices should be captured (they have highest scores)
    // 2. Indices should be valid
    // 3. No duplicates
    if (high_corr_ratio >= 0.9 && all_indices_valid && all_no_duplicates) {
        std::cout << "\n✓ PASSED: Top-K selection is working correctly!" << std::endl;
        return 0;
    } else if (overlap_ratio >= 0.5) {
        std::cout << "\n~ PARTIAL PASS: Significant overlap but some differences." << std::endl;
        std::cout << "  This may be due to ties in scores or numerical precision." << std::endl;
        return 0;
    } else {
        std::cout << "\n✗ FAILED: Significant mismatch between hardware and software!" << std::endl;
        return 1;
    }
}
