#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <random>
#include <set>

#include <gflags/gflags.h>
#include <tapa.h>

#include "indexer_vanilla.h"

DEFINE_string(bitstream, "", "path to bitstream file, run csim if empty");

// Software reference implementation
void indexer_top_ref(
    const int L,
    const std::vector<std::vector<ap_int<16>>>& query_vecs,  // [NUM_INDEX_HEAD][HEAD_DIM]
    const std::vector<std::vector<ap_int<16>>>& key_vecs,    // [L][HEAD_DIM]
    const std::vector<float>& weights,                        // [NUM_INDEX_HEAD]
    std::vector<int>& topk_ids)                              // [TOP_K]
{
    // Compute weighted indexing scores
    std::vector<std::pair<float, int>> scores(L);
    
    for (int k = 0; k < L; k++) {
        float total_score = 0.0f;
        
        for (int h = 0; h < NUM_INDEX_HEAD; h++) {
            // Compute dot product qk (using int64_t to avoid overflow)
            int64_t qk = 0;
            for (int d = 0; d < HEAD_DIM; d++) {
                qk += query_vecs[h][d].to_int() * key_vecs[k][d].to_int();
            }
            
            // Apply ReLU
            if (qk < 0) {
                qk = 0;
            }
            
            // Weight and accumulate
            total_score += weights[h] * (float)qk;
        }

        scores[k] = std::make_pair(total_score, k);
    }
    
    // Find top K
    std::sort(scores.begin(), scores.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
    
    topk_ids.resize(TOP_K);
    for (int i = 0; i < TOP_K; i++) {
        topk_ids[i] = scores[i].second;
    }
}

int main(int argc, char* argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    
    int L = 4096;  // Number of key vectors
    if (argc > 1) {
        L = std::atoi(argv[1]);
    }
    
    std::cout << "Indexer Testbench" << std::endl;
    std::cout << "L (key vectors): " << L << std::endl;
    std::cout << "NUM_INDEX_HEAD: " << NUM_INDEX_HEAD << std::endl;
    std::cout << "HEAD_DIM: " << HEAD_DIM << std::endl;
    std::cout << "TOP_K: " << TOP_K << std::endl;
    
    // Initialize random number generator
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_int_distribution<int> dis_int(-128, 127);  // Range for ap_int<16>
    std::uniform_real_distribution<float> dis_float(0.0f, 2.0f);
    
    // Generate random query vectors (16 heads x 128 dimensions) as integers
    std::vector<std::vector<ap_int<16>>> query_vecs(NUM_INDEX_HEAD, std::vector<ap_int<16>>(HEAD_DIM));
    for (int h = 0; h < NUM_INDEX_HEAD; h++) {
        for (int d = 0; d < HEAD_DIM; d++) {
            query_vecs[h][d] = ap_int<16>(dis_int(gen));
        }
    }
    
    // Generate random key vectors (4096 x 128 dimensions) as integers
    std::vector<std::vector<ap_int<16>>> key_vecs(L, std::vector<ap_int<16>>(HEAD_DIM));
    for (int k = 0; k < L; k++) {
        for (int d = 0; d < HEAD_DIM; d++) {
            key_vecs[k][d] = ap_int<16>(dis_int(gen));
        }
    }
    
    // Generate random weights (16 weights for 16 heads)
    std::vector<float> weights(NUM_INDEX_HEAD);
    for (int h = 0; h < NUM_INDEX_HEAD; h++) {
        weights[h] = dis_float(gen);  // Use positive weights
    }
    
    // Prepare hardware inputs
    // qk_vec_mem layout: 
    // - First: Query vectors packed across NUM_INDEX_HEAD dimension
    //   For each of HEAD_DIM elements, pack all 16 heads together in one vec_t
    // - Then: Key vectors packed across L dimension
    //   For each of HEAD_DIM elements, pack 16 consecutive keys together in one vec_t
    const int query_lines = HEAD_DIM;  // HEAD_DIM lines, each containing 16 heads
    const int key_lines = L * HEAD_DIM / 16;  // L vectors * HEAD_DIM / 16 elements per vec_t
    const int total_qk_lines = query_lines + key_lines;
    
    // Allocate memory for 8 channels
    std::vector<std::vector<tapa::vec_t<ap_int<16>, 16>>> qk_vec_hw(8);
    for (int ch = 0; ch < 8; ch++) {
        qk_vec_hw[ch].resize(total_qk_lines / 8);
    }
    
    // Pack query vectors: transpose to [HEAD_DIM][NUM_INDEX_HEAD]
    for (int d = 0; d < HEAD_DIM; d++) {
        tapa::vec_t<ap_int<16>, 16> packed;
        for (int h = 0; h < NUM_INDEX_HEAD; h++) {
            packed[h] = query_vecs[h][d];
        }
        int ch = d % 8;
        int idx = d / 8;
        qk_vec_hw[ch][idx] = packed;
    }
    
    // Pack key vectors: for each dimension, pack 16 consecutive keys
    for (int k = 0; k < L; k += 16) {
        for (int d = 0; d < HEAD_DIM; d++) {
            tapa::vec_t<ap_int<16>, 16> packed;
            for (int i = 0; i < 16; i++) {
                packed[i] = key_vecs[k + i][d];
            }
            int global_line = query_lines + (k / 16) * HEAD_DIM + d;
            int ch = global_line % 8;
            int idx = global_line / 8;
            qk_vec_hw[ch][idx] = packed;
        }
    }
    
    // Pack weights into hardware format
    std::vector<tapa::vec_t<float, 16>> weight_hw(1);
    for (int i = 0; i < NUM_INDEX_HEAD; i++) {
        weight_hw[0][i] = weights[i];
    }
    
    // Allocate output memory
    std::vector<tapa::vec_t<int, 16>> topk_id_hw(TOP_K / 16);
    
    std::cout << "\nRunning hardware kernel..." << std::endl;
    
    // Invoke the kernel
    int64_t kernel_time_ns = 0;
    kernel_time_ns = tapa::invoke(indexer_top, FLAGS_bitstream,
                 L,
                 tapa::read_only_mmaps<tapa::vec_t<ap_int<16>, 16>, 8>(qk_vec_hw),
                 tapa::read_only_mmap<tapa::vec_t<float, 16>>(weight_hw),
                 tapa::write_only_mmap<tapa::vec_t<int, 16>>(topk_id_hw));
    
    std::cout << "Hardware kernel completed." << std::endl;
    std::clog << "kernel time: " << kernel_time_ns * 1e-9 << " s" << std::endl;

    
    // Extract hardware results
    std::vector<int> topk_ids_hw(TOP_K);
    for (int i = 0; i < TOP_K / 16; i++) {
        for (int j = 0; j < 16; j++) {
            topk_ids_hw[i * 16 + j] = topk_id_hw[i][j];
        }
    }
    
    // Compute software reference with fixed-point conversion for fair comparison
    std::cout << "\nRunning software reference..." << std::endl;
    
    std::vector<int> topk_ids_sw;
    indexer_top_ref(L, query_vecs, key_vecs, weights, topk_ids_sw);
    
    std::cout << "Software reference completed." << std::endl;
    
    // Compare results
    std::cout << "\n=== Results ===" << std::endl;
    std::cout << "Hardware Top-64 IDs:" << std::endl;
    for (int i = 0; i < std::min(64, TOP_K); i++) {
        std::cout << topk_ids_hw[i] << " ";
    }
    std::cout << "..." << std::endl;
    
    std::cout << "\nSoftware Top-64 IDs:" << std::endl;
    for (int i = 0; i < std::min(64, TOP_K); i++) {
        std::cout << topk_ids_sw[i] << " ";
    }
    std::cout << "..." << std::endl;
    
    // Check correctness
    // Since the top-K order might vary slightly due to ties, we check if the sets overlap significantly
    std::set<int> hw_set(topk_ids_hw.begin(), topk_ids_hw.end());
    std::set<int> sw_set(topk_ids_sw.begin(), topk_ids_sw.end());
    
    int matches = 0;
    for (int id : hw_set) {
        if (sw_set.count(id)) {
            matches++;
        }
    }
    
    float overlap_ratio = static_cast<float>(matches) / TOP_K;
    std::cout << "\nOverlap: " << matches << "/" << TOP_K << " (" << (overlap_ratio * 100) << "%)" << std::endl;
    
    // Check if exact match
    bool exact_match = true;
    for (int i = 0; i < TOP_K; i++) {
        if (topk_ids_hw[i] != topk_ids_sw[i]) {
            exact_match = false;
            break;
        }
    }
    
    if (exact_match) {
        std::cout << "\n✓ PASSED: Exact match between hardware and software!" << std::endl;
        return 0;
    } else if (overlap_ratio >= 0.95) {
        std::cout << "\n✓ PASSED: High overlap (>95%) between hardware and software!" << std::endl;
        std::cout << "  (Small differences may be due to floating point precision and tie-breaking)" << std::endl;
        return 0;
    } else {
        std::cout << "\n✗ FAILED: Significant mismatch between hardware and software!" << std::endl;
        return 1;
    }
}
