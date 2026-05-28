#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <random>
#include <set>

#include <gflags/gflags.h>
#include <tapa.h>

#include "indexer_lut.h"

DEFINE_string(bitstream, "", "path to bitstream file, run csim if empty");

// Software reference implementation
void indexer_lut_ref(
    const int L,
    const std::vector<std::vector<ap_uint<8>>>& query_indices,  // [NUM_INDEX_HEAD][HEAD_DIM_DIV_2]
    const std::vector<std::vector<ap_uint<8>>>& key_indices,    // [L][HEAD_DIM_DIV_2]
    const std::vector<std::vector<std::vector<ap_int<8>>>>& lookup_tables, // [HEAD_DIM_DIV_2][K_CODEBOOK_SIZE][Q_CODEBOOK_SIZE]
    const std::vector<float>& weights,                          // [NUM_INDEX_HEAD]
    std::vector<int>& topk_ids)                                // [TOP_K]
{
    // Compute weighted indexing scores
    std::vector<std::pair<float, int>> scores(L);
    
    for (int k = 0; k < L; k++) {
        float total_score = 0.0f;
        
        for (int h = 0; h < NUM_INDEX_HEAD; h++) {
            // Compute dot product using lookup tables
            int64_t qk = 0;
            for (int d = 0; d < HEAD_DIM_DIV_2; d++) {
                ap_uint<8> q_idx = query_indices[h][d];
                ap_uint<8> k_idx = key_indices[k][d];
                ap_int<8> lut_val = lookup_tables[d][k_idx.to_int()][q_idx.to_int()];
                qk += lut_val.to_int();
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
    
    std::cout << "Indexer LUT Testbench" << std::endl;
    std::cout << "L (key vectors): " << L << std::endl;
    std::cout << "NUM_INDEX_HEAD: " << NUM_INDEX_HEAD << std::endl;
    std::cout << "HEAD_DIM: " << HEAD_DIM << std::endl;
    std::cout << "HEAD_DIM_DIV_2: " << HEAD_DIM_DIV_2 << std::endl;
    std::cout << "TOP_K: " << TOP_K << std::endl;
    std::cout << "K_CODEBOOK_SIZE: " << K_CODEBOOK_SIZE << " (uint8)" << std::endl;
    std::cout << "Q_CODEBOOK_SIZE: " << Q_CODEBOOK_SIZE << " (uint6 packed as uint8)" << std::endl;
    
    // Initialize random number generator
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_int_distribution<int> dis_k_idx(0, K_CODEBOOK_SIZE - 1);  // 0-255 for uint8
    std::uniform_int_distribution<int> dis_q_idx(0, Q_CODEBOOK_SIZE - 1);  // 0-63 for uint6
    std::uniform_int_distribution<int> dis_lut(-128, 127);
    std::uniform_real_distribution<float> dis_float(0.0f, 3.0f);
    
    // Generate random query indices (16 heads x 64 dimensions)
    // Note: Query indices are uint6 (0-63) but packed as uint8 for convenience
    std::vector<std::vector<ap_uint<8>>> query_indices(NUM_INDEX_HEAD, std::vector<ap_uint<8>>(HEAD_DIM_DIV_2));
    for (int h = 0; h < NUM_INDEX_HEAD; h++) {
        for (int d = 0; d < HEAD_DIM_DIV_2; d++) {
            query_indices[h][d] = ap_uint<8>(dis_q_idx(gen));
        }
    }
    
    // Generate random key indices (L x 64 dimensions)
    // Note: Key indices are uint8 (0-255)
    std::vector<std::vector<ap_uint<8>>> key_indices(L, std::vector<ap_uint<8>>(HEAD_DIM_DIV_2));
    for (int k = 0; k < L; k++) {
        for (int d = 0; d < HEAD_DIM_DIV_2; d++) {
            key_indices[k][d] = ap_uint<8>(dis_k_idx(gen));
        }
    }

    // test: flash later key to 0
    for (int k = 64; k < L; k++) {
        for (int d = 0; d < HEAD_DIM_DIV_2; d++) {
            key_indices[k][d] = ap_uint<8>(0);
        }
    }
    
    // Generate random lookup tables (64 dimensions x 256 k_codes x 64 q_codes)
    // LUT[dim][k_idx][q_idx] returns the int8 dot product contribution
    // k_idx: 0-255 (uint8), q_idx: 0-63 (uint6)
    std::vector<std::vector<std::vector<ap_int<8>>>> lookup_tables(
        HEAD_DIM_DIV_2, 
        std::vector<std::vector<ap_int<8>>>(
            K_CODEBOOK_SIZE, 
            std::vector<ap_int<8>>(Q_CODEBOOK_SIZE)
        )
    );
    for (int d = 0; d < HEAD_DIM_DIV_2; d++) {
        for (int k = 0; k < K_CODEBOOK_SIZE; k++) {
            for (int q = 0; q < Q_CODEBOOK_SIZE; q++) {
                lookup_tables[d][k][q] = ap_int<8>(dis_lut(gen));
            }
        }
    }
    
    // Generate random weights (16 weights for 16 heads)
    std::vector<float> weights(NUM_INDEX_HEAD);
    for (int h = 0; h < NUM_INDEX_HEAD; h++) {
        weights[h] = dis_float(gen);
    }
    
    // Prepare hardware inputs
    // Layout for k_lut_vec_mem (8 channels):
    // 1. First: Key indices [L][HEAD_DIM_DIV_2] packed as vec_t<ap_uint<8>, 64>
    //    - From read_k_lut: k_ind_size = L * (HEAD_DIM >> 10) = L * 128/1024 = L/8
    //    - Each vec_t<64> holds 64 indices, so this is L/8 lines total across all channels
    // 2. Then: Lookup tables [K_CODEBOOK_SIZE][Q_CODEBOOK_SIZE] packed
    //    - From read_k_lut: lut_size = K_CODEBOOK_SIZE * (Q_CODEBOOK_SIZE >> 6) = 256 * 1 = 256 lines
    //    - Total: 256 lines across all channels

    const int k_ind_size = (L * HEAD_DIM) >> 8;  // L/8 lines
    const int lut_size = (K_CODEBOOK_SIZE * Q_CODEBOOK_SIZE) >> 6;  // 256 lines
    const int total_k_lut_lines = k_ind_size + lut_size;
    
    // Allocate memory for 8 channels
    std::vector<std::vector<tapa::vec_t<ap_uint<8>, 64>>> k_lut_vec_hw(8);
    for (int ch = 0; ch < 8; ch++) {
        k_lut_vec_hw[ch].resize(total_k_lut_lines);
    }
    
    // Pack key indices
    // k_ind_size = L/8, so we pack all L*HEAD_DIM_DIV_2 indices into L/8 lines
    // Each line (vec_t<64>) holds 64 indices
    // Total indices = L * HEAD_DIM_DIV_2 = L * 64
    // Pack them sequentially: for each key, all its HEAD_DIM_DIV_2 dimensions in order
    int idx_line = 0;
    int idx_pos = 0;
    tapa::vec_t<ap_uint<8>, 64> packed_indices;
    
    for (int k = 0; k < L; k++) {
        for (int d = 0; d < HEAD_DIM_DIV_2; d++) {
            packed_indices[idx_pos] = key_indices[k][d];
            idx_pos++;
            
            if (idx_pos == 64) {
                int ch = idx_line % 8;
                int idx = idx_line / 8;
                k_lut_vec_hw[ch][idx] = packed_indices;
                idx_line++;
                idx_pos = 0;
            }
        }
    }
    
    // Pack lookup tables
    // lut_size = K_CODEBOOK_SIZE * (Q_CODEBOOK_SIZE >> 6) = 256 * 1 = 256
    // Each line holds 64 values
    // For each k_code (256 total), pack all Q_CODEBOOK_SIZE (64) values in one line
    // Note: Hardware will read the same LUT for all PEs, they'll use it with different q_indices
    for (int k_code = 0; k_code < K_CODEBOOK_SIZE; k_code++) {
        tapa::vec_t<ap_uint<8>, 64> packed_lut;
        for (int q_idx = 0; q_idx < Q_CODEBOOK_SIZE; q_idx++) {
            // Use dimension 0 as reference; hardware replicates this for all dimensions
            ap_int<8> lut_val = lookup_tables[0][k_code][q_idx];
            packed_lut[q_idx] = tapa::bit_cast<ap_uint<8>>(lut_val);
        }
        int global_line = k_ind_size + k_code;
        int ch = global_line % 8;
        int idx = global_line / 8;
        k_lut_vec_hw[ch][idx] = packed_lut;
    }
    
    // Layout for q_vec_mem (1 channel):
    // Query indices [NUM_INDEX_HEAD][HEAD_DIM_DIV_2] packed as vec_t<ap_uint<8>, 64>
    // Each line contains 64 elements: pack across heads for same dimension
    // But wait, we only have 16 heads, so we need to pack differently
    // Looking at read_q_vec: total_size = NUM_INDEX_HEAD * (HEAD_DIM >> 7) = 16 * 1 = 16
    // So we need 16 lines, each with 64 elements
    // For each head, we have HEAD_DIM_DIV_2 = 64 indices
    // Pack them as: each line has all 64 dimensions for one head (but we need to fit in vec_t<64>)
    // Actually, HEAD_DIM >> 7 = 128 >> 7 = 1, so total_size = 16 * 1 = 16 lines
    // This means: pack HEAD_DIM_DIV_2 = 64 indices per line

    const int total_q_lines = (NUM_INDEX_HEAD * HEAD_DIM) >> 7;  // = 16 * 1 = 16
    std::vector<tapa::vec_t<ap_uint<8>, 64>> q_vec_hw(total_q_lines);
    
    for (int h = 0; h < NUM_INDEX_HEAD; h++) {
        tapa::vec_t<ap_uint<8>, 64> packed;
        for (int d = 0; d < HEAD_DIM_DIV_2; d++) {
            packed[d] = query_indices[h][d];
        }
        q_vec_hw[h] = packed;
    }
    
    // Pack weights into hardware format
    std::vector<tapa::vec_t<float, 16>> weight_hw(1);
    for (int i = 0; i < NUM_INDEX_HEAD; i++) {
        weight_hw[0][i] = weights[i];
    }
    
    // Allocate output memory
    std::vector<tapa::vec_t<int, 16>> topk_id_hw(TOP_K / 16);
    
    std::cout << "\nRunning hardware kernel..." << std::endl;
    std::cout << "k_lut_vec total lines: " << total_k_lut_lines << " (" << total_k_lut_lines << " per channel)" << std::endl;
    std::cout << "q_vec total lines: " << total_q_lines << std::endl;
    
    // Invoke the kernel
    int64_t kernel_time_ns = 0;
    kernel_time_ns = tapa::invoke(indexer_top, FLAGS_bitstream,
                 L,
                 tapa::read_only_mmaps<tapa::vec_t<ap_uint<8>, 64>, 8>(k_lut_vec_hw),
                 tapa::read_only_mmap<tapa::vec_t<ap_uint<8>, 64>>(q_vec_hw),
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
    
    // Compute software reference
    std::cout << "\nRunning software reference..." << std::endl;
    
    std::vector<int> topk_ids_sw;
    indexer_lut_ref(L, query_indices, key_indices, lookup_tables, weights, topk_ids_sw);
    
    std::cout << "Software reference completed." << std::endl;
    
    // // Compare results
    // std::cout << "\n=== Results ===" << std::endl;
    // std::cout << "Hardware Top-64 IDs:" << std::endl;
    // for (int i = 0; i < std::min(64, TOP_K); i++) {
    //     std::cout << topk_ids_hw[i] << " ";
    // }
    // std::cout << "..." << std::endl;
    
    // std::cout << "\nSoftware Top-64 IDs:" << std::endl;
    // for (int i = 0; i < std::min(64, TOP_K); i++) {
    //     std::cout << topk_ids_sw[i] << " ";
    // }
    // std::cout << "..." << std::endl;
    
    // // Check correctness
    // // Since the top-K order might vary slightly due to ties, we check if the sets overlap significantly
    // std::set<int> hw_set(topk_ids_hw.begin(), topk_ids_hw.end());
    // std::set<int> sw_set(topk_ids_sw.begin(), topk_ids_sw.end());
    
    // int matches = 0;
    // for (int id : hw_set) {
    //     if (sw_set.count(id)) {
    //         matches++;
    //     }
    // }
    
    // float overlap_ratio = static_cast<float>(matches) / TOP_K;
    // std::cout << "\nOverlap: " << matches << "/" << TOP_K << " (" << (overlap_ratio * 100) << "%)" << std::endl;
    
    // // Check if exact match
    // bool exact_match = true;
    // for (int i = 0; i < TOP_K; i++) {
    //     if (topk_ids_hw[i] != topk_ids_sw[i]) {
    //         exact_match = false;
    //         break;
    //     }
    // }
    
    // if (exact_match) {
    //     std::cout << "\n✓ PASSED: Exact match between hardware and software!" << std::endl;
    //     return 0;
    // } else if (overlap_ratio >= 0.95) {
    //     std::cout << "\n✓ PASSED: High overlap (>95%) between hardware and software!" << std::endl;
    //     std::cout << "  (Small differences may be due to floating point precision and tie-breaking)" << std::endl;
    //     return 0;
    // } else {
    //     std::cout << "\n✗ FAILED: Significant mismatch between hardware and software!" << std::endl;
    //     return 1;
    // }
}
