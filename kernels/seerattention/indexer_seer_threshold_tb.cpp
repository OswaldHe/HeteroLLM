#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <random>
#include <bitset>

#include <gflags/gflags.h>
#include <tapa.h>

// Include the kernel header
#include "indexer_seer_threshold.h"

DEFINE_string(bitstream, "", "path to bitstream file, run csim if empty");

// Software reference implementation
void indexer_top_ref(
    const int L,
    const int NUM_QUERY,
    const int THRESHOLD,
    const std::vector<std::vector<ap_int<16>>>& past_keys,  // [L][HEAD_DIM]
    const std::vector<std::vector<ap_int<16>>>& queries,     // [NUM_QUERY][HEAD_DIM]
    const std::vector<std::vector<ap_int<16>>>& new_keys,    // [NUM_QUERY][HEAD_DIM]
    std::vector<std::vector<bool>>& sparse_bitmap)           // [NUM_QUERY][(L+NUM_QUERY+127)/128][128]
{
    sparse_bitmap.clear();
    
    // For each query
    for (int q_idx = 0; q_idx < NUM_QUERY; q_idx++) {
        // Current position includes past keys + queries processed so far
        int current_pos = L + (q_idx / 8);
        
        // Create a temporary key buffer that includes past keys and new keys
        std::vector<std::vector<ap_int<16>>> all_keys = past_keys;
        
        // Add new keys up to current position
        for (int i = 0; i <= q_idx && i < NUM_QUERY; i++) {
            int pos = L + (i / 8);
            if (pos < all_keys.size()) {
                all_keys[pos] = new_keys[i];
            } else {
                all_keys.push_back(new_keys[i]);
            }
        }
        
        std::vector<bool> query_bitmap;
        
        // Compute dot products for all keys up to current_pos
        int num_blocks = (current_pos + 128) / 128;
        
        for (int block = 0; block < num_blocks; block++) {
            for (int k_idx = 0; k_idx < 128; k_idx++) {
                int abs_k_idx = block * 128 + k_idx;
                
                if (abs_k_idx > current_pos) {
                    query_bitmap.push_back(false);
                    continue;
                }
                
                // Compute dot product between query and key
                int64_t dot_product = 0;
                for (int d = 0; d < HEAD_DIM; d++) {
                    int16_t q_val = queries[q_idx][d].to_int();
                    int16_t k_val = (abs_k_idx < all_keys.size()) ? 
                                    all_keys[abs_k_idx][d].to_int() : 0;
                    dot_product += (int64_t)q_val * (int64_t)k_val;
                }
                
                // Check threshold
                query_bitmap.push_back(dot_product > THRESHOLD);
            }
        }
        
        sparse_bitmap.push_back(query_bitmap);
    }
}

int main(int argc, char* argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    
    int L = 256;  // Number of past key vectors (must be multiple of 256 for the kernel)
    int NUM_QUERY = 8;  // Number of queries
    int THRESHOLD = 100000;  // Threshold for dot product
    
    if (argc > 1) {
        L = std::atoi(argv[1]);
    }
    if (argc > 2) {
        NUM_QUERY = std::atoi(argv[2]);
    }
    if (argc > 3) {
        THRESHOLD = std::atoi(argv[3]);
    }
    
    // Ensure L is a multiple of 256 (required by kernel: L >> 8)
    L = (L + 255) & ~255;
    
    std::cout << "Indexer SEER Threshold Testbench" << std::endl;
    std::cout << "L (past key vectors): " << L << std::endl;
    std::cout << "NUM_QUERY: " << NUM_QUERY << std::endl;
    std::cout << "HEAD_DIM: " << HEAD_DIM << std::endl;
    std::cout << "THRESHOLD: " << THRESHOLD << std::endl;
    
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
    
    // Create some vectors with high dot products to ensure some bits are set
    // Make first query highly correlated with some past keys
    for (int k = 0; k < std::min(5, L); k++) {
        for (int d = 0; d < HEAD_DIM; d++) {
            past_keys[k][d] = queries[0][d];  // Perfect correlation
        }
    }
    
    // Prepare hardware inputs
    // qk_vec_mem layout (across 8 channels):
    // - First: All past keys (L vectors)
    //   Each vector is HEAD_DIM elements, but each 32-bit word contains two 16-bit values
    //   So each vector needs HEAD_DIM_DIV_2 words, packed in groups of 16 across channels
    // - Then: For each query (q0, k0, q1, k1, ...)
    //   Each q and k also need HEAD_DIM_DIV_2 words
    
    const int words_per_vec = HEAD_DIM_DIV_2;  // 64 words (32-bit) per vector
    const int past_lines = L * words_per_vec / 16;  // L vectors * 64 words / 16 words per line
    const int query_pair_lines = NUM_QUERY * 2 * words_per_vec / 16;  // NUM_QUERY pairs * 2 * 64 / 16
    const int total_lines = past_lines + query_pair_lines;
    const int lines_per_channel = ((L/8 + NUM_QUERY*2) * HEAD_DIM) >> 5;
    
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
    
    // Pack past keys
    int line_idx = 0;
    for (int k = 0; k < L; k++) {
        // Each key vector: pack adjacent pairs of dimensions into 32-bit words
        for (int d = 0; d < HEAD_DIM_DIV_2; d += 16) {
            tapa::vec_t<ap_uint<32>, 16> packed;
            for (int i = 0; i < 16; i++) {
                packed[i] = pack_two_int16(past_keys[k][d*2 + i], past_keys[k][d*2 + i + 1]);
            }
            int ch = line_idx % 8;
            int idx = line_idx / 8;
            qk_vec_hw[ch][idx] = packed;
            line_idx++;
        }
    }
    
    // Pack query and new key pairs
    for (int q = 0; q < NUM_QUERY; q++) {
        // Pack query
        for (int d = 0; d < HEAD_DIM_DIV_2; d += 16) {
            tapa::vec_t<ap_uint<32>, 16> packed;
            for (int i = 0; i < 16; i++) {
                packed[i] = pack_two_int16(queries[q][d*2 + i], queries[q][d*2 + i + 1]);
            }
            int ch = line_idx % 8;
            int idx = line_idx / 8;
            qk_vec_hw[ch][idx] = packed;
            line_idx++;
        }
        
        // Pack new key
        for (int d = 0; d < HEAD_DIM_DIV_2; d += 16) {
            tapa::vec_t<ap_uint<32>, 16> packed;
            for (int i = 0; i < 16; i++) {
                packed[i] = pack_two_int16(new_keys[q][d*2 + i], new_keys[q][d*2 + i + 1]);
            }
            int ch = line_idx % 8;
            int idx = line_idx / 8;
            qk_vec_hw[ch][idx] = packed;
            line_idx++;
        }
    }
    
    // Calculate output size
    // New formula from write_id: total_size = L * NUM_QUERY + (NUM_QUERY-1)*NUM_QUERY/16 + NUM_QUERY*128
    int lines_per_output_channel = (L/8) * NUM_QUERY + NUM_QUERY/1024 + NUM_QUERY*128;
    lines_per_output_channel = (lines_per_output_channel >> 7);
    int total_bitmap_lines = lines_per_output_channel;
    // Allocate output memory for 8 channels
    std::vector<std::vector<ap_uint<128>>> sparse_bitmap_hw(8);
    for (int ch = 0; ch < 8; ch++) {
        sparse_bitmap_hw[ch].resize(lines_per_output_channel);
    }
    
    std::cout << "\nRunning hardware kernel..." << std::endl;
    std::cout << "Total input lines per channel: " << lines_per_channel << std::endl;
    std::cout << "Total output bitmap lines: " << total_bitmap_lines << std::endl;
    std::cout << "Output lines per channel: " << lines_per_output_channel << std::endl;
    
    // Invoke the kernel
    int64_t kernel_time_ns = 0;
    kernel_time_ns = tapa::invoke(indexer_top, FLAGS_bitstream,
                 L/8,
                 NUM_QUERY,
                 THRESHOLD,
                 tapa::read_only_mmaps<tapa::vec_t<ap_uint<32>, 16>, 8>(qk_vec_hw),
                 tapa::write_only_mmaps<ap_uint<128>, 8>(sparse_bitmap_hw));
    
    std::cout << "Hardware kernel completed." << std::endl;
    std::clog << "kernel time: " << kernel_time_ns * 1e-9 << " s" << std::endl;
    
    // Compute software reference
    std::cout << "\nRunning software reference..." << std::endl;
    
    std::vector<std::vector<bool>> sparse_bitmap_sw;
    indexer_top_ref(L, NUM_QUERY, THRESHOLD, past_keys, queries, new_keys, sparse_bitmap_sw);
    
    std::cout << "Software reference completed." << std::endl;
    
    // Compare results
    std::cout << "\n=== Results ===" << std::endl;
    
    int total_bits_checked = 0;
    int total_bits_matched = 0;
    int total_ones_hw = 0;
    int total_ones_sw = 0;
    
    // Extract and compare bitmaps for all queries
    // The hardware writes bitmaps across 8 channels
    int hw_bitmap_idx = 0;
    
    for (int q_idx = 0; q_idx < NUM_QUERY; q_idx++) {
        int current_pos = L + (q_idx / 8);
        int num_blocks = (current_pos + 128) / 128;
        
        if (q_idx < 3) {  // Print details for first 3 queries
            std::cout << "\nQuery " << q_idx << " comparison (current_pos=" << current_pos << ", blocks=" << num_blocks << "):" << std::endl;
        }
        
        for (int block = 0; block < num_blocks; block++) {
            // Determine which channel and index
            int ch = hw_bitmap_idx % 8;
            int idx = hw_bitmap_idx / 8;
            
            if (idx >= lines_per_output_channel) {
                std::cout << "Warning: idx (" << idx << ") exceeds lines_per_output_channel (" << lines_per_output_channel << ")" << std::endl;
                break;
            }
            
            ap_uint<128> hw_bitmap = sparse_bitmap_hw[ch][idx];
            hw_bitmap_idx++;
            
            if (q_idx < 3 && block < 2) {  // Print first 2 blocks for first 3 queries
                std::cout << "  Block " << block << " HW (ch=" << ch << ", idx=" << idx << "): ";
                for (int i = 0; i < 128; i++) {
                    std::cout << (hw_bitmap[i] ? "1" : "0");
                    if ((i + 1) % 32 == 0) std::cout << " ";
                }
                std::cout << std::endl;
            }
            
            if (block < sparse_bitmap_sw[q_idx].size() / 128) {
                if (q_idx < 3 && block < 2) {
                    std::cout << "  Block " << block << " SW: ";
                }
                
                for (int i = 0; i < 128; i++) {
                    int sw_idx = block * 128 + i;
                    bool sw_bit = (sw_idx < sparse_bitmap_sw[q_idx].size()) ? 
                                  sparse_bitmap_sw[q_idx][sw_idx] : false;
                    
                    if (q_idx < 3 && block < 2) {
                        std::cout << (sw_bit ? "1" : "0");
                        if ((i + 1) % 32 == 0) std::cout << " ";
                    }
                    
                    if (hw_bitmap[i] == sw_bit) {
                        total_bits_matched++;
                    }
                    if (hw_bitmap[i]) total_ones_hw++;
                    if (sw_bit) total_ones_sw++;
                    total_bits_checked++;
                }
                
                if (q_idx < 3 && block < 2) {
                    std::cout << std::endl;
                }
            }
        }
    }
    
    std::cout << "\n=== Statistics ===" << std::endl;
    std::cout << "Total bits checked: " << total_bits_checked << std::endl;
    std::cout << "Matching bits: " << total_bits_matched << std::endl;
    std::cout << "Accuracy: " << (total_bits_checked > 0 ? 
                (100.0 * total_bits_matched / total_bits_checked) : 0.0) << "%" << std::endl;
    std::cout << "HW ones count: " << total_ones_hw << std::endl;
    std::cout << "SW ones count: " << total_ones_sw << std::endl;
    
    // Check correctness
    float accuracy = (total_bits_checked > 0) ? 
                     (float)total_bits_matched / total_bits_checked : 0.0f;
    
    if (accuracy >= 0.99) {
        std::cout << "\n✓ PASSED: High accuracy (>99%) between hardware and software!" << std::endl;
        return 0;
    } else {
        std::cout << "\n✗ FAILED: Significant mismatch between hardware and software!" << std::endl;
        std::cout << "  Accuracy: " << (accuracy * 100) << "%" << std::endl;
        return 1;
    }
}
