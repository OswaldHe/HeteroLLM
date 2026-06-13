#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <random>
#include <set>
#include <cstdint>

#include <gflags/gflags.h>
#include <tapa.h>

// Include the kernel header
#include "indexer_lserve_long.h"

DEFINE_string(bitstream, "", "path to bitstream file, run csim if empty");

constexpr int NUM_CHANNELS = 8;
constexpr int NUM_ACTIVE_CHANNELS = 5;

int round_up(int value, int alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

int l_logical_per_channel(int L_per_channel) {
    if (L_per_channel < MAX_LEN_PER_CNL_BRAM) {
        return L_per_channel;
    }
    if (L_per_channel < MAX_LEN_PER_CNL) {
        return MAX_LEN_PER_CNL_BRAM + (L_per_channel - MAX_LEN_PER_CNL_BRAM) * 4;
    }
    return MAX_LEN_PER_CNL_BRAM
         + (MAX_LEN_PER_CNL - MAX_LEN_PER_CNL_BRAM) * 4
         + (L_per_channel - MAX_LEN_PER_CNL) * 16;
}

int total_input_lines(int L_per_channel, int NUM_QUERY) {
    if (L_per_channel > MAX_LEN_PER_CNL) {
        return ((MAX_LEN_PER_CNL * HEAD_DIM) >> 5)
             + (((L_per_channel - MAX_LEN_PER_CNL + 2) * HEAD_DIM * NUM_QUERY) >> 5)
             + ((HEAD_DIM / 4) * NUM_QUERY);
    }
    return ((L_per_channel + NUM_QUERY * 2) * HEAD_DIM) >> 5;
}

int logical_pos_for_query(int L_per_channel, int L_logical_per_channel, int q_idx) {
    if (L_per_channel > MAX_LEN_PER_CNL) {
        return L_logical_per_channel + 16 * (q_idx / 8);
    }
    if (L_per_channel > MAX_LEN_PER_CNL_BRAM) {
        return L_logical_per_channel + 4 * (q_idx / 8);
    }
    return L_logical_per_channel;
}

int stream_id_for_pos(int channel, int local_pos) {
    int r_k = 0;
    int idx = 0;
    if (local_pos < MAX_LEN_PER_CNL_BRAM) {
        r_k = local_pos >> 7;
        idx = local_pos & 127;
    } else if (local_pos < MAX_LEN_PER_CNL) {
        const int rel = local_pos - MAX_LEN_PER_CNL_BRAM;
        r_k = (MAX_LEN_PER_CNL_BRAM >> 7) + (rel >> 5);
        idx = rel & 31;
    } else {
        const int rel = local_pos - MAX_LEN_PER_CNL;
        r_k = MAX_ONCHIP_LEN + (rel >> 3);
        idx = rel & 7;
    }

    const int pack_i = idx >> 4;
    const int half = (idx >> 3) & 1;
    return (r_k * 8 + pack_i) * 16 + channel * 2 + half;
}

int64_t dot_product(
    const std::vector<ap_int<16>>& lhs,
    const std::vector<ap_int<16>>& rhs
) {
    int64_t dot = 0;
    for (int d = 0; d < HEAD_DIM; d++) {
        dot += (int64_t)lhs[d].to_int() * (int64_t)rhs[d].to_int();
    }
    return dot;
}

// Software reference for the packed score-stream IDs emitted by topk_parallel_cmp.
void indexer_top_ref(
    const int L_per_channel,
    const int L_logical_per_channel,
    const int NUM_QUERY,
    const std::vector<std::vector<ap_int<16>>>& past_keys,  // [active L][HEAD_DIM]
    const std::vector<std::vector<ap_int<16>>>& queries,     // [NUM_QUERY][HEAD_DIM]
    const std::vector<std::vector<ap_int<16>>>& new_keys,    // [NUM_QUERY][HEAD_DIM]
    std::vector<std::vector<int>>& topk_ids)                 // [NUM_QUERY][TOP_K]
{
    topk_ids.clear();
    topk_ids.resize(NUM_QUERY);

    for (int q_idx = 0; q_idx < NUM_QUERY; q_idx++) {
        const int current_logical_pos =
            logical_pos_for_query(L_per_channel, L_logical_per_channel, q_idx);
        const int total_pages = (current_logical_pos + 128) >> 7;
        const int stream_len = total_pages * 8 * 16;
        std::vector<float> stream_scores(stream_len, -2e10f);

        for (int ch = 0; ch < NUM_ACTIVE_CHANNELS; ch++) {
            for (int r_k = 0; r_k < total_pages; r_k++) {
                int mask_pos = 128;
                if (r_k >= (MAX_LEN_PER_CNL_BRAM >> 7) && r_k < MAX_ONCHIP_LEN) {
                    mask_pos = 32;
                } else if (r_k >= MAX_ONCHIP_LEN) {
                    mask_pos = 8;
                }

                const int page_remaining = current_logical_pos + 128 - r_k * 128;
                for (int i = 0; i < 8; i++) {
                    for (int half = 0; half < 2; half++) {
                        const int idx = i * 16 + half * 8;
                        if (idx < page_remaining && idx < mask_pos) {
                            const int id = (r_k * 8 + i) * 16 + ch * 2 + half;
                            stream_scores[id] = -1e10f;
                        }
                    }
                }
            }

            const int global_base = ch * L_per_channel;
            for (int local_pos = 0; local_pos < L_per_channel; local_pos++) {
                const int id = stream_id_for_pos(ch, local_pos);
                if (id < stream_len) {
                    const float score = (float)dot_product(
                        queries[q_idx], past_keys[global_base + local_pos]);
                    stream_scores[id] = std::max(stream_scores[id], score);
                }
            }

            for (int group = 0; group <= q_idx / 8; group++) {
                const int source_query = (group == q_idx / 8) ? q_idx : group * 8 + 7;
                if (source_query >= NUM_QUERY) {
                    continue;
                }
                const int local_pos = L_per_channel + group;
                const int id = stream_id_for_pos(ch, local_pos);
                if (id < stream_len) {
                    const float score = (float)dot_product(queries[q_idx], new_keys[source_query]);
                    stream_scores[id] = std::max(stream_scores[id], score);
                }
            }
        }

        std::vector<std::pair<float, int>> score_ids;
        score_ids.reserve(stream_scores.size());
        for (int id = 0; id < (int)stream_scores.size(); id++) {
            score_ids.push_back({stream_scores[id], id});
        }
        std::sort(score_ids.begin(), score_ids.end(), [](const auto& a, const auto& b) {
            if (a.first != b.first) {
                return a.first > b.first;
            }
            return a.second < b.second;
        });

        for (int i = 0; i < TOP_K && i < (int)score_ids.size(); i++) {
            topk_ids[q_idx].push_back(score_ids[i].second);
        }
    }
}

int main(int argc, char* argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    
    int L = 1024;  // Number of active past key vectors.
    int NUM_QUERY = 8;  // Number of queries
    
    if (argc > 1) {
        L = std::atoi(argv[1]);
    }
    if (argc > 2) {
        NUM_QUERY = std::atoi(argv[2]);
    }
    
    L = round_up(L, NUM_ACTIVE_CHANNELS * 128);
    if (L <= TOP_K * 16) {
        L = round_up(TOP_K * 16 + 1, NUM_ACTIVE_CHANNELS * 128);
    }

    const int L_per_channel = L / NUM_ACTIVE_CHANNELS;
    const int L_logical_per_channel = l_logical_per_channel(L_per_channel);
    const int lines_per_channel = total_input_lines(L_per_channel, NUM_QUERY);

    std::cout << "Indexer LServe Long Testbench" << std::endl;
    std::cout << "Active past key vectors: " << L << std::endl;
    std::cout << "Active channels: " << NUM_ACTIVE_CHANNELS << " / " << NUM_CHANNELS << std::endl;
    std::cout << "L per active channel: " << L_per_channel << std::endl;
    std::cout << "L logical per channel: " << L_logical_per_channel << std::endl;
    std::cout << "Input lines per channel: " << lines_per_channel << std::endl;
    std::cout << "NUM_QUERY: " << NUM_QUERY << std::endl;
    std::cout << "HEAD_DIM: " << HEAD_DIM << std::endl;
    std::cout << "TOP_K (packed score IDs): " << TOP_K << std::endl;
    
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

    // Zero out keys beyond 1024 for padding
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
    
    // Make some score groups perfectly correlated with query 0 without filling
    // the whole top-k boundary with tied scores.
    for (int b = 0; b < std::min(TOP_K / 4, L / 16); b++) {
        int block_idx = b * 2;  // Spread them out
        if (block_idx * 16 < L) {
            for (int i = 0; i < 16; i++) {
                int key_idx = block_idx * 16 + i;
                if (key_idx < L) {
                    for (int d = 0; d < HEAD_DIM; d++) {
                        past_keys[key_idx][d] = queries[0][d];  // Perfect correlation
                    }
                }
            }
        }
    }

    // Prepare hardware inputs
    // Channels 0..4 carry real data. Channels 5..7 are zero-filled and consumed
    // by dummy_consumer in the kernel.

    std::vector<std::vector<tapa::vec_t<ap_uint<32>, 16>>> qk_vec_hw(NUM_CHANNELS);
    for (int ch = 0; ch < NUM_CHANNELS; ch++) {
        qk_vec_hw[ch].resize(lines_per_channel);
    }
    
    // Helper function to pack two 16-bit integers into one 32-bit word
    auto pack_two_int16 = [](ap_int<16> low, ap_int<16> high) -> ap_uint<32> {
        ap_uint<32> result;
        result(15, 0) = low.to_uint();
        result(31, 16) = high.to_uint();
        return result;
    };

    auto write_zero_pack = [&](int ch, int& write_idx) {
        tapa::vec_t<ap_uint<32>, 16> packed;
        for (int i = 0; i < 16; i++) {
            packed[i] = 0;
        }
        qk_vec_hw[ch][write_idx++] = packed;
    };

    auto write_standard_vector = [&](int ch, int& write_idx,
                                     const std::vector<ap_int<16>>& vec) {
        for (int word_group = 0; word_group < HEAD_DIM_DIV_2; word_group += 16) {
            tapa::vec_t<ap_uint<32>, 16> packed;
            for (int i = 0; i < 16; i++) {
                const int dim_idx = (word_group + i) * 2;
                packed[i] = pack_two_int16(vec[dim_idx], vec[dim_idx + 1]);
            }
            qk_vec_hw[ch][write_idx++] = packed;
        }
    };

    auto write_bram_vectors = [&](int ch, int& write_idx, int global_start, int count) {
        for (int group = 0; group < count; group += 16) {
            for (int word = 0; word < HEAD_DIM_DIV_2; word++) {
                tapa::vec_t<ap_uint<32>, 16> packed;
                for (int lane = 0; lane < 16; lane++) {
                    const auto& vec = past_keys[global_start + group + lane];
                    const int dim_idx = word * 2;
                    packed[lane] = pack_two_int16(vec[dim_idx], vec[dim_idx + 1]);
                }
                qk_vec_hw[ch][write_idx++] = packed;
            }
        }
    };

    auto write_group8_vectors = [&](int ch, int& write_idx, int global_start, int count) {
        for (int group = 0; group < count; group += 8) {
            for (int word_pair = 0; word_pair < HEAD_DIM_DIV_4; word_pair++) {
                tapa::vec_t<ap_uint<32>, 16> packed;
                for (int lane = 0; lane < 8; lane++) {
                    const auto& vec = past_keys[global_start + group + lane];
                    const int word0_dim = (word_pair * 2) * 2;
                    const int word1_dim = (word_pair * 2 + 1) * 2;
                    packed[lane * 2] = pack_two_int16(vec[word0_dim], vec[word0_dim + 1]);
                    packed[lane * 2 + 1] = pack_two_int16(vec[word1_dim], vec[word1_dim + 1]);
                }
                qk_vec_hw[ch][write_idx++] = packed;
            }
        }
    };

    auto write_group8_repeat_vector = [&](int ch, int& write_idx,
                                          const std::vector<ap_int<16>>& vec) {
        for (int word_pair = 0; word_pair < HEAD_DIM_DIV_4; word_pair++) {
            tapa::vec_t<ap_uint<32>, 16> packed;
            for (int lane = 0; lane < 8; lane++) {
                const int word0_dim = (word_pair * 2) * 2;
                const int word1_dim = (word_pair * 2 + 1) * 2;
                packed[lane * 2] = pack_two_int16(vec[word0_dim], vec[word0_dim + 1]);
                packed[lane * 2 + 1] = pack_two_int16(vec[word1_dim], vec[word1_dim + 1]);
            }
            qk_vec_hw[ch][write_idx++] = packed;
        }
    };

    for (int ch = 0; ch < NUM_CHANNELS; ch++) {
        for (int line = 0; line < lines_per_channel; line++) {
            int write_idx = line;
            write_zero_pack(ch, write_idx);
        }
    }

    for (int ch = 0; ch < NUM_ACTIVE_CHANNELS; ch++) {
        int write_idx = 0;
        const int global_base = ch * L_per_channel;
        const int bram_len = std::min(L_per_channel, MAX_LEN_PER_CNL_BRAM);
        write_bram_vectors(ch, write_idx, global_base, bram_len);

        const int uram_len = std::min(
            std::max(L_per_channel - MAX_LEN_PER_CNL_BRAM, 0),
            MAX_LEN_PER_CNL_URAM);
        if (uram_len > 0) {
            write_group8_vectors(ch, write_idx, global_base + MAX_LEN_PER_CNL_BRAM, uram_len);
        }
        
        for (int q = 0; q < NUM_QUERY; q++) {
            write_standard_vector(ch, write_idx, queries[q]);
            write_standard_vector(ch, write_idx, new_keys[q]);

            if (L_per_channel > MAX_LEN_PER_CNL) {
                const int tail_len = L_per_channel - MAX_LEN_PER_CNL;
                write_group8_vectors(ch, write_idx, global_base + MAX_LEN_PER_CNL, tail_len);
                write_group8_repeat_vector(ch, write_idx, new_keys[q]);
            }
        }

        if (write_idx != lines_per_channel) {
            std::cerr << "Channel " << ch << " wrote " << write_idx
                      << " lines, expected " << lines_per_channel << std::endl;
            return 1;
        }
    }
    
    const int output_size = NUM_QUERY * ((TOP_K + 15) / 16);
    std::vector<tapa::vec_t<int, 16>> topk_ids_hw(output_size);
    
    std::cout << "\nRunning hardware kernel..." << std::endl;
    std::cout << "Total input lines per channel: " << lines_per_channel << std::endl;
    std::cout << "Output size (vec_t<int,16>): " << output_size << std::endl;
    std::cout << "Total output IDs: " << NUM_QUERY * TOP_K << std::endl;
    
    // Invoke the kernel
    int64_t kernel_time_ns = 0;
    kernel_time_ns = tapa::invoke(indexer_top, FLAGS_bitstream,
                 L_per_channel,  // L is per channel
                 L_logical_per_channel,
                 NUM_QUERY,
                 tapa::read_only_mmaps<tapa::vec_t<ap_uint<32>, 16>, 8>(qk_vec_hw),
                 tapa::write_only_mmap<tapa::vec_t<int, 16>>(topk_ids_hw));
    
    std::cout << "Hardware kernel completed." << std::endl;
    std::clog << "kernel time: " << kernel_time_ns * 1e-9 << " s" << std::endl;
    
    std::vector<std::vector<int>> hw_topk_ids(NUM_QUERY);
    int hw_idx = 0;
    for (int q = 0; q < NUM_QUERY; q++) {
        for (int i = 0; i < (TOP_K / 16); i++) {
            for (int j = 0; j < 16; j++) {
                hw_topk_ids[q].push_back(topk_ids_hw[hw_idx][j]);
            }
            hw_idx++;
        }
    }
    
    // Compute software reference
    std::cout << "\nRunning software reference..." << std::endl;
    
    std::vector<std::vector<int>> sw_topk_ids;
    indexer_top_ref(
        L_per_channel, L_logical_per_channel, NUM_QUERY,
        past_keys, queries, new_keys, sw_topk_ids);
    
    std::cout << "Software reference completed." << std::endl;
    
    // Compare results for each query
    std::cout << "\n=== Results ===" << std::endl;
    
    int total_overlap = 0;
    bool all_exact = true;
    bool all_indices_valid = true;
    bool all_no_duplicates = true;
    
    for (int q = 0; q < NUM_QUERY; q++) {
        std::set<int> hw_set(hw_topk_ids[q].begin(), hw_topk_ids[q].end());
        std::set<int> sw_set(sw_topk_ids[q].begin(), sw_topk_ids[q].end());
        all_exact &= (hw_set == sw_set);
        
        int overlap_count = 0;
        for (int idx : hw_set) {
            if (sw_set.find(idx) != sw_set.end()) {
                overlap_count++;
            }
        }
        total_overlap += overlap_count;

        const int current_logical_pos =
            logical_pos_for_query(L_per_channel, L_logical_per_channel, q);
        const int valid_stream_len = ((current_logical_pos + 128) >> 7) * 8 * 16;
        for (int id : hw_topk_ids[q]) {
            if (id < 0 || id >= valid_stream_len) {
                all_indices_valid = false;
            }
        }
        
        if (hw_set.size() != hw_topk_ids[q].size()) {
            all_no_duplicates = false;
        }
        
        if (q < 3) {  // Print details for first 3 queries
            std::cout << "\nQuery " << q << ":" << std::endl;
            std::cout << "  HW returned " << hw_topk_ids[q].size() << " IDs" << std::endl;
            std::cout << "  SW returned " << sw_topk_ids[q].size() << " IDs" << std::endl;
            std::cout << "  Overlap: " << overlap_count << " / " << TOP_K << std::endl;
            
            std::cout << "  First 10 HW IDs: ";
            for (int i = 0; i < std::min(10, (int)hw_topk_ids[q].size()); i++) {
                std::cout << hw_topk_ids[q][i] << " ";
            }
            std::cout << std::endl;
            
            std::cout << "  First 10 SW IDs: ";
            for (int i = 0; i < std::min(10, (int)sw_topk_ids[q].size()); i++) {
                std::cout << sw_topk_ids[q][i] << " ";
            }
            std::cout << std::endl;
        }

        if (overlap_count != TOP_K) {
            std::cout << "\nMismatch in query " << q << ":" << std::endl;
            std::cout << "  Missing from HW: ";
            for (int id : sw_set) {
                if (hw_set.find(id) == hw_set.end()) {
                    std::cout << id << " ";
                }
            }
            std::cout << std::endl;
            std::cout << "  Extra in HW: ";
            for (int id : hw_set) {
                if (sw_set.find(id) == sw_set.end()) {
                    std::cout << id << " ";
                }
            }
            std::cout << std::endl;
        }
    }
    
    std::cout << "\n=== Statistics ===" << std::endl;
    std::cout << "Total overlap across all queries: " << total_overlap << " / " << (NUM_QUERY * TOP_K) << std::endl;
    float overlap_ratio = (float)total_overlap / (NUM_QUERY * TOP_K);
    std::cout << "Average overlap ratio: " << (overlap_ratio * 100) << "%" << std::endl;
    if (!all_indices_valid) {
        std::cout << "Warning: Some hardware output contains invalid IDs" << std::endl;
    }
    
    if (!all_no_duplicates) {
        std::cout << "Warning: Some hardware output contains duplicate IDs" << std::endl;
    }
    
    if (all_exact && all_indices_valid && all_no_duplicates) {
        std::cout << "\nPASSED: Packed top-k IDs match the software reference." << std::endl;
        return 0;
    }

    std::cout << "\nFAILED: Hardware packed top-k IDs do not match the software reference." << std::endl;
    return 1;
}
