/**
 * @file paged_kv_index_builder.cpp
 * @brief Implementation of PagedKVIndexBuilder kernels
 * 
 * This kernel takes KVCacheData, extracts the K cache (LxD matrix), and for every
 * page_size_ tokens, performs average pooling to merge them into one token, then
 * applies linear projection using the weight matrix. The generated tokens are
 * concatenated to create a FlatIndexMemory.
 */

#include "../build_memory.h"
#include <iostream>
#include <cmath>

namespace heteromm {
namespace step {

void PagedKVIndexBuilder::run_test_kernel(
    const data_type::KVCacheData<float>& raw_data,
    data_type::FlatIndexMemory<float>& memory
) {
    // Extract K cache from KVCacheData (LxD matrix where L is context length, D is head dim)
    const auto& kv_cache = raw_data.export_data();
    const auto& keys = kv_cache.keys;  // L x D
    
    size_t context_length = keys.size();

    size_t head_dim = keys.empty() ? 0 : keys[0].size();
    
    if (context_length == 0 || head_dim == 0) {
        std::clog << "[PagedKVIndexBuilder] Warning: Empty KV cache data." << std::endl;
        memory.set_data({});
        return;
    }
    
    // Calculate output dimension from weight matrix
    size_t output_dim = weight_.size();  // weight_ is output_dim x head_dim
    if (output_dim == 0 || weight_[0].size() != head_dim) {
        std::clog << "[PagedKVIndexBuilder] Warning: Weight matrix dimension mismatch. "
                  << "Expected weight[?][" << head_dim << "], got weight[" 
                  << output_dim << "][" << (weight_.empty() ? 0 : weight_[0].size()) << "]" << std::endl;
        memory.set_data({});
        return;
    }
    
    // Calculate number of pages (each page contains page_size_ tokens)
    size_t num_pages = (context_length + page_size_ - 1) / page_size_;
    
    std::vector<std::vector<float>> result;
    result.reserve(num_pages);
    
    for (size_t page_idx = 0; page_idx < num_pages; ++page_idx) {
        size_t start_token = page_idx * page_size_;
        size_t end_token = std::min(start_token + page_size_, context_length);
        size_t tokens_in_page = end_token - start_token;
        
        // Step 1: Average pooling - merge tokens in this page into one token
        std::vector<float> pooled_token(head_dim, 0.0f);
        for (size_t token_idx = start_token; token_idx < end_token; ++token_idx) {
            for (size_t d = 0; d < head_dim; ++d) {
                pooled_token[d] += keys[token_idx][d];
            }
        }
        // Divide by number of tokens to get average
        for (size_t d = 0; d < head_dim; ++d) {
            pooled_token[d] /= static_cast<float>(tokens_in_page);
        }
        
        // Step 2: Linear projection using weight matrix
        // output = weight_ @ pooled_token (matrix-vector multiplication)
        // weight_ is output_dim x head_dim, pooled_token is head_dim
        // result is output_dim
        std::vector<float> projected_token(output_dim, 0.0f);
        for (size_t o = 0; o < output_dim; ++o) {
            for (size_t d = 0; d < head_dim; ++d) {
                projected_token[o] += weight_[o][d] * pooled_token[d];
            }
        }
        
        result.push_back(projected_token);
    }
    
    // Create FlatIndexMemory with the concatenated projected tokens
    memory.set_data(result);
}

void PagedKVIndexBuilder::run_cpu_kernel(
    const data_type::KVCacheData<float>& raw_data,
    data_type::FlatIndexMemory<float>& memory
) {
    // TODO: Implement CPU kernel
    run_test_kernel(raw_data, memory);
    return;
}

void PagedKVIndexBuilder::run_gpu_kernel(
    const data_type::KVCacheData<float>& raw_data,
    data_type::FlatIndexMemory<float>& memory
) {
    // TODO: Implement GPU kernel
    std::clog << "[PagedKVIndexBuilder] GPU kernel not implemented yet." << std::endl;
    return;
}

void PagedKVIndexBuilder::run_fpga_kernel(
    const data_type::KVCacheData<float>& raw_data,
    data_type::FlatIndexMemory<float>& memory
) {
    // TODO: Implement FPGA kernel
    std::clog << "[PagedKVIndexBuilder] FPGA kernel not implemented yet." << std::endl;
    return;
}

}  // namespace step
}  // namespace heteromm
