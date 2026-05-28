/**
 * @file block_sparse_attention.cpp
 * @brief Implementation of BlockSparseAttention kernels
 * 
 * Implements block sparse attention that:
 * 1. Takes KV cache, query, and top-k block indices
 * 2. Extracts the selected blocks of keys and values
 * 3. Computes attention scores (Q @ K^T)
 * 4. Applies softmax
 * 5. Computes output (attention_weights @ V)
 */

#include "../apply_memory.h"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace heteromm {
namespace step {

void BlockSparseAttention::run_test_kernel(
    const data_type::KVCacheData<float>& retrieved_data,
    const data_type::TopKIndex& index,
    const data_type::VectorInputOutputData<float>& input,
    data_type::VectorInputOutputData<float>& output
) {
    // Get the KV cache data
    const auto& kv_cache = retrieved_data.export_data();
    const auto& keys = kv_cache.keys;      // [context_length, head_dim]
    const auto& values = kv_cache.values;  // [context_length, head_dim]
    
    // Get top-k block indices
    const auto& topk_indices = index.export_data();
    
    // Get input query vector
    const auto& query = input.export_data();  // [head_dim]
    
    size_t head_dim = query.size();
    size_t context_length = keys.size();
    size_t num_blocks = topk_indices.size();
    
    if (context_length == 0 || head_dim == 0 || num_blocks == 0) {
        // Return zero output if no data
        std::vector<float> zero_output(head_dim, 0.0f);
        output.set_data(zero_output);
        return;
    }

    // Calculate scaling factor for attention (1/sqrt(d_k))
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    
    // Collect all token indices from selected blocks
    std::vector<int> selected_token_indices;
    for (int block_idx : topk_indices) {
        // Each block corresponds to page_size_ consecutive tokens
        int start_token = block_idx * static_cast<int>(page_size_);
        int end_token = std::min(start_token + static_cast<int>(page_size_), 
                                  static_cast<int>(context_length));
        
        for (int t = start_token; t < end_token; ++t) {
            selected_token_indices.push_back(t);
        }
    }
    
    size_t num_selected_tokens = selected_token_indices.size();
    
    if (num_selected_tokens == 0) {
        std::vector<float> zero_output(head_dim, 0.0f);
        output.set_data(zero_output);
        return;
    }
    
    // Step 1: Compute attention scores for selected tokens
    // scores[i] = Q @ K[selected_token_indices[i]]^T * scale
    std::vector<float> attention_scores(num_selected_tokens);
    
    for (size_t i = 0; i < num_selected_tokens; ++i) {
        int token_idx = selected_token_indices[i];
        const auto& key_vec = keys[token_idx];
        
        float dot_product = 0.0f;
        size_t dim = std::min(head_dim, key_vec.size());
        for (size_t d = 0; d < dim; ++d) {
            dot_product += query[d] * key_vec[d];
        }
        attention_scores[i] = dot_product * scale;
    }
    
    // Step 2: Apply softmax to attention scores
    // softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
    
    // Find max for numerical stability
    float max_score = *std::max_element(attention_scores.begin(), attention_scores.end());
    
    // Compute exp(score - max) and sum
    std::vector<float> exp_scores(num_selected_tokens);
    float sum_exp = 0.0f;
    
    for (size_t i = 0; i < num_selected_tokens; ++i) {
        exp_scores[i] = std::exp(attention_scores[i] - max_score);
        sum_exp += exp_scores[i];
    }
    
    // Normalize to get attention weights
    std::vector<float> attention_weights(num_selected_tokens);
    for (size_t i = 0; i < num_selected_tokens; ++i) {
        attention_weights[i] = exp_scores[i] / sum_exp;
    }
    
    // Step 3: Compute weighted sum of values
    // output = sum(attention_weights[i] * V[selected_token_indices[i]])
    std::vector<float> result(head_dim, 0.0f);
    
    for (size_t i = 0; i < num_selected_tokens; ++i) {
        int token_idx = selected_token_indices[i];
        const auto& value_vec = values[token_idx];
        float weight = attention_weights[i];
        
        size_t dim = std::min(head_dim, value_vec.size());
        for (size_t d = 0; d < dim; ++d) {
            result[d] += weight * value_vec[d];
        }
    }
    
    output.set_data(result);
}

void BlockSparseAttention::run_cpu_kernel(
    const data_type::KVCacheData<float>& retrieved_data,
    const data_type::TopKIndex& index,
    const data_type::VectorInputOutputData<float>& input,
    data_type::VectorInputOutputData<float>& output
) {
    // TODO: Implement optimized CPU kernel
    run_test_kernel(retrieved_data, index, input, output);
    return;
}

void BlockSparseAttention::run_gpu_kernel(
    const data_type::KVCacheData<float>& retrieved_data,
    const data_type::TopKIndex& index,
    const data_type::VectorInputOutputData<float>& input,
    data_type::VectorInputOutputData<float>& output
) {
    // TODO: Implement GPU kernel
    std::clog << "[BlockSparseAttention] GPU kernel not implemented. exiting." << std::endl;
    return;
}

void BlockSparseAttention::run_fpga_kernel(
    const data_type::KVCacheData<float>& retrieved_data,
    const data_type::TopKIndex& index,
    const data_type::VectorInputOutputData<float>& input,
    data_type::VectorInputOutputData<float>& output
) {
    // TODO: Implement FPGA kernel
    std::clog << "[BlockSparseAttention] FPGA kernel not implemented. exiting." << std::endl;
    return;
}

}  // namespace step
}  // namespace heteromm
