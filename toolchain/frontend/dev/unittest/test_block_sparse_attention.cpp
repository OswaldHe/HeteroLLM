#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "../steps/apply_memory.h"
#include <cmath>

using namespace heteromm;

// Helper function to compute expected block sparse attention output
std::vector<float> compute_expected_output(
    const data_type::KVCache<float>& kv_cache,
    const std::vector<int>& topk_indices,
    const std::vector<float>& query,
    size_t page_size
) {
    size_t head_dim = query.size();
    size_t context_length = kv_cache.keys.size();
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Collect selected token indices
    std::vector<int> selected_tokens;
    for (int block_idx : topk_indices) {
        int start = block_idx * static_cast<int>(page_size);
        int end = std::min(start + static_cast<int>(page_size), static_cast<int>(context_length));
        for (int t = start; t < end; ++t) {
            selected_tokens.push_back(t);
        }
    }

    // Compute attention scores
    std::vector<float> scores(selected_tokens.size());
    for (size_t i = 0; i < selected_tokens.size(); ++i) {
        float dot = 0.0f;
        for (size_t d = 0; d < head_dim; ++d) {
            dot += query[d] * kv_cache.keys[selected_tokens[i]][d];
        }
        scores[i] = dot * scale;
    }

    // Softmax
    float max_score = *std::max_element(scores.begin(), scores.end());
    float sum_exp = 0.0f;
    std::vector<float> weights(scores.size());
    for (size_t i = 0; i < scores.size(); ++i) {
        weights[i] = std::exp(scores[i] - max_score);
        sum_exp += weights[i];
    }
    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] /= sum_exp;
    }

    // Weighted sum of values
    std::vector<float> output(head_dim, 0.0f);
    for (size_t i = 0; i < selected_tokens.size(); ++i) {
        for (size_t d = 0; d < head_dim; ++d) {
            output[d] += weights[i] * kv_cache.values[selected_tokens[i]][d];
        }
    }
    return output;
}

TEST_SUITE("BlockSparseAttention") {

    TEST_CASE("Basic block sparse attention - single block") {
        // Create KV cache: 4 tokens, head_dim = 3
        data_type::KVCache<float> kv_data;
        kv_data.keys = {
            {1.0f, 0.0f, 0.0f},
            {0.0f, 1.0f, 0.0f},
            {0.0f, 0.0f, 1.0f},
            {1.0f, 1.0f, 0.0f}
        };
        kv_data.values = {
            {1.0f, 2.0f, 3.0f},
            {4.0f, 5.0f, 6.0f},
            {7.0f, 8.0f, 9.0f},
            {10.0f, 11.0f, 12.0f}
        };
        data_type::KVCacheData<float> retrieved_data(kv_data);

        // Query vector
        std::vector<float> query = {1.0f, 0.0f, 0.0f};
        data_type::VectorInputOutputData<float> input(query);

        // Top-k indices: select block 0 (tokens 0, 1)
        std::vector<int> topk = {0};
        data_type::TopKIndex index(topk);

        // page_size = 2
        size_t page_size = 2;

        // Compute expected output
        std::vector<float> expected = compute_expected_output(kv_data, topk, query, page_size);
        data_type::VectorInputOutputData<float> output(expected);

        step::BlockSparseAttention attention(page_size);
        int result = attention.execute(retrieved_data, index, input, output, true, true);

        CHECK(result == 0);  // Should pass
    }

    TEST_CASE("Block sparse attention - multiple blocks") {
        // Create KV cache: 8 tokens, head_dim = 2
        data_type::KVCache<float> kv_data;
        kv_data.keys = {
            {1.0f, 0.0f},
            {0.5f, 0.5f},
            {0.0f, 1.0f},
            {0.3f, 0.7f},
            {0.8f, 0.2f},
            {0.6f, 0.4f},
            {0.2f, 0.8f},
            {0.4f, 0.6f}
        };
        kv_data.values = {
            {1.0f, 1.0f},
            {2.0f, 2.0f},
            {3.0f, 3.0f},
            {4.0f, 4.0f},
            {5.0f, 5.0f},
            {6.0f, 6.0f},
            {7.0f, 7.0f},
            {8.0f, 8.0f}
        };
        data_type::KVCacheData<float> retrieved_data(kv_data);

        // Query vector
        std::vector<float> query = {0.707f, 0.707f};
        data_type::VectorInputOutputData<float> input(query);

        // Top-k indices: select blocks 0 and 2 (tokens 0-1 and 4-5)
        std::vector<int> topk = {0, 2};
        data_type::TopKIndex index(topk);

        // page_size = 2
        size_t page_size = 2;

        // Compute expected output
        std::vector<float> expected = compute_expected_output(kv_data, topk, query, page_size);
        data_type::VectorInputOutputData<float> output(expected);

        step::BlockSparseAttention attention(page_size);
        int result = attention.execute(retrieved_data, index, input, output, true, true);

        CHECK(result == 0);  // Should pass
    }

    TEST_CASE("Block sparse attention - partial last block") {
        // Create KV cache: 5 tokens, head_dim = 2 (last block has only 1 token)
        data_type::KVCache<float> kv_data;
        kv_data.keys = {
            {1.0f, 0.0f},
            {0.0f, 1.0f},
            {0.5f, 0.5f},
            {0.3f, 0.7f},
            {0.9f, 0.1f}
        };
        kv_data.values = {
            {1.0f, 2.0f},
            {3.0f, 4.0f},
            {5.0f, 6.0f},
            {7.0f, 8.0f},
            {9.0f, 10.0f}
        };
        data_type::KVCacheData<float> retrieved_data(kv_data);

        // Query vector
        std::vector<float> query = {1.0f, 0.0f};
        data_type::VectorInputOutputData<float> input(query);

        // Top-k indices: select block 2 (only token 4)
        std::vector<int> topk = {2};
        data_type::TopKIndex index(topk);

        // page_size = 2
        size_t page_size = 2;

        // Compute expected output
        std::vector<float> expected = compute_expected_output(kv_data, topk, query, page_size);
        data_type::VectorInputOutputData<float> output(expected);

        step::BlockSparseAttention attention(page_size);
        int result = attention.execute(retrieved_data, index, input, output, true, true);

        CHECK(result == 0);  // Should pass
    }

    TEST_CASE("Block sparse attention - larger page size") {
        // Create KV cache: 6 tokens, head_dim = 4
        data_type::KVCache<float> kv_data;
        kv_data.keys = {
            {1.0f, 0.0f, 0.0f, 0.0f},
            {0.0f, 1.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 1.0f, 0.0f},
            {0.0f, 0.0f, 0.0f, 1.0f},
            {0.5f, 0.5f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.5f, 0.5f}
        };
        kv_data.values = {
            {1.0f, 1.0f, 1.0f, 1.0f},
            {2.0f, 2.0f, 2.0f, 2.0f},
            {3.0f, 3.0f, 3.0f, 3.0f},
            {4.0f, 4.0f, 4.0f, 4.0f},
            {5.0f, 5.0f, 5.0f, 5.0f},
            {6.0f, 6.0f, 6.0f, 6.0f}
        };
        data_type::KVCacheData<float> retrieved_data(kv_data);

        // Query vector
        std::vector<float> query = {0.5f, 0.5f, 0.5f, 0.5f};
        data_type::VectorInputOutputData<float> input(query);

        // Top-k indices: select block 0 (tokens 0-2)
        std::vector<int> topk = {0};
        data_type::TopKIndex index(topk);

        // page_size = 3
        size_t page_size = 3;

        // Compute expected output
        std::vector<float> expected = compute_expected_output(kv_data, topk, query, page_size);
        data_type::VectorInputOutputData<float> output(expected);

        step::BlockSparseAttention attention(page_size);
        int result = attention.execute(retrieved_data, index, input, output, true, true);

        CHECK(result == 0);  // Should pass
    }

    TEST_CASE("Block sparse attention - all blocks selected") {
        // Create KV cache: 4 tokens, head_dim = 2
        data_type::KVCache<float> kv_data;
        kv_data.keys = {
            {1.0f, 0.0f},
            {0.0f, 1.0f},
            {0.5f, 0.5f},
            {0.3f, 0.7f}
        };
        kv_data.values = {
            {1.0f, 2.0f},
            {3.0f, 4.0f},
            {5.0f, 6.0f},
            {7.0f, 8.0f}
        };
        data_type::KVCacheData<float> retrieved_data(kv_data);

        // Query vector
        std::vector<float> query = {0.6f, 0.8f};
        data_type::VectorInputOutputData<float> input(query);

        // Top-k indices: select all blocks
        std::vector<int> topk = {0, 1};
        data_type::TopKIndex index(topk);

        // page_size = 2
        size_t page_size = 2;

        // Compute expected output
        std::vector<float> expected = compute_expected_output(kv_data, topk, query, page_size);
        data_type::VectorInputOutputData<float> output(expected);

        step::BlockSparseAttention attention(page_size);
        int result = attention.execute(retrieved_data, index, input, output, true, true);

        CHECK(result == 0);  // Should pass
    }

    TEST_CASE("Block sparse attention - uniform attention weights") {
        // Create KV cache with identical keys to get uniform attention
        data_type::KVCache<float> kv_data;
        kv_data.keys = {
            {1.0f, 1.0f},
            {1.0f, 1.0f},
            {1.0f, 1.0f},
            {1.0f, 1.0f}
        };
        kv_data.values = {
            {1.0f, 0.0f},
            {0.0f, 1.0f},
            {1.0f, 1.0f},
            {2.0f, 2.0f}
        };
        data_type::KVCacheData<float> retrieved_data(kv_data);

        // Query vector
        std::vector<float> query = {1.0f, 1.0f};
        data_type::VectorInputOutputData<float> input(query);

        // Top-k indices: select both blocks
        std::vector<int> topk = {0, 1};
        data_type::TopKIndex index(topk);

        // page_size = 2
        size_t page_size = 2;

        // With uniform attention, output should be average of all values
        // Expected: [(1+0+1+2)/4, (0+1+1+2)/4] = [1.0, 1.0]
        std::vector<float> expected = {1.0f, 1.0f};
        data_type::VectorInputOutputData<float> output(expected);

        step::BlockSparseAttention attention(page_size);
        int result = attention.execute(retrieved_data, index, input, output, true, true);

        CHECK(result == 0);  // Should pass
    }

    TEST_CASE("Block sparse attention - single token blocks") {
        // Create KV cache: 3 tokens, page_size = 1
        data_type::KVCache<float> kv_data;
        kv_data.keys = {
            {1.0f, 0.0f, 0.0f},
            {0.0f, 1.0f, 0.0f},
            {0.0f, 0.0f, 1.0f}
        };
        kv_data.values = {
            {10.0f, 20.0f, 30.0f},
            {40.0f, 50.0f, 60.0f},
            {70.0f, 80.0f, 90.0f}
        };
        data_type::KVCacheData<float> retrieved_data(kv_data);

        // Query vector that mostly aligns with first key
        std::vector<float> query = {1.0f, 0.1f, 0.0f};
        data_type::VectorInputOutputData<float> input(query);

        // Top-k indices: select blocks 0 and 1 (tokens 0 and 1)
        std::vector<int> topk = {0, 1};
        data_type::TopKIndex index(topk);

        // page_size = 1
        size_t page_size = 1;

        // Compute expected output
        std::vector<float> expected = compute_expected_output(kv_data, topk, query, page_size);
        data_type::VectorInputOutputData<float> output(expected);

        step::BlockSparseAttention attention(page_size);
        int result = attention.execute(retrieved_data, index, input, output, true, true);

        CHECK(result == 0);  // Should pass
    }

    TEST_CASE("Block sparse attention - high dimensional vectors") {
        // Create KV cache: 4 tokens, head_dim = 8
        data_type::KVCache<float> kv_data;
        kv_data.keys = {
            {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f},
            {0.8f, 0.7f, 0.6f, 0.5f, 0.4f, 0.3f, 0.2f, 0.1f},
            {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f},
            {0.1f, 0.1f, 0.1f, 0.1f, 0.9f, 0.9f, 0.9f, 0.9f}
        };
        kv_data.values = {
            {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
            {2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f},
            {3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f},
            {4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f}
        };
        data_type::KVCacheData<float> retrieved_data(kv_data);

        // Query vector
        std::vector<float> query = {0.3f, 0.3f, 0.3f, 0.3f, 0.3f, 0.3f, 0.3f, 0.3f};
        data_type::VectorInputOutputData<float> input(query);

        // Top-k indices: select all blocks
        std::vector<int> topk = {0, 1};
        data_type::TopKIndex index(topk);

        // page_size = 2
        size_t page_size = 2;

        // Compute expected output
        std::vector<float> expected = compute_expected_output(kv_data, topk, query, page_size);
        data_type::VectorInputOutputData<float> output(expected);

        step::BlockSparseAttention attention(page_size);
        int result = attention.execute(retrieved_data, index, input, output, true, true);

        CHECK(result == 0);  // Should pass
    }

}
