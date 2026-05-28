#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "../steps/build_memory.h"

using namespace heteromm;

TEST_SUITE("PagedKVIndexBuilder") {

    TEST_CASE("Basic paged KV index building") {
        // Create KV cache: 4 tokens, head_dim = 3
        data_type::KVCache<float> kv_data;
        kv_data.keys = {
            {1.0f, 2.0f, 3.0f},
            {4.0f, 5.0f, 6.0f},
            {7.0f, 8.0f, 9.0f},
            {10.0f, 11.0f, 12.0f}
        };
        kv_data.values = kv_data.keys;  // Values not used in this kernel
        data_type::KVCacheData<float> raw_data(kv_data);

        // Weight matrix: 2x3 (output_dim=2, head_dim=3)
        std::vector<std::vector<float>> weight = {
            {1.0f, 0.0f, 0.0f},
            {0.0f, 1.0f, 0.0f}
        };

        // page_size = 2: tokens [0,1] -> page 0, tokens [2,3] -> page 1
        // Page 0 average: [(1+4)/2, (2+5)/2, (3+6)/2] = [2.5, 3.5, 4.5]
        // Page 1 average: [(7+10)/2, (8+11)/2, (9+12)/2] = [8.5, 9.5, 10.5]
        // After projection with identity-like weight:
        // Page 0: [2.5, 3.5]
        // Page 1: [8.5, 9.5]
        std::vector<std::vector<float>> expected = {
            {2.5f, 3.5f},
            {8.5f, 9.5f}
        };
        data_type::FlatIndexMemory<float> memory(expected);

        step::PagedKVIndexBuilder builder(weight, 2);
        int result = builder.execute(raw_data, memory, true, true);

        CHECK(result == 0);  // Should pass
    }

    TEST_CASE("Single page (context_length <= page_size)") {
        // Create KV cache: 2 tokens, head_dim = 2
        data_type::KVCache<float> kv_data;
        kv_data.keys = {
            {2.0f, 4.0f},
            {6.0f, 8.0f}
        };
        kv_data.values = kv_data.keys;
        data_type::KVCacheData<float> raw_data(kv_data);

        // Weight matrix: 2x2 (identity)
        std::vector<std::vector<float>> weight = {
            {1.0f, 0.0f},
            {0.0f, 1.0f}
        };

        // page_size = 4: all tokens in one page
        // Average: [(2+6)/2, (4+8)/2] = [4.0, 6.0]
        // After projection: [4.0, 6.0]
        std::vector<std::vector<float>> expected = {
            {4.0f, 6.0f}
        };
        data_type::FlatIndexMemory<float> memory(expected);

        step::PagedKVIndexBuilder builder(weight, 4);
        int result = builder.execute(raw_data, memory, true, true);

        CHECK(result == 0);  // Should pass
    }

    TEST_CASE("Partial last page") {
        // Create KV cache: 5 tokens, head_dim = 2
        data_type::KVCache<float> kv_data;
        kv_data.keys = {
            {1.0f, 1.0f},
            {2.0f, 2.0f},
            {3.0f, 3.0f},
            {4.0f, 4.0f},
            {5.0f, 5.0f}
        };
        kv_data.values = kv_data.keys;
        data_type::KVCacheData<float> raw_data(kv_data);

        // Weight matrix: 2x2 (identity)
        std::vector<std::vector<float>> weight = {
            {1.0f, 0.0f},
            {0.0f, 1.0f}
        };

        // page_size = 2: pages are [0,1], [2,3], [4]
        // Page 0 average: [1.5, 1.5]
        // Page 1 average: [3.5, 3.5]
        // Page 2 average: [5.0, 5.0] (only 1 token)
        std::vector<std::vector<float>> expected = {
            {1.5f, 1.5f},
            {3.5f, 3.5f},
            {5.0f, 5.0f}
        };
        data_type::FlatIndexMemory<float> memory(expected);

        step::PagedKVIndexBuilder builder(weight, 2);
        int result = builder.execute(raw_data, memory, true, true);

        CHECK(result == 0);  // Should pass
    }

    TEST_CASE("Linear projection with non-identity weight") {
        // Create KV cache: 2 tokens, head_dim = 3
        data_type::KVCache<float> kv_data;
        kv_data.keys = {
            {1.0f, 2.0f, 3.0f},
            {1.0f, 2.0f, 3.0f}
        };
        kv_data.values = kv_data.keys;
        data_type::KVCacheData<float> raw_data(kv_data);

        // Weight matrix: 2x3
        std::vector<std::vector<float>> weight = {
            {1.0f, 1.0f, 1.0f},  // Sum all dimensions
            {1.0f, -1.0f, 0.0f}  // Difference of first two
        };

        // page_size = 2: one page with average [1.0, 2.0, 3.0]
        // Projection: [1+2+3, 1-2+0] = [6.0, -1.0]
        std::vector<std::vector<float>> expected = {
            {6.0f, -1.0f}
        };
        data_type::FlatIndexMemory<float> memory(expected);

        step::PagedKVIndexBuilder builder(weight, 2);
        int result = builder.execute(raw_data, memory, true, true);

        CHECK(result == 0);  // Should pass
    }

    TEST_CASE("Functional test with incorrect expected output") {
        data_type::KVCache<float> kv_data;
        kv_data.keys = {
            {1.0f, 2.0f},
            {3.0f, 4.0f}
        };
        kv_data.values = kv_data.keys;
        data_type::KVCacheData<float> raw_data(kv_data);

        std::vector<std::vector<float>> weight = {
            {1.0f, 0.0f},
            {0.0f, 1.0f}
        };

        // Wrong expected output (should be [2.0, 3.0])
        std::vector<std::vector<float>> wrong_expected = {
            {1.0f, 1.0f}
        };
        data_type::FlatIndexMemory<float> memory(wrong_expected);

        step::PagedKVIndexBuilder builder(weight, 2);
        int result = builder.execute(raw_data, memory, true, true);

        CHECK(result == 1);  // Should fail
    }

    TEST_CASE("Kernel type switching") {
        std::vector<std::vector<float>> weight = {{1.0f}};
        step::PagedKVIndexBuilder builder(weight, 4);

        CHECK(builder.current_kernel() == step::KernelType::CPU);

        builder.set_current_kernel(step::KernelType::GPU);
        CHECK(builder.current_kernel() == step::KernelType::GPU);

        builder.set_current_kernel(step::KernelType::FPGA);
        CHECK(builder.current_kernel() == step::KernelType::FPGA);
    }

    TEST_CASE("Page size of 1 (no pooling)") {
        // Create KV cache: 3 tokens, head_dim = 2
        data_type::KVCache<float> kv_data;
        kv_data.keys = {
            {1.0f, 2.0f},
            {3.0f, 4.0f},
            {5.0f, 6.0f}
        };
        kv_data.values = kv_data.keys;
        data_type::KVCacheData<float> raw_data(kv_data);

        // Weight matrix: 2x2 (identity)
        std::vector<std::vector<float>> weight = {
            {1.0f, 0.0f},
            {0.0f, 1.0f}
        };

        // page_size = 1: each token is its own page
        // No averaging, just projection
        std::vector<std::vector<float>> expected = {
            {1.0f, 2.0f},
            {3.0f, 4.0f},
            {5.0f, 6.0f}
        };
        data_type::FlatIndexMemory<float> memory(expected);

        step::PagedKVIndexBuilder builder(weight, 1);
        int result = builder.execute(raw_data, memory, true, true);

        CHECK(result == 0);  // Should pass
    }

    TEST_CASE("Dimension reduction via projection") {
        // Create KV cache: 4 tokens, head_dim = 4
        data_type::KVCache<float> kv_data;
        kv_data.keys = {
            {1.0f, 2.0f, 3.0f, 4.0f},
            {1.0f, 2.0f, 3.0f, 4.0f},
            {2.0f, 4.0f, 6.0f, 8.0f},
            {2.0f, 4.0f, 6.0f, 8.0f}
        };
        kv_data.values = kv_data.keys;
        data_type::KVCacheData<float> raw_data(kv_data);

        // Weight matrix: 1x4 (reduce to 1 dimension by summing)
        std::vector<std::vector<float>> weight = {
            {1.0f, 1.0f, 1.0f, 1.0f}
        };

        // page_size = 2
        // Page 0 average: [1, 2, 3, 4], projection: 10.0
        // Page 1 average: [2, 4, 6, 8], projection: 20.0
        std::vector<std::vector<float>> expected = {
            {10.0f},
            {20.0f}
        };
        data_type::FlatIndexMemory<float> memory(expected);

        step::PagedKVIndexBuilder builder(weight, 2);
        int result = builder.execute(raw_data, memory, true, true);

        CHECK(result == 0);  // Should pass
    }

}
