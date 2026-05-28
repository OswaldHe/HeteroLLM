#include "../steps/apply_memory.h"

using namespace heteromm;

int main(int argc, char** argv) {
// Create KV cache: 8 tokens, head_dim = 2
    data_type::KVCache<float> kv_data;
    kv_data.keys = std::vector<std::vector<float>>(65536, std::vector<float>(768));
    kv_data.values = std::vector<std::vector<float>>(65536, std::vector<float>(768));
    data_type::KVCacheData<float> retrieved_data(kv_data);

    // Query vector
    std::vector<float> query = std::vector<float>(768, 1.0f);
    data_type::VectorInputOutputData<float> input(query);

    std::vector<int> topk = std::vector<int>(64, 1);  // Dummy indices
    for(int i = 0; i < 64; ++i) {
        topk[i] = i;  // Select the first 64 blocks
    }
    data_type::TopKIndex index(topk);

    size_t page_size = 8;

    // Compute expected output
    std::vector<float> expected = std::vector<float>(768, 0.f);
    data_type::VectorInputOutputData<float> output(expected);

    step::BlockSparseAttention attention(page_size);
    attention.execute(retrieved_data, index, input, output, true, false);

    int key_size = 64 * page_size * 768 * sizeof(float);
    int value_size = 64 * page_size * 768 * sizeof(float);
    int index_size = 64 * sizeof(int);
    int input_size = 768 * sizeof(float);
    int output_size = 768 * sizeof(float);

    std::clog << "========== MEMORY PROFILE RESULTS ==========" << std::endl;
    std::clog << "retrieved_data: " << value_size + key_size << " Byte" << std::endl;
    std::clog << "retrieved_index: " << index_size << " Byte" << std::endl;
    std::clog << "target_data: " << input_size << " Byte" << std::endl;
    std::clog << "output: " << output_size << " Byte" << std::endl;
    std::clog << "  ----------------------------------------" << std::endl;
    std::clog << "TOTAL_MEM: " << (key_size + value_size + index_size + input_size + output_size) << " Byte" << std::endl;
    std::clog << "===========================================" << std::endl;
}