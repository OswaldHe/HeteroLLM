#include "../steps/build_memory.h"

using namespace heteromm;

int main(int argc, char** argv) {
    data_type::KVCache<float> kv_data;
    kv_data.keys = std::vector<std::vector<float>>(65536, std::vector<float>(768));
    kv_data.values = kv_data.keys;  // Values not used in this kernel
    data_type::KVCacheData<float> raw_data(kv_data);

    // Weight matrix: 2x3 (output_dim=2, head_dim=3)
    std::vector<std::vector<float>> weight = std::vector<std::vector<float>>(128, std::vector<float>(768, 1.0f));  // Identity-like projection

    std::vector<std::vector<float>> expected = std::vector<std::vector<float>>(8192, std::vector<float>(128, 768.0f));  // Each output is sum of 768 inputs
    data_type::FlatIndexMemory<float> memory(expected);

    step::PagedKVIndexBuilder builder(weight, 8);
    builder.execute(raw_data, memory, true, false);

    std::clog << "========== MEMORY PROFILE RESULTS ==========" << std::endl;
    std::clog << "retrieved_data: " << 65536*768*sizeof(float) << " Byte" << std::endl;
    std::clog << "weight: " << 128*768*sizeof(float) << " Byte" << std::endl;
    std::clog << "memory: " << 8192*128*sizeof(float) << " Byte" << std::endl;
    std::clog << "  ----------------------------------------" << std::endl;
    std::clog << "TOTAL_MEM: " << (65536*768*sizeof(float) + 128*768*sizeof(float) + 8192*128*sizeof(float)) << " Byte" << std::endl;
    std::clog << "===========================================" << std::endl;
}