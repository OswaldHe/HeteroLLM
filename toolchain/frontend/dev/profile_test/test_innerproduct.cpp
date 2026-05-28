#include "../steps/compute_score.h"

using namespace heteromm;

int main(int argc, char** argv) {
    // Initialize the test framework
    std::vector<std::vector<float>> mem_data(8192, std::vector<float>(128));
    data_type::FlatIndexMemory<float> memory(mem_data);

    std::vector<float> query_data(128);
    data_type::VectorQuery<float> query(query_data);

    // Pre-compute ground truth: 1+2+3=6, 4+5+6=15, -1-2-3=-6
    std::vector<float> ground_truth(8192);
    data_type::VectorScore<float> score(ground_truth);

    step::InnerProductCompute compute;
    compute.execute(memory, query, score, true, false);

    std::clog << "========== MEMORY PROFILE RESULTS ==========" << std::endl;
    std::clog << "memory: " << 8192*128*sizeof(float) << " Byte" << std::endl;
    std::clog << "query: " << 128*sizeof(float) << " Byte" << std::endl;
    std::clog << "score: " << 8192*sizeof(float) << " Byte" << std::endl;
    std::clog << "  ----------------------------------------" << std::endl;
    std::clog << "TOTAL_MEM: " << (8192*128*sizeof(float) + 128*sizeof(float) + 8192*sizeof(float)) << " Byte" << std::endl;
    std::clog << "===========================================" << std::endl;
}