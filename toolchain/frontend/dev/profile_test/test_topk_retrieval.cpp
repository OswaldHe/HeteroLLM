#include "../steps/memory_retrieval.h"

using namespace heteromm;

int main(int argc, char** argv) {
    // Initialize the test framework
    std::vector<float> score_data = std::vector<float>(8192, 1.0f);  // Dummy scores
    data_type::VectorScore<float> score(score_data);

    // Top-3 should be indices [1, 3, 2] (scores 5.0, 4.0, 3.0)
    std::vector<int> ground_truth = std::vector<int>(64, 1);  // Dummy indices
    data_type::TopKIndex index(ground_truth);

    step::TopKRetrieval retrieval(64);
    retrieval.execute(score, index, true, false);

    std::clog << "========== MEMORY PROFILE RESULTS ==========" << std::endl;
    std::clog << "score: " << 8192*sizeof(float) << " Byte" << std::endl;
    std::clog << "retrieved_index: " << 64*sizeof(int) << " Byte" << std::endl;
    std::clog << "  ----------------------------------------" << std::endl;
    std::clog << "TOTAL_MEM: " << (8192*sizeof(float) + 64*sizeof(int)) << " Byte" << std::endl;
    std::clog << "===========================================" << std::endl;
}