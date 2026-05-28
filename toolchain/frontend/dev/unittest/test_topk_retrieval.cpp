#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "../steps/memory_retrieval.h"

using namespace heteromm;

TEST_SUITE("TopKRetrieval") {

    TEST_CASE("Basic top-k retrieval") {
        // Create scores: [1.0, 5.0, 3.0, 4.0, 2.0]
        std::vector<float> score_data = {1.0f, 5.0f, 3.0f, 4.0f, 2.0f};
        data_type::VectorScore<float> score(score_data);

        // Top-3 should be indices [1, 3, 2] (scores 5.0, 4.0, 3.0)
        std::vector<int> ground_truth = {1, 3, 2};
        data_type::TopKIndex index(ground_truth);

        step::TopKRetrieval retrieval(3);
        int result = retrieval.execute(score, index, true, true);

        CHECK(result == 0);  // Should pass
    }

    TEST_CASE("Top-k with k larger than array size") {
        std::vector<float> score_data = {3.0f, 1.0f, 2.0f};
        data_type::VectorScore<float> score(score_data);

        // k=5 but only 3 elements, should return all 3 in descending order
        std::vector<int> ground_truth = {0, 2, 1};  // scores 3.0, 2.0, 1.0
        data_type::TopKIndex index(ground_truth);

        step::TopKRetrieval retrieval(5);
        int result = retrieval.execute(score, index, true, true);

        CHECK(result == 0);  // Should pass
    }

    TEST_CASE("Top-1 retrieval") {
        std::vector<float> score_data = {2.0f, 8.0f, 5.0f, 1.0f};
        data_type::VectorScore<float> score(score_data);

        // Top-1 should be index 1 (score 8.0)
        std::vector<int> ground_truth = {1};
        data_type::TopKIndex index(ground_truth);

        step::TopKRetrieval retrieval(1);
        int result = retrieval.execute(score, index, true, true);

        CHECK(result == 0);  // Should pass
    }

    TEST_CASE("Top-k with all equal scores") {
        std::vector<float> score_data = {5.0f, 5.0f, 5.0f, 5.0f};
        data_type::VectorScore<float> score(score_data);

        // With equal scores, any 2 indices are valid - use first 2
        std::vector<int> ground_truth = {0, 1};
        data_type::TopKIndex index(ground_truth);

        step::TopKRetrieval retrieval(2);
        int result = retrieval.execute(score, index, true, true);

        CHECK(result == 0);  // Should pass
    }

    TEST_CASE("Top-k with negative scores") {
        std::vector<float> score_data = {-1.0f, -5.0f, -2.0f, -3.0f};
        data_type::VectorScore<float> score(score_data);

        // Top-2 should be indices [0, 2] (scores -1.0, -2.0)
        std::vector<int> ground_truth = {0, 2};
        data_type::TopKIndex index(ground_truth);

        step::TopKRetrieval retrieval(2);
        int result = retrieval.execute(score, index, true, true);

        CHECK(result == 0);  // Should pass
    }

    TEST_CASE("Functional test with incorrect expected indices") {
        std::vector<float> score_data = {1.0f, 5.0f, 3.0f};
        data_type::VectorScore<float> score(score_data);

        // Wrong ground truth
        std::vector<int> wrong_indices = {0, 2};  // should be [1, 2]
        data_type::TopKIndex index(wrong_indices);

        step::TopKRetrieval retrieval(2);
        int result = retrieval.execute(score, index, true, true);

        CHECK(result == 1);  // Should fail
    }

    TEST_CASE("Kernel type switching") {
        step::TopKRetrieval retrieval(5);
        
        CHECK(retrieval.current_kernel() == step::KernelType::CPU);
        
        retrieval.set_current_kernel(step::KernelType::GPU);
        CHECK(retrieval.current_kernel() == step::KernelType::GPU);
        
        retrieval.set_current_kernel(step::KernelType::FPGA);
        CHECK(retrieval.current_kernel() == step::KernelType::FPGA);
    }

    TEST_CASE("K value getter and setter") {
        step::TopKRetrieval retrieval(10);
        CHECK(retrieval.k() == 10);
        
        retrieval.set_k(20);
        CHECK(retrieval.k() == 20);
    }

}
