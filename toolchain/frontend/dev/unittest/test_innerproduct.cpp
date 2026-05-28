#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "../steps/compute_score.h"

using namespace heteromm;

TEST_SUITE("InnerProductCompute") {

    TEST_CASE("Basic dot product computation") {
        // Create a simple 3x4 memory (3 vectors of dimension 4)
        std::vector<std::vector<float>> mem_data = {
            {1.0f, 0.0f, 0.0f, 0.0f},
            {0.0f, 1.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 1.0f, 0.0f}
        };
        data_type::FlatIndexMemory<float> memory(mem_data);

        // Create a query vector
        std::vector<float> query_data = {1.0f, 2.0f, 3.0f, 4.0f};
        data_type::VectorQuery<float> query(query_data);

        // Pre-compute ground truth: dot([1,0,0,0], [1,2,3,4]) = 1, etc.
        std::vector<float> ground_truth = {1.0f, 2.0f, 3.0f};
        data_type::VectorScore<float> score(ground_truth);

        // Create the compute step and execute
        step::InnerProductCompute compute;
        int result = compute.execute(memory, query, score, true, true);

        CHECK(result == 0);  // Should pass
    }

    TEST_CASE("Dot product with non-trivial vectors") {
        std::vector<std::vector<float>> mem_data = {
            {1.0f, 2.0f, 3.0f},
            {4.0f, 5.0f, 6.0f},
            {-1.0f, -2.0f, -3.0f}
        };
        data_type::FlatIndexMemory<float> memory(mem_data);

        std::vector<float> query_data = {1.0f, 1.0f, 1.0f};
        data_type::VectorQuery<float> query(query_data);

        // Pre-compute ground truth: 1+2+3=6, 4+5+6=15, -1-2-3=-6
        std::vector<float> ground_truth = {6.0f, 15.0f, -6.0f};
        data_type::VectorScore<float> score(ground_truth);

        step::InnerProductCompute compute;
        int result = compute.execute(memory, query, score, true, true);

        CHECK(result == 0);  // Should pass
    }

    TEST_CASE("Empty memory") {
        std::vector<std::vector<float>> mem_data = {};
        data_type::FlatIndexMemory<float> memory(mem_data);

        std::vector<float> query_data = {1.0f, 2.0f, 3.0f};
        data_type::VectorQuery<float> query(query_data);

        // Pre-compute ground truth: empty
        std::vector<float> ground_truth = {};
        data_type::VectorScore<float> score(ground_truth);

        step::InnerProductCompute compute;
        int result = compute.execute(memory, query, score, true, true);

        CHECK(result == 0);  // Should pass
    }

    TEST_CASE("Single vector memory") {
        std::vector<std::vector<float>> mem_data = {
            {2.0f, 3.0f, 4.0f}
        };
        data_type::FlatIndexMemory<float> memory(mem_data);

        std::vector<float> query_data = {1.0f, 2.0f, 3.0f};
        data_type::VectorQuery<float> query(query_data);

        // Pre-compute ground truth: 2*1 + 3*2 + 4*3 = 2+6+12 = 20
        std::vector<float> ground_truth = {20.0f};
        data_type::VectorScore<float> score(ground_truth);

        step::InnerProductCompute compute;
        int result = compute.execute(memory, query, score, true, true);

        CHECK(result == 0);  // Should pass
    }

    TEST_CASE("Functional test with correct expected score") {
        std::vector<std::vector<float>> mem_data = {
            {1.0f, 0.0f},
            {0.0f, 1.0f}
        };
        data_type::FlatIndexMemory<float> memory(mem_data);

        std::vector<float> query_data = {3.0f, 4.0f};
        data_type::VectorQuery<float> query(query_data);

        // Pre-compute ground truth: dot([1,0], [3,4]) = 3, dot([0,1], [3,4]) = 4
        std::vector<float> ground_truth = {3.0f, 4.0f};
        data_type::VectorScore<float> score(ground_truth);

        step::InnerProductCompute compute;
        int result = compute.execute(memory, query, score, true, true);

        CHECK(result == 0);  // Functional test should pass
    }

    TEST_CASE("Functional test with incorrect expected score") {
        std::vector<std::vector<float>> mem_data = {
            {1.0f, 0.0f},
            {0.0f, 1.0f}
        };
        data_type::FlatIndexMemory<float> memory(mem_data);

        std::vector<float> query_data = {3.0f, 4.0f};
        data_type::VectorQuery<float> query(query_data);

        // Pre-populate score with wrong expected values
        std::vector<float> wrong_scores = {100.0f, 200.0f};
        data_type::VectorScore<float> score(wrong_scores);

        step::InnerProductCompute compute;
        int result = compute.execute(memory, query, score, true, true);

        CHECK(result == 1);  // Functional test should fail
    }

    TEST_CASE("Kernel type switching") {
        step::InnerProductCompute compute;
        
        CHECK(compute.current_kernel() == step::KernelType::CPU);
        
        compute.set_current_kernel(step::KernelType::GPU);
        CHECK(compute.current_kernel() == step::KernelType::GPU);
        
        compute.set_current_kernel(step::KernelType::FPGA);
        CHECK(compute.current_kernel() == step::KernelType::FPGA);
    }

}



