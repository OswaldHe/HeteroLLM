#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "../steps/memory_retrieval.h"

using namespace heteromm;

TEST_SUITE("ThresholdRetrieval") {

    TEST_CASE("Basic threshold retrieval") {
        // Scores: [1.0, 5.0, 3.0, 4.0, 2.0], threshold = 3.0
        // Indices >= 3.0: 1 (5.0), 2 (3.0), 3 (4.0)
        std::vector<float> score_data = {1.0f, 5.0f, 3.0f, 4.0f, 2.0f};
        data_type::VectorScore<float> score(score_data);

        // Bitmap: bit 1, 2, 3 set = 0b01110 = 14
        std::vector<unsigned long long> ground_truth_bitmap = {0b01110ULL};
        data_type::ThresholdBitmapIndex index(ground_truth_bitmap, 5);

        step::ThresholdRetrieval retrieval(3.0f);
        int result = retrieval.execute(score, index, true, true);

        CHECK(result == 0);  // Should pass
    }

    TEST_CASE("Threshold retrieval with no matches") {
        std::vector<float> score_data = {1.0f, 2.0f, 3.0f};
        data_type::VectorScore<float> score(score_data);

        // Threshold = 10.0, no scores match
        std::vector<unsigned long long> ground_truth_bitmap = {0ULL};
        data_type::ThresholdBitmapIndex index(ground_truth_bitmap, 3);

        step::ThresholdRetrieval retrieval(10.0f);
        int result = retrieval.execute(score, index, true, true);

        CHECK(result == 0);  // Should pass
    }

    TEST_CASE("Threshold retrieval with all matches") {
        std::vector<float> score_data = {5.0f, 6.0f, 7.0f, 8.0f};
        data_type::VectorScore<float> score(score_data);

        // Threshold = 1.0, all scores match
        // Bitmap: all 4 bits set = 0b1111 = 15
        std::vector<unsigned long long> ground_truth_bitmap = {0b1111ULL};
        data_type::ThresholdBitmapIndex index(ground_truth_bitmap, 4);

        step::ThresholdRetrieval retrieval(1.0f);
        int result = retrieval.execute(score, index, true, true);

        CHECK(result == 0);  // Should pass
    }

    TEST_CASE("Threshold retrieval with negative threshold") {
        std::vector<float> score_data = {-5.0f, -2.0f, 0.0f, 3.0f};
        data_type::VectorScore<float> score(score_data);

        // Threshold = -3.0, indices 1, 2, 3 match
        // Bitmap: bits 1, 2, 3 set = 0b1110 = 14
        std::vector<unsigned long long> ground_truth_bitmap = {0b1110ULL};
        data_type::ThresholdBitmapIndex index(ground_truth_bitmap, 4);

        step::ThresholdRetrieval retrieval(-3.0f);
        int result = retrieval.execute(score, index, true, true);

        CHECK(result == 0);  // Should pass
    }

    TEST_CASE("Threshold retrieval with exact threshold match") {
        std::vector<float> score_data = {1.0f, 3.0f, 3.0f, 2.0f};
        data_type::VectorScore<float> score(score_data);

        // Threshold = 3.0, indices 1, 2 match (exactly 3.0)
        // Bitmap: bits 1, 2 set = 0b0110 = 6
        std::vector<unsigned long long> ground_truth_bitmap = {0b0110ULL};
        data_type::ThresholdBitmapIndex index(ground_truth_bitmap, 4);

        step::ThresholdRetrieval retrieval(3.0f);
        int result = retrieval.execute(score, index, true, true);

        CHECK(result == 0);  // Should pass
    }

    TEST_CASE("Threshold retrieval with more than 64 elements") {
        // Create 100 scores
        std::vector<float> score_data(100);
        for (int i = 0; i < 100; ++i) {
            score_data[i] = static_cast<float>(i);
        }
        data_type::VectorScore<float> score(score_data);

        // Threshold = 50.0, indices 50-99 match
        // Need 2 words: word 0 has bits 50-63 set, word 1 has bits 0-35 set (indices 64-99)
        std::vector<unsigned long long> ground_truth_bitmap(2, 0);
        // Word 0: bits 50-63 set
        for (int i = 50; i < 64; ++i) {
            ground_truth_bitmap[0] |= (1ULL << i);
        }
        // Word 1: bits 0-35 set (for indices 64-99)
        for (int i = 0; i < 36; ++i) {
            ground_truth_bitmap[1] |= (1ULL << i);
        }
        data_type::ThresholdBitmapIndex index(ground_truth_bitmap, 100);

        step::ThresholdRetrieval retrieval(50.0f);
        int result = retrieval.execute(score, index, true, true);

        CHECK(result == 0);  // Should pass
    }

    TEST_CASE("Functional test with incorrect expected bitmap") {
        std::vector<float> score_data = {1.0f, 5.0f, 3.0f};
        data_type::VectorScore<float> score(score_data);

        // Wrong bitmap (should be 0b110 for threshold 3.0)
        std::vector<unsigned long long> wrong_bitmap = {0b111ULL};
        data_type::ThresholdBitmapIndex index(wrong_bitmap, 3);

        step::ThresholdRetrieval retrieval(3.0f);
        int result = retrieval.execute(score, index, true, true);

        CHECK(result == 1);  // Should fail
    }

    TEST_CASE("Kernel type switching") {
        step::ThresholdRetrieval retrieval(5.0f);
        
        CHECK(retrieval.current_kernel() == step::KernelType::CPU);
        
        retrieval.set_current_kernel(step::KernelType::GPU);
        CHECK(retrieval.current_kernel() == step::KernelType::GPU);
        
        retrieval.set_current_kernel(step::KernelType::FPGA);
        CHECK(retrieval.current_kernel() == step::KernelType::FPGA);
    }

    TEST_CASE("Threshold value getter and setter") {
        step::ThresholdRetrieval retrieval(5.0f);
        CHECK(retrieval.threshold() == doctest::Approx(5.0f));
        
        retrieval.set_threshold(10.0f);
        CHECK(retrieval.threshold() == doctest::Approx(10.0f));
    }

}
