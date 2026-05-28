/**
 * @file inner_product_compute.cpp
 * @brief Implementation of InnerProductCompute kernels
 */

#include "../compute_score.h"

namespace heteromm {
namespace step {

void InnerProductCompute::run_test_kernel(
    const data_type::FlatIndexMemory<float>& memory,
    const data_type::VectorQuery<float>& query,
    data_type::VectorScore<float>& score
) {
    const auto& mem_data = memory.export_data();
    const auto& query_vec = query.export_data();
    std::vector<float> scores(mem_data.size(), 0.0f);

    size_t i_bound = mem_data.size();
    size_t j_bound = query_vec.size();

    for (size_t i = 0; i < i_bound; ++i) {
        float dot = 0.0f;
        for (size_t j = 0; j < j_bound; ++j) {
            dot += mem_data[i][j] * query_vec[j];
        }
        scores[i] = dot;
    }
    score.set_data(scores);
}

void InnerProductCompute::run_cpu_kernel(
    const data_type::FlatIndexMemory<float>& memory,
    const data_type::VectorQuery<float>& query,
    data_type::VectorScore<float>& score
) {
    // TODO: Implement CPU kernel
    run_test_kernel(memory, query, score);
    return;
}

void InnerProductCompute::run_gpu_kernel(
    const data_type::FlatIndexMemory<float>& memory,
    const data_type::VectorQuery<float>& query,
    data_type::VectorScore<float>& score
) {
    // TODO: Implement GPU kernel
    std::clog << "[InnerProductCompute] GPU kernel not implemented. exiting." << std::endl;
    return;
}

void InnerProductCompute::run_fpga_kernel(
    const data_type::FlatIndexMemory<float>& memory,
    const data_type::VectorQuery<float>& query,
    data_type::VectorScore<float>& score
) {
    // TODO: Implement FPGA kernel
    std::clog << "[InnerProductCompute] FPGA kernel not implemented. exiting." << std::endl;
    return;
}

}  // namespace step
}  // namespace heteromm
