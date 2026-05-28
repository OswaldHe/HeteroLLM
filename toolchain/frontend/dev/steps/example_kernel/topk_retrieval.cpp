/**
 * @file topk_retrieval.cpp
 * @brief Implementation of TopKRetrieval kernels
 */

#include "../memory_retrieval.h"
#include <algorithm>
#include <numeric>

namespace heteromm {
namespace step {

void TopKRetrieval::run_test_kernel(
    const data_type::VectorScore<float>& score,
    data_type::TopKIndex& index
) {
    const auto& scores = score.export_data();
    size_t n = scores.size();
    size_t k = std::min(k_, n);

    // Create index array
    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);

    // Partial sort to get top-k indices
    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(), [&scores](int a, int b) {return scores[a] > scores[b]; });

    // Take only top-k
    indices.resize(k);
    index.set_data(indices);
}

void TopKRetrieval::run_cpu_kernel(
    const data_type::VectorScore<float>& score,
    data_type::TopKIndex& index
) {
    // TODO: Implement optimized CPU kernel
    run_test_kernel(score, index);
    return;
}

void TopKRetrieval::run_gpu_kernel(
    const data_type::VectorScore<float>& score,
    data_type::TopKIndex& index
) {
    // TODO: Implement GPU kernel
    std::clog << "[TopKRetrieval] GPU kernel not implemented. exiting." << std::endl;
    return;
}

void TopKRetrieval::run_fpga_kernel(
    const data_type::VectorScore<float>& score,
    data_type::TopKIndex& index
) {
    // TODO: Implement FPGA kernel
    std::clog << "[TopKRetrieval] FPGA kernel not implemented. exiting." << std::endl;
    return;
}

}  // namespace step
}  // namespace heteromm
