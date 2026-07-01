#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <random>
#include <set>
#include <chrono>

#include <gflags/gflags.h>
#include <tapa.h>

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include "indexer_lut.h"

// Error checking macros
#define HIP_CHECK(call) \
do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        std::cerr << "HIP Error at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::cerr << "Error code: " << err << " (" << hipGetErrorString(err) << ")" << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define RBLAS_CHECK(call) \
do { \
    rocblas_status status = call; \
    if (status != rocblas_status_success) { \
        std::cerr << "rocBLAS Error at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::cerr << "Error code: " << status << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

DEFINE_string(bitstream, "", "path to bitstream file, run csim if empty");

// GPU kernel to select sparse vectors based on indices
__global__ void select_vectors_kernel(
    const float* __restrict__ K_full,
    const int* __restrict__ indices,
    float* __restrict__ K_sparse,
    int K_cols,
    int K_sparse_rows,
    int K_full_rows
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < K_sparse_rows && col < K_cols) {
        int source_row = indices[row];
        if (source_row >= 0 && source_row < K_full_rows) {
            K_sparse[row * K_cols + col] = K_full[source_row * K_cols + col];
        }
    }
}

// Software reference implementation
void indexer_lut_ref(
    const int L,
    const std::vector<std::vector<ap_uint<8>>>& query_indices,  // [NUM_INDEX_HEAD][HEAD_DIM_DIV_2]
    const std::vector<std::vector<ap_uint<8>>>& key_indices,    // [L][HEAD_DIM_DIV_2]
    const std::vector<std::vector<std::vector<ap_int<8>>>>& lookup_tables, // [HEAD_DIM_DIV_2][K_CODEBOOK_SIZE][Q_CODEBOOK_SIZE]
    const std::vector<float>& weights,                          // [NUM_INDEX_HEAD]
    std::vector<int>& topk_ids)                                // [TOP_K]
{
    // Compute weighted indexing scores
    std::vector<std::pair<float, int>> scores(L);
    
    for (int k = 0; k < L; k++) {
        float total_score = 0.0f;
        
        for (int h = 0; h < NUM_INDEX_HEAD; h++) {
            // Compute dot product using lookup tables
            int64_t qk = 0;
            for (int d = 0; d < HEAD_DIM_DIV_2; d++) {
                ap_uint<8> q_idx = query_indices[h][d];
                ap_uint<8> k_idx = key_indices[k][d];
                ap_int<8> lut_val = lookup_tables[d][k_idx.to_int()][q_idx.to_int()];
                qk += lut_val.to_int();
            }
            
            // Apply ReLU
            if (qk < 0) {
                qk = 0;
            }
            
            // Weight and accumulate
            total_score += weights[h] * (float)qk;
        }

        scores[k] = std::make_pair(total_score, k);
    }
    
    // Find top K
    std::sort(scores.begin(), scores.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    
    topk_ids.resize(TOP_K);
    for (int i = 0; i < TOP_K; i++) {
        topk_ids[i] = scores[i].second;
    }
}

int main(int argc, char* argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    
    int L = 4096;  // Number of key vectors
    if (argc > 1) {
        L = std::atoi(argv[1]);
    }
    
    std::cout << "Indexer LUT Testbench" << std::endl;
    std::cout << "L (key vectors): " << L << std::endl;
    std::cout << "NUM_INDEX_HEAD: " << NUM_INDEX_HEAD << std::endl;
    std::cout << "HEAD_DIM: " << HEAD_DIM << std::endl;
    std::cout << "HEAD_DIM_DIV_2: " << HEAD_DIM_DIV_2 << std::endl;
    std::cout << "TOP_K: " << TOP_K << std::endl;
    std::cout << "K_CODEBOOK_SIZE: " << K_CODEBOOK_SIZE << " (uint8)" << std::endl;
    std::cout << "Q_CODEBOOK_SIZE: " << Q_CODEBOOK_SIZE << " (uint6 packed as uint8)" << std::endl;
    
    // Initialize random number generator
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_int_distribution<int> dis_k_idx(0, K_CODEBOOK_SIZE - 1);  // 0-255 for uint8
    std::uniform_int_distribution<int> dis_q_idx(0, Q_CODEBOOK_SIZE - 1);  // 0-63 for uint6
    std::uniform_int_distribution<int> dis_lut(-128, 127);
    std::uniform_real_distribution<float> dis_float(0.0f, 3.0f);
    
    // Generate random query indices (16 heads x 64 dimensions)
    // Note: Query indices are uint6 (0-63) but packed as uint8 for convenience
    std::vector<std::vector<ap_uint<8>>> query_indices(NUM_INDEX_HEAD, std::vector<ap_uint<8>>(HEAD_DIM_DIV_2));
    for (int h = 0; h < NUM_INDEX_HEAD; h++) {
        for (int d = 0; d < HEAD_DIM_DIV_2; d++) {
            query_indices[h][d] = ap_uint<8>(dis_q_idx(gen));
        }
    }
    
    // Generate random key indices (L x 64 dimensions)
    // Note: Key indices are uint8 (0-255)
    std::vector<std::vector<ap_uint<8>>> key_indices(L, std::vector<ap_uint<8>>(HEAD_DIM_DIV_2));
    for (int k = 0; k < L; k++) {
        for (int d = 0; d < HEAD_DIM_DIV_2; d++) {
            key_indices[k][d] = ap_uint<8>(dis_k_idx(gen));
        }
    }

    // test: flash later key to 0
    for (int k = 64; k < L; k++) {
        for (int d = 0; d < HEAD_DIM_DIV_2; d++) {
            key_indices[k][d] = ap_uint<8>(0);
        }
    }
    
    // Generate random lookup tables (64 dimensions x 256 k_codes x 64 q_codes)
    // LUT[dim][k_idx][q_idx] returns the int8 dot product contribution
    // k_idx: 0-255 (uint8), q_idx: 0-63 (uint6)
    std::vector<std::vector<std::vector<ap_int<8>>>> lookup_tables(
        HEAD_DIM_DIV_2, 
        std::vector<std::vector<ap_int<8>>>(
            K_CODEBOOK_SIZE, 
            std::vector<ap_int<8>>(Q_CODEBOOK_SIZE)
        )
    );
    for (int d = 0; d < HEAD_DIM_DIV_2; d++) {
        for (int k = 0; k < K_CODEBOOK_SIZE; k++) {
            for (int q = 0; q < Q_CODEBOOK_SIZE; q++) {
                lookup_tables[d][k][q] = ap_int<8>(dis_lut(gen));
            }
        }
    }
    
    // Generate random weights (16 weights for 16 heads)
    std::vector<float> weights(NUM_INDEX_HEAD);
    for (int h = 0; h < NUM_INDEX_HEAD; h++) {
        weights[h] = dis_float(gen);
    }
    
    // Prepare hardware inputs
    // Layout for k_lut_vec_mem (8 channels):
    // 1. First: Key indices [L][HEAD_DIM_DIV_2] packed as vec_t<ap_uint<8>, 64>
    //    - From read_k_lut: k_ind_size = L * (HEAD_DIM >> 10) = L * 128/1024 = L/8
    //    - Each vec_t<64> holds 64 indices, so this is L/8 lines total across all channels
    // 2. Then: Lookup tables [K_CODEBOOK_SIZE][Q_CODEBOOK_SIZE] packed
    //    - From read_k_lut: lut_size = K_CODEBOOK_SIZE * (Q_CODEBOOK_SIZE >> 6) = 256 * 1 = 256 lines
    //    - Total: 256 lines across all channels

    const int k_ind_size = (L * HEAD_DIM) >> 10;  // L/8 lines
    const int lut_size = (K_CODEBOOK_SIZE * Q_CODEBOOK_SIZE) >> 6;  // 256 lines
    const int total_k_lut_lines = k_ind_size + lut_size;
    
    // Allocate memory for 8 channels
    std::vector<std::vector<tapa::vec_t<ap_uint<8>, 64>>> k_lut_vec_hw(8);
    for (int ch = 0; ch < 8; ch++) {
        k_lut_vec_hw[ch].resize(total_k_lut_lines);
    }
    
    // Pack key indices
    // k_ind_size = L/8, so we pack all L*HEAD_DIM_DIV_2 indices into L/8 lines
    // Each line (vec_t<64>) holds 64 indices
    // Total indices = L * HEAD_DIM_DIV_2 = L * 64
    // Pack them sequentially: for each key, all its HEAD_DIM_DIV_2 dimensions in order
    int idx_line = 0;
    int idx_pos = 0;
    tapa::vec_t<ap_uint<8>, 64> packed_indices;
    
    for (int k = 0; k < L; k++) {
        for (int d = 0; d < HEAD_DIM_DIV_2; d++) {
            packed_indices[idx_pos] = key_indices[k][d];
            idx_pos++;
            
            if (idx_pos == 64) {
                int ch = idx_line % 8;
                int idx = idx_line / 8;
                k_lut_vec_hw[ch][idx] = packed_indices;
                idx_line++;
                idx_pos = 0;
            }
        }
    }
    
    // Pack lookup tables
    // lut_size = K_CODEBOOK_SIZE * (Q_CODEBOOK_SIZE >> 6) = 256 * 1 = 256
    // Each line holds 64 values
    // For each k_code (256 total), pack all Q_CODEBOOK_SIZE (64) values in one line
    // Note: Hardware will read the same LUT for all PEs, they'll use it with different q_indices
    for (int k_code = 0; k_code < K_CODEBOOK_SIZE; k_code++) {
        tapa::vec_t<ap_uint<8>, 64> packed_lut;
        for (int q_idx = 0; q_idx < Q_CODEBOOK_SIZE; q_idx++) {
            // Use dimension 0 as reference; hardware replicates this for all dimensions
            ap_int<8> lut_val = lookup_tables[0][k_code][q_idx];
            packed_lut[q_idx] = tapa::bit_cast<ap_uint<8>>(lut_val);
        }
        int global_line = k_ind_size + k_code;
        int ch = global_line % 8;
        int idx = global_line / 8;
        k_lut_vec_hw[ch][idx] = packed_lut;
    }
    
    // Layout for q_vec_mem (1 channel):
    // Query indices [NUM_INDEX_HEAD][HEAD_DIM_DIV_2] packed as vec_t<ap_uint<8>, 64>
    // Each line contains 64 elements: pack across heads for same dimension
    // But wait, we only have 16 heads, so we need to pack differently
    // Looking at read_q_vec: total_size = NUM_INDEX_HEAD * (HEAD_DIM >> 7) = 16 * 1 = 16
    // So we need 16 lines, each with 64 elements
    // For each head, we have HEAD_DIM_DIV_2 = 64 indices
    // Pack them as: each line has all 64 dimensions for one head (but we need to fit in vec_t<64>)
    // Actually, HEAD_DIM >> 7 = 128 >> 7 = 1, so total_size = 16 * 1 = 16 lines
    // This means: pack HEAD_DIM_DIV_2 = 64 indices per line

    const int total_q_lines = (NUM_INDEX_HEAD * HEAD_DIM) >> 7;  // = 16 * 1 = 16
    std::vector<tapa::vec_t<ap_uint<8>, 64>> q_vec_hw(total_q_lines);
    
    for (int h = 0; h < NUM_INDEX_HEAD; h++) {
        tapa::vec_t<ap_uint<8>, 64> packed;
        for (int d = 0; d < HEAD_DIM_DIV_2; d++) {
            packed[d] = query_indices[h][d];
        }
        q_vec_hw[h] = packed;
    }
    
    // Pack weights into hardware format
    std::vector<tapa::vec_t<float, 16>> weight_hw(1);
    for (int i = 0; i < NUM_INDEX_HEAD; i++) {
        weight_hw[0][i] = weights[i];
    }
    
    // Allocate output memory
    std::vector<tapa::vec_t<int, 16>> topk_id_hw(TOP_K / 16);
    
    std::cout << "\nRunning hardware kernel..." << std::endl;
    std::cout << "k_lut_vec total lines: " << total_k_lut_lines << " (" << total_k_lut_lines << " per channel)" << std::endl;
    std::cout << "q_vec total lines: " << total_q_lines << std::endl;
    
    // Invoke the kernel
    int64_t kernel_time_ns = 0;
    kernel_time_ns = tapa::invoke(indexer_top, FLAGS_bitstream,
                 L,
                 tapa::read_only_mmaps<tapa::vec_t<ap_uint<8>, 64>, 8>(k_lut_vec_hw),
                 tapa::read_only_mmap<tapa::vec_t<ap_uint<8>, 64>>(q_vec_hw),
                 tapa::read_only_mmap<tapa::vec_t<float, 16>>(weight_hw),
                 tapa::write_only_mmap<tapa::vec_t<int, 16>>(topk_id_hw));
    
    std::cout << "Hardware kernel completed." << std::endl;
    std::clog << "kernel time: " << kernel_time_ns * 1e-9 << " s" << std::endl;
    
    // Extract hardware results
    std::vector<int> topk_ids_hw(TOP_K);
    for (int i = 0; i < TOP_K / 16; i++) {
        for (int j = 0; j < 16; j++) {
            topk_ids_hw[i * 16 + j] = topk_id_hw[i][j];
        }
    }
    
    // // Compute software reference
    // std::cout << "\nRunning software reference..." << std::endl;
    
    // std::vector<int> topk_ids_sw;
    // indexer_lut_ref(L, query_indices, key_indices, lookup_tables, weights, topk_ids_sw);
    
    // std::cout << "Software reference completed." << std::endl;
    
    // // Compare results
    // std::cout << "\n=== Results ===" << std::endl;
    // std::cout << "Hardware Top-64 IDs:" << std::endl;
    // for (int i = 0; i < std::min(64, TOP_K); i++) {
    //     std::cout << topk_ids_hw[i] << " ";
    // }
    // std::cout << "..." << std::endl;
    
    // std::cout << "\nSoftware Top-64 IDs:" << std::endl;
    // for (int i = 0; i < std::min(64, TOP_K); i++) {
    //     std::cout << topk_ids_sw[i] << " ";
    // }
    // std::cout << "..." << std::endl;
    
    // // Check correctness
    // // Since the top-K order might vary slightly due to ties, we check if the sets overlap significantly
    // std::set<int> hw_set(topk_ids_hw.begin(), topk_ids_hw.end());
    // std::set<int> sw_set(topk_ids_sw.begin(), topk_ids_sw.end());
    
    // int matches = 0;
    // for (int id : hw_set) {
    //     if (sw_set.count(id)) {
    //         matches++;
    //     }
    // }
    
    // float overlap_ratio = static_cast<float>(matches) / TOP_K;
    // std::cout << "\nOverlap: " << matches << "/" << TOP_K << " (" << (overlap_ratio * 100) << "%)" << std::endl;
    
    // // Check if exact match
    // bool exact_match = true;
    // for (int i = 0; i < TOP_K; i++) {
    //     if (topk_ids_hw[i] != topk_ids_sw[i]) {
    //         exact_match = false;
    //         break;
    //     }
    // }
    
    // if (exact_match) {
    //     std::cout << "\n✓ PASSED: Exact match between hardware and software!" << std::endl;
    // } else if (overlap_ratio >= 0.95) {
    //     std::cout << "\n✓ PASSED: High overlap (>95%) between hardware and software!" << std::endl;
    //     std::cout << "  (Small differences may be due to floating point precision and tie-breaking)" << std::endl;
    // } else {
    //     std::cout << "\n✗ FAILED: Significant mismatch between hardware and software!" << std::endl;
    //     return 1;
    // }


    // transfer and start GPU kernel
    std::cout << "\n=== GPU GEMV Kernel ===" << std::endl;
    
    // Define matrix dimensions
    const int K_rows = L;
    const int K_cols = 2048;
    const int Q_size = 2048;
    const int K_sparse_rows = 64;
    
    // Generate random floating point input data
    std::uniform_real_distribution<float> dis_gemv(-1.0f, 1.0f);
    std::vector<float> K_full;
    K_full.reserve(K_rows * K_cols);
    std::vector<float> Q(Q_size);
    
    for (int i = 0; i < K_rows * K_cols; i++) {
        K_full[i] = dis_gemv(gen);
    }
    
    for (int i = 0; i < Q_size; i++) {
        Q[i] = dis_gemv(gen);
    }

    std::cout << "Move data to GPU and run selection kernel..." << std::endl;
    
    // Allocate device memory for full K matrix, indices, and Q vector
    float *d_K_full, *d_K_sparse, *d_Q, *d_result;
    int *d_topk_ids;
    
    HIP_CHECK(hipMalloc(&d_K_full, K_rows * K_cols * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_K_sparse, K_sparse_rows * K_cols * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_Q, Q_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_result, K_sparse_rows * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_topk_ids, K_sparse_rows * sizeof(int)));
    
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    
    // Copy K_full, Q, and indices to device
    HIP_CHECK(hipMemcpy(d_K_full, K_full.data(), K_rows * K_cols * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_Q, Q.data(), Q_size * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipEventRecord(start));
    HIP_CHECK(hipMemcpy(d_topk_ids, topk_ids_hw.data(), K_sparse_rows * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    
    float memcpy_time_ms = 0;
    HIP_CHECK(hipEventElapsedTime(&memcpy_time_ms, start, stop));
    std::cout << "H2D memcpy time: " << memcpy_time_ms << " ms" << std::endl;
    
    // Launch GPU kernel to select 64 subvectors
    dim3 blockDim(16, 16);  // 16x16 = 256 threads per block
    dim3 gridDim((K_sparse_rows + blockDim.x - 1) / blockDim.x,
                 (K_cols + blockDim.y - 1) / blockDim.y);
    
    // Warm-up run
    hipLaunchKernelGGL(select_vectors_kernel, gridDim, blockDim, 0, 0,
                       d_K_full, d_topk_ids, d_K_sparse, K_cols, K_sparse_rows, K_rows);
    HIP_CHECK(hipDeviceSynchronize());
    
    // Measure selection kernel time
    HIP_CHECK(hipEventRecord(start));
    hipLaunchKernelGGL(select_vectors_kernel, gridDim, blockDim, 0, 0,
                       d_K_full, d_topk_ids, d_K_sparse, K_cols, K_sparse_rows, K_rows);
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    
    float selection_time_ms = 0;
    HIP_CHECK(hipEventElapsedTime(&selection_time_ms, start, stop));
    std::cout << "GPU subvector selection time: " << selection_time_ms << " ms" << std::endl;
    
    // Free K_full on device as we don't need it anymore
    HIP_CHECK(hipFree(d_K_full));
    HIP_CHECK(hipFree(d_topk_ids));

    // Create rocBLAS handle
    rocblas_handle handle;
    RBLAS_CHECK(rocblas_create_handle(&handle));
    
    // GEMV parameters
    // rocBLAS expects column-major matrices, but we have row-major
    // For row-major A * x, we compute: (A^T)^T * x = (x^T * A^T)^T
    // Or use rocblas_operation_transpose to treat row-major as column-major transposed
    // y = alpha * A * x + beta * y
    // With transpose: y = alpha * A^T * x + beta * y (where A^T in column-major = A in row-major)
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // For row-major matrix stored as [rows][cols], we need to:
    // 1. Use rocblas_operation_transpose to indicate the matrix is transposed
    // 2. Swap m and n dimensions
    // 3. Set lda to the number of rows (since in memory, consecutive elements in a row are K_sparse_rows apart in column-major view)
    
    // Warm-up run
    RBLAS_CHECK(rocblas_sgemv(handle, rocblas_operation_transpose,
                  K_cols, K_sparse_rows,  // n, m (swapped for transpose)
                  &alpha,
                  d_K_sparse, K_cols,  // lda = K_cols (leading dimension in row-major storage)
                  d_Q, 1,
                  &beta,
                  d_result, 1));
    HIP_CHECK(hipDeviceSynchronize());
    
    // Measure kernel time
    
    HIP_CHECK(hipEventRecord(start));
    RBLAS_CHECK(rocblas_sgemv(handle, rocblas_operation_transpose,
                  K_cols, K_sparse_rows,
                  &alpha,
                  d_K_sparse, K_cols,
                  d_Q, 1,
                  &beta,
                  d_result, 1));
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    
    float gemv_time_ms = 0;
    HIP_CHECK(hipEventElapsedTime(&gemv_time_ms, start, stop));
    
    std::cout << "GEMV kernel time: " << gemv_time_ms << " ms" << std::endl;
    std::cout << "GEMV throughput: " << (2.0 * K_sparse_rows * K_cols / gemv_time_ms / 1e6) << " GFLOPS" << std::endl;
    
    // Copy result back to host
    std::vector<float> result(K_sparse_rows);
    HIP_CHECK(hipMemcpy(result.data(), d_result, K_sparse_rows * sizeof(float), hipMemcpyDeviceToHost));
    
    // Print first few results
    std::cout << "First 10 GEMV results: ";
    for (int i = 0; i < std::min(10, K_sparse_rows); i++) {
        std::cout << result[i] << " ";
    }
    std::cout << std::endl;
    
    // Cleanup
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    RBLAS_CHECK(rocblas_destroy_handle(handle));
    HIP_CHECK(hipFree(d_K_sparse));
    HIP_CHECK(hipFree(d_Q));
    HIP_CHECK(hipFree(d_result));
    
    std::cout << "\nGPU GEMV kernel completed successfully!" << std::endl;

    std::cout << "\nSummary:" << std::endl;
    std::cout << "FPGA kernel time: " << kernel_time_ns * 1e-6 << " ms" << std::endl;
    std::cout << "PCIe time: " << memcpy_time_ms << " ms" << std::endl;
    std::cout << "GPU sparse time: " << selection_time_ms << " ms" << std::endl;
    std::cout << "GPU GEMV time: " << gemv_time_ms << " ms" << std::endl;

    std::cout << "e2e latency: " << (kernel_time_ns * 1e-6 + memcpy_time_ms + selection_time_ms + gemv_time_ms) << " ms" << std::endl;
    return 0;
}
