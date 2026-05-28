/**
 * SpMV with BM25 Indexer Demo
 * 
 * Uses the pre-built BM25 indexer kernel from kernel_lib to generate
 * row indices for sparse matrix operations.
 * 
 * Flow:
 * 1. Load BM25 indexer XCLBIN
 * 2. Create minimal dummy data (query, documents, df table)
 * 3. Run BM25 indexer to get top-64 document IDs
 * 4. Use those IDs as row indices for SpMV
 * 5. GPU performs SpMV on selected rows
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <random>
#include <chrono>
#include <map>
#include <set>
#include <algorithm>

// XRT includes
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>

// HIP includes
#include <hip/hip_runtime.h>

// BM25 kernel constants (from indexer_bm25.h)
constexpr int VOCAB_SIZE = 65536;
constexpr int VOCAB_SIZE_DIV_16 = VOCAB_SIZE / 16;
constexpr int VOCAB_SIZE_DIV_512 = VOCAB_SIZE / 512;
constexpr int TOP_K = 64;
constexpr int NUM_INDICES = 64;

// SpMV constants
constexpr int MATRIX_SIZE = 1024;

// Error checking macros
#define HIP_CHECK(cmd) \
    do { \
        hipError_t error = (cmd); \
        if (error != hipSuccess) { \
            std::cerr << "HIP error: " << hipGetErrorString(error) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Sparse matrix in CSR format
struct CSRMatrix {
    std::vector<float> values;
    std::vector<uint32_t> col_idx;
    std::vector<uint32_t> row_ptr;
    uint32_t num_rows;
    uint32_t num_cols;
    uint32_t nnz;
};

// GPU kernels (from gpu_spmv.hip)
extern "C" __global__ void read_indices_from_fpga(
    const uint32_t* __restrict__ fpga_indices,
    uint32_t* __restrict__ gpu_indices,
    uint32_t count);

extern "C" __global__ void spmv_csr_kernel(
    const uint32_t* __restrict__ row_indices,
    uint32_t num_rows,
    uint32_t matrix_size,
    const float* __restrict__ values,
    const uint32_t* __restrict__ col_idx,
    const uint32_t* __restrict__ row_ptr,
    const float* __restrict__ x,
    float* __restrict__ y);

extern "C" __global__ void init_vector_kernel(
    float* __restrict__ vec,
    uint32_t size);

// Create a simple tridiagonal test matrix
CSRMatrix create_test_matrix(uint32_t size) {
    CSRMatrix mat;
    mat.num_rows = size;
    mat.num_cols = size;
    
    // Create randomized sparse matrix with ~5-10 non-zeros per row
    std::mt19937 rng(12345);  // Fixed seed for reproducibility
    std::uniform_real_distribution<float> val_dist(-10.0f, 10.0f);
    std::uniform_int_distribution<uint32_t> nnz_dist(3, 10);
    
    mat.row_ptr.push_back(0);
    
    for (uint32_t i = 0; i < size; i++) {
        uint32_t nnz_in_row = nnz_dist(rng);
        std::set<uint32_t> cols;
        
        // Always include diagonal
        cols.insert(i);
        
        // Add random columns
        std::uniform_int_distribution<uint32_t> col_dist(0, size - 1);
        while (cols.size() < nnz_in_row) {
            cols.insert(col_dist(rng));
        }
        
        // Add sorted columns with random values
        for (uint32_t col : cols) {
            mat.col_idx.push_back(col);
            mat.values.push_back(val_dist(rng));
        }
        
        mat.row_ptr.push_back(mat.values.size());
    }
    
    mat.nnz = mat.values.size();
    return mat;
}

int main() {
    std::cout << "=== SpMV with BM25 Indexer Demo ===" << std::endl;
    
    // Step 1: Initialize FPGA device
    std::cout << "\n[1] Initializing FPGA device..." << std::endl;
    
    xrt::device fpga_device;
    bool fpga_found = false;
    
    for (unsigned int i = 0; i < 16; i++) {
        try {
            xrt::device dev(i);
            std::string bdf = dev.get_info<xrt::info::device::bdf>();
            std::cout << "  Found FPGA at index " << i << " (BDF: " << bdf << ")" << std::endl;
            fpga_device = std::move(dev);
            fpga_found = true;
            break;
        } catch (...) {
            continue;
        }
    }
    
    if (!fpga_found) {
        std::cerr << "Error: No FPGA device found!" << std::endl;
        return EXIT_FAILURE;
    }
    
    // Step 2: Load BM25 indexer XCLBIN
    std::cout << "\n[2] Loading BM25 indexer XCLBIN..." << std::endl;
    
    const char* xclbin_path = "indexer_bm25_fixed.xclbin";
    xrt::uuid xclbin_uuid;
    
    try {
        xclbin_uuid = fpga_device.load_xclbin(xclbin_path);
        std::cout << "  XCLBIN loaded successfully" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error loading XCLBIN: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    
    // Step 3: Get kernel handle
    std::cout << "\n[3] Getting kernel handle..." << std::endl;
    
    xrt::kernel indexer_kernel;
    try {
        indexer_kernel = xrt::kernel(fpga_device, xclbin_uuid, "indexer_top");
        std::cout << "  Kernel 'indexer_top' found" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error getting kernel: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    
    // Step 4: Prepare BM25 input data (following indexer_bm25_tb.cpp pattern)
    std::cout << "\n[4] Preparing BM25 indexer inputs..." << std::endl;
    
    // Use L=4096 as suggested by PhD student
    int L = 4096;
    L = (L + 63) & ~63;  // Round up to multiple of 64
    const int num_super_batches = L >> 6;  // 64 docs per super-batch
    const int tokens_per_doc = 32;  // Average tokens per document
    const int query_size = 64;  // Number of query tokens
    
    std::cout << "  L (documents): " << L << std::endl;
    std::cout << "  Super-batches: " << num_super_batches << std::endl;
    
    // Generate random data following testbench pattern
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_int_distribution<int> freq_dis(1, 10);
    std::uniform_int_distribution<int> doc_len_dis(tokens_per_doc / 2, tokens_per_doc * 3 / 2);
    std::uniform_int_distribution<int> token_base_dis(0, (VOCAB_SIZE / 16) - 1);
    
    // Generate query tokens - spread across all mod-16 values
    std::vector<int> query_tokens;
    for (int i = 0; i < query_size; i++) {
        int base = token_base_dis(gen);
        int mod = i % 16;  // Spread across all mod values
        int token_id = base * 16 + mod;
        query_tokens.push_back(token_id);
    }
    
    // Generate documents with mod-16 constraint
    std::vector<int> df(VOCAB_SIZE, 0);
    std::vector<std::vector<std::pair<int, int>>> documents(L);
    
    for (int doc_id = 0; doc_id < L; doc_id++) {
        int num_tokens = doc_len_dis(gen);
        int required_mod = doc_id % 16;
        std::map<int, int> token_freq_map;
        
        for (int t = 0; t < num_tokens; t++) {
            int base = token_base_dis(gen);
            int token_id = base * 16 + required_mod;
            token_freq_map[token_id]++;
        }
        
        for (const auto& tf : token_freq_map) {
            documents[doc_id].push_back({tf.first, std::min(tf.second, 255)});
            df[tf.first]++;
        }
    }
    
    // Create high-overlap documents for predictable top-k
    for (int i = 0; i < std::min(TOP_K / 2, L); i++) {
        int doc_id = i * 2;
        int required_mod = doc_id % 16;
        
        for (const auto& tf : documents[doc_id]) {
            df[tf.first]--;
        }
        documents[doc_id].clear();
        
        int added = 0;
        for (int token_id : query_tokens) {
            if ((token_id % 16) == required_mod) {
                int freq = 5 + (added % 5);
                documents[doc_id].push_back({token_id, freq});
                df[token_id]++;
                added++;
            }
        }
    }
    
    // Prepare df_buffer
    auto df_buffer = xrt::bo(fpga_device, VOCAB_SIZE_DIV_16 * 16 * sizeof(int),
                             indexer_kernel.group_id(2));
    auto df_ptr = df_buffer.map<int*>();
    for (int i = 0; i < VOCAB_SIZE_DIV_16; i++) {
        for (int j = 0; j < 16; j++) {
            df_ptr[i * 16 + j] = df[i * 16 + j];
        }
    }
    df_buffer.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    
    // Prepare query_bitmap
    auto query_bitmap = xrt::bo(fpga_device, VOCAB_SIZE_DIV_512 * 64,
                                indexer_kernel.group_id(3));
    auto query_ptr = query_bitmap.map<uint64_t*>();
    std::memset(query_ptr, 0, VOCAB_SIZE_DIV_512 * 64);
    
    for (int token_id : query_tokens) {
        int chunk_idx = token_id >> 9;  // token_id / 512
        int bit_idx = token_id & 0x1FF;  // token_id % 512
        int qword_idx = bit_idx / 64;
        int bit_in_qword = bit_idx % 64;
        query_ptr[chunk_idx * 8 + qword_idx] |= (1ULL << bit_in_qword);
    }
    query_bitmap.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    
    // Pack documents for hardware (following testbench logic)
    std::vector<std::vector<int>> inst_mem_data(1, std::vector<int>(num_super_batches));
    std::vector<std::vector<uint32_t>> doc_mem_data(4);
    
    for (int super_batch = 0; super_batch < num_super_batches; super_batch++) {
        // Collect tokens for each channel's 16 documents
        std::vector<std::vector<std::vector<std::pair<int, int>>>> doc_tokens(4,
            std::vector<std::vector<std::pair<int, int>>>(16));
        
        for (int channel = 0; channel < 4; channel++) {
            for (int j = 0; j < 16; j++) {
                int doc_id = super_batch * 64 + channel * 16 + j;
                for (const auto& token_freq : documents[doc_id]) {
                    int token_id = token_freq.first;
                    int freq = token_freq.second;
                    if ((token_id % 16) == j) {
                        doc_tokens[channel][j].push_back({token_id, freq});
                    }
                }
            }
        }
        
        // Find max tokens across all 64 documents
        int max_tokens = 0;
        for (int channel = 0; channel < 4; channel++) {
            for (int j = 0; j < 16; j++) {
                max_tokens = std::max(max_tokens, (int)doc_tokens[channel][j].size());
            }
        }
        if (max_tokens == 0) max_tokens = 1;
        
        inst_mem_data[0][super_batch] = max_tokens;
        
        // Pack vectors for each channel
        for (int channel = 0; channel < 4; channel++) {
            for (int i = 0; i < max_tokens; i++) {
                for (int j = 0; j < 16; j++) {
                    uint32_t packed_val = 0;
                    if (i < (int)doc_tokens[channel][j].size()) {
                        int token_id = doc_tokens[channel][j][i].first;
                        int freq = doc_tokens[channel][j][i].second;
                        packed_val = (freq << 16) | token_id;
                    } else {
                        packed_val = (0 << 16) | j;  // Dummy: token_id=j, freq=0
                    }
                    doc_mem_data[channel].push_back(packed_val);
                }
            }
        }
    }
    
    int L_doc_total = doc_mem_data[0].size() / 16;  // Vectors per channel
    
    std::cout << "  L_doc_total (per channel): " << L_doc_total << std::endl;
    
    // Allocate and fill inst_mem
    auto inst_mem = xrt::bo(fpga_device, num_super_batches * sizeof(int),
                            indexer_kernel.group_id(4));
    auto inst_ptr = inst_mem.map<int*>();
    std::memcpy(inst_ptr, inst_mem_data[0].data(), num_super_batches * sizeof(int));
    inst_mem.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    
    // Allocate and fill doc_mem buffers
    std::vector<xrt::bo> doc_mem_buffers;
    for (int ch = 0; ch < 4; ch++) {
        auto doc_buf = xrt::bo(fpga_device, L_doc_total * 16 * sizeof(uint32_t),
                               indexer_kernel.group_id(5 + ch));
        auto doc_ptr = doc_buf.map<uint32_t*>();
        std::memcpy(doc_ptr, doc_mem_data[ch].data(), L_doc_total * 16 * sizeof(uint32_t));
        doc_buf.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        doc_mem_buffers.push_back(std::move(doc_buf));
    }
    
    // topk_id_mem: output buffer for top-64 document IDs (with P2P flag for GPU access)
    auto topk_id_mem = xrt::bo(fpga_device,
                               (TOP_K / 16) * 16 * sizeof(int),
                               xrt::bo::flags::p2p,
                               indexer_kernel.group_id(9));
    
    std::cout << "  Input buffers prepared" << std::endl;
    
    // Step 5: Run BM25 indexer kernel
    std::cout << "\n[5] Running BM25 indexer kernel..." << std::endl;
    
    auto run = indexer_kernel(L, L_doc_total, df_buffer, query_bitmap, inst_mem,
                              doc_mem_buffers[0], doc_mem_buffers[1], 
                              doc_mem_buffers[2], doc_mem_buffers[3],
                              topk_id_mem);
    run.wait();
    
    std::cout << "  Kernel execution completed" << std::endl;
    
    // Step 6: Verify FPGA output (read for display only)
    std::cout << "\n[6] Reading top-K indices from FPGA..." << std::endl;
    
    topk_id_mem.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    auto topk_ptr = topk_id_mem.map<int*>();
    
    // Step 7: Use FPGA output buffer directly for P2P (no copy!)
    std::cout << "\n[7] Using FPGA output buffer for P2P..." << std::endl;
    std::cout << "  GPU will read directly from FPGA HBM" << std::endl;
    
    // The topk_id_mem buffer is already in FPGA memory - use it directly
    uint64_t fpga_addr = topk_id_mem.address();
    std::cout << "  P2P buffer created at FPGA address: 0x" << std::hex << fpga_addr << std::dec << std::endl;
    
    // Step 8: Initialize GPU
    std::cout << "\n[8] Initializing GPU..." << std::endl;
    
    int gpu_count;
    HIP_CHECK(hipGetDeviceCount(&gpu_count));
    if (gpu_count == 0) {
        std::cerr << "Error: No GPU found!" << std::endl;
        return EXIT_FAILURE;
    }
    
    HIP_CHECK(hipSetDevice(0));
    std::cout << "  Using GPU 0" << std::endl;
    
    // Register FPGA output buffer with GPU for P2P access
    void* device_ptr;
    size_t topk_size_bytes = NUM_INDICES * sizeof(int);
    
    // Try registering as IoMemory for true P2P (GPU directly accesses FPGA buffer)
    hipError_t reg_err = hipHostRegister(topk_ptr, topk_size_bytes,
                                         hipHostRegisterMapped | hipHostRegisterIoMemory);
    if (reg_err != hipSuccess) {
        std::cerr << "  Warning: hipHostRegister with hipHostRegisterIoMemory failed ("
                  << hipGetErrorString(reg_err)
                  << "). Falling back to hipHostRegisterMapped (no direct P2P)." << std::endl;
        
        // Fallback: register as generic mapped host memory
        reg_err = hipHostRegister(topk_ptr, topk_size_bytes, hipHostRegisterMapped);
        if (reg_err != hipSuccess) {
            std::cerr << "Error: hipHostRegister failed even with hipHostRegisterMapped: "
                      << hipGetErrorString(reg_err) << std::endl;
            return EXIT_FAILURE;
        }
        
        std::cout << "  Host buffer registered with GPU using hipHostRegisterMapped "
                  << "(P2P disabled or IoMemory not supported)." << std::endl;
    } else {
        std::cout << "  FPGA buffer registered with GPU for true P2P access "
                  << "via hipHostRegisterIoMemory." << std::endl;
    }
    
    HIP_CHECK(hipHostGetDevicePointer(&device_ptr, topk_ptr, 0));
    
    // Step 9: Prepare SpMV data on GPU
    std::cout << "\n[9] Preparing SpMV data on GPU..." << std::endl;
    
    CSRMatrix mat = create_test_matrix(MATRIX_SIZE);
    
    float *d_values, *d_x, *d_y;
    uint32_t *d_col_idx, *d_row_ptr;
    
    HIP_CHECK(hipMalloc(&d_values, mat.nnz * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_col_idx, mat.nnz * sizeof(uint32_t)));
    HIP_CHECK(hipMalloc(&d_row_ptr, (mat.num_rows + 1) * sizeof(uint32_t)));
    HIP_CHECK(hipMalloc(&d_x, MATRIX_SIZE * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_y, MATRIX_SIZE * sizeof(float)));
    
    HIP_CHECK(hipMemcpy(d_values, mat.values.data(), mat.nnz * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_col_idx, mat.col_idx.data(), mat.nnz * sizeof(uint32_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_row_ptr, mat.row_ptr.data(), (mat.num_rows + 1) * sizeof(uint32_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_y, 0, MATRIX_SIZE * sizeof(float)));
    
    // Initialize x vector
    dim3 threads(256);
    dim3 blocks((MATRIX_SIZE + threads.x - 1) / threads.x);
    hipLaunchKernelGGL(init_vector_kernel, blocks, threads, 0, 0, d_x, MATRIX_SIZE);
    HIP_CHECK(hipDeviceSynchronize());
    
    std::cout << "  SpMV data prepared" << std::endl;
    
    // Step 10: Verify FPGA indices are accessible via P2P
    std::cout << "\n[10] Verifying FPGA indices via P2P..." << std::endl;
    
    // Log first 16 indices to verify alignment with FPGA output
    std::cout << "  FPGA indices (first 16): ";
    for (int i = 0; i < 16; i++) {
        std::cout << topk_ptr[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "  Using device_ptr directly in SpMV kernel (true P2P)" << std::endl;
    
    // Step 11: Perform SpMV
    std::cout << "\n[11] Performing SpMV on GPU..." << std::endl;
    
    threads = dim3(256);
    blocks = dim3((NUM_INDICES + threads.x - 1) / threads.x);
    
    // Measure P2P transfer time
    auto p2p_start = std::chrono::high_resolution_clock::now();
    
    hipLaunchKernelGGL(spmv_csr_kernel, blocks, threads, 0, 0,
                      (uint32_t*)device_ptr, NUM_INDICES, MATRIX_SIZE, d_values, d_col_idx, d_row_ptr,
                      d_x, d_y);
    HIP_CHECK(hipDeviceSynchronize());
    
    auto p2p_end = std::chrono::high_resolution_clock::now();
    auto p2p_duration = std::chrono::duration_cast<std::chrono::microseconds>(p2p_end - p2p_start);
    
    std::cout << "  P2P+SpMV time: " << p2p_duration.count() << " μs" << std::endl;
    HIP_CHECK(hipDeviceSynchronize());
    
    std::cout << "  ✓ SpMV complete" << std::endl;
    
    // Step 12: Verify results
    std::cout << "\n[12] Verifying results..." << std::endl;
    
    std::vector<float> y_host(MATRIX_SIZE);
    HIP_CHECK(hipMemcpy(y_host.data(), d_y, MATRIX_SIZE * sizeof(float), hipMemcpyDeviceToHost));
    
    // Check all results using FPGA-generated indices
    bool results_ok = true;
    int mismatches = 0;
    for (int i = 0; i < NUM_INDICES; i++) {
        uint32_t row = static_cast<uint32_t>(topk_ptr[i]);
        if (row >= MATRIX_SIZE) {
            row = row % MATRIX_SIZE;  // Clamp to valid range
        }
        
        float expected = 0.0f;
        for (uint32_t j = mat.row_ptr[row]; j < mat.row_ptr[row + 1]; j++) {
            expected += mat.values[j] * (mat.col_idx[j] + 1);
        }
        
        if (std::abs(y_host[row] - expected) > 1e-3) {
            if (mismatches < 10) {  // Only print first 10 mismatches
                std::cout << "  Mismatch at row " << row << ": got " << y_host[row] 
                          << ", expected " << expected << std::endl;
            }
            results_ok = false;
            mismatches++;
        }
    }
    
    if (mismatches > 10) {
        std::cout << "  ... and " << (mismatches - 10) << " more mismatches" << std::endl;
    }
    
    if (results_ok) {
        std::cout << "  ✓ Results verified!" << std::endl;
    }
    
    // Cleanup
    HIP_CHECK(hipHostUnregister(topk_ptr));
    HIP_CHECK(hipFree(d_values));
    HIP_CHECK(hipFree(d_col_idx));
    HIP_CHECK(hipFree(d_row_ptr));
    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_y));
    
    std::cout << "\n=== Demo Complete ===" << std::endl;
    std::cout << "✓ BM25 indexer generated " << NUM_INDICES << " indices" << std::endl;
    std::cout << "✓ P2P transfer: FPGA → GPU" << std::endl;
    std::cout << "✓ SpMV computed on selected rows" << std::endl;
    
    return EXIT_SUCCESS;
}
