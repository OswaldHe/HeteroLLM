/**
 * engram_gpu_test.cpp
 * GPU-only testbench for Engram GPU kernels:
 *   1. rms_norm_gate_kernel
 *   2. short_conv_kernel
 *   3. residual_add_kernel
 *
 * Usage: ./build/engram_gpu_test [L]
 *        ./build/engram_gpu_test --L=<seq_len>
 *
 * L defaults to 256. L is the sequence length.
 */

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <random>
#include <cstdint>

// Constants (must match engram_kernels.hip)
#define HIDDEN_DIM     1024
#define NUM_GROUP      4
#define CONV_KERNEL_SZ 4
#define CONV_DILATION  3
#define RMS_EPS        1e-5f

// Forward-declare GPU kernels (defined in engram_kernels.hip)
extern "C" __global__ void rms_norm_gate_kernel(
    const float* __restrict__ fpga_keys,
    const float* __restrict__ fpga_value,
    const float* __restrict__ hidden_states,
    const float* __restrict__ norm1_weight,
    const float* __restrict__ norm2_weight,
    float* __restrict__ gated_value,
    float* __restrict__ gates_out,
    uint32_t L);

extern "C" __global__ void short_conv_kernel(
    const float* __restrict__ gated_value,
    const float* __restrict__ conv_weight,
    const float* __restrict__ conv_norm_weight,
    float* __restrict__ conv_output,
    uint32_t L);

extern "C" __global__ void residual_add_kernel(
    const float* __restrict__ gated_value,
    const float* __restrict__ conv_output,
    const float* __restrict__ hidden_states,
    float* __restrict__ output,
    uint32_t L);

// ============================================================================
// Utilities
// ============================================================================

#define HIP_CHECK(cmd) do { \
    hipError_t _e = (cmd); \
    if (_e != hipSuccess) { \
        fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__, \
                hipGetErrorString(_e)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

static bool verify(const char* name,
                   const std::vector<float>& sw,
                   const std::vector<float>& gpu,
                   float tol = 1e-3f)
{
    int mismatches = 0;
    float max_err = 0.0f;
    for (size_t i = 0; i < sw.size(); i++) {
        float err = fabsf(sw[i] - gpu[i]);
        if (err > max_err) max_err = err;
        if (err > tol) {
            if (mismatches < 5)
                printf("    [%zu] sw=%.6f gpu=%.6f err=%.2e\n", i, sw[i], gpu[i], err);
            mismatches++;
        }
    }
    bool pass = (mismatches == 0);
    printf("  %-26s  max_err=%.2e  mismatches=%d/%zu  %s\n",
           name, max_err, mismatches, sw.size(), pass ? "PASS" : "FAIL");
    return pass;
}

// ============================================================================
// Software reference implementations
// ============================================================================

// Kernel 1 reference: RMSNorm + gating
static void sw_rms_norm_gate(
    const std::vector<float>& fpga_keys,      // [L, G, D]
    const std::vector<float>& fpga_value,     // [L, D]
    const std::vector<float>& hidden_states,  // [L, G, D]
    const std::vector<float>& norm1_weight,   // [G, D]
    const std::vector<float>& norm2_weight,   // [G, D]
    std::vector<float>& gated_value,          // [L, G, D]
    std::vector<float>& gates_out,            // [L, G]
    int L)
{
    const int D = HIDDEN_DIM, G = NUM_GROUP;
    for (int t = 0; t < L; t++) {
        for (int g = 0; g < G; g++) {
            const float* kp = fpga_keys.data()     + (t * G + g) * D;
            const float* hp = hidden_states.data() + (t * G + g) * D;
            const float* n1 = norm1_weight.data()  + g * D;
            const float* n2 = norm2_weight.data()  + g * D;

            float ksq = 0.0f, hsq = 0.0f;
            for (int d = 0; d < D; d++) { ksq += kp[d] * kp[d]; hsq += hp[d] * hp[d]; }
            float kinv = 1.0f / sqrtf(ksq / D + RMS_EPS);
            float hinv = 1.0f / sqrtf(hsq / D + RMS_EPS);

            float dot = 0.0f;
            for (int d = 0; d < D; d++)
                dot += kp[d] * kinv * n1[d] * hp[d] * hinv * n2[d];

            float score = dot / sqrtf((float)D);
            float sign_val = (score >= 0.0f) ? 1.0f : -1.0f;
            float abs_sc = fabsf(score);
            if (abs_sc < 1e-6f) abs_sc = 1e-6f;
            score = sqrtf(abs_sc) * sign_val;
            float gate = 1.0f / (1.0f + expf(-score));

            gates_out[t * G + g] = gate;

            const float* vp  = fpga_value.data() + t * D;
            float*       gvp = gated_value.data() + (t * G + g) * D;
            for (int d = 0; d < D; d++) gvp[d] = gate * vp[d];
        }
    }
}

// Kernel 2 reference: depthwise causal conv1d + SiLU
static void sw_short_conv(
    const std::vector<float>& gated_value,      // [L, G, D]
    const std::vector<float>& conv_weight,       // [G*D, KERNEL_SZ]
    const std::vector<float>& conv_norm_weight,  // [G, D]
    std::vector<float>& conv_output,             // [L, G, D]
    int L)
{
    const int D = HIDDEN_DIM, G = NUM_GROUP;
    for (int t = 0; t < L; t++) {
        for (int g = 0; g < G; g++) {
            for (int d = 0; d < D; d++) {
                int channel = g * D + d;
                const float* cw = conv_weight.data() + channel * CONV_KERNEL_SZ;
                float acc = 0.0f;

                for (int k = 0; k < CONV_KERNEL_SZ; k++) {
                    int src_t = t - k * CONV_DILATION;
                    if (src_t < 0) continue;

                    // Per-group RMSNorm at src_t
                    float sq = 0.0f;
                    for (int dd = 0; dd < D; dd++) {
                        float v = gated_value[src_t * G * D + g * D + dd];
                        sq += v * v;
                    }
                    float rinv = 1.0f / sqrtf(sq / D + RMS_EPS);
                    float raw    = gated_value[src_t * G * D + g * D + d];
                    float normed = raw * rinv * conv_norm_weight[g * D + d];
                    acc += cw[k] * normed;
                }

                // SiLU: x * sigmoid(x)
                float sig = 1.0f / (1.0f + expf(-acc));
                conv_output[t * G * D + g * D + d] = acc * sig;
            }
        }
    }
}

// Kernel 3 reference: element-wise residual add
static void sw_residual_add(
    const std::vector<float>& gated_value,
    const std::vector<float>& conv_output,
    const std::vector<float>& hidden_states,
    std::vector<float>& output,
    int L)
{
    int total = L * NUM_GROUP * HIDDEN_DIM;
    for (int i = 0; i < total; i++)
        output[i] = gated_value[i] + conv_output[i] + hidden_states[i];
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char* argv[])
{
    // Parse L: positional arg, --L=N, or -L N
    int L = 256;
    for (int i = 1; i < argc; i++) {
        if (strncmp(argv[i], "--L=", 4) == 0) {
            L = atoi(argv[i] + 4);
        } else if ((strcmp(argv[i], "--L") == 0 || strcmp(argv[i], "-L") == 0)
                   && i + 1 < argc) {
            L = atoi(argv[++i]);
        } else {
            int v = atoi(argv[i]);
            if (v > 0) L = v;
        }
    }
    if (L <= 0) { fprintf(stderr, "Invalid L=%d\n", L); return EXIT_FAILURE; }

    const int D = HIDDEN_DIM;
    const int G = NUM_GROUP;

    printf("=== Engram GPU Testbench ===\n");
    printf("  L=%d  G=%d  D=%d  CONV_K=%d  CONV_DIL=%d\n",
           L, G, D, CONV_KERNEL_SZ, CONV_DILATION);

    // Random data
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    auto rand_vec = [&](int n) {
        std::vector<float> v(n);
        for (auto& x : v) x = dist(gen);
        return v;
    };

    auto h_fpga_keys     = rand_vec(L * G * D);
    auto h_fpga_value    = rand_vec(L * D);
    auto h_hidden_states = rand_vec(L * G * D);
    auto h_norm1_weight  = rand_vec(G * D);
    auto h_norm2_weight  = rand_vec(G * D);
    auto h_conv_weight   = rand_vec(G * D * CONV_KERNEL_SZ);
    auto h_conv_norm_w   = rand_vec(G * D);

    // ── Software reference ──
    printf("\n[1] Running SW reference...\n");
    std::vector<float> sw_gated(L * G * D, 0.0f);
    std::vector<float> sw_gates(L * G,     0.0f);
    std::vector<float> sw_conv(L * G * D,  0.0f);
    std::vector<float> sw_out(L * G * D,   0.0f);

    sw_rms_norm_gate(h_fpga_keys, h_fpga_value, h_hidden_states,
                     h_norm1_weight, h_norm2_weight,
                     sw_gated, sw_gates, L);
    sw_short_conv(sw_gated, h_conv_weight, h_conv_norm_w, sw_conv, L);
    sw_residual_add(sw_gated, sw_conv, h_hidden_states, sw_out, L);

    // ── GPU allocations ──
    float *d_fpga_keys, *d_fpga_value, *d_hidden;
    float *d_n1w, *d_n2w;
    float *d_gated, *d_gates;
    float *d_conv_w, *d_conv_nw, *d_conv_out;
    float *d_out;

    HIP_CHECK(hipMalloc(&d_fpga_keys,  (size_t)L * G * D * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_fpga_value, (size_t)L * D     * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_hidden,     (size_t)L * G * D * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_n1w,        (size_t)G * D     * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_n2w,        (size_t)G * D     * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_gated,      (size_t)L * G * D * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_gates,      (size_t)L * G     * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_conv_w,     (size_t)G * D * CONV_KERNEL_SZ * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_conv_nw,    (size_t)G * D     * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_conv_out,   (size_t)L * G * D * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_out,        (size_t)L * G * D * sizeof(float)));

    // Copy inputs to device
    HIP_CHECK(hipMemcpy(d_fpga_keys,  h_fpga_keys.data(),     (size_t)L * G * D * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_fpga_value, h_fpga_value.data(),    (size_t)L * D     * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_hidden,     h_hidden_states.data(), (size_t)L * G * D * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_n1w,        h_norm1_weight.data(),  (size_t)G * D     * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_n2w,        h_norm2_weight.data(),  (size_t)G * D     * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_conv_w,     h_conv_weight.data(),   (size_t)G * D * CONV_KERNEL_SZ * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_conv_nw,    h_conv_norm_w.data(),   (size_t)G * D     * sizeof(float), hipMemcpyHostToDevice));

    // ── HIP event handles for latency measurement ──
    hipEvent_t ev_start, ev_stop;
    HIP_CHECK(hipEventCreate(&ev_start));
    HIP_CHECK(hipEventCreate(&ev_stop));
    float ms_gate = 0.0f, ms_conv = 0.0f, ms_add = 0.0f;

    // ── Kernel 1: rms_norm_gate ──
    printf("\n[2] Launching rms_norm_gate_kernel (grid=%d, block=256)...\n", L * G);
    HIP_CHECK(hipEventRecord(ev_start, 0));
    hipLaunchKernelGGL(rms_norm_gate_kernel,
                       dim3(L * G), dim3(256), 0, 0,
                       d_fpga_keys, d_fpga_value, d_hidden,
                       d_n1w, d_n2w,
                       d_gated, d_gates, (uint32_t)L);
    HIP_CHECK(hipEventRecord(ev_stop, 0));
    HIP_CHECK(hipEventSynchronize(ev_stop));
    HIP_CHECK(hipEventElapsedTime(&ms_gate, ev_start, ev_stop));

    // ── Kernel 2: short_conv ──
    {
        int total = L * G * D;
        int block = 256;
        int grid  = (total + block - 1) / block;
        printf("[3] Launching short_conv_kernel (grid=%d, block=%d)...\n", grid, block);
        HIP_CHECK(hipEventRecord(ev_start, 0));
        hipLaunchKernelGGL(short_conv_kernel,
                           dim3(grid), dim3(block), 0, 0,
                           d_gated, d_conv_w, d_conv_nw, d_conv_out, (uint32_t)L);
        HIP_CHECK(hipEventRecord(ev_stop, 0));
        HIP_CHECK(hipEventSynchronize(ev_stop));
        HIP_CHECK(hipEventElapsedTime(&ms_conv, ev_start, ev_stop));
    }

    // ── Kernel 3: residual_add ──
    {
        int total = L * G * D;
        int block = 256;
        int grid  = (total + block - 1) / block;
        printf("[4] Launching residual_add_kernel (grid=%d, block=%d)...\n", grid, block);
        HIP_CHECK(hipEventRecord(ev_start, 0));
        hipLaunchKernelGGL(residual_add_kernel,
                           dim3(grid), dim3(block), 0, 0,
                           d_gated, d_conv_out, d_hidden, d_out, (uint32_t)L);
        HIP_CHECK(hipEventRecord(ev_stop, 0));
        HIP_CHECK(hipEventSynchronize(ev_stop));
        HIP_CHECK(hipEventElapsedTime(&ms_add, ev_start, ev_stop));
    }

    HIP_CHECK(hipEventDestroy(ev_start));
    HIP_CHECK(hipEventDestroy(ev_stop));

    // ── Copy results back ──
    std::vector<float> gpu_gated(L * G * D);
    std::vector<float> gpu_gates(L * G);
    std::vector<float> gpu_conv(L * G * D);
    std::vector<float> gpu_out(L * G * D);

    HIP_CHECK(hipMemcpy(gpu_gated.data(), d_gated,    (size_t)L * G * D * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(gpu_gates.data(), d_gates,    (size_t)L * G     * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(gpu_conv.data(),  d_conv_out, (size_t)L * G * D * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(gpu_out.data(),   d_out,      (size_t)L * G * D * sizeof(float), hipMemcpyDeviceToHost));

    // ── Latency summary ──
    float ms_total = ms_gate + ms_conv + ms_add;
    printf("\n[5] GPU kernel latency (L=%d):\n", L);
    printf("  %-26s  %8.3f ms\n", "rms_norm_gate_kernel", ms_gate);
    printf("  %-26s  %8.3f ms\n", "short_conv_kernel",    ms_conv);
    printf("  %-26s  %8.3f ms\n", "residual_add_kernel",  ms_add);
    printf("  %-26s  %8.3f ms\n", "total",                ms_total);

    // ── Verify ──
    printf("\n[6] Verifying GPU vs SW reference...\n");
    bool pass = true;
    pass &= verify("rms_norm_gate (gated_value)", sw_gated, gpu_gated);
    pass &= verify("rms_norm_gate (gates)",       sw_gates, gpu_gates);
    pass &= verify("short_conv (conv_output)",    sw_conv,  gpu_conv);
    pass &= verify("residual_add (output)",       sw_out,   gpu_out);

    // ── Cleanup ──
    hipFree(d_fpga_keys);  hipFree(d_fpga_value); hipFree(d_hidden);
    hipFree(d_n1w);        hipFree(d_n2w);
    hipFree(d_gated);      hipFree(d_gates);
    hipFree(d_conv_w);     hipFree(d_conv_nw);    hipFree(d_conv_out);
    hipFree(d_out);

    printf("\nGPU testbench %s\n", pass ? "PASSED" : "FAILED");
    return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
