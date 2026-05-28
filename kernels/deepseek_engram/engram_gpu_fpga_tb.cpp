/**
 * Engram End-to-End GPU-FPGA Testbench
 *
 * Pipeline:
 *   [FPGA] token IDs → n-gram hash → 16× embedding lookup → KV projection
 *          output: [L, KV_PROJ_ROWS] = [L, (NUM_GROUP+1)*HIDDEN_DIM] = [L, 5120]
 *            layout: keys[0..3] each [HIDDEN_DIM], then value [HIDDEN_DIM]
 *
 *   [GPU]  1. rms_norm_gate_kernel : RMSNorm(key,norm1) + RMSNorm(query,norm2)
 *                                    → gate → gated_value = gate * fpga_value
 *          2. short_conv_kernel    : per-group RMSNorm → depthwise conv1d(k=4,dil=3) → SiLU
 *          3. residual_add_kernel  : output = gated_value + conv_output + hidden_states
 *
 * Modes:
 *   --bitstream=""    : FPGA C-simulation via TAPA (no hardware required)
 *   --bitstream=<f>   : FPGA hardware execution via TAPA + XRT
 *
 * Build (see Makefile target gpu_fpga_test):
 *   $(HIPCC) -std=c++17 -O2 -g $(TAPA_INC) -isystem$(HLS_INC_DIR) \
 *       -o build/engram_gpu_fpga_tb engram_gpu_fpga_tb.cpp engram_kernels.hip \
 *       $(TAPA_LIB) $(GFLAGS_LIB)
 *
 * Run:
 *   ./build/engram_gpu_fpga_tb                          # csim mode
 *   ./build/engram_gpu_fpga_tb --bitstream=engram.xclbin # hw mode
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <random>
#include <algorithm>
#include <cassert>
#include <chrono>

#include <gflags/gflags.h>
#include <tapa.h>
#include <hip/hip_runtime.h>

#include "engram.h"

// GPU kernel constants (must match definitions in engram_kernels.hip)
#define CONV_KERNEL_SZ 4
#define CONV_DILATION  3

// ============================================================================
// Flags
// ============================================================================
DEFINE_string(bitstream, "", "path to FPGA bitstream; empty = TAPA C-simulation");

// ============================================================================
// HIP error check
// ============================================================================
#define HIP_CHECK(cmd)                                                      \
    do {                                                                    \
        hipError_t e = (cmd);                                               \
        if (e != hipSuccess) {                                              \
            std::cerr << "HIP error " << hipGetErrorString(e)              \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";    \
            std::exit(EXIT_FAILURE);                                        \
        }                                                                   \
    } while (0)

// ============================================================================
// Aligned allocator (reused from engram_tb.cpp pattern)
// ============================================================================
template <typename T>
using aligned_vector = std::vector<T, tapa::aligned_allocator<T>>;

// ============================================================================
// GPU kernel declarations (defined in engram_kernels.hip)
// ============================================================================
extern "C" {
__global__ void rms_norm_gate_kernel(
    const float* fpga_keys, const float* fpga_value,
    const float* hidden_states,
    const float* norm1_weight, const float* norm2_weight,
    float* gated_value, float* gates_out,
    uint32_t L);

__global__ void short_conv_kernel(
    const float* gated_value,
    const float* conv_weight, const float* conv_norm_weight,
    float* conv_output,
    uint32_t L);

__global__ void residual_add_kernel(
    const float* gated_value, const float* conv_output,
    const float* hidden_states,
    float* output,
    uint32_t L);
}

// ============================================================================
// SW Reference: n-gram hash (from engram_tb.cpp)
// ============================================================================
static void sw_ngram_hash(
    const int* token_ids, int L,
    int pad_id,
    int64_t mult_0, int64_t mult_1, int64_t mult_2,
    std::vector<std::vector<int>>& hash_ids
) {
    hash_ids.resize(L, std::vector<int>(TOTAL_HEADS));
    int64_t shift_0 = pad_id, shift_1 = pad_id, shift_2 = pad_id;

    for (int t = 0; t < L; t++) {
        shift_2 = shift_1; shift_1 = shift_0; shift_0 = token_ids[t];
        int64_t mix_2 = (shift_0 * mult_0) ^ (shift_1 * mult_1);
        int64_t mix_3 = mix_2 ^ (shift_2 * mult_2);

        for (int j = 0; j < NUM_HEAD; j++) {
            int64_t r = mix_2 % primes_vocab_size[0][j];
            if (r < 0) r += primes_vocab_size[0][j];
            hash_ids[t][j] = (int)r;
        }
        for (int j = 0; j < NUM_HEAD; j++) {
            int64_t r = mix_3 % primes_vocab_size[1][j];
            if (r < 0) r += primes_vocab_size[1][j];
            hash_ids[t][NUM_HEAD + j] = (int)r;
        }
    }
}

// ============================================================================
// SW Reference: embedding lookup (from engram_tb.cpp)
// ============================================================================
static void sw_embed_lookup(
    const std::vector<std::vector<int>>& hash_ids,
    const std::vector<std::vector<float>>& embed_tables_sw,
    int L,
    std::vector<float>& flat_output
) {
    flat_output.resize(L * ENGRAM_HIDDEN);
    for (int t = 0; t < L; t++) {
        float* out_ptr = flat_output.data() + t * ENGRAM_HIDDEN;
        int offset = 0;
        for (int h = 0; h < TOTAL_HEADS; h++) {
            int idx = hash_ids[t][h];
            const float* emb = embed_tables_sw[h].data() + idx * EMBED_DIM;
            for (int d = 0; d < EMBED_DIM; d++) out_ptr[offset + d] = emb[d];
            offset += EMBED_DIM;
        }
    }
}

// ============================================================================
// SW Reference: KV projection (from engram_tb.cpp)
// Channel h covers embedding columns for head pair (h, h+8).
// Weight slice kv_weights_sw[h]: [KV_PROJ_ROWS x KV_WEIGHT_COLS_PER_CHANNEL]
// Output rows layout: [group0_key, group1_key, ..., group3_key, value]
// ============================================================================
static void sw_kv_projection(
    const std::vector<float>& flat_embeddings,
    const std::vector<std::vector<float>>& kv_weights_sw,
    int L,
    std::vector<float>& kv_output
) {
    kv_output.assign(L * KV_PROJ_ROWS, 0.0f);
    for (int t = 0; t < L; t++) {
        const float* emb = flat_embeddings.data() + t * ENGRAM_HIDDEN;
        float* out = kv_output.data() + t * KV_PROJ_ROWS;

        for (int h = 0; h < KV_WEIGHT_CHANNELS; h++) {
            const float* w = kv_weights_sw[h].data();
            const int lo_off = h * EMBED_DIM;
            const int hi_off = (h + KV_WEIGHT_CHANNELS) * EMBED_DIM;
            for (int r = 0; r < KV_PROJ_ROWS; r++) {
                float acc = 0.0f;
                const int row_base = r * KV_WEIGHT_COLS_PER_CHANNEL;
                for (int c = 0; c < EMBED_DIM; c++)
                    acc += w[row_base + c] * emb[lo_off + c];
                for (int c = 0; c < EMBED_DIM; c++)
                    acc += w[row_base + EMBED_DIM + c] * emb[hi_off + c];
                out[r] += acc;
            }
        }
    }
}

// ============================================================================
// SW Reference: RMSNorm helper
// ============================================================================
static void sw_rms_norm(const float* in, const float* weight, float* out, int D, float eps = 1e-5f) {
    float sq_sum = 0.0f;
    for (int d = 0; d < D; d++) sq_sum += in[d] * in[d];
    float rms_inv = 1.0f / std::sqrt(sq_sum / D + eps);
    for (int d = 0; d < D; d++) out[d] = in[d] * rms_inv * weight[d];
}

// ============================================================================
// SW Reference: RMSNorm + Gating
// Inputs:
//   sw_kv [L, KV_PROJ_ROWS]: FPGA output (keys for 4 groups + value)
//   hidden [L, NUM_GROUP, HIDDEN_DIM]: backbone hidden states
//   norm1_weight [NUM_GROUP, HIDDEN_DIM]
//   norm2_weight [NUM_GROUP, HIDDEN_DIM]
// Outputs:
//   gated_value [L, NUM_GROUP, HIDDEN_DIM]
//   gates [L, NUM_GROUP]
// ============================================================================
static void sw_rms_norm_gate(
    const std::vector<float>& sw_kv,
    const std::vector<float>& hidden,
    const std::vector<float>& norm1_weight,
    const std::vector<float>& norm2_weight,
    int L,
    std::vector<float>& gated_value,
    std::vector<float>& gates
) {
    gated_value.resize(L * NUM_GROUP * HIDDEN_DIM);
    gates.resize(L * NUM_GROUP);

    std::vector<float> nk(HIDDEN_DIM), nq(HIDDEN_DIM);

    for (int t = 0; t < L; t++) {
        // FPGA output layout: keys[0..G-1] then value, each HIDDEN_DIM wide
        const float* fpga_val = sw_kv.data() + t * KV_PROJ_ROWS + NUM_GROUP * HIDDEN_DIM;

        for (int g = 0; g < NUM_GROUP; g++) {
            const float* key_ptr = sw_kv.data() + t * KV_PROJ_ROWS + g * HIDDEN_DIM;
            const float* hid_ptr = hidden.data() + (t * NUM_GROUP + g) * HIDDEN_DIM;
            const float* n1w     = norm1_weight.data() + g * HIDDEN_DIM;
            const float* n2w     = norm2_weight.data() + g * HIDDEN_DIM;

            sw_rms_norm(key_ptr, n1w, nk.data(), HIDDEN_DIM);
            sw_rms_norm(hid_ptr, n2w, nq.data(), HIDDEN_DIM);

            // dot product -> gate
            double dot = 0.0;
            for (int d = 0; d < HIDDEN_DIM; d++) dot += (double)nk[d] * nq[d];
            float score = (float)(dot / std::sqrt((double)HIDDEN_DIM));
            float sign_val = (score >= 0.0f) ? 1.0f : -1.0f;
            float abs_sc = std::fabs(score);
            if (abs_sc < 1e-6f) abs_sc = 1e-6f;
            score = std::sqrt(abs_sc) * sign_val;
            float gate = 1.0f / (1.0f + std::exp(-score));

            gates[t * NUM_GROUP + g] = gate;

            float* gv = gated_value.data() + (t * NUM_GROUP + g) * HIDDEN_DIM;
            for (int d = 0; d < HIDDEN_DIM; d++)
                gv[d] = gate * fpga_val[d];
        }
    }
}

// ============================================================================
// SW Reference: Short Convolution
// Matches ShortConv.forward:
//   per-group RMSNorm → depthwise causal conv1d (k=4, dil=3) → SiLU
// Inputs:
//   gated_value [L, NUM_GROUP, HIDDEN_DIM]
//   conv_weight [NUM_GROUP * HIDDEN_DIM, CONV_KERNEL_SZ]
//   conv_norm_weight [NUM_GROUP, HIDDEN_DIM]
// Output:
//   conv_output [L, NUM_GROUP, HIDDEN_DIM]
// ============================================================================
static void sw_short_conv(
    const std::vector<float>& gated_value,
    const std::vector<float>& conv_weight,
    const std::vector<float>& conv_norm_weight,
    int L,
    std::vector<float>& conv_output
) {
    conv_output.resize(L * NUM_GROUP * HIDDEN_DIM);
    const int K   = CONV_KERNEL_SZ;   // 4
    const int DIL = CONV_DILATION;    // 3

    // Precompute per-position per-group RMSNorm denominators
    // rms_inv[t * NUM_GROUP + g]
    std::vector<float> rms_inv(L * NUM_GROUP);
    for (int t = 0; t < L; t++) {
        for (int g = 0; g < NUM_GROUP; g++) {
            float sq = 0.0f;
            for (int d = 0; d < HIDDEN_DIM; d++) {
                float v = gated_value[(t * NUM_GROUP + g) * HIDDEN_DIM + d];
                sq += v * v;
            }
            rms_inv[t * NUM_GROUP + g] = 1.0f / std::sqrt(sq / HIDDEN_DIM + 1e-5f);
        }
    }

    for (int t = 0; t < L; t++) {
        for (int g = 0; g < NUM_GROUP; g++) {
            for (int d = 0; d < HIDDEN_DIM; d++) {
                int ch = g * HIDDEN_DIM + d;
                const float* cw = conv_weight.data() + ch * K;

                float acc = 0.0f;
                for (int k = 0; k < K; k++) {
                    int src_t = t - k * DIL;
                    if (src_t < 0) continue;
                    float raw    = gated_value[(src_t * NUM_GROUP + g) * HIDDEN_DIM + d];
                    float ri     = rms_inv[src_t * NUM_GROUP + g];
                    float normed = raw * ri * conv_norm_weight[g * HIDDEN_DIM + d];
                    acc += cw[k] * normed;
                }

                // SiLU
                float sig = 1.0f / (1.0f + std::exp(-acc));
                conv_output[(t * NUM_GROUP + g) * HIDDEN_DIM + d] = acc * sig;
            }
        }
    }
}

// ============================================================================
// SW Reference: Residual Add
// ============================================================================
static void sw_residual_add(
    const std::vector<float>& gated_value,
    const std::vector<float>& conv_output,
    const std::vector<float>& hidden,
    int L,
    std::vector<float>& output
) {
    int N = L * NUM_GROUP * HIDDEN_DIM;
    output.resize(N);
    for (int i = 0; i < N; i++)
        output[i] = gated_value[i] + conv_output[i] + hidden[i];
}

// ============================================================================
// Utility: compare with tolerance
// ============================================================================
static bool compare_results(
    const char* name,
    const std::vector<float>& ref, const std::vector<float>& got,
    float tol, int max_print = 5
) {
    int mismatches = 0;
    float max_err  = 0.0f;
    for (size_t i = 0; i < ref.size(); i++) {
        float err = std::fabs(ref[i] - got[i]);
        if (err > max_err) max_err = err;
        if (err > tol) {
            if (mismatches < max_print)
                std::cout << "  [" << name << "] MISMATCH i=" << i
                          << " ref=" << ref[i] << " got=" << got[i]
                          << " err=" << err << "\n";
            mismatches++;
        }
    }
    std::cout << "  [" << name << "] max_err=" << max_err
              << "  mismatches=" << mismatches << "/" << ref.size();
    if (mismatches == 0) std::cout << "  -> PASS\n";
    else                 std::cout << "  -> FAIL\n";
    return (mismatches == 0);
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char* argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    const int L      = 8;
    const int pad_id = 2;
    const int64_t mult_0 = 123457LL;
    const int64_t mult_1 = 654323LL;
    const int64_t mult_2 = 314159LL;

    const float FPGA_TOL   = 1e-2f;  // KV projection accumulation tolerance
    const float GPU_TOL    = 1e-2f;  // float32 accumulation tolerance

    std::cout << "=== Engram End-to-End GPU-FPGA Testbench ===\n";
    std::cout << "  L=" << L
              << "  NUM_GROUP=" << NUM_GROUP
              << "  HIDDEN_DIM=" << HIDDEN_DIM
              << "  KV_PROJ_ROWS=" << KV_PROJ_ROWS << "\n";
    std::cout << "  mode=" << (FLAGS_bitstream.empty() ? "TAPA C-sim" : "FPGA hw")
              << "\n\n";

    std::mt19937 gen(12345);
    std::uniform_int_distribution<int>  token_dist(0, 999);
    std::uniform_real_distribution<float> val_dist(-0.1f, 0.1f);

    // ── Token IDs ──────────────────────────────────────────────────────────
    aligned_vector<int> token_ids(L);
    for (int i = 0; i < L; i++) token_ids[i] = token_dist(gen);

    // ── Embedding tables: 16 heads ─────────────────────────────────────────
    std::vector<std::vector<float>> embed_tables_sw(TOTAL_HEADS);
    std::vector<aligned_vector<tapa::vec_t<float,16>>> embed_lo_hw(KV_WEIGHT_CHANNELS);
    std::vector<aligned_vector<tapa::vec_t<float,16>>> embed_hi_hw(KV_WEIGHT_CHANNELS);

    for (int h = 0; h < TOTAL_HEADS; h++) {
        int ngram_idx = h / NUM_HEAD;
        int head_j    = h % NUM_HEAD;
        int64_t vs    = primes_vocab_size[ngram_idx][head_j];

        embed_tables_sw[h].resize(vs * EMBED_DIM);
        auto& hw_bank = (h < KV_WEIGHT_CHANNELS)
                        ? embed_lo_hw[h]
                        : embed_hi_hw[h - KV_WEIGHT_CHANNELS];
        hw_bank.resize(vs * EMBED_DIM_DIV_16);

        for (int64_t i = 0; i < vs; i++) {
            for (int d = 0; d < EMBED_DIM; d++) {
                float v = val_dist(gen);
                embed_tables_sw[h][i * EMBED_DIM + d] = v;
                int vi = i * EMBED_DIM_DIV_16 + d / 16;
                int ki = d % 16;
                hw_bank[vi][ki] = v;
            }
        }
    }

    // ── KV weight channels: 8 channels ────────────────────────────────────
    std::vector<std::vector<float>> kv_weights_sw(KV_WEIGHT_CHANNELS);
    std::vector<aligned_vector<tapa::vec_t<float,16>>> kv_weights_hw(KV_WEIGHT_CHANNELS);

    for (int h = 0; h < KV_WEIGHT_CHANNELS; h++) {
        kv_weights_sw[h].resize(KV_PROJ_ROWS * KV_WEIGHT_COLS_PER_CHANNEL);
        kv_weights_hw[h].resize(KV_WEIGHT_VECS_PER_CHANNEL);

        for (int r = 0; r < KV_PROJ_ROWS; r++) {
            for (int c = 0; c < KV_WEIGHT_COLS_PER_CHANNEL; c++) {
                float v = val_dist(gen);
                kv_weights_sw[h][r * KV_WEIGHT_COLS_PER_CHANNEL + c] = v;
                int vi = r * KV_WEIGHT_COLS_PER_CHANNEL_DIV_16 + c / 16;
                int ki = c % 16;
                kv_weights_hw[h][vi][ki] = v;
            }
        }
    }

    // ── GPU-specific parameters ────────────────────────────────────────────
    // hidden_states [L, NUM_GROUP, HIDDEN_DIM]  (backbone query)
    std::vector<float> hidden_states(L * NUM_GROUP * HIDDEN_DIM);
    for (auto& v : hidden_states) v = val_dist(gen);

    // norm1_weight [NUM_GROUP, HIDDEN_DIM]  (Engram.norm1, one per group)
    std::vector<float> norm1_weight(NUM_GROUP * HIDDEN_DIM);
    for (auto& v : norm1_weight) v = val_dist(gen) + 1.0f;  // init near 1

    // norm2_weight [NUM_GROUP, HIDDEN_DIM]  (Engram.norm2, one per group)
    std::vector<float> norm2_weight(NUM_GROUP * HIDDEN_DIM);
    for (auto& v : norm2_weight) v = val_dist(gen) + 1.0f;

    // conv_weight [NUM_GROUP * HIDDEN_DIM, CONV_KERNEL_SZ]  (ShortConv.conv)
    std::vector<float> conv_weight(NUM_GROUP * HIDDEN_DIM * CONV_KERNEL_SZ);
    for (auto& v : conv_weight) v = val_dist(gen);

    // conv_norm_weight [NUM_GROUP, HIDDEN_DIM]  (ShortConv.norms)
    std::vector<float> conv_norm_weight(NUM_GROUP * HIDDEN_DIM);
    for (auto& v : conv_norm_weight) v = val_dist(gen) + 1.0f;

    // ── [1] SW Reference ───────────────────────────────────────────────────
    std::cout << "[1] Running full SW reference pipeline...\n";

    std::vector<std::vector<int>> sw_hash_ids;
    sw_ngram_hash(token_ids.data(), L, pad_id, mult_0, mult_1, mult_2, sw_hash_ids);

    std::vector<float> sw_flat;
    sw_embed_lookup(sw_hash_ids, embed_tables_sw, L, sw_flat);

    std::vector<float> sw_kv;
    sw_kv_projection(sw_flat, kv_weights_sw, L, sw_kv);

    std::vector<float> sw_gated_value, sw_gates;
    sw_rms_norm_gate(sw_kv, hidden_states, norm1_weight, norm2_weight,
                     L, sw_gated_value, sw_gates);

    std::vector<float> sw_conv_output;
    sw_short_conv(sw_gated_value, conv_weight, conv_norm_weight, L, sw_conv_output);

    std::vector<float> sw_output;
    sw_residual_add(sw_gated_value, sw_conv_output, hidden_states, L, sw_output);

    std::cout << "  SW done. Sample output[t=0,g=0,d=0..3]: ";
    for (int d = 0; d < 4; d++) std::cout << sw_output[d] << " ";
    std::cout << "\n\n";

    // ── [2] FPGA Execution ─────────────────────────────────────────────────
    std::cout << "[2] Invoking engram_fpga_top (" << (FLAGS_bitstream.empty() ? "csim" : "hw") << ")...\n";

    int output_vecs = L * KV_PROJ_ROWS_DIV_16;
    aligned_vector<tapa::vec_t<float,16>> hw_kv_output(output_vecs);
    for (auto& v : hw_kv_output) for (int k = 0; k < 16; k++) v[k] = 0.0f;

    auto t_fpga_start = std::chrono::steady_clock::now();

    int64_t fpga_ns = tapa::invoke(
        engram_fpga_top,
        FLAGS_bitstream,
        L, pad_id,
        (ap_int<64>)mult_0, (ap_int<64>)mult_1, (ap_int<64>)mult_2,
        tapa::read_only_mmap<int>(token_ids),
        tapa::read_only_mmaps<tapa::vec_t<float,16>, KV_WEIGHT_CHANNELS>(embed_lo_hw),
        tapa::read_only_mmaps<tapa::vec_t<float,16>, KV_WEIGHT_CHANNELS>(embed_hi_hw),
        tapa::read_only_mmaps<tapa::vec_t<float,16>, KV_WEIGHT_CHANNELS>(kv_weights_hw),
        tapa::write_only_mmap<tapa::vec_t<float,16>>(hw_kv_output)
    );

    auto t_fpga_end = std::chrono::steady_clock::now();
    double fpga_wall_ms = std::chrono::duration<double,std::milli>(
        t_fpga_end - t_fpga_start).count();

    std::cout << "  FPGA kernel returned in " << fpga_ns << " ns"
              << " (wall: " << fpga_wall_ms << " ms)\n\n";

    // ── [3] Validate FPGA KV output ────────────────────────────────────────
    std::cout << "[3] Validating FPGA KV projection output...\n";

    // Unpack vec_t output into flat float for comparison
    std::vector<float> fpga_kv_flat(L * KV_PROJ_ROWS);
    for (int i = 0; i < output_vecs; i++) {
        for (int k = 0; k < 16; k++) {
            fpga_kv_flat[i * 16 + k] = hw_kv_output[i][k];
        }
    }

    bool fpga_pass = compare_results("FPGA KV", sw_kv, fpga_kv_flat, FPGA_TOL);

    std::cout << "  FPGA KV sample [t=0, rows 0..3]: ";
    for (int r = 0; r < 4; r++) std::cout << fpga_kv_flat[r] << " ";
    std::cout << "\n\n";

    if (!fpga_pass) {
        std::cerr << "FPGA KV projection FAILED. Aborting GPU test.\n";
        return EXIT_FAILURE;
    }

    // ── [4] GPU Setup ──────────────────────────────────────────────────────
    std::cout << "[4] Setting up GPU...\n";

    int gpu_device = 0;
    HIP_CHECK(hipSetDevice(gpu_device));

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, gpu_device));
    std::cout << "  GPU: " << prop.name << "\n";

    // Allocate device buffers
    size_t sz_kv        = L * KV_PROJ_ROWS     * sizeof(float);
    size_t sz_lghd      = L * NUM_GROUP * HIDDEN_DIM * sizeof(float);
    size_t sz_norm_g    = NUM_GROUP * HIDDEN_DIM * sizeof(float);
    size_t sz_conv_w    = NUM_GROUP * HIDDEN_DIM * CONV_KERNEL_SZ * sizeof(float);
    size_t sz_gates     = L * NUM_GROUP * sizeof(float);

    // Split FPGA output into keys [L, NUM_GROUP, HIDDEN_DIM] and value [L, HIDDEN_DIM]
    // Layout in fpga_kv_flat: for each token t, rows [g*D, (g+1)*D) = key_g, rows [G*D, (G+1)*D) = value
    // Re-arrange into contiguous keys and value arrays for the GPU kernel
    std::vector<float> fpga_keys_flat(L * NUM_GROUP * HIDDEN_DIM);
    std::vector<float> fpga_value_flat(L * HIDDEN_DIM);
    for (int t = 0; t < L; t++) {
        const float* src = fpga_kv_flat.data() + t * KV_PROJ_ROWS;
        // Keys: first NUM_GROUP * HIDDEN_DIM floats
        std::copy(src, src + NUM_GROUP * HIDDEN_DIM,
                  fpga_keys_flat.data() + t * NUM_GROUP * HIDDEN_DIM);
        // Value: last HIDDEN_DIM floats
        std::copy(src + NUM_GROUP * HIDDEN_DIM, src + KV_PROJ_ROWS,
                  fpga_value_flat.data() + t * HIDDEN_DIM);
    }

    float *d_fpga_keys, *d_fpga_value;
    float *d_hidden_states;
    float *d_norm1_weight, *d_norm2_weight;
    float *d_conv_weight, *d_conv_norm_weight;
    float *d_gated_value, *d_gates_out;
    float *d_conv_output;
    float *d_output;

    HIP_CHECK(hipMalloc(&d_fpga_keys,       L * NUM_GROUP * HIDDEN_DIM * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_fpga_value,      L * HIDDEN_DIM * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_hidden_states,   sz_lghd));
    HIP_CHECK(hipMalloc(&d_norm1_weight,    sz_norm_g));
    HIP_CHECK(hipMalloc(&d_norm2_weight,    sz_norm_g));
    HIP_CHECK(hipMalloc(&d_conv_weight,     sz_conv_w));
    HIP_CHECK(hipMalloc(&d_conv_norm_weight,sz_norm_g));
    HIP_CHECK(hipMalloc(&d_gated_value,     sz_lghd));
    HIP_CHECK(hipMalloc(&d_gates_out,       sz_gates));
    HIP_CHECK(hipMalloc(&d_conv_output,     sz_lghd));
    HIP_CHECK(hipMalloc(&d_output,          sz_lghd));

    // Copy inputs to device
    HIP_CHECK(hipMemcpy(d_fpga_keys,    fpga_keys_flat.data(),
                        L * NUM_GROUP * HIDDEN_DIM * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_fpga_value,   fpga_value_flat.data(),
                        L * HIDDEN_DIM * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_hidden_states,hidden_states.data(), sz_lghd, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_norm1_weight, norm1_weight.data(), sz_norm_g, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_norm2_weight, norm2_weight.data(), sz_norm_g, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_conv_weight,  conv_weight.data(),  sz_conv_w, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_conv_norm_weight, conv_norm_weight.data(), sz_norm_g, hipMemcpyHostToDevice));
    std::cout << "  GPU buffers allocated and initialized.\n\n";

    // HIP timing events
    hipEvent_t ev_gate_s, ev_gate_e;
    hipEvent_t ev_conv_s, ev_conv_e;
    hipEvent_t ev_res_s,  ev_res_e;
    HIP_CHECK(hipEventCreate(&ev_gate_s)); HIP_CHECK(hipEventCreate(&ev_gate_e));
    HIP_CHECK(hipEventCreate(&ev_conv_s)); HIP_CHECK(hipEventCreate(&ev_conv_e));
    HIP_CHECK(hipEventCreate(&ev_res_s));  HIP_CHECK(hipEventCreate(&ev_res_e));

    // ── [5] GPU Kernel: rms_norm_gate_kernel ──────────────────────────────
    std::cout << "[5] GPU kernel: rms_norm_gate_kernel...\n";
    {
        // One block per (t,g), 256 threads
        dim3 grid(L * NUM_GROUP);
        dim3 block(256);
        HIP_CHECK(hipEventRecord(ev_gate_s));
        hipLaunchKernelGGL(rms_norm_gate_kernel, grid, block, 0, 0,
            d_fpga_keys, d_fpga_value, d_hidden_states,
            d_norm1_weight, d_norm2_weight,
            d_gated_value, d_gates_out,
            (uint32_t)L);
        HIP_CHECK(hipEventRecord(ev_gate_e));
        HIP_CHECK(hipEventSynchronize(ev_gate_e));
    }

    // ── [6] GPU Kernel: short_conv_kernel ────────────────────────────────
    std::cout << "[6] GPU kernel: short_conv_kernel...\n";
    {
        int total_elems = L * NUM_GROUP * HIDDEN_DIM;
        int block_sz = 256;
        int grid_sz  = (total_elems + block_sz - 1) / block_sz;
        HIP_CHECK(hipEventRecord(ev_conv_s));
        hipLaunchKernelGGL(short_conv_kernel,
            dim3(grid_sz), dim3(block_sz), 0, 0,
            d_gated_value, d_conv_weight, d_conv_norm_weight,
            d_conv_output, (uint32_t)L);
        HIP_CHECK(hipEventRecord(ev_conv_e));
        HIP_CHECK(hipEventSynchronize(ev_conv_e));
    }

    // ── [7] GPU Kernel: residual_add_kernel ──────────────────────────────
    std::cout << "[7] GPU kernel: residual_add_kernel...\n";
    {
        int total_elems = L * NUM_GROUP * HIDDEN_DIM;
        int block_sz = 256;
        int grid_sz  = (total_elems + block_sz - 1) / block_sz;
        HIP_CHECK(hipEventRecord(ev_res_s));
        hipLaunchKernelGGL(residual_add_kernel,
            dim3(grid_sz), dim3(block_sz), 0, 0,
            d_gated_value, d_conv_output, d_hidden_states,
            d_output, (uint32_t)L);
        HIP_CHECK(hipEventRecord(ev_res_e));
        HIP_CHECK(hipEventSynchronize(ev_res_e));
    }

    // ── [8] Copy results back ─────────────────────────────────────────────
    std::vector<float> gpu_gated_value(L * NUM_GROUP * HIDDEN_DIM);
    std::vector<float> gpu_conv_output(L * NUM_GROUP * HIDDEN_DIM);
    std::vector<float> gpu_output(L * NUM_GROUP * HIDDEN_DIM);

    HIP_CHECK(hipMemcpy(gpu_gated_value.data(), d_gated_value, sz_lghd, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(gpu_conv_output.data(), d_conv_output, sz_lghd, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(gpu_output.data(),      d_output,      sz_lghd, hipMemcpyDeviceToHost));

    // ── [9] Verify GPU results ────────────────────────────────────────────
    std::cout << "\n[9] Verifying GPU outputs vs SW reference...\n";

    bool gv_pass   = compare_results("gated_value",  sw_gated_value,  gpu_gated_value,  GPU_TOL);
    bool conv_pass = compare_results("short_conv",   sw_conv_output,  gpu_conv_output,  GPU_TOL);
    bool out_pass  = compare_results("final_output", sw_output,       gpu_output,       GPU_TOL);

    // ── [10] Latency Summary ──────────────────────────────────────────────
    float ms_gate, ms_conv, ms_res;
    HIP_CHECK(hipEventElapsedTime(&ms_gate, ev_gate_s, ev_gate_e));
    HIP_CHECK(hipEventElapsedTime(&ms_conv, ev_conv_s, ev_conv_e));
    HIP_CHECK(hipEventElapsedTime(&ms_res,  ev_res_s,  ev_res_e));
    float ms_gpu_total = ms_gate + ms_conv + ms_res;
    double ms_e2e = (fpga_ns / 1e6) + ms_gpu_total;

    std::cout << "\n=== Latency Breakdown ===\n";
    std::cout << "  FPGA (tapa::invoke):       " << fpga_ns / 1000 << " us"
              << "  (wall: " << fpga_wall_ms << " ms)\n";
    std::cout << "  GPU rms_norm_gate:         " << ms_gate * 1e3f << " us\n";
    std::cout << "  GPU short_conv:            " << ms_conv * 1e3f << " us\n";
    std::cout << "  GPU residual_add:          " << ms_res  * 1e3f << " us\n";
    std::cout << "  GPU total:                 " << ms_gpu_total * 1e3f << " us\n";
    std::cout << "  E2E (FPGA wall + GPU):     " << ms_e2e << " ms\n";

    // ── Cleanup ───────────────────────────────────────────────────────────
    (void)hipFree(d_fpga_keys);    (void)hipFree(d_fpga_value);
    (void)hipFree(d_hidden_states);
    (void)hipFree(d_norm1_weight); (void)hipFree(d_norm2_weight);
    (void)hipFree(d_conv_weight);  (void)hipFree(d_conv_norm_weight);
    (void)hipFree(d_gated_value);  (void)hipFree(d_gates_out);
    (void)hipFree(d_conv_output);  (void)hipFree(d_output);
    (void)hipEventDestroy(ev_gate_s); (void)hipEventDestroy(ev_gate_e);
    (void)hipEventDestroy(ev_conv_s); (void)hipEventDestroy(ev_conv_e);
    (void)hipEventDestroy(ev_res_s);  (void)hipEventDestroy(ev_res_e);

    // ── Final verdict ─────────────────────────────────────────────────────
    bool all_pass = fpga_pass && gv_pass && conv_pass && out_pass;
    std::cout << "\n";
    if (all_pass)
        std::cout << "Engram GPU-FPGA testbench PASSED!\n";
    else
        std::cout << "Engram GPU-FPGA testbench FAILED!\n";

    return all_pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
