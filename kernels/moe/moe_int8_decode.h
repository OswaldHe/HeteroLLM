#ifndef __MOE_INT8_DECODE_H__
#define __MOE_INT8_DECODE_H__

#include <tapa.h>
#include <ap_int.h>
#include <hls_math.h>
#include <cstdint>
#include <cmath>

// ============================================================================
// MoE Kernel — Opt 10 (INT8 weights, INT16 activations) — Decode-only
// Target: Alveo U55C (xcu55c-fsvh2892-2L-e)
// Framework: TAPA dataflow
//
// This is a clone of moe_kernel_dec_spec.h with weight precision dropped from
// INT16 → INT8 and elements-per-beat doubled 32 → 64. Activations remain INT16
// (32 per 512-bit beat). Per inner-loop iteration the GEMV tasks fetch TWO
// consecutive INT16 x beats and ONE INT8 weight beat per port; the inner k-loop
// is unrolled 64-wide and selects xk = (k<32) ? xva[k] : xvb[k-32].
// ============================================================================

// ------------------- Compile-time configuration ------------------------------
constexpr int HIDDEN          = 7168;   // Token hidden dimension
constexpr int MOE_INTER       = 2048;   // Expert intermediate dimension
constexpr int N_EXPERTS_TOTAL = 256;    // Total routed experts
constexpr int LOCAL_EXPERTS   = 128;    // Experts resident on this card
constexpr int TOP_K           = 8;      // Activated experts per token
constexpr int LOCAL_K         = TOP_K / 2;  // 4: experts per FPGA per token
constexpr int N_EXPERT_SLOTS  = 1;      // v1 baseline: single compute lane
constexpr int N_PARALLEL_SLOTS = LOCAL_K;  // 4 parallel compute lanes
constexpr float ROUTE_SCALE   = 2.5f;

// ---- INT16 activation packing (unchanged from Opt 9) -----------------------
constexpr int VEC_WIDTH       = 32;     // INT16 elements per 512-bit beat
constexpr int VEC_WIDTH_FP    = 16;     // FP32 elements per 512-bit beat
constexpr int HIDDEN_VECS     = HIDDEN / VEC_WIDTH;        // 224 (INT16)
constexpr int HIDDEN_VECS_FP  = HIDDEN / VEC_WIDTH_FP;     // 448 (FP32)
constexpr int INTER_VECS      = MOE_INTER / VEC_WIDTH;     // 64 (INT16)
constexpr int INTER_VECS_FP   = MOE_INTER / VEC_WIDTH_FP;  // 128 (FP32)

// ---- INT8 weight packing (NEW for Opt 10) ----------------------------------
constexpr int VEC_WIDTH_W       = 64;                          // INT8 per beat
constexpr int HIDDEN_VECS_W     = HIDDEN / VEC_WIDTH_W;        // 112
constexpr int INTER_VECS_W      = MOE_INTER / VEC_WIDTH_W;     // 32
constexpr int W_GATE_HALF_VECS_W = (N_EXPERTS_TOTAL/2) * HIDDEN_VECS_W; // 14,336

// 3-way row counts (unchanged — they are row counts, not vector counts).
constexpr int W1_THIRD_ROWS    = (MOE_INTER + 2) / 3;       // 683
constexpr int W2_SIXTH_ROWS    = ((HIDDEN / 2) + 2) / 3;    // 1195

// INT8 vector-beat counts per third / sixth.
constexpr int W1_THIRD_VECS_W    = W1_THIRD_ROWS * HIDDEN_VECS_W;   // 76,496
constexpr int W3_THIRD_VECS_W    = W1_THIRD_VECS_W;
constexpr int W2_SIXTH_VECS_W    = W2_SIXTH_ROWS * INTER_VECS_W;    // 38,240
constexpr int W1_W2C_PORT_VECS_W = W1_THIRD_VECS_W + W2_SIXTH_VECS_W;  // 114,736
constexpr int W3_W2C_PORT_VECS_W = W1_W2C_PORT_VECS_W;

// Dequant parameter packing: 6 floats per slot — (scale, shift) × (W1, W3, W2)
constexpr int DEQUANT_PARAMS_PER_SLOT = 6;

// Weight loader trigger: relative expert_id (0..LOCAL_EXPERTS-1) to load, or -1 to stop
using weight_trigger_t = int;

// ------------------- Vector type aliases -------------------------------------
using int16_vec_t = tapa::vec_t<ap_int<16>, VEC_WIDTH>;       // 32 × INT16 = 512 b
using int8_vec_t  = tapa::vec_t<ap_int<8>,  VEC_WIDTH_W>;     // 64 × INT8  = 512 b
using fp32_vec_t  = tapa::vec_t<float, VEC_WIDTH_FP>;         // 16 × FP32  = 512 b
using float2_t    = tapa::vec_t<float, 2>;
using float3_t    = tapa::vec_t<float, 3>;

// ------------------- Packed structs for inter-task streams -------------------
using token_id_t  = ap_int<14>;
using expert_id_t = ap_int<10>;

struct route_pkt_t {
    expert_id_t local_expert_ids[LOCAL_K];
    float       weights[LOCAL_K];
};

struct dispatch_hdr_t {
    token_id_t  token_id;
    expert_id_t global_expert_id;
    float       routing_weight;
    float       x_scale;
    bool        done;
};

struct expert_out_hdr_t {
    token_id_t token_id;
    float      routing_weight;
    bool       done;
};

// ============================================================================
// Top-level kernel declaration
//   All weight ports are now int8_vec_t (64 × INT8 = 512 b).
//   x_mem, bias_dequant_mem, y_mem_a/b are unchanged.
// ============================================================================
void moe_fpga_top(
    const int   local_expert_base,
    const float x_scale,
    const float w_gate_scale,
    const float w_gate_shift,

    tapa::mmap<int16_vec_t> x_mem,
    tapa::mmap<int8_vec_t>  W_gate_mem,
    tapa::mmap<int8_vec_t>  W_gate_mem_b,
    tapa::mmap<float>       bias_dequant_mem,

    tapa::mmaps<int8_vec_t, N_PARALLEL_SLOTS> W1_W2a_expert_mem,
    tapa::mmaps<int8_vec_t, N_PARALLEL_SLOTS> W1_W2b_expert_mem,
    tapa::mmaps<int8_vec_t, N_PARALLEL_SLOTS> W1_W2c_expert_mem,
    tapa::mmaps<int8_vec_t, N_PARALLEL_SLOTS> W3_W2a_expert_mem,
    tapa::mmaps<int8_vec_t, N_PARALLEL_SLOTS> W3_W2b_expert_mem,
    tapa::mmaps<int8_vec_t, N_PARALLEL_SLOTS> W3_W2c_expert_mem,

    tapa::mmap<fp32_vec_t> y_mem_a,
    tapa::mmap<fp32_vec_t> y_mem_b
);

#endif // __MOE_INT8_DECODE_H__
