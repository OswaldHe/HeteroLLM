#ifndef __MOE_KERNEL_H__
#define __MOE_KERNEL_H__

#include <tapa.h>
#include <ap_int.h>
#include <hls_math.h>
#include <cstdint>
#include <cmath>

// ============================================================================
// MoE Kernel — DeepSeek V3.2 Routed Experts (v1: N_EXPERT_SLOTS=1)
// Target: Alveo U55C (xcu55c-fsvh2892-2L-e)
// Framework: TAPA dataflow
// ============================================================================

// ------------------- Compile-time configuration ------------------------------
constexpr int HIDDEN          = 7168;   // Token hidden dimension
constexpr int MOE_INTER       = 2048;   // Expert intermediate dimension
constexpr int N_EXPERTS_TOTAL = 256;    // Total routed experts
constexpr int LOCAL_EXPERTS   = 128;    // Experts resident on this card
constexpr int TOP_K           = 8;      // Activated experts per token
constexpr int LOCAL_K         = TOP_K / 2;  // 4: experts per FPGA per token
// Group-based routing constants (unused in the 2-FPGA simplified router):
// constexpr int N_GROUPS        = 8;
// constexpr int N_LIMITED_GROUPS= 4;
// constexpr int EXPERTS_PER_GROUP = N_EXPERTS_TOTAL / N_GROUPS;
constexpr int N_EXPERT_SLOTS  = 1;      // v1 baseline: single compute lane
constexpr int N_PARALLEL_SLOTS = LOCAL_K;  // 4 parallel compute lanes
constexpr float ROUTE_SCALE   = 2.5f;

constexpr int VEC_WIDTH       = 32;     // INT16 elements per 512-bit beat
constexpr int VEC_WIDTH_FP    = 16;     // FP32 elements per 512-bit beat
constexpr int HIDDEN_VECS     = HIDDEN / VEC_WIDTH;        // 224 (INT16)
constexpr int HIDDEN_VECS_FP  = HIDDEN / VEC_WIDTH_FP;     // 448 (FP32)
constexpr int INTER_VECS      = MOE_INTER / VEC_WIDTH;     // 64 (INT16)
constexpr int INTER_VECS_FP   = MOE_INTER / VEC_WIDTH_FP;  // 128 (FP32)

// Total INT16 vector beats in the gate weight matrix (row-major, 256 × HIDDEN).
constexpr int W_GATE_VECS     = N_EXPERTS_TOTAL * HIDDEN_VECS; // 57,344

// Per-weight INT16 vector-beat counts for one expert slot.
constexpr int W1_VECS         = MOE_INTER * HIDDEN_VECS;  // 131,072
constexpr int W3_VECS         = MOE_INTER * HIDDEN_VECS;  // 131,072
constexpr int W2_VECS         = HIDDEN    * INTER_VECS;   // 458,752

// Opt 3: half-matrix beat counts (even-row and odd-row sub-matrices).
constexpr int W1_HALF_VECS    = W1_VECS / 2;              // 65,536
constexpr int W3_HALF_VECS    = W3_VECS / 2;              // 65,536
constexpr int W2_HALF_VECS    = W2_VECS / 2;              // 229,376

// Dequant parameter packing: 6 floats per slot — (scale, shift) × (W1, W3, W2)
constexpr int DEQUANT_PARAMS_PER_SLOT = 6;

// Weight loader trigger: relative expert_id (0..LOCAL_EXPERTS-1) to load, or -1 to stop
using weight_trigger_t = int;

// ------------------- Vector type aliases -------------------------------------
using int16_vec_t = tapa::vec_t<ap_int<16>, VEC_WIDTH>;    // 32 × INT16 = 512 bits
using fp32_vec_t  = tapa::vec_t<float, VEC_WIDTH_FP>;      // 16 × FP32  = 512 bits
using float2_t    = tapa::vec_t<float, 2>;                  // pair of FP32 scalars (Opt 3 acc streams)

// ------------------- Packed structs for inter-task streams -------------------
//
// Bit-width rationale (v1 assumptions: N ≤ 4096 tokens, 256 experts):
//   - token_id      : holds [0, N) plus −1 sentinel → signed 14 bits
//                     (range −8192..8191, covers max N = 4096 + sentinel).
//   - expert id     : holds [0, 256) plus −1 sentinel           → signed 10 bits
//                     (range −512..511).
using token_id_t  = ap_int<14>;
using expert_id_t = ap_int<10>;

// route_pkt_t: one token's routing decision, filtered to local experts only.
// With the simplified 2-FPGA router every packet carries exactly LOCAL_K
// local experts; tokens are emitted in order by gate_compute and consumed in
// the same order by dispatch_compute, so no explicit token_id is needed.
//   local_expert_ids[i] — global expert ID (0..255) for the i-th local activation.
//   weights[i]          — routing weight (sigmoid score × ROUTE_SCALE) for that expert.
struct route_pkt_t {
    expert_id_t local_expert_ids[LOCAL_K];
    float       weights[LOCAL_K];
};

// dispatch_hdr_t: one (token, expert) pair sent from dispatcher to the expert
// compute lane. Carries the routing weight and the per-token quantization scale
// so the compute task can do everything independently. The companion INT16
// activation vectors arrive on a parallel stream (one header → HIDDEN_VECS
// int16 vectors).
struct dispatch_hdr_t {
    token_id_t  token_id;
    expert_id_t global_expert_id;
    float       routing_weight;
    float       x_scale;   // per-token quantization scale: x_fp32 ≈ x_int16 * x_scale
    bool        done;      // sentinel: true when there are no more dispatches
};

// expert_out_hdr_t: after full FFN (W2 output) for one (token, expert) activation.
// The dense 7168-element output is streamed on a companion vector stream.
struct expert_out_hdr_t {
    token_id_t token_id;
    float      routing_weight;
    bool       done;      // sentinel: true when the slot finished all dispatches
};

// ============================================================================
// Top-level kernel declaration
// ============================================================================
void moe_fpga_top(
    const int   N,
    const int   local_expert_base,
    const float x_scale,          // global dequant scale for INT16 x:  x_fp32 ≈ x_int16 * x_scale
    const float w_gate_scale,     // dequant scale for INT16 W_gate
    const float w_gate_shift,     // dequant shift for INT16 W_gate (0.0f for symmetric quant)

    // Input activations (INT16)   — logical PC 0
    tapa::mmap<int16_vec_t> x_mem,
    // Gate weight (INT16, 256 × HIDDEN row-major) — logical PC 0
    tapa::mmap<int16_vec_t> W_gate_mem,
    // Combined bias + dequant params (FP32) on a single HBM port.
    // Layout: [ bias[0..N_EXPERTS_TOTAL-1] | dequant[0..DEQUANT_PARAMS_PER_SLOT-1] ]
    // Total length = N_EXPERTS_TOTAL + DEQUANT_PARAMS_PER_SLOT (= 262 floats).
    tapa::mmap<float> bias_dequant_mem,

    // Expert weight slots — six HBM ports per parallel compute lane (Opt 3: even/odd split).
    // Each even-row port holds MOE_INTER/2 (or HIDDEN/2) rows × their beat counts.
    // Layout per expert: rows 0,2,4,...  in W1_expert_mem; rows 1,3,5,... in W1b_expert_mem.
    tapa::mmaps<int16_vec_t, N_PARALLEL_SLOTS> W1_expert_mem,
    tapa::mmaps<int16_vec_t, N_PARALLEL_SLOTS> W1b_expert_mem,
    tapa::mmaps<int16_vec_t, N_PARALLEL_SLOTS> W3_expert_mem,
    tapa::mmaps<int16_vec_t, N_PARALLEL_SLOTS> W3b_expert_mem,
    tapa::mmaps<int16_vec_t, N_PARALLEL_SLOTS> W2_expert_mem,
    tapa::mmaps<int16_vec_t, N_PARALLEL_SLOTS> W2b_expert_mem,

    // Output partial y (FP32, routed-expert contribution only) — PC 5
    tapa::mmap<fp32_vec_t> y_mem
);

#endif // __MOE_KERNEL_H__
