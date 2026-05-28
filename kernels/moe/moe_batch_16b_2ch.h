#ifndef __MOE_BATCH_16B_2CH_H__
#define __MOE_BATCH_16B_2CH_H__

#include <tapa.h>
#include <ap_int.h>
#include <hls_math.h>
#include <cstdint>
#include <cmath>

// ============================================================================
// moe_batch_16b_2ch — 2-channel (a/b) variant of moe_batch_16b
// Drops the c-channel of W1_W2 and W3_W2: each parallel FFN slot now uses
// 4 HBM ports for weights (instead of 6), trading throughput for free
// channels. Top-level mmap count: 4 + 4*N_PARALLEL_SLOTS + 2 = 22 (<= 32).
//
// All other behavior (routing, dequant, y accumulation, packet layouts) is
// preserved bit-exactly relative to moe_batch_16b.
// Target: Alveo U55C (xcu55c-fsvh2892-2L-e)
// Framework: TAPA dataflow
// ============================================================================

// ------------------- Compile-time configuration ------------------------------
// Production sizing (revert to these before HLS synthesis).
constexpr int HIDDEN          = 7168;    // Token hidden dimension
constexpr int MOE_INTER       = 2048;    // Expert intermediate dimension
constexpr int N_EXPERTS_TOTAL = 256;     // Total routed experts
constexpr int LOCAL_EXPERTS   = 128;     // Experts resident on this card
constexpr int TOP_K           = 8;       // Activated experts per token
constexpr int LOCAL_K         = TOP_K / 2;   // 4: experts per FPGA per token
constexpr int N_EXPERT_SLOTS  = 1;       // v1 baseline: single compute lane
constexpr int N_PARALLEL_SLOTS = LOCAL_K;    // 4 parallel compute lanes
constexpr float ROUTE_SCALE   = 2.5f;

// Max batch size supported by the on-chip x_cache and y_acc structures.
constexpr int N_TILE_MAX      = 8;
constexpr int MAX_ROUNDS      = N_TILE_MAX;

constexpr int VEC_WIDTH       = 32;     // INT16 elements per 512-bit beat
constexpr int VEC_WIDTH_FP    = 16;     // FP32 elements per 512-bit beat
constexpr int HIDDEN_VECS     = HIDDEN / VEC_WIDTH;
constexpr int HIDDEN_VECS_FP  = HIDDEN / VEC_WIDTH_FP;
constexpr int VEC_HALF        = VEC_WIDTH / 2;            // 16: half-width int16 vector
constexpr int INTER_VECS      = MOE_INTER / VEC_WIDTH;
constexpr int INTER_VECS_FP   = MOE_INTER / VEC_WIDTH_FP;
constexpr int INTER_HALF_VECS = MOE_INTER / VEC_HALF;
constexpr int HIDDEN_PACKED   = HIDDEN / 2;               // ap_uint<32> words per token's x

// Total INT16 vector beats in the gate weight matrix (row-major, 256 × HIDDEN).
constexpr int W_GATE_VECS      = N_EXPERTS_TOTAL * HIDDEN_VECS;
constexpr int W_GATE_HALF_VECS = (N_EXPERTS_TOTAL / 2) * HIDDEN_VECS;

// Per-weight INT16 vector-beat counts for one expert slot.
constexpr int W1_VECS = MOE_INTER * HIDDEN_VECS;
constexpr int W3_VECS = MOE_INTER * HIDDEN_VECS;
constexpr int W2_VECS = HIDDEN    * INTER_VECS;

// ---- 2-way row interleaving (a/b), replaces the older 3-way a/b/c geometry ----
constexpr int W1_AB_HALF_ROWS    = (MOE_INTER + 1) / 2;
constexpr int W3_AB_HALF_ROWS    = W1_AB_HALF_ROWS;
constexpr int W2_AB_QUARTER_ROWS = ((HIDDEN / 2) + 1) / 2;

constexpr int W1_AB_HALF_VECS    = W1_AB_HALF_ROWS * HIDDEN_VECS;
constexpr int W3_AB_HALF_VECS    = W3_AB_HALF_ROWS * HIDDEN_VECS;
constexpr int W2_AB_QUARTER_VECS = W2_AB_QUARTER_ROWS * INTER_VECS;

constexpr int W1_W2_AB_PORT_VECS = W1_AB_HALF_VECS + W2_AB_QUARTER_VECS;
constexpr int W3_W2_AB_PORT_VECS = W3_AB_HALF_VECS + W2_AB_QUARTER_VECS;

constexpr int DEQUANT_PARAMS_PER_SLOT = 6;
constexpr int N_BIAS_DEQUANT = N_EXPERTS_TOTAL + DEQUANT_PARAMS_PER_SLOT;

using weight_trigger_t = int;

// ------------------- Vector type aliases -------------------------------------
using int16_vec_t      = tapa::vec_t<ap_int<16>, VEC_WIDTH>;
using int16_half_vec_t = tapa::vec_t<ap_int<16>, VEC_HALF>;
using fp32_vec_t       = tapa::vec_t<float, VEC_WIDTH_FP>;
using bf16_vec_t       = tapa::vec_t<ap_uint<16>, VEC_WIDTH_FP>;
using float2_t         = tapa::vec_t<float, 2>;
using float3_t         = tapa::vec_t<float, 3>;   // unused after 2ch refactor; kept harmless

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

struct round_desc_t {
    expert_id_t local_expert_id[N_PARALLEL_SLOTS];
    token_id_t  token_id      [N_PARALLEL_SLOTS];
    float       routing_weight[N_PARALLEL_SLOTS];
    bool        slot_active   [N_PARALLEL_SLOTS];
    bool        last_round;
};

constexpr int TOKEN_LANES = 4;
constexpr int TOKEN_LANE_PAIRS = TOKEN_LANES / 2;
struct round_hdr_t {
    token_id_t token_ids      [TOKEN_LANES];
    float      routing_weights[TOKEN_LANES];
    float      x_scale;
    ap_uint<3> token_count;
    bool       slot_active;
    bool       last_round;
    bool       tile_done;
    int        trigger_eid;
};

struct partial_pkt_t {
    token_id_t token_id;
    bf16_vec_t y;
    bool       slot_active;
    bool       last_round;
    bool       tile_done;
};

// ============================================================================
// Top-level kernel declaration — 2ch variant (drops W1_W2c, W3_W2c).
// mmap count: 4 + 4 * N_PARALLEL_SLOTS + 2 = 22.
// ============================================================================
void moe_fpga_top(
    const int   N,
    const int   local_expert_base,
    const float x_scale,
    const float w_gate_scale,
    const float w_gate_shift,

    tapa::mmap<int16_vec_t> x_mem,
    tapa::mmap<int16_vec_t> W_gate_mem,
    tapa::mmap<int16_vec_t> W_gate_mem_b,
    tapa::mmap<float>       bias_dequant_mem,

    tapa::mmaps<int16_vec_t, N_PARALLEL_SLOTS> W1_W2a_expert_mem,
    tapa::mmaps<int16_vec_t, N_PARALLEL_SLOTS> W1_W2b_expert_mem,
    tapa::mmaps<int16_vec_t, N_PARALLEL_SLOTS> W3_W2a_expert_mem,
    tapa::mmaps<int16_vec_t, N_PARALLEL_SLOTS> W3_W2b_expert_mem,

    tapa::mmap<fp32_vec_t> y_mem_a,
    tapa::mmap<fp32_vec_t> y_mem_b
);

#endif // __MOE_BATCH_16B_2CH_H__
