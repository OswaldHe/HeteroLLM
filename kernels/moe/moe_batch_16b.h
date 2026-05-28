#ifndef __MOE_BATCH_16B_H__
#define __MOE_BATCH_16B_H__

#include <tapa.h>
#include <ap_int.h>
#include <hls_math.h>
#include <cstdint>
#include <cmath>

// ============================================================================
// moe_batch_16b — batched-token MoE kernel (N>1)
// Phase A of the dynamic-dispatch optimization (plan_batch.md §8.1):
//   - Adds batch_scheduler / slot_dispatcher / y_accumulator tasks.
//   - FFN slots remain single-token-wide (no MAC widening yet).
//   - Removes the inter-slot partial-sum chain; per-round partials go to
//     a centralized y_accumulator that holds y_acc[N][HIDDEN] in BRAM.
// All inter-task TAPA stream element types are kept under 2048 bits.
// Largest element: partial_pkt_t = fp32_vec_t (512 bits) + small fields.
// Target: Alveo U55C (xcu55c-fsvh2892-2L-e)
// Framework: TAPA dataflow
// ============================================================================

// ------------------- Compile-time configuration ------------------------------
constexpr int HIDDEN          = 7168;    // Token hidden dimension
constexpr int MOE_INTER       = 2048;    // Expert intermediate dimension
constexpr int N_EXPERTS_TOTAL = 256;    // Total routed experts
constexpr int LOCAL_EXPERTS   = 128;    // Experts resident on this card
constexpr int TOP_K           = 8;      // Activated experts per token
constexpr int LOCAL_K         = TOP_K / 2;  // 4: experts per FPGA per token
constexpr int N_EXPERT_SLOTS  = 1;      // v1 baseline: single compute lane
constexpr int N_PARALLEL_SLOTS = LOCAL_K;  // 4 parallel compute lanes
constexpr float ROUTE_SCALE   = 2.5f;

// Max batch size supported by the on-chip x_cache and y_acc structures.
// We size for N up to N_TILE_MAX so a single invocation can handle a tile
// of tokens. With N_TILE_MAX=8 the dynamic-dispatch sees at most 32 routings.
constexpr int N_TILE_MAX      = 8;
// Worst-case rounds: each round dispatches up to LOCAL_K (=4) routings.
// With N_TILE_MAX=8 tokens × LOCAL_K=4 routings = 32 routings, the round
// count is bounded by ceil(32/4) = 8 (best) and up to N_TILE_MAX = 8 in
// pathological skew where one expert gets all 8 tokens. Round count <= 8.
constexpr int MAX_ROUNDS      = N_TILE_MAX;

constexpr int VEC_WIDTH       = 32;     // INT16 elements per 512-bit beat
constexpr int VEC_WIDTH_FP    = 16;     // FP32 elements per 512-bit beat
constexpr int HIDDEN_VECS     = HIDDEN / VEC_WIDTH;        // 224 (INT16)
constexpr int HIDDEN_VECS_FP  = HIDDEN / VEC_WIDTH_FP;     // 448 (FP32)
constexpr int VEC_HALF         = VEC_WIDTH / 2;             // 16: half-width int16 vector
constexpr int INTER_VECS      = MOE_INTER / VEC_WIDTH;     // 64  (INT16)
constexpr int INTER_VECS_FP   = MOE_INTER / VEC_WIDTH_FP;  // 128 (FP32)
constexpr int INTER_HALF_VECS = MOE_INTER / VEC_HALF;      // 128 (half-width INT16)
constexpr int HIDDEN_PACKED   = HIDDEN / 2;                // 3584: ap_uint<32> words per token's x

// Total INT16 vector beats in the gate weight matrix (row-major, 256 × HIDDEN).
constexpr int W_GATE_VECS     = N_EXPERTS_TOTAL * HIDDEN_VECS; // 57,344

// Half-matrix beat counts (one HBM port covers half of the experts).
constexpr int W_GATE_HALF_VECS = (N_EXPERTS_TOTAL / 2) * HIDDEN_VECS; // 28,672

// Per-weight INT16 vector-beat counts for one expert slot.
constexpr int W1_VECS         = MOE_INTER * HIDDEN_VECS;  // 458,752
constexpr int W3_VECS         = MOE_INTER * HIDDEN_VECS;  // 458,752
constexpr int W2_VECS         = HIDDEN    * INTER_VECS;   // 458,752

constexpr int W1_HALF_VECS    = W1_VECS / 2;              // 229,376
constexpr int W3_HALF_VECS    = W3_VECS / 2;              // 229,376
constexpr int W2_HALF_VECS    = W2_VECS / 2;              // 229,376

constexpr int W2_QUARTER_VECS = W2_HALF_VECS / 2;          // 114,688

constexpr int W1_W2_PORT_VECS = W1_HALF_VECS + W2_QUARTER_VECS;  // 344,064
constexpr int W3_W2_PORT_VECS = W3_HALF_VECS + W2_QUARTER_VECS;  // 344,064

// Opt 5: 3-way row interleaving (a/b/c).
constexpr int W1_THIRD_VECS    = ((MOE_INTER + 2) / 3) * HIDDEN_VECS;   // 683 * 224
constexpr int W3_THIRD_VECS    = W1_THIRD_VECS;
constexpr int W2_SIXTH_VECS    = (((HIDDEN / 2) + 2) / 3) * INTER_VECS; // 1195 * 64
constexpr int W1_W2C_PORT_VECS = W1_THIRD_VECS + W2_SIXTH_VECS;
constexpr int W3_W2C_PORT_VECS = W3_THIRD_VECS + W2_SIXTH_VECS;

constexpr int W1_THIRD_ROWS    = (MOE_INTER + 2) / 3;       // 683
constexpr int W2_SIXTH_ROWS    = ((HIDDEN / 2) + 2) / 3;    // 1195

constexpr int DEQUANT_PARAMS_PER_SLOT = 6;
constexpr int N_BIAS_DEQUANT = N_EXPERTS_TOTAL + DEQUANT_PARAMS_PER_SLOT; // 6 per expert slot (W1/W3 scale+shift, W2 scale+shift, route scale+shift)

using weight_trigger_t = int;

// ------------------- Vector type aliases -------------------------------------
using int16_vec_t      = tapa::vec_t<ap_int<16>, VEC_WIDTH>;
using int16_half_vec_t = tapa::vec_t<ap_int<16>, VEC_HALF>;
using fp32_vec_t       = tapa::vec_t<float, VEC_WIDTH_FP>;
using bf16_vec_t       = tapa::vec_t<ap_uint<16>, VEC_WIDTH_FP>;
using float2_t         = tapa::vec_t<float, 2>;
using float3_t         = tapa::vec_t<float, 3>;

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

// Per-round descriptor emitted by batch_scheduler. In Phase A each slot
// processes exactly one (token, expert) pair per round. slot_active=false
// indicates a "filler" slot — we still drive the FFN with a dummy routing
// (token 0, weight 0.0, expert 0) so the slot's pipelined tasks see a
// uniform stream length; the zero weight makes its y contribution vanish.
// Total bits: 4 * (10 + 14 + 32 + 1) + 1 ≈ 229 bits  (well under 2048)
struct round_desc_t {
    expert_id_t local_expert_id[N_PARALLEL_SLOTS];
    token_id_t  token_id      [N_PARALLEL_SLOTS];
    float       routing_weight[N_PARALLEL_SLOTS];
    bool        slot_active   [N_PARALLEL_SLOTS];
    bool        last_round;
};

// Per-round header forwarded from slot_dispatcher to ffn_x_split and onward.
// Phase B: extended to carry up to 4 token-lanes (one expert serves 1..4 tokens
// per round, sharing weights). Inactive lanes have routing_weight=0 (their FFN
// MAC still runs on placeholder x but contributes 0 to y_accumulator).
//
// Bits: 4*14 + 4*32 + 32 + 3 + 1 + 1 = 56 + 128 + 32 + 5 = 221 bits  (<< 2048)
constexpr int TOKEN_LANES = 4;  // Phase B 4-token spatial widening
constexpr int TOKEN_LANE_PAIRS = TOKEN_LANES / 2;  // Packed lane pairs for y_buf
struct round_hdr_t {
    token_id_t token_ids      [TOKEN_LANES];
    float      routing_weights[TOKEN_LANES];
    float      x_scale;
    ap_uint<3> token_count;   // 0..4 ; 0 ⇒ slot inactive (no expert chosen)
    bool       slot_active;   // false ⇒ entire slot is filler this round
    bool       last_round;
    bool       tile_done;     // last round of the current tile
    int        trigger_eid;   // expert_id for weight trigger relay via x_split
};

// Partial output packet from ffn_w*_w2_proj to y_accumulator.
// BF16 packing: 14 + 16*16 + 3 = 273 bits (reduced from 529)
struct partial_pkt_t {
    token_id_t token_id;
    bf16_vec_t y;
    bool       slot_active;
    bool       last_round;
    bool       tile_done;     // last round of the current tile
};

// ============================================================================
// Top-level kernel declaration (batched: N tokens per invocation)
// ============================================================================
void moe_fpga_top(
    const int   N,                // runtime token count for this batch
    const int   local_expert_base,
    const float x_scale,          // global dequant scale shared across all N tokens
    const float w_gate_scale,
    const float w_gate_shift,

    tapa::mmap<int16_vec_t> x_mem,
    tapa::mmap<int16_vec_t> W_gate_mem,
    tapa::mmap<int16_vec_t> W_gate_mem_b,
    tapa::mmap<float>       bias_dequant_mem,

    tapa::mmaps<int16_vec_t, N_PARALLEL_SLOTS> W1_W2a_expert_mem,
    tapa::mmaps<int16_vec_t, N_PARALLEL_SLOTS> W1_W2b_expert_mem,
    tapa::mmaps<int16_vec_t, N_PARALLEL_SLOTS> W1_W2c_expert_mem,
    tapa::mmaps<int16_vec_t, N_PARALLEL_SLOTS> W3_W2a_expert_mem,
    tapa::mmaps<int16_vec_t, N_PARALLEL_SLOTS> W3_W2b_expert_mem,
    tapa::mmaps<int16_vec_t, N_PARALLEL_SLOTS> W3_W2c_expert_mem,

    tapa::mmap<fp32_vec_t> y_mem_a,
    tapa::mmap<fp32_vec_t> y_mem_b
);

#endif // __MOE_BATCH_16B_H__
