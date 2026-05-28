// ============================================================================
// MoE FPGA Kernel — batch_16b (INT16, batched-token variant of dec_spec / Opt 9)
//
// Phase A of the dynamic-dispatch plan (plan_batch.md §8.1):
//   * batch_scheduler      — caches N tokens of x, builds a per-expert pending
//                            bank from gate's route_pkts, and emits one
//                            round_desc per round (single token per slot).
//   * slot_dispatcher (×4) — per round, reads round_desc + the routed token's
//                            x, emits 6 weight triggers + a round_hdr to its
//                            FFN slot. Filler rounds (slot_active=false) still
//                            drive the slot with dummy data (weight=0).
//   * y_accumulator_half   — receives one half's per-round partial_pkt streams
//                            from all 4 slots, accumulates into y_acc[N][H/2]
//                            (BRAM, cyclic-16 partition), flushes token-
//                            sequentially per tile.
//   * Removed:  dispatch_compute, partial_seed_half. The partial-sum chain
//                            between slots no longer exists.
//   * FFN slots remain single-token-wide. ffn_w*_w2_proj now consumes rounds
//                            until last_round and emits to y_accumulator
//                            instead of folding through a partial_chain.
// ============================================================================

#include "moe_batch_16b.h"

static inline float moe_sigmoidf(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}
static inline float moe_siluf(float x) {
    return x * moe_sigmoidf(x);
}
static inline float bf16_to_f32(ap_uint<16> bf16) {
    ap_uint<32> bits = ((ap_uint<32>)bf16) << 16;
    return tapa::bit_cast<float>(bits);
}

// ============================================================================
// Reader: x_gate_reader — N tokens of x, total N * HIDDEN_VECS beats.
// ============================================================================
void x_gate_reader(
    const int N,
    tapa::async_mmap<int16_vec_t>& x_mem,
    tapa::ostream<int16_vec_t>&    out
) {
    const int total = N * HIDDEN_VECS;
    for (int i_req = 0, i_resp = 0; i_resp < total;) {
#pragma HLS pipeline II=1
        if ((i_req < total) & !x_mem.read_addr.full()) {
            x_mem.read_addr.try_write(i_req); ++i_req;
        }
        if (!x_mem.read_data.empty()) {
            int16_vec_t xv; x_mem.read_data.try_read(xv);
            out.write(xv);
            ++i_resp;
        }
    }
}

// ============================================================================
// Reader: w_gate_reader — re-streams the half-matrix N times (once per token).
// ============================================================================
void w_gate_reader(
    const int N,
    tapa::async_mmap<int16_vec_t>& w_gate_mem,
    tapa::ostream<int16_vec_t>&    out
) {
    for (int t = 0; t < N; t++) {
        for (int i_req = 0, i_resp = 0; i_resp < W_GATE_HALF_VECS;) {
#pragma HLS pipeline II=1
            if ((i_req < W_GATE_HALF_VECS) & !w_gate_mem.read_addr.full()) {
                w_gate_mem.read_addr.try_write(i_req); ++i_req;
            }
            if (!w_gate_mem.read_data.empty()) {
                int16_vec_t wv; w_gate_mem.read_data.try_read(wv);
                out.write(wv);
                ++i_resp;
            }
        }
    }
}

// ============================================================================
// Reader: bias_dequant_reader — unchanged (single-shot).
// ============================================================================
void bias_dequant_reader(
    tapa::async_mmap<float>& mem,
    tapa::ostream<float>&    bias_out,
    tapa::ostream<float>&    dequant_out
) {
    for (int i_req = 0, i_resp = 0; i_resp < N_BIAS_DEQUANT;) {
#pragma HLS pipeline II=1
        if ((i_req < N_BIAS_DEQUANT) & !mem.read_addr.full()) {
            mem.read_addr.try_write(i_req); ++i_req;
        }
        if (!mem.read_data.empty()) {
            float v; mem.read_data.try_read(v);
            if (i_resp < N_EXPERTS_TOTAL)
                bias_out.write(v);
            else
                dequant_out.write(v);
            ++i_resp;
        }
    }
}

void dequant_sink(
    tapa::istream<float>& dequant_in
) {
    for (int i = 0; i < DEQUANT_PARAMS_PER_SLOT; i++) {
#pragma HLS pipeline II=1
        dequant_in.read();
    }
}

// ============================================================================
// Compute: gate_compute (unchanged from sequential baseline).
//   Outer loop over t: re-prefetch x into x_buf (forwarding to x_out for the
//   downstream batch_scheduler), run the 2-way parallel GEMV consuming W_gate
//   FIFOs once per token, then top-K and emit one route_pkt per token.
// ============================================================================
void gate_compute(
    const int   N,
    const int   local_expert_base,
    const float x_scale,
    const float w_gate_scale,
    const float w_gate_shift,
    tapa::istream<int16_vec_t>&    x_in,
    tapa::istream<int16_vec_t>&    w_gate_in,
    tapa::istream<int16_vec_t>&    w_gate_in_b,
    tapa::istream<float>&          bias_in,
    tapa::ostream<route_pkt_t>&    route_out,
    tapa::ostream<int16_vec_t>&    x_out
) {
    // Bias is global — load once.
    float bias[N_EXPERTS_TOTAL];
#pragma HLS array_partition variable=bias block factor=2 dim=1
    for (int i = 0; i < N_EXPERTS_TOTAL; i++) {
#pragma HLS pipeline II=1
        bias[i] = bias_in.read();
    }

    int16_vec_t x_buf[HIDDEN_VECS];
#pragma HLS bind_storage variable=x_buf type=RAM_2P impl=BRAM

    const float joint_scale = x_scale * w_gate_scale;

    for (int t = 0; t < N; t++) {
        // Prefetch this token's x and forward it to the scheduler.
        for (int v = 0; v < HIDDEN_VECS; v++) {
#pragma HLS pipeline II=1
            int16_vec_t xv = x_in.read();
            x_buf[v] = xv;
            x_out.write(xv);
        }

        float scores[N_EXPERTS_TOTAL];
#pragma HLS bind_storage variable=scores type=RAM_1P impl=BRAM
#pragma HLS array_partition variable=scores block factor=2 dim=1
        for (int e_lo = 0; e_lo < N_EXPERTS_TOTAL / 2; e_lo++) {
            ap_int<44> sum_lo = 0;
            ap_int<44> sum_hi = 0;
            for (int v = 0; v < HIDDEN_VECS; v++) {
#pragma HLS pipeline II=1
                int16_vec_t xv    = x_buf[v];
                int16_vec_t wv_lo = w_gate_in.read();
                int16_vec_t wv_hi = w_gate_in_b.read();
                ap_int<37> p_lo = 0, p_hi = 0;
                for (int k = 0; k < 8; k++) {
                    for (int l = 0; l < 4; l++) {
                        ap_int<16> x_val = xv[k*4+l];
                        ap_int<16> w_lo  = wv_lo[k*4+l];
                        ap_int<16> w_hi  = wv_hi[k*4+l];
                        p_lo += x_val * w_lo;
                        p_hi += x_val * w_hi;
                    }
                }
                sum_lo += p_lo;
                sum_hi += p_hi;
            }
            int e_hi = e_lo + N_EXPERTS_TOTAL / 2;
            scores[e_lo] = moe_sigmoidf((float)sum_lo * joint_scale + w_gate_shift) + bias[e_lo];
            scores[e_hi] = moe_sigmoidf((float)sum_hi * joint_scale + w_gate_shift) + bias[e_hi];
        }

        float top_lo_w [LOCAL_K];
        int   top_lo_ids[LOCAL_K];
#pragma HLS array_partition variable=top_lo_w  complete
#pragma HLS array_partition variable=top_lo_ids complete
        float top_hi_w [LOCAL_K];
        int   top_hi_ids[LOCAL_K];
#pragma HLS array_partition variable=top_hi_w  complete
#pragma HLS array_partition variable=top_hi_ids complete

        for (int i = 0; i < LOCAL_K; i++) {
#pragma HLS unroll
            top_lo_w  [i] = scores[i];
            top_lo_ids[i] = i;
            top_hi_w  [i] = scores[LOCAL_EXPERTS + i];
            top_hi_ids[i] = LOCAL_EXPERTS + i;
        }
        float lo_min_s; int lo_min_i;
        float hi_min_s; int hi_min_i;
        {
            float sl01, sl23, sh01, sh23; int il01, il23, ih01, ih23;
            if (top_lo_w[0] < top_lo_w[1]) { sl01 = top_lo_w[0]; il01 = 0; } else { sl01 = top_lo_w[1]; il01 = 1; }
            if (top_lo_w[2] < top_lo_w[3]) { sl23 = top_lo_w[2]; il23 = 2; } else { sl23 = top_lo_w[3]; il23 = 3; }
            if (sl01 < sl23)              { lo_min_s = sl01; lo_min_i = il01; } else { lo_min_s = sl23; lo_min_i = il23; }
            if (top_hi_w[0] < top_hi_w[1]) { sh01 = top_hi_w[0]; ih01 = 0; } else { sh01 = top_hi_w[1]; ih01 = 1; }
            if (top_hi_w[2] < top_hi_w[3]) { sh23 = top_hi_w[2]; ih23 = 2; } else { sh23 = top_hi_w[3]; ih23 = 3; }
            if (sh01 < sh23)              { hi_min_s = sh01; hi_min_i = ih01; } else { hi_min_s = sh23; hi_min_i = ih23; }
        }
        for (int r = 1; r < (LOCAL_EXPERTS / LOCAL_K); r++) {
            float cand_lo_s[LOCAL_K];
            int   cand_lo_e[LOCAL_K];
            float cand_hi_s[LOCAL_K];
            int   cand_hi_e[LOCAL_K];
#pragma HLS array_partition variable=cand_lo_s complete
#pragma HLS array_partition variable=cand_lo_e complete
#pragma HLS array_partition variable=cand_hi_s complete
#pragma HLS array_partition variable=cand_hi_e complete
            for (int j = 0; j < LOCAL_K; j++) {
#pragma HLS unroll
                int elo = r * LOCAL_K + j;
                int ehi = LOCAL_EXPERTS + r * LOCAL_K + j;
                cand_lo_s[j] = scores[elo]; cand_lo_e[j] = elo;
                cand_hi_s[j] = scores[ehi]; cand_hi_e[j] = ehi;
            }
            bool fast_lo = false, fast_hi = false;
            for (int j = 0; j < LOCAL_K; j++) {
#pragma HLS unroll
                fast_lo |= (cand_lo_s[j] > lo_min_s);
                fast_hi |= (cand_hi_s[j] > hi_min_s);
            }
            if (fast_lo) {
                for (int j = 0; j < LOCAL_K; j++) {
#pragma HLS pipeline off
                    if (cand_lo_s[j] > lo_min_s) {
                        top_lo_w  [lo_min_i] = cand_lo_s[j];
                        top_lo_ids[lo_min_i] = cand_lo_e[j];
                        float s01, s23; int i01, i23;
                        if (top_lo_w[0] < top_lo_w[1]) { s01 = top_lo_w[0]; i01 = 0; } else { s01 = top_lo_w[1]; i01 = 1; }
                        if (top_lo_w[2] < top_lo_w[3]) { s23 = top_lo_w[2]; i23 = 2; } else { s23 = top_lo_w[3]; i23 = 3; }
                        if (s01 < s23) { lo_min_s = s01; lo_min_i = i01; } else { lo_min_s = s23; lo_min_i = i23; }
                    }
                }
            }
            if (fast_hi) {
                for (int j = 0; j < LOCAL_K; j++) {
#pragma HLS pipeline off
                    if (cand_hi_s[j] > hi_min_s) {
                        top_hi_w  [hi_min_i] = cand_hi_s[j];
                        top_hi_ids[hi_min_i] = cand_hi_e[j];
                        float s01, s23; int i01, i23;
                        if (top_hi_w[0] < top_hi_w[1]) { s01 = top_hi_w[0]; i01 = 0; } else { s01 = top_hi_w[1]; i01 = 1; }
                        if (top_hi_w[2] < top_hi_w[3]) { s23 = top_hi_w[2]; i23 = 2; } else { s23 = top_hi_w[3]; i23 = 3; }
                        if (s01 < s23) { hi_min_s = s01; hi_min_i = i01; } else { hi_min_s = s23; hi_min_i = i23; }
                    }
                }
            }
        }

        float wsum = 0.0f;
        for (int i = 0; i < LOCAL_K; i++) {
#pragma HLS unroll
            wsum += top_lo_w[i] + top_hi_w[i];
        }
        float wscale = (wsum > 1e-12f) ? (ROUTE_SCALE / wsum) : 0.0f;

        route_pkt_t pkt;
        if (local_expert_base == 0) {
            for (int i = 0; i < LOCAL_K; i++) {
#pragma HLS unroll
                pkt.local_expert_ids[i] = (expert_id_t)top_lo_ids[i];
                pkt.weights[i]          = top_lo_w[i] * wscale;
            }
        } else {
            for (int i = 0; i < LOCAL_K; i++) {
#pragma HLS unroll
                pkt.local_expert_ids[i] = (expert_id_t)top_hi_ids[i];
                pkt.weights[i]          = top_hi_w[i] * wscale;
            }
        }
        route_out.write(pkt);
    }
}

// ============================================================================
// dispatch_core — tile-streaming dispatch (replaces dispatch_compute).
//
// Processes N tokens in tiles of N_TILE_MAX. Per tile:
//   Phase 1: zero pending bank, ingest tile_n tokens (x_cache + route_pkts)
//   Phase 2: dispatch rounds (argmax top-4 experts, emit hdr+triggers+x)
//            tile_done is pre-computed before emitting each round's headers.
// After all tiles: emit one last_round sentinel (-1 triggers + dummy x).
//
// Weight loaders see a continuous trigger stream with NO per-tile sentinel.
// Only the final -1 sentinel at kernel end causes them to exit.
// ============================================================================
void dispatch_core(
    const int   N,
    const int   local_expert_base,
    const float x_scale,
    tapa::istream<route_pkt_t>&                          route_in,
    tapa::istream<int16_vec_t>&                          x_in,
    tapa::ostreams<round_hdr_t,      N_PARALLEL_SLOTS>&  hdr_out,
    tapa::ostreams<int16_vec_t,      N_PARALLEL_SLOTS>&  x_out
) {
    int16_vec_t x_cache[N_TILE_MAX][HIDDEN_VECS];
#pragma HLS bind_storage variable=x_cache type=RAM_2P impl=BRAM

    token_id_t pending_token[LOCAL_EXPERTS][TOP_K];
    float      pending_weight[LOCAL_EXPERTS][TOP_K];
    ap_uint<4> pending_count [LOCAL_EXPERTS];
#pragma HLS bind_storage variable=pending_token  type=RAM_2P impl=BRAM
#pragma HLS bind_storage variable=pending_weight type=RAM_2P impl=BRAM
#pragma HLS array_partition variable=pending_count complete

    const int n_tiles = (N + N_TILE_MAX - 1) / N_TILE_MAX;

    for (int tile = 0; tile < n_tiles; tile++) {
#pragma HLS loop_tripcount min=1 max=16
        const int tile_base = tile * N_TILE_MAX;
        const int tile_n = ((N - tile_base) < N_TILE_MAX)
                               ? (N - tile_base) : N_TILE_MAX;

        // ===== Phase 1: zero pending + ingest =====
        for (int e = 0; e < LOCAL_EXPERTS; e++) {
#pragma HLS pipeline II=1
            pending_count[e] = 0;
        }

        for (int t = 0; t < tile_n; t++) {
            for (int v = 0; v < HIDDEN_VECS; v++) {
#pragma HLS pipeline II=1
                x_cache[t][v] = x_in.read();
            }
            route_pkt_t pkt = route_in.read();
            for (int i = 0; i < LOCAL_K; i++) {
#pragma HLS pipeline off
                int eid = (int)pkt.local_expert_ids[i];
                ap_uint<4> c = pending_count[eid];
                pending_token [eid][c] = (token_id_t)t;
                pending_weight[eid][c] = pkt.weights[i];
                pending_count[eid] = c + 1;
            }
        }

        // ===== Phase 2: dispatch rounds for this tile =====
        bool any_pending = false;
        for (int e = 0; e < LOCAL_EXPERTS; e++) {
#pragma HLS pipeline II=1
            if (pending_count[e] != 0) any_pending = true;
        }

        while (any_pending) {
#pragma HLS loop_tripcount min=1 max=8
            // 4 sequential argmax passes to pick top-4 experts.
            bool picked_mask[LOCAL_EXPERTS];
#pragma HLS array_partition variable=picked_mask cyclic factor=8
            for (int e = 0; e < LOCAL_EXPERTS; e++) {
#pragma HLS pipeline II=1
                picked_mask[e] = false;
            }

            int chosen_expert[N_PARALLEL_SLOTS];
            ap_uint<4> chosen_count[N_PARALLEL_SLOTS];
#pragma HLS array_partition variable=chosen_expert complete
#pragma HLS array_partition variable=chosen_count  complete
            for (int s = 0; s < N_PARALLEL_SLOTS; s++) {
#pragma HLS unroll
                chosen_expert[s] = -1;
                chosen_count[s]  = 0;
            }

            for (int s = 0; s < N_PARALLEL_SLOTS; s++) {
#pragma HLS pipeline off
                int best_e = -1;
                ap_uint<4> best_c = 0;
                for (int e = 0; e < LOCAL_EXPERTS; e++) {
#pragma HLS pipeline II=1
                    ap_uint<4> c = pending_count[e];
                    bool ok = (!picked_mask[e]) && (c != 0) && (c > best_c);
                    if (ok) { best_c = c; best_e = e; }
                }
                chosen_expert[s] = best_e;
                chosen_count[s]  = best_c;
                if (best_e >= 0) picked_mask[best_e] = true;
            }

            // Build per-slot routing info.
            token_id_t  slot_toks [N_PARALLEL_SLOTS][TOKEN_LANES];
            float       slot_ws   [N_PARALLEL_SLOTS][TOKEN_LANES];
            expert_id_t slot_eid  [N_PARALLEL_SLOTS];
            bool        slot_act  [N_PARALLEL_SLOTS];
            ap_uint<3>  slot_cnt  [N_PARALLEL_SLOTS];
#pragma HLS array_partition variable=slot_toks complete dim=0
#pragma HLS array_partition variable=slot_ws   complete dim=0
#pragma HLS array_partition variable=slot_eid  complete
#pragma HLS array_partition variable=slot_act  complete
#pragma HLS array_partition variable=slot_cnt  complete
            for (int s = 0; s < N_PARALLEL_SLOTS; s++) {
#pragma HLS pipeline off
                int eid = chosen_expert[s];
                if (eid < 0) {
                    slot_eid[s] = (expert_id_t)0;
                    slot_act[s] = false;
                    slot_cnt[s] = 0;
                    for (int lane = 0; lane < TOKEN_LANES; lane++) {
#pragma HLS unroll
                        slot_toks[s][lane] = (token_id_t)0;
                        slot_ws  [s][lane] = 0.0f;
                    }
                } else {
                    slot_eid[s] = (expert_id_t)eid;
                    slot_act[s] = true;
                    ap_uint<4> have = chosen_count[s];
                    ap_uint<3> popped = (have >= (ap_uint<4>)TOKEN_LANES)
                                            ? (ap_uint<3>)TOKEN_LANES
                                            : (ap_uint<3>)(unsigned int)have;
                    slot_cnt[s] = popped;
                    for (int lane = 0; lane < TOKEN_LANES; lane++) {
#pragma HLS unroll
                        if (lane < (int)popped) {
                            ap_uint<4> idx = have - (ap_uint<4>)(lane + 1);
                            slot_toks[s][lane] = pending_token [eid][idx];
                            slot_ws  [s][lane] = pending_weight[eid][idx];
                        } else {
                            slot_toks[s][lane] = (token_id_t)0;
                            slot_ws  [s][lane] = 0.0f;
                        }
                    }
                }
            }

            // Decrement pending_count.
            for (int s = 0; s < N_PARALLEL_SLOTS; s++) {
#pragma HLS pipeline off
                int eid = chosen_expert[s];
                if (eid >= 0) {
                    pending_count[eid] = chosen_count[s] - (ap_uint<4>)slot_cnt[s];
                }
            }

            // Pre-compute tile_done: sum remaining pending vs dispatched.
            int total_remaining = 0;
            for (int e = 0; e < LOCAL_EXPERTS; e++) {
#pragma HLS pipeline II=1
                total_remaining += (int)pending_count[e];
            }
            bool tile_done_flag = (total_remaining == 0);

            // Emit per-slot header + 6 triggers.
            for (int s = 0; s < N_PARALLEL_SLOTS; s++) {
#pragma HLS unroll
                round_hdr_t hdr;
                for (int lane = 0; lane < TOKEN_LANES; lane++) {
#pragma HLS unroll
                    hdr.token_ids      [lane] = slot_toks[s][lane];
                    hdr.routing_weights[lane] = slot_ws  [s][lane];
                }
                hdr.x_scale     = x_scale;
                hdr.token_count = slot_cnt[s];
                hdr.slot_active = slot_act[s];
                hdr.last_round  = false;
                hdr.tile_done   = tile_done_flag;
                int rel_eid = (int)slot_eid[s] - local_expert_base;
                if (!slot_act[s]) rel_eid = 0;
                hdr.trigger_eid = rel_eid;
                hdr_out[s].write(hdr);
            }

            // Stream per-slot x.
            for (int lane = 0; lane < TOKEN_LANES; lane++) {
                for (int v = 0; v < HIDDEN_VECS; v++) {
#pragma HLS pipeline II=1
                    for (int s = 0; s < N_PARALLEL_SLOTS; s++) {
#pragma HLS unroll
                        int t_id = (lane < (int)slot_cnt[s])
                                       ? (int)slot_toks[s][lane] : 0;
                        x_out[s].write(x_cache[t_id][v]);
                    }
                }
            }

            any_pending = !tile_done_flag;
        }  // while any_pending
    }  // for tile

    // ===== Emit terminating sentinel (last_round=true) — ONCE at kernel end =====
    for (int s = 0; s < N_PARALLEL_SLOTS; s++) {
#pragma HLS unroll
        round_hdr_t hdr;
        for (int lane = 0; lane < TOKEN_LANES; lane++) {
#pragma HLS unroll
            hdr.token_ids      [lane] = (token_id_t)0;
            hdr.routing_weights[lane] = 0.0f;
        }
        hdr.x_scale     = x_scale;
        hdr.token_count = 0;
        hdr.slot_active = false;
        hdr.last_round  = true;
        hdr.tile_done   = false;
        hdr.trigger_eid = -1;
        hdr_out[s].write(hdr);
    }
    for (int lane = 0; lane < TOKEN_LANES; lane++) {
        for (int v = 0; v < HIDDEN_VECS; v++) {
#pragma HLS pipeline II=1
            int16_vec_t z;
            for (int k = 0; k < VEC_WIDTH; k++) {
#pragma HLS unroll
                z[k] = (ap_int<16>)0;
            }
            for (int s = 0; s < N_PARALLEL_SLOTS; s++) {
#pragma HLS unroll
                x_out[s].write(z);
            }
        }
    }
}

// ============================================================================
// Reader: w1_w2c_weight_loader (round-driven) — reads triggers until a
// special "last" trigger is consumed. Since the dispatcher emits one trigger
// per round (active or filler), we count rounds by streaming a fixed number
// per call. To avoid needing an end-marker, we run a fixed upper-bound loop:
// the slot_dispatcher emits exactly (rounds_used + 1 last_round) triggers,
// and the reader processes triggers until the matching loaded data is
// consumed downstream. Here we use a simple count-driven loop based on the
// number of triggers received, where each trigger triggers one full
// W1_W2C_PORT_VECS load.
// We rely on TAPA backpressure to stop reading once downstream stops; in
// csim, we close the stream when slot_dispatcher exits. To avoid a hang,
// we loop "while !trigger_in.eot()" via try_read poll. Actually, in TAPA
// streams there is no cheap eot signal; the simplest pattern is to use the
// MAX_ROUNDS+1 upper bound (plan: rounds <= N_TILE_MAX, plus one last
// terminator), so loop exactly (MAX_ROUNDS + 1) times.
// ============================================================================
void w1_w2c_weight_loader(
    tapa::istream<weight_trigger_t>& trigger_in,
    tapa::async_mmap<int16_vec_t>&   W_mem,
    tapa::ostream<int16_vec_t>&      W_fifo
) {
    while (true) {
#pragma HLS loop_tripcount min=1 max=33
        weight_trigger_t trig = trigger_in.read();
        if (trig < 0) { break; }  // sentinel: terminator, no load
        const int base = trig * W1_W2C_PORT_VECS;
        for (int i_req = 0, i_resp = 0; i_resp < W1_W2C_PORT_VECS;) {
#pragma HLS pipeline II=1
            if ((i_req < W1_W2C_PORT_VECS) & !W_mem.read_addr.full()) {
                W_mem.read_addr.try_write(base + i_req); ++i_req;
            }
            if (!W_mem.read_data.empty()) {
                int16_vec_t tmp; W_mem.read_data.try_read(tmp);
                W_fifo.write(tmp);
                ++i_resp;
            }
        }
    }
}

// ============================================================================
// Reader: w3_w2c_weight_loader (round-driven) — symmetric.
// ============================================================================
void w3_w2c_weight_loader(
    tapa::istream<weight_trigger_t>& trigger_in,
    tapa::async_mmap<int16_vec_t>&   W_mem,
    tapa::ostream<int16_vec_t>&      W_fifo
) {
    while (true) {
#pragma HLS loop_tripcount min=1 max=33
        weight_trigger_t trig = trigger_in.read();
        if (trig < 0) { break; }  // sentinel: terminator, no load
        const int base = trig * W3_W2C_PORT_VECS;
        for (int i_req = 0, i_resp = 0; i_resp < W3_W2C_PORT_VECS;) {
#pragma HLS pipeline II=1
            if ((i_req < W3_W2C_PORT_VECS) & !W_mem.read_addr.full()) {
                W_mem.read_addr.try_write(base + i_req); ++i_req;
            }
            if (!W_mem.read_data.empty()) {
                int16_vec_t tmp; W_mem.read_data.try_read(tmp);
                W_fifo.write(tmp);
                ++i_resp;
            }
        }
    }
}

// ============================================================================
// dequant_split / dequant_sink — UNCHANGED. Dequant params global per slot.
// ============================================================================
void dequant_split(
    const float x_scale,                  // [b32] pre-multiply w1/w3_scale*x_scale here
    tapa::istream<float>& dequant_in,
    tapa::ostream<float>& dequant_fwd,
    tapa::ostream<float>& w1_dq_out,
    tapa::ostream<float>& w3_dq_out,
    tapa::ostream<float>& w2_dq_a_out,
    tapa::ostream<float>& w2_dq_b_out
) {
    for (int i = 0; i < DEQUANT_PARAMS_PER_SLOT; i++) {
#pragma HLS pipeline II=1
        float v = dequant_in.read();
        dequant_fwd.write(v);
        if (i == 0) {
            w1_dq_out.write(v * x_scale);  // [b32] w1_scale * x_scale folded
        } else if (i == 1) {
            w1_dq_out.write(v);            // w1_shift unchanged
        } else if (i == 2) {
            w3_dq_out.write(v * x_scale);  // [b32] w3_scale * x_scale folded
        } else if (i == 3) {
            w3_dq_out.write(v);            // w3_shift unchanged
        } else {
            w2_dq_a_out.write(v);
            w2_dq_b_out.write(v);
        }
    }
}

// ============================================================================
// ffn_x_split (round-driven): per round (excluding last_round) forwards
// HIDDEN_VECS x beats and a per-stream header. On last_round=true, forwards
// the header to downstream (signalling stop) but still drains HIDDEN_VECS
// x beats from x_in (kept uniform by slot_dispatcher).
// ============================================================================
void ffn_x_split(
    tapa::istream<round_hdr_t>&    hdr_in,
    tapa::istream<int16_vec_t>&    x_in,
    tapa::ostream<round_hdr_t>&    hdr_w1_out,
    tapa::ostream<round_hdr_t>&    hdr_w3_out,
    tapa::ostream<round_hdr_t>&    hdr_swiglu_out,
    tapa::ostream<int16_vec_t>&    x1_out,
    tapa::ostream<int16_vec_t>&    x3_out,
    tapa::ostream<weight_trigger_t>& trigger1a_out,
    tapa::ostream<weight_trigger_t>& trigger1b_out,
    tapa::ostream<weight_trigger_t>& trigger1c_out,
    tapa::ostream<weight_trigger_t>& trigger3a_out,
    tapa::ostream<weight_trigger_t>& trigger3b_out,
    tapa::ostream<weight_trigger_t>& trigger3c_out
) {
    bool stop = false;
    int rnd = 0;
    while (!stop) {
#pragma HLS loop_tripcount min=1 max=33
        round_hdr_t hdr = hdr_in.read();
        hdr_w1_out.write(hdr);
        hdr_w3_out.write(hdr);
        hdr_swiglu_out.write(hdr);
        int trig = hdr.trigger_eid;
        trigger1a_out.write(trig);
        trigger1b_out.write(trig);
        trigger1c_out.write(trig);
        trigger3a_out.write(trig);
        trigger3b_out.write(trig);
        trigger3c_out.write(trig);
        for (int lane = 0; lane < TOKEN_LANES; lane++) {
            for (int v = 0; v < HIDDEN_VECS; v++) {
#pragma HLS pipeline II=1
                int16_vec_t xv = x_in.read();
                x1_out.write(xv);
                x3_out.write(xv);
            }
        }
        if (hdr.last_round) stop = true;
        rnd++;
    }
}

// ============================================================================
// ffn_w1_w2_proj (round-driven): processes rounds until last_round=true.
// On a non-last round (regardless of slot_active), runs the 2-stage GEMV +
// emits HIDDEN_VECS_FP/2 partial_pkt's to y_accumulator. When slot_active is
// false, routing_weight is 0.0f so the contribution is zero.
// On last_round=true, drains HIDDEN_VECS x's and emits one "tail" partial_pkt
// with last_round=true so y_accumulator knows this slot is done.
// ============================================================================
// ============= bisect-8 ffn_w1_w2_proj =============
void ffn_w1_w2_proj(
    tapa::istream<round_hdr_t>& hdr_in,
    tapa::istream<int16_vec_t>& x1_in,
    tapa::istream<int16_vec_t>& W1_W2a_fifo,
    tapa::istream<int16_vec_t>& W1_W2b_fifo,
    tapa::istream<int16_vec_t>& W1_W2c_fifo,
    tapa::istream<float>&       w1_dq_in,
    tapa::istream<int16_half_vec_t>& inter_a_in,
    tapa::istream<float>&       inter_scale_a_in,
    tapa::istream<float>&       w2_dq_a_in,
    tapa::ostream<float3_t>&    acc1_out,
    tapa::ostream<partial_pkt_t>& partial_a_out
) {
    // Read dequant scales/shifts ONCE at task start.
    float w1_scale = w1_dq_in.read();
    float w1_shift = w1_dq_in.read();
    float w2_scale = w2_dq_a_in.read();
    float w2_shift = w2_dq_a_in.read();

    ap_uint<32> x_cur[TOKEN_LANES][HIDDEN_PACKED];
#pragma HLS bind_storage variable=x_cur type=RAM_2P impl=BRAM
#pragma HLS array_partition variable=x_cur complete dim=1
#pragma HLS array_partition variable=x_cur cyclic factor=16 dim=2
    int16_half_vec_t inter_buf[TOKEN_LANES][INTER_HALF_VECS];
#pragma HLS array_partition variable=inter_buf complete dim=1
#pragma HLS array_partition variable=inter_buf cyclic factor=2 dim=2
    ap_uint<64> y_buf[TOKEN_LANE_PAIRS][3 * W2_SIXTH_ROWS];
#pragma HLS bind_storage variable=y_buf type=RAM_2P impl=URAM
#pragma HLS array_partition variable=y_buf complete dim=1
#pragma HLS array_partition variable=y_buf cyclic factor=16 dim=2

    bool stop = false;
    int rnd = 0;
    while (!stop) {
#pragma HLS loop_tripcount min=1 max=33
        round_hdr_t hdr = hdr_in.read();
        for (int lane = 0; lane < TOKEN_LANES; lane++) {
            for (int v = 0; v < HIDDEN_VECS; v++) {
#pragma HLS pipeline II=1
                int16_vec_t xv = x1_in.read();
                for (int k = 0; k < VEC_WIDTH/2; k++) {
#pragma HLS unroll
                    ap_uint<32> packed;
                    packed(15,0) = xv[2*k];
                    packed(31,16) = xv[2*k+1];
                    x_cur[lane][v * (VEC_WIDTH/2) + k] = packed;
                }
            }
        }
        if (hdr.last_round) {
            partial_pkt_t tail;
            tail.token_id = (token_id_t)0;
            tail.slot_active = false;
            tail.last_round = true;
            tail.tile_done = false;
            for (int k = 0; k < VEC_WIDTH_FP; k++) {
#pragma HLS unroll
                tail.y[k] = (ap_uint<16>)0;
            }
            partial_a_out.write(tail);
            stop = true;
            continue;
        } else {
            for (int stage = 0; stage < 2; stage++) {
                float scale_w_lane[TOKEN_LANES];
#pragma HLS array_partition variable=scale_w_lane complete
                if (stage == 1) {
                    for (int lane = 0; lane < TOKEN_LANES; lane++) {
#pragma HLS pipeline II=1
                        scale_w_lane[lane] = w2_scale * inter_scale_a_in.read();
                    }
                    for (int lane = 0; lane < TOKEN_LANES; lane++) {
                        for (int vh = 0; vh < INTER_HALF_VECS; vh++) {
#pragma HLS pipeline II=1
                            inter_buf[lane][vh] = inter_a_in.read();
                        }
                    }
                }
                int n_rows = (stage == 0) ? W1_THIRD_ROWS : W2_SIXTH_ROWS;
                int n_vecs = (stage == 0) ? HIDDEN_VECS   : INTER_VECS;
                int row_iters = n_vecs + ((stage == 0) ? TOKEN_LANES : 0);
                ap_int<44> sum_a[TOKEN_LANES], sum_b[TOKEN_LANES], sum_c[TOKEN_LANES];
#pragma HLS array_partition variable=sum_a complete
#pragma HLS array_partition variable=sum_b complete
#pragma HLS array_partition variable=sum_c complete
                for (int lane = 0; lane < TOKEN_LANES; lane++) {
#pragma HLS unroll
                    sum_a[lane] = 0; sum_b[lane] = 0; sum_c[lane] = 0;
                }
                int rh = 0, pos = 0;
                const int total_iters = n_rows * row_iters;
                for (int it = 0; it < total_iters; it++) {
#pragma HLS loop_tripcount min=1376 max=155724
#pragma HLS pipeline II=1
                    if (pos == 0) {
                        for (int lane = 0; lane < TOKEN_LANES; lane++) {
#pragma HLS unroll
                            sum_a[lane] = 0; sum_b[lane] = 0; sum_c[lane] = 0;
                        }
                    }
                    if (pos < n_vecs) {
                        int16_vec_t wav = W1_W2a_fifo.read();
                        int16_vec_t wbv = W1_W2b_fifo.read();
                        int16_vec_t wcv = W1_W2c_fifo.read();
                        for (int lane = 0; lane < TOKEN_LANES; lane++) {
#pragma HLS unroll
                            int16_vec_t xv;
                            if (stage == 0) {
                                for (int k = 0; k < VEC_WIDTH/2; k++) {
#pragma HLS unroll
                                    ap_uint<32> packed = x_cur[lane][pos * (VEC_WIDTH/2) + k];
                                    xv[2*k] = ap_int<16>(packed(15,0));
                                    xv[2*k+1] = ap_int<16>(packed(31,16));
                                }
                            } else {
                                int16_half_vec_t lo = inter_buf[lane][pos * 2];
                                int16_half_vec_t hi = inter_buf[lane][pos * 2 + 1];
                                for (int k = 0; k < VEC_HALF; k++) {
#pragma HLS unroll
                                    xv[k] = lo[k];
                                    xv[VEC_HALF + k] = hi[k];
                                }
                            }
                            ap_int<37> pa = 0, pb = 0, pc = 0;
                            for (int k = 0; k < VEC_WIDTH; k++) {
#pragma HLS unroll
                                pa += xv[k] * wav[k];
                                pb += xv[k] * wbv[k];
                                pc += xv[k] * wcv[k];
                            }
                            sum_a[lane] += pa; sum_b[lane] += pb; sum_c[lane] += pc;
                        }
                        if ((stage == 1) && (pos == n_vecs - 1)) {
                            for (int pl = 0; pl < TOKEN_LANE_PAIRS; pl++) {
#pragma HLS unroll
                                int l0 = pl * 2, l1 = pl * 2 + 1;
                                float sw0 = scale_w_lane[l0], sw1 = scale_w_lane[l1];
                                float va0 = (float)sum_a[l0] * sw0 + w2_shift;
                                float va1 = (float)sum_a[l1] * sw1 + w2_shift;
                                float vb0 = (float)sum_b[l0] * sw0 + w2_shift;
                                float vb1 = (float)sum_b[l1] * sw1 + w2_shift;
                                float vc0 = (float)sum_c[l0] * sw0 + w2_shift;
                                float vc1 = (float)sum_c[l1] * sw1 + w2_shift;
                                ap_uint<64> pa;
                                pa(31,0) = tapa::bit_cast<ap_uint<32>>(va0);
                                pa(63,32) = tapa::bit_cast<ap_uint<32>>(va1);
                                y_buf[pl][3*rh] = pa;
                                ap_uint<64> pb;
                                pb(31,0) = tapa::bit_cast<ap_uint<32>>(vb0);
                                pb(63,32) = tapa::bit_cast<ap_uint<32>>(vb1);
                                y_buf[pl][3*rh + 1] = pb;
                                ap_uint<64> pc;
                                pc(31,0) = tapa::bit_cast<ap_uint<32>>(vc0);
                                pc(63,32) = tapa::bit_cast<ap_uint<32>>(vc1);
                                y_buf[pl][3*rh + 2] = pc;
                            }
                        }
                    } else {
                        int lane = pos - n_vecs;
                        float3_t triple;
                        triple[0] = (float)sum_a[lane] * w1_scale + w1_shift;
                        triple[1] = (float)sum_b[lane] * w1_scale + w1_shift;
                        triple[2] = (float)sum_c[lane] * w1_scale + w1_shift;
                        acc1_out.write(triple);
                    }
                    if (pos == row_iters - 1) { rh++; pos = 0; }
                    else { pos++; }
                }
            }
            // Per-round emit partial_pkt's: TOKEN_LANES x HIDDEN_VECS_FP/2.
            for (int lane2 = 0; lane2 < TOKEN_LANES; lane2++) {
                float w_route = hdr.routing_weights[lane2];
                token_id_t tok = hdr.token_ids[lane2];
                bool lane_active = hdr.slot_active && (lane2 < (int)hdr.token_count);
                for (int v2 = 0; v2 < HIDDEN_VECS_FP/2; v2++) {
#pragma HLS pipeline II=1
                    partial_pkt_t pkt;
                    pkt.token_id = tok;
                    pkt.slot_active = lane_active;
                    pkt.last_round = false;
                    pkt.tile_done = hdr.tile_done;
                    for (int k = 0; k < VEC_WIDTH_FP; k++) {
#pragma HLS unroll
                        ap_uint<64> packed = y_buf[lane2 / 2][v2 * VEC_WIDTH_FP + k];
                        ap_uint<32> bits = (lane2 & 1) ? ap_uint<32>(packed(63, 32)) : ap_uint<32>(packed(31, 0));
                        float val = w_route * tapa::bit_cast<float>(bits);
                        ap_uint<32> fp32 = tapa::bit_cast<ap_uint<32>>(val);
                        pkt.y[k] = (ap_uint<16>)(fp32 >> 16);
                    }
                    partial_a_out.write(pkt);
                }
            }
        }
        rnd++;
    }
    return;
}
// ============= end bisect-8 ffn_w1_w2_proj =============

// ============================================================================
// ffn_w3_w2_proj (round-driven): symmetric to ffn_w1_w2_proj.
// ============= bisect-8 ffn_w3_w2_proj =============
void ffn_w3_w2_proj(
    tapa::istream<round_hdr_t>& hdr_in,
    tapa::istream<int16_vec_t>& x3_in,
    tapa::istream<int16_vec_t>& W3_W2a_fifo,
    tapa::istream<int16_vec_t>& W3_W2b_fifo,
    tapa::istream<int16_vec_t>& W3_W2c_fifo,
    tapa::istream<float>&       w3_dq_in,
    tapa::istream<int16_half_vec_t>& inter_b_in,
    tapa::istream<float>&       inter_scale_b_in,
    tapa::istream<float>&       w2_dq_b_in,
    tapa::ostream<float3_t>&    acc3_out,
    tapa::ostream<partial_pkt_t>& partial_b_out
) {
    float w3_scale = w3_dq_in.read();
    float w3_shift = w3_dq_in.read();
    float w2_scale = w2_dq_b_in.read();
    float w2_shift = w2_dq_b_in.read();

    ap_uint<32> x_cur[TOKEN_LANES][HIDDEN_PACKED];
#pragma HLS bind_storage variable=x_cur type=RAM_2P impl=BRAM
#pragma HLS array_partition variable=x_cur complete dim=1
#pragma HLS array_partition variable=x_cur cyclic factor=16 dim=2
    int16_half_vec_t inter_buf[TOKEN_LANES][INTER_HALF_VECS];
#pragma HLS array_partition variable=inter_buf complete dim=1
#pragma HLS array_partition variable=inter_buf cyclic factor=2 dim=2
    ap_uint<64> y_buf[TOKEN_LANE_PAIRS][3 * W2_SIXTH_ROWS];
#pragma HLS bind_storage variable=y_buf type=RAM_2P impl=URAM
#pragma HLS array_partition variable=y_buf complete dim=1
#pragma HLS array_partition variable=y_buf cyclic factor=16 dim=2

    bool stop = false;
    int rnd = 0;
    while (!stop) {
#pragma HLS loop_tripcount min=1 max=33
        round_hdr_t hdr = hdr_in.read();
        // [bisect-16] hoist x3_in.read() (mirror)
        for (int lane = 0; lane < TOKEN_LANES; lane++) {
            for (int v = 0; v < HIDDEN_VECS; v++) {
#pragma HLS pipeline II=1
                int16_vec_t xv = x3_in.read();
                // Pack 32 int16 values into 16 ap_uint<32> words
                for (int k = 0; k < VEC_WIDTH/2; k++) {
#pragma HLS unroll
                    ap_uint<32> packed;
                    packed(15,0) = xv[2*k];
                    packed(31,16) = xv[2*k+1];
                    x_cur[lane][v * (VEC_WIDTH/2) + k] = packed;
                }
            }
        }
        if (hdr.last_round) {
            partial_pkt_t tail;
            tail.token_id = (token_id_t)0;
            tail.slot_active = false;
            tail.last_round = true;
            tail.tile_done = false;
            for (int k = 0; k < VEC_WIDTH_FP; k++) {
#pragma HLS unroll
                tail.y[k] = (ap_uint<16>)0;
            }
            partial_b_out.write(tail);
            stop = true;
            continue;
        } else {
            for (int stage = 0; stage < 2; stage++) {
                float scale_w_lane[TOKEN_LANES];
#pragma HLS array_partition variable=scale_w_lane complete
                if (stage == 1) {
                    for (int lane = 0; lane < TOKEN_LANES; lane++) {
#pragma HLS pipeline II=1
                        scale_w_lane[lane] = w2_scale * inter_scale_b_in.read();
                    }
                    for (int lane = 0; lane < TOKEN_LANES; lane++) {
                        for (int vh = 0; vh < INTER_HALF_VECS; vh++) {
#pragma HLS pipeline II=1
                            inter_buf[lane][vh] = inter_b_in.read();
                        }
                    }
                }
                int n_rows = (stage == 0) ? W1_THIRD_ROWS : W2_SIXTH_ROWS;
                int n_vecs = (stage == 0) ? HIDDEN_VECS   : INTER_VECS;
                int row_iters = n_vecs + ((stage == 0) ? TOKEN_LANES : 0);
                ap_int<44> sum_a[TOKEN_LANES], sum_b[TOKEN_LANES], sum_c[TOKEN_LANES];
#pragma HLS array_partition variable=sum_a complete
#pragma HLS array_partition variable=sum_b complete
#pragma HLS array_partition variable=sum_c complete
                for (int lane = 0; lane < TOKEN_LANES; lane++) {
#pragma HLS unroll
                    sum_a[lane] = 0; sum_b[lane] = 0; sum_c[lane] = 0;
                }
                int rh = 0, pos = 0;
                const int total_iters = n_rows * row_iters;
                for (int it = 0; it < total_iters; it++) {
#pragma HLS loop_tripcount min=1376 max=155724
#pragma HLS pipeline II=1
                    if (pos == 0) {
                        for (int lane = 0; lane < TOKEN_LANES; lane++) {
#pragma HLS unroll
                            sum_a[lane] = 0; sum_b[lane] = 0; sum_c[lane] = 0;
                        }
                    }
                    if (pos < n_vecs) {
                        int16_vec_t wav = W3_W2a_fifo.read();
                        int16_vec_t wbv = W3_W2b_fifo.read();
                        int16_vec_t wcv = W3_W2c_fifo.read();
                        for (int lane = 0; lane < TOKEN_LANES; lane++) {
#pragma HLS unroll
                            int16_vec_t xv;
                            if (stage == 0) {
                                for (int k = 0; k < VEC_WIDTH/2; k++) {
#pragma HLS unroll
                                    ap_uint<32> packed = x_cur[lane][pos * (VEC_WIDTH/2) + k];
                                    xv[2*k] = ap_int<16>(packed(15,0));
                                    xv[2*k+1] = ap_int<16>(packed(31,16));
                                }
                            } else {
                                int16_half_vec_t lo = inter_buf[lane][pos * 2];
                                int16_half_vec_t hi = inter_buf[lane][pos * 2 + 1];
                                for (int k = 0; k < VEC_HALF; k++) {
#pragma HLS unroll
                                    xv[k] = lo[k];
                                    xv[VEC_HALF + k] = hi[k];
                                }
                            }
                            ap_int<37> pa = 0, pb = 0, pc = 0;
                            for (int k = 0; k < VEC_WIDTH; k++) {
#pragma HLS unroll
                                pa += xv[k] * wav[k];
                                pb += xv[k] * wbv[k];
                                pc += xv[k] * wcv[k];
                            }
                            sum_a[lane] += pa; sum_b[lane] += pb; sum_c[lane] += pc;
                        }
                        if ((stage == 1) && (pos == n_vecs - 1)) {
                            for (int pl = 0; pl < TOKEN_LANE_PAIRS; pl++) {
#pragma HLS unroll
                                int l0 = pl * 2, l1 = pl * 2 + 1;
                                float sw0 = scale_w_lane[l0], sw1 = scale_w_lane[l1];
                                float va0 = (float)sum_a[l0] * sw0 + w2_shift;
                                float va1 = (float)sum_a[l1] * sw1 + w2_shift;
                                float vb0 = (float)sum_b[l0] * sw0 + w2_shift;
                                float vb1 = (float)sum_b[l1] * sw1 + w2_shift;
                                float vc0 = (float)sum_c[l0] * sw0 + w2_shift;
                                float vc1 = (float)sum_c[l1] * sw1 + w2_shift;
                                ap_uint<64> pa;
                                pa(31,0) = tapa::bit_cast<ap_uint<32>>(va0);
                                pa(63,32) = tapa::bit_cast<ap_uint<32>>(va1);
                                y_buf[pl][3*rh] = pa;
                                ap_uint<64> pb;
                                pb(31,0) = tapa::bit_cast<ap_uint<32>>(vb0);
                                pb(63,32) = tapa::bit_cast<ap_uint<32>>(vb1);
                                y_buf[pl][3*rh + 1] = pb;
                                ap_uint<64> pc;
                                pc(31,0) = tapa::bit_cast<ap_uint<32>>(vc0);
                                pc(63,32) = tapa::bit_cast<ap_uint<32>>(vc1);
                                y_buf[pl][3*rh + 2] = pc;
                            }
                        }
                    } else {
                        int lane = pos - n_vecs;
                        float3_t triple;
                        triple[0] = (float)sum_a[lane] * w3_scale + w3_shift;
                        triple[1] = (float)sum_b[lane] * w3_scale + w3_shift;
                        triple[2] = (float)sum_c[lane] * w3_scale + w3_shift;
                        acc3_out.write(triple);
                    }
                    if (pos == row_iters - 1) { rh++; pos = 0; }
                    else { pos++; }
                }
            }
            // Per-round emit partial_pkts (mirror ffn_w1_w2 final emit).
            for (int lane2 = 0; lane2 < TOKEN_LANES; lane2++) {
                float w_route = hdr.routing_weights[lane2];
                token_id_t tok = hdr.token_ids[lane2];
                bool lane_active = hdr.slot_active && (lane2 < (int)hdr.token_count);
                for (int v2 = 0; v2 < HIDDEN_VECS_FP/2; v2++) {
#pragma HLS pipeline II=1
                    partial_pkt_t pkt;
                    pkt.token_id = tok;
                    pkt.slot_active = lane_active;
                    pkt.last_round = false;
                    pkt.tile_done = hdr.tile_done;
                    for (int k = 0; k < VEC_WIDTH_FP; k++) {
#pragma HLS unroll
                        ap_uint<64> packed = y_buf[lane2 / 2][v2 * VEC_WIDTH_FP + k];
                        ap_uint<32> bits = (lane2 & 1) ? ap_uint<32>(packed(63, 32)) : ap_uint<32>(packed(31, 0));
                        float val = w_route * tapa::bit_cast<float>(bits);
                        ap_uint<32> fp32 = tapa::bit_cast<ap_uint<32>>(val);
                        pkt.y[k] = (ap_uint<16>)(fp32 >> 16);
                    }
                    partial_b_out.write(pkt);
                }
            }
        }
        rnd++;
    }
    return;
}
// ============= end bisect-8 ffn_w3_w2_proj =============
// ============================================================================

// ============================================================================
// ffn_swiglu (round-driven): per round, consume W1_THIRD_ROWS triples each
// from acc1 and acc3, compute swiglu+max+requantize, emit INTER_VECS int16
// vectors and one inter_scale to each downstream port. Uses last_round flag
// to terminate.
// ============================================================================
void ffn_swiglu(
    tapa::istream<round_hdr_t>&      hdr_in,
    tapa::istream<float3_t>&         acc1_in,
    tapa::istream<float3_t>&         acc3_in,
    tapa::ostream<int16_half_vec_t>& inter_a_out,
    tapa::ostream<int16_half_vec_t>& inter_b_out,
    tapa::ostream<float>&            inter_scale_a_out,
    tapa::ostream<float>&            inter_scale_b_out
) {
    // Phase B: per-lane FP32 inter buffer for time-multiplexed quantisation
    // across TOKEN_LANES token lanes. Producers now emit row-major triples
    // (rh first, then lane), so this task consumes the same order.
    // W1/W3 stage-1 expects TOKEN_LANES inter_scales BEFORE the int16 stream,
    // so we stash all lanes' FP32 results then emit scales first, vecs second.
    float inter_lane_buf[TOKEN_LANES][3 * W1_THIRD_ROWS];
#pragma HLS bind_storage variable=inter_lane_buf type=RAM_2P impl=BRAM
#pragma HLS array_partition variable=inter_lane_buf cyclic factor=32 dim=2

    bool stop = false;
    int rnd = 0;
    while (!stop) {
#pragma HLS loop_tripcount min=1 max=33
        round_hdr_t hdr = hdr_in.read();
        if (hdr.last_round) { stop = true; rnd++; continue; }

        float lane_scale[TOKEN_LANES];
#pragma HLS array_partition variable=lane_scale complete
        float inter_max[TOKEN_LANES];
#pragma HLS array_partition variable=inter_max complete

        for (int lane = 0; lane < TOKEN_LANES; lane++) {
#pragma HLS unroll
            inter_max[lane] = 0.0f;
        }

        for (int rh = 0; rh < W1_THIRD_ROWS; rh++) {
            for (int lane = 0; lane < TOKEN_LANES; lane++) {
#pragma HLS pipeline II=1
                float3_t p1 = acc1_in.read();
                float3_t p3 = acc3_in.read();
                float va = moe_siluf(p1[0]) * p3[0];
                float vb = moe_siluf(p1[1]) * p3[1];
                float vc = moe_siluf(p1[2]) * p3[2];
                inter_lane_buf[lane][3*rh]     = va;
                inter_lane_buf[lane][3*rh + 1] = vb;
                inter_lane_buf[lane][3*rh + 2] = vc;
                float aa = (va >= 0.0f) ? va : -va;
                float ab = (vb >= 0.0f) ? vb : -vb;
                float ac = (vc >= 0.0f) ? vc : -vc;
                float mx_ab = (aa > ab) ? aa : ab;
                float mx    = (mx_ab > ac) ? mx_ab : ac;
                if (mx > inter_max[lane]) inter_max[lane] = mx;
            }
        }

        for (int lane = 0; lane < TOKEN_LANES; lane++) {
#pragma HLS unroll
            lane_scale[lane] = (inter_max[lane] > 0.0f) ? (inter_max[lane] / 32767.0f) : 1.0f;
        }

        for (int lane = 0; lane < TOKEN_LANES; lane++) {
#pragma HLS pipeline
            inter_scale_a_out.write(lane_scale[lane]);
            inter_scale_b_out.write(lane_scale[lane]);
        }

        for (int lane = 0; lane < TOKEN_LANES; lane++) {
            float inter_inv = 1.0f / lane_scale[lane];
            for (int v = 0; v < INTER_HALF_VECS; v++) {
#pragma HLS pipeline II=1
                int16_half_vec_t iv;
                for (int k = 0; k < VEC_HALF; k++) {
#pragma HLS unroll
                    float f = inter_lane_buf[lane][v * VEC_HALF + k] * inter_inv;
                    int q = (int)(f >= 0.0f ? f + 0.5f : f - 0.5f);
                    if (q >  32767) q =  32767;
                    if (q < -32768) q = -32768;
                    iv[k] = (ap_int<16>)q;
                }
                inter_a_out.write(iv);
                inter_b_out.write(iv);
            }
        }
        rnd++;
    }
}

// ============================================================================
// swiglu_hdr_split — fan-out the round_hdr from ffn_x_split's W1 path to
// ffn_swiglu (so swiglu knows when to stop). Simple pass-through.
// We instead arrange the topology so ffn_swiglu reads from a dedicated
// hdr_swiglu stream produced by ffn_x_split.
// ============================================================================

// ============================================================================
// y_accumulator_half — tile-based accumulate + per-tile flush for one y half.
// Processes rounds in a tile loop. Per tile: zero y_acc, accumulate all
// non-last rounds until tile_done is seen, then flush tile_n tokens to the
// matching y_writer_half stream in token-major, vector-minor order.
// ============================================================================
void y_accumulator_half(
    const int N,
    tapa::istreams<partial_pkt_t, N_PARALLEL_SLOTS>& partial_in,
    tapa::ostream<fp32_vec_t>&                       y_out
) {
    float y_acc[N_TILE_MAX][HIDDEN / 2];
#pragma HLS bind_storage variable=y_acc type=RAM_2P impl=BRAM
#pragma HLS array_partition variable=y_acc cyclic factor=16 dim=2

    const int n_tiles = (N + N_TILE_MAX - 1) / N_TILE_MAX;
    const int half_vecs = HIDDEN_VECS_FP / 2;

    for (int tile = 0; tile < n_tiles; tile++) {
#pragma HLS loop_tripcount min=1 max=16
        const int tile_n = ((N - tile * N_TILE_MAX) < N_TILE_MAX)
                               ? (N - tile * N_TILE_MAX) : N_TILE_MAX;

        // Zero y_acc for this tile.
        for (int t = 0; t < N_TILE_MAX; t++) {
            for (int d = 0; d < HIDDEN / 2; d++) {
#pragma HLS pipeline II=1
                y_acc[t][d] = 0.0f;
            }
        }

        // Accumulate rounds until tile_done.
        bool tile_over = false;
        while (!tile_over) {
#pragma HLS loop_tripcount min=1 max=8
            for (int s = 0; s < N_PARALLEL_SLOTS; s++) {
#pragma HLS pipeline off
                // Read first packet for this slot's non-last round.
                partial_pkt_t pkt0 = partial_in[s].read();

                if (pkt0.last_round) {
                    tile_over = true;
                    continue;
                }

                // Accumulate first packet (lane 0, v 0).
                if (pkt0.slot_active) {
                    int t_id = (int)pkt0.token_id;
                    for (int k = 0; k < VEC_WIDTH_FP; k++) {
#pragma HLS unroll
                        y_acc[t_id][k] += bf16_to_f32(pkt0.y[k]);
                    }
                }
                bool saw_tile_done = pkt0.tile_done;

                // Remaining packets for lane 0.
                for (int v = 1; v < half_vecs; v++) {
#pragma HLS pipeline II=1
                    partial_pkt_t pkt = partial_in[s].read();
                    if (pkt.slot_active) {
                        int t_id = (int)pkt.token_id;
                        int base = v * VEC_WIDTH_FP;
                        for (int k = 0; k < VEC_WIDTH_FP; k++) {
#pragma HLS unroll
                            y_acc[t_id][base + k] += bf16_to_f32(pkt.y[k]);
                        }
                    }
                }
                // Lanes 1..3 packets.
                for (int lane = 1; lane < TOKEN_LANES; lane++) {
                    for (int v = 0; v < half_vecs; v++) {
#pragma HLS pipeline II=1
                        partial_pkt_t pkt = partial_in[s].read();
                        if (pkt.slot_active) {
                            int t_id = (int)pkt.token_id;
                            int base = v * VEC_WIDTH_FP;
                            for (int k = 0; k < VEC_WIDTH_FP; k++) {
#pragma HLS unroll
                                y_acc[t_id][base + k] += bf16_to_f32(pkt.y[k]);
                            }
                        }
                    }
                }

                if (saw_tile_done) tile_over = true;
            }  // for s
        }  // while !tile_over

        // Flush y_acc for this tile's tokens, in y_writer_half order.
        for (int t = 0; t < tile_n; t++) {
            for (int v = 0; v < half_vecs; v++) {
#pragma HLS pipeline II=1
                fp32_vec_t yv;
                int base = v * VEC_WIDTH_FP;
                for (int k = 0; k < VEC_WIDTH_FP; k++) {
#pragma HLS unroll
                    yv[k] = y_acc[t][base + k];
                }
                y_out.write(yv);
            }
        }
    }  // for tile

    // Drain the last_round sentinel tail pkt from every slot.
    // dispatch_core emits one last_round sentinel after all tiles; each FFN
    // proj task converts it to a single tail pkt per slot per half.
    for (int s = 0; s < N_PARALLEL_SLOTS; s++) {
#pragma HLS pipeline off
        partial_in[s].read();
    }
}

// ============================================================================
// y_writer_half: writes N * (HIDDEN_VECS_FP/2) fp32 vectors per HBM channel.
// Unchanged from baseline.
// ============================================================================
void y_writer_half(
    const int N,
    tapa::istream<fp32_vec_t>&    y_in,
    tapa::async_mmap<fp32_vec_t>& y_mem
) {
    const int total = N * (HIDDEN_VECS_FP / 2);

    for (int i_req = 0, i_resp = 0; i_resp < total;) {
#pragma HLS pipeline II=1
        if ((i_req < total) & !y_mem.write_addr.full() & !y_mem.write_data.full()
            & !y_in.empty()) {
            fp32_vec_t ov; y_in.try_read(ov);
            y_mem.write_addr.try_write(i_req);
            y_mem.write_data.try_write(ov);
            ++i_req;
        }
        if (!y_mem.write_resp.empty()) {
            bool ok = false;
            uint8_t resp = y_mem.write_resp.read(ok);
            if (ok) i_resp += (int)resp + 1;
        }
    }
}

// ============================================================================
// Top-level task graph (Phase A).
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
    tapa::mmap<float>                              bias_dequant_mem,
    tapa::mmaps<int16_vec_t, N_PARALLEL_SLOTS>     W1_W2a_expert_mem,
    tapa::mmaps<int16_vec_t, N_PARALLEL_SLOTS>     W1_W2b_expert_mem,
    tapa::mmaps<int16_vec_t, N_PARALLEL_SLOTS>     W1_W2c_expert_mem,
    tapa::mmaps<int16_vec_t, N_PARALLEL_SLOTS>     W3_W2a_expert_mem,
    tapa::mmaps<int16_vec_t, N_PARALLEL_SLOTS>     W3_W2b_expert_mem,
    tapa::mmaps<int16_vec_t, N_PARALLEL_SLOTS>     W3_W2c_expert_mem,
    tapa::mmap<fp32_vec_t>                         y_mem_a,
    tapa::mmap<fp32_vec_t>                         y_mem_b
) {
    tapa::stream<int16_vec_t, 8>     x_gate_fifo  ("x_gate_fifo");
    tapa::stream<int16_vec_t>        w_gate_fifo  ("w_gate_fifo");
    tapa::stream<int16_vec_t>        w_gate_fifo_b("w_gate_fifo_b");
    tapa::stream<float>              bias_fifo    ("bias_fifo");

    tapa::streams<float, N_PARALLEL_SLOTS + 1, 2> dequant_chain("dequant_chain");

    tapa::streams<float, N_PARALLEL_SLOTS, 2> w1_dq_fifos   ("w1_dq");
    tapa::streams<float, N_PARALLEL_SLOTS, 2> w3_dq_fifos   ("w3_dq");
    tapa::streams<float, N_PARALLEL_SLOTS, 2> w2_dq_a_fifos ("w2_dq_a");
    tapa::streams<float, N_PARALLEL_SLOTS, 2> w2_dq_b_fifos ("w2_dq_b");

    // Gate -> scheduler
    tapa::stream<route_pkt_t, 8> route_out  ("route_out");
    tapa::stream<int16_vec_t, 8> x_to_sched ("x_to_sched");

    // dispatch_compute -> ffn_x_split (header) + x (per-slot)
    tapa::streams<round_hdr_t,  N_PARALLEL_SLOTS, 4> hdr_to_xsplit("hdr_to_xsplit");
    tapa::streams<int16_vec_t,  N_PARALLEL_SLOTS, 4> x_to_xsplit  ("x_to_xsplit");

    // ffn_x_split -> w1/w3 projections (per-port headers + x)
    tapa::streams<round_hdr_t, N_PARALLEL_SLOTS, 4> hdr_to_w1   ("hdr_to_w1");
    tapa::streams<round_hdr_t, N_PARALLEL_SLOTS, 4> hdr_to_w3   ("hdr_to_w3");
    tapa::streams<int16_vec_t, N_PARALLEL_SLOTS, 4> x1_fifos    ("x1");
    tapa::streams<int16_vec_t, N_PARALLEL_SLOTS, 4> x3_fifos    ("x3");

    // Weight triggers
    tapa::streams<weight_trigger_t, N_PARALLEL_SLOTS, 2> trigger1a_fifos("trigger1a");
    tapa::streams<weight_trigger_t, N_PARALLEL_SLOTS, 2> trigger1b_fifos("trigger1b");
    tapa::streams<weight_trigger_t, N_PARALLEL_SLOTS, 2> trigger1c_fifos("trigger1c");
    tapa::streams<weight_trigger_t, N_PARALLEL_SLOTS, 2> trigger3a_fifos("trigger3a");
    tapa::streams<weight_trigger_t, N_PARALLEL_SLOTS, 2> trigger3b_fifos("trigger3b");
    tapa::streams<weight_trigger_t, N_PARALLEL_SLOTS, 2> trigger3c_fifos("trigger3c");

    tapa::streams<int16_vec_t, N_PARALLEL_SLOTS> W1_W2a_fifos("W1_W2a_fifo");
    tapa::streams<int16_vec_t, N_PARALLEL_SLOTS> W1_W2b_fifos("W1_W2b_fifo");
    tapa::streams<int16_vec_t, N_PARALLEL_SLOTS> W1_W2c_fifos("W1_W2c_fifo");
    tapa::streams<int16_vec_t, N_PARALLEL_SLOTS> W3_W2a_fifos("W3_W2a_fifo");
    tapa::streams<int16_vec_t, N_PARALLEL_SLOTS> W3_W2b_fifos("W3_W2b_fifo");
    tapa::streams<int16_vec_t, N_PARALLEL_SLOTS> W3_W2c_fifos("W3_W2c_fifo");

    // Stage-1 paths for swiglu
    tapa::streams<float3_t, N_PARALLEL_SLOTS, 2> acc1_fifos("acc1");
    tapa::streams<float3_t, N_PARALLEL_SLOTS, 2> acc3_fifos("acc3");
    // hdr to swiglu — we route hdr_to_w1's twin through a separate stream so
    // swiglu knows when to stop. We'll have ffn_x_split emit 3 hdr copies:
    // hdr_to_w1, hdr_to_w3, hdr_to_swiglu. Implementation detail: we add a
    // separate stream below and a ffn_x_split overload.
    tapa::streams<round_hdr_t, N_PARALLEL_SLOTS, 4> hdr_to_swiglu("hdr_to_swiglu");

    tapa::streams<int16_half_vec_t, N_PARALLEL_SLOTS, 8> inter_a_fifos  ("inter_a");
    tapa::streams<int16_half_vec_t, N_PARALLEL_SLOTS, 8> inter_b_fifos("inter_b");
    // Phase B: swiglu writes TOKEN_LANES (=4) inter_scales contiguously
    // before W1/W3 stage-1 begins reading; depth >= TOKEN_LANES avoids stall.
    tapa::streams<float,       N_PARALLEL_SLOTS, 8> inter_scale_a_fifos("inter_scale_a");
    tapa::streams<float,       N_PARALLEL_SLOTS, 8> inter_scale_b_fifos("inter_scale_b");

    // Per-slot partial outputs to y_accumulator (replaces partial_chain).
    tapa::streams<partial_pkt_t, N_PARALLEL_SLOTS, 4> partial_a_streams("partial_a");
    tapa::streams<partial_pkt_t, N_PARALLEL_SLOTS, 4> partial_b_streams("partial_b");

    // y_accumulator_half -> y_writer
    tapa::stream<fp32_vec_t, 8> y_a_out ("y_a_out");
    tapa::stream<fp32_vec_t, 8> y_b_out ("y_b_out");

    tapa::task()
        .invoke<tapa::join>(x_gate_reader,       N, x_mem, x_gate_fifo)
        .invoke<tapa::join>(w_gate_reader,       N, W_gate_mem,   w_gate_fifo)
        .invoke<tapa::join>(w_gate_reader,       N, W_gate_mem_b, w_gate_fifo_b)
        .invoke<tapa::join>(bias_dequant_reader, bias_dequant_mem,
                            bias_fifo, dequant_chain)
        .invoke<tapa::join>(gate_compute,        N, local_expert_base,
                            x_scale, w_gate_scale, w_gate_shift,
                            x_gate_fifo, w_gate_fifo, w_gate_fifo_b, bias_fifo,
                            route_out, x_to_sched)
        .invoke<tapa::join>(dispatch_core, N, local_expert_base, x_scale,
                            route_out, x_to_sched,
                            hdr_to_xsplit, x_to_xsplit)
        .invoke<tapa::join, N_PARALLEL_SLOTS>(
            dequant_split, x_scale,
            dequant_chain, dequant_chain,
            w1_dq_fifos, w3_dq_fifos, w2_dq_a_fifos, w2_dq_b_fifos)
        .invoke<tapa::join>(dequant_sink, dequant_chain)
        .invoke<tapa::join, N_PARALLEL_SLOTS>(
            ffn_x_split,
            hdr_to_xsplit, x_to_xsplit,
            hdr_to_w1, hdr_to_w3, hdr_to_swiglu,
            x1_fifos, x3_fifos,
            trigger1a_fifos, trigger1b_fifos, trigger1c_fifos,
            trigger3a_fifos, trigger3b_fifos, trigger3c_fifos)
        .invoke<tapa::join, N_PARALLEL_SLOTS>(
            w1_w2c_weight_loader, trigger1a_fifos, W1_W2a_expert_mem, W1_W2a_fifos)
        .invoke<tapa::join, N_PARALLEL_SLOTS>(
            w1_w2c_weight_loader, trigger1b_fifos, W1_W2b_expert_mem, W1_W2b_fifos)
        .invoke<tapa::join, N_PARALLEL_SLOTS>(
            w1_w2c_weight_loader, trigger1c_fifos, W1_W2c_expert_mem, W1_W2c_fifos)
        .invoke<tapa::join, N_PARALLEL_SLOTS>(
            w3_w2c_weight_loader, trigger3a_fifos, W3_W2a_expert_mem, W3_W2a_fifos)
        .invoke<tapa::join, N_PARALLEL_SLOTS>(
            w3_w2c_weight_loader, trigger3b_fifos, W3_W2b_expert_mem, W3_W2b_fifos)
        .invoke<tapa::join, N_PARALLEL_SLOTS>(
            w3_w2c_weight_loader, trigger3c_fifos, W3_W2c_expert_mem, W3_W2c_fifos)
        .invoke<tapa::join, N_PARALLEL_SLOTS>(
            ffn_w1_w2_proj,
            hdr_to_w1,
            x1_fifos, W1_W2a_fifos, W1_W2b_fifos, W1_W2c_fifos,
            w1_dq_fifos,
            inter_a_fifos, inter_scale_a_fifos,
            w2_dq_a_fifos,
            acc1_fifos,
            partial_a_streams)
        .invoke<tapa::join, N_PARALLEL_SLOTS>(
            ffn_w3_w2_proj,
            hdr_to_w3,
            x3_fifos, W3_W2a_fifos, W3_W2b_fifos, W3_W2c_fifos,
            w3_dq_fifos,
            inter_b_fifos, inter_scale_b_fifos,
            w2_dq_b_fifos,
            acc3_fifos,
            partial_b_streams)
        .invoke<tapa::join, N_PARALLEL_SLOTS>(
            ffn_swiglu,
            hdr_to_swiglu,
            acc1_fifos, acc3_fifos,
            inter_a_fifos, inter_b_fifos,
            inter_scale_a_fifos, inter_scale_b_fifos)
        .invoke<tapa::join>(y_accumulator_half, N, partial_a_streams, y_a_out)
        .invoke<tapa::join>(y_accumulator_half, N, partial_b_streams, y_b_out)
        .invoke<tapa::join>(y_writer_half, N, y_a_out, y_mem_a)
        .invoke<tapa::join>(y_writer_half, N, y_b_out, y_mem_b);
}
