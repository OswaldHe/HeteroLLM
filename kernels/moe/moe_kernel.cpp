// ============================================================================
// MoE FPGA Kernel — DeepSeek V3.2 Routed Experts (v1: N_EXPERT_SLOTS=1)
//
// Refactor: off-chip memory access is separated from computation. Every task
// that owns an async_mmap port only issues read/write requests and forwards
// the data through streams; compute tasks carry zero async_mmap ports.
// ============================================================================

#include "moe_kernel.h"

static inline float moe_sigmoidf(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}
static inline float moe_siluf(float x) {
    return x * moe_sigmoidf(x);
}

// ============================================================================
// Reader: x_gate_reader
//   Streams N * HIDDEN_VECS INT16 vectors from x_mem to gate_compute.
//   No computation.
// ============================================================================
void x_gate_reader(
    const int N,
    tapa::async_mmap<int16_vec_t>& x_mem,
    tapa::ostream<int16_vec_t>&    out
) {
    const int total = N * HIDDEN_VECS;
    for (int i_req = 0, i_resp = 0; i_resp < total;) {
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=224 max=917504
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
// Reader: w_gate_reader
//   Streams the INT16 gate weight matrix exactly once (weight-stationary).
//   The compute task caches it in on-chip URAM and reuses it across all tokens.
//   No computation.
// ============================================================================
void w_gate_reader(
    tapa::async_mmap<int16_vec_t>& w_gate_mem,
    tapa::ostream<int16_vec_t>&    out
) {
    for (int i_req = 0, i_resp = 0; i_resp < W_GATE_VECS;) {
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=57344 max=57344
        if ((i_req < W_GATE_VECS) & !w_gate_mem.read_addr.full()) {
            w_gate_mem.read_addr.try_write(i_req); ++i_req;
        }
        if (!w_gate_mem.read_data.empty()) {
            int16_vec_t wv; w_gate_mem.read_data.try_read(wv);
            out.write(wv);
            ++i_resp;
        }
    }
}

// ============================================================================
// Reader: bias_dequant_reader
//   Streams the combined [bias | dequant] FP32 buffer from a single HBM port,
//   demuxing each response by index: indices [0, N_EXPERTS_TOTAL) → bias_out,
//   indices [N_EXPERTS_TOTAL, total) → dequant_out. No computation.
// ============================================================================
// Single dequant stream (no fan-out here). The 6 dequant floats are forwarded
// down a chain of expert_ffn_compute tasks — each lane reads them, caches
// them locally, and forwards them on to the next lane. The final lane's
// downstream FIFO is drained by `dequant_sink`.
void bias_dequant_reader(
    tapa::async_mmap<float>& mem,       // combined: [bias(256) | dequant(6)]
    tapa::ostream<float>&    bias_out,
    tapa::ostream<float>&    dequant_out
) {
    const int total = N_EXPERTS_TOTAL + DEQUANT_PARAMS_PER_SLOT;
    for (int i_req = 0, i_resp = 0; i_resp < total;) {
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=262 max=262
        if ((i_req < total) & !mem.read_addr.full()) {
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

// Drain task: consumes the DEQUANT_PARAMS_PER_SLOT floats forwarded out of
// the last expert_ffn_compute lane in the chain. Without this, the final
// lane's output FIFO would be unread and TAPA would refuse to terminate.
void dequant_sink(
    tapa::istream<float>& dequant_in
) {
    for (int i = 0; i < DEQUANT_PARAMS_PER_SLOT; i++) {
#pragma HLS pipeline II=1
        float tmp = dequant_in.read();
        (void)tmp;
    }
}

// ============================================================================
// Compute: gate_compute (INT16 MAC + dequant)
//   Pure computation — no mmap. Consumes INT16 x (streamed), INT16 W_gate
//   (streamed), and bias (streamed once at startup).
//   Scores all experts via INT16 MAC + dequant (global x_scale, w_gate_scale),
//   runs the simplified 2-FPGA top-K router (4 per half), and emits one
//   route_pkt_t per token. Also forwards the INT16 x stream to dispatch_compute
//   on x_out (so x only needs to be read from HBM once, by x_gate_reader).
// ============================================================================
void gate_compute(
    const int   N,
    const int   local_expert_base,
    const float x_scale,
    const float w_gate_scale,
    const float w_gate_shift,
    tapa::istream<int16_vec_t>&    x_in,
    tapa::istream<int16_vec_t>&    w_gate_in,
    tapa::istream<float>&          bias_in,
    tapa::ostream<route_pkt_t>&    route_out,
    tapa::ostream<int16_vec_t>&    x_out
) {
    // One-time: buffer the bias vector from the bias_reader stream.
    float bias[N_EXPERTS_TOTAL];
#pragma HLS bind_storage variable=bias type=RAM_1P impl=BRAM
    for (int i = 0; i < N_EXPERTS_TOTAL; i++) {
#pragma HLS pipeline II=1
        bias[i] = bias_in.read();
    }
    // Weight-stationary: load INT16 gate weight matrix once into on-chip URAM
    ap_uint<64> W_gate_cache[8][N_EXPERTS_TOTAL][HIDDEN_VECS];
#pragma HLS bind_storage variable=W_gate_cache type=RAM_2P impl=URAM
#pragma HLS array_partition variable=W_gate_cache complete dim=1

    for (int e = 0; e < N_EXPERTS_TOTAL; e++) {
        for (int v = 0; v < HIDDEN_VECS; v++) {
#pragma HLS pipeline II=1
            auto w_vec = w_gate_in.read();
            for(int k = 0; k < 8; k++) {
                ap_uint<64> w_val = 0;
                for(int l = 0; l < 4; l++) {
                    w_val(l*16+15, l*16) = tapa::bit_cast<ap_uint<16>>(w_vec[k*4+l]);
                }
                W_gate_cache[k][e][v] = w_val;
            }
        }
    }

    const float joint_scale = x_scale * w_gate_scale;

    for (int t = 0; t < N; t++) {
#pragma HLS loop_tripcount min=1 max=4096
        // Drain HIDDEN_VECS INT16 vectors of this token into x_buf and
        // simultaneously forward them to dispatch_compute on x_out.
        int16_vec_t x_buf[HIDDEN_VECS];
#pragma HLS bind_storage variable=x_buf type=RAM_2P impl=BRAM
        for (int v = 0; v < HIDDEN_VECS; v++) {
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=224 max=224
            int16_vec_t xv = x_in.read();
            x_buf[v] = xv;
            x_out.write(xv);
        }
        // Score all N_EXPERTS_TOTAL experts: INT16 MAC → dequant → sigmoid + bias
        float scores[N_EXPERTS_TOTAL];
#pragma HLS bind_storage variable=scores type=RAM_1P impl=BRAM
        for (int e = 0; e < N_EXPERTS_TOTAL; e++) {
#pragma HLS loop_tripcount min=256 max=256
            ap_int<44> sum = 0;
            for (int v = 0; v < HIDDEN_VECS; v++) {
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=224 max=224
                int16_vec_t xv = x_buf[v];
                ap_int<37> p = 0;
                for (int k = 0; k < 8; k++) {
                    for (int l = 0; l < 4; l++) {
                        ap_int<16> x_val = xv[k*4+l];
                        ap_int<16> w_val = ap_int<16>(W_gate_cache[k][e][v]((l*16)+15, l*16));
                        p += x_val * w_val;
                    }
                }
                sum += p;
            }
            scores[e] = moe_sigmoidf((float)sum * joint_scale + w_gate_shift) + bias[e];
        }
        // ----------------------------------------------------------------
        // Parallel top-LOCAL_K per half, mirroring topk_parallel_cmp from
        // indexer_vanilla.h. A single fused loop walks both halves in
        // lockstep, processing LOCAL_K candidates from each half per
        // iteration:
        //   * Seed: first LOCAL_K scores of each half populate the
        //     top-K buffer; initial (min_score, min_idx) comes from a
        //     parallel binary reduction.
        //   * Each subsequent iteration OR-reduces a per-half fast_check
        //     "any candidate > current min_score". On a hit, enter a
        //     `#pragma HLS pipeline off` inner loop over the LOCAL_K
        //     candidates that replaces top[min_idx] one at a time and
        //     refreshes min via a parallel binary reduction.
        // Lower half is expert range [0, LOCAL_EXPERTS); upper half is
        // [LOCAL_EXPERTS, N_EXPERTS_TOTAL).
        // ----------------------------------------------------------------
        float top_lo_w [LOCAL_K];
        int   top_lo_ids[LOCAL_K];
#pragma HLS array_partition variable=top_lo_w  complete
#pragma HLS array_partition variable=top_lo_ids complete
        float top_hi_w [LOCAL_K];
        int   top_hi_ids[LOCAL_K];
#pragma HLS array_partition variable=top_hi_w  complete
#pragma HLS array_partition variable=top_hi_ids complete

        // Seed each half from its first LOCAL_K scores.
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
        // Fused scan: (LOCAL_EXPERTS / LOCAL_K) - 1 iterations, 4 candidates
        // per half per iteration (starting at offset LOCAL_K from each half).
        for (int r = 1; r < (LOCAL_EXPERTS / LOCAL_K); r++) {
#pragma HLS loop_tripcount min=31 max=31
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

        // Normalize using all 8 selected scores (sum of both halves).
        float wsum = 0.0f;
        for (int i = 0; i < LOCAL_K; i++) {
#pragma HLS unroll
            wsum += top_lo_w[i] + top_hi_w[i];
        }
        float wscale = (wsum > 1e-12f) ? (ROUTE_SCALE / wsum) : 0.0f;

        // Emit local top-LOCAL_K based on this card's local_expert_base.
        // Tokens are emitted in order; the consumer infers token_id from its
        // own loop index, so no token_id/num_local fields are needed.
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
// Compute: dispatch_compute
//   Pure computation (no async_mmap). Consumes route packets from gate_compute
//   and INT16 x forwarded from gate_compute (no HBM access — gate_compute is
//   the single reader of x_mem). Since x is already INT16 with a global
//   x_scale, no per-token quantization is needed; this task is pure fan-out:
//   for each token, buffer the INT16 x once and emit LOCAL_K (hdr + HIDDEN_VECS
//   INT16 vecs) bundles.
// ============================================================================
void dispatch_compute(
    const int   N,
    const int   local_expert_base,
    const float x_scale,
    tapa::istream<route_pkt_t>&                          route_in,
    tapa::istream<int16_vec_t>&                          x_in,
    tapa::ostreams<dispatch_hdr_t, LOCAL_K>&             hdr_out,
    tapa::ostreams<int16_vec_t,    LOCAL_K>&             x_out,
    // Trigger fan-out: one per (matrix, lane). Sent directly to the weight
    // loaders so they can prefetch weights asynchronously, decoupled from
    // expert_ffn_compute's pipeline.
    tapa::ostreams<weight_trigger_t, LOCAL_K>&           trigger1_out,
    tapa::ostreams<weight_trigger_t, LOCAL_K>&           trigger3_out,
    tapa::ostreams<weight_trigger_t, LOCAL_K>&           trigger2_out,
    // Opt 3: triggers for odd-row sub-matrix loaders.
    tapa::ostreams<weight_trigger_t, LOCAL_K>&           trigger1b_out,
    tapa::ostreams<weight_trigger_t, LOCAL_K>&           trigger3b_out,
    tapa::ostreams<weight_trigger_t, LOCAL_K>&           trigger2b_out
) {
    for (int t = 0; t < N; t++) {
#pragma HLS loop_tripcount min=1 max=4096

        // Buffer this token's INT16 activation so we can replay it to all
        // LOCAL_K parallel expert lanes simultaneously.
        int16_vec_t x_int_buf[HIDDEN_VECS];
#pragma HLS bind_storage variable=x_int_buf type=RAM_2P impl=BRAM
        for (int v = 0; v < HIDDEN_VECS; v++) {
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=224 max=224
            x_int_buf[v] = x_in.read();
        }

        route_pkt_t route = route_in.read();

        // One header per parallel lane, plus three trigger writes (one per
        // matrix loader). Triggers go straight from dispatch to the loaders
        // so loaders never wait for compute to finish a token before issuing
        // the next prefetch.
        for (int i = 0; i < LOCAL_K; i++) {
#pragma HLS unroll
            int eid_global = (int)route.local_expert_ids[i];
            int rel_eid    = eid_global - local_expert_base;
            dispatch_hdr_t hdr;
            hdr.token_id         = (token_id_t)t;
            hdr.global_expert_id = (expert_id_t)eid_global;
            hdr.routing_weight   = route.weights[i];
            hdr.x_scale          = x_scale;
            hdr.done             = false;
            hdr_out[i].write(hdr);
            trigger1_out[i].write(rel_eid);
            trigger3_out[i].write(rel_eid);
            trigger2_out[i].write(rel_eid);
            trigger1b_out[i].write(rel_eid);
            trigger3b_out[i].write(rel_eid);
            trigger2b_out[i].write(rel_eid);
        }

        // Replicate the activation tensor to all LOCAL_K lanes in lockstep.
        for (int v = 0; v < HIDDEN_VECS; v++) {
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=224 max=224
            int16_vec_t xv = x_int_buf[v];
            for (int i = 0; i < LOCAL_K; i++) {
#pragma HLS unroll
                x_out[i].write(xv);
            }
        }
    }
}

// ============================================================================
// Readers: w1/w3/w2_weight_loader
//   One HBM port per matrix. Each loader consumes a per-iteration trigger
//   (relative expert id), then streams its matrix's beats into a dedicated
//   downstream FIFO. With three independent ports per lane, W1 and W3 can
//   be fetched in parallel (feeding the fused MAC loop), and W2 prefetches
//   while Phase 1 runs.
// ============================================================================
void w1_weight_loader(
    const int N,
    tapa::istream<weight_trigger_t>& trigger_in,
    tapa::async_mmap<int16_vec_t>&   W1_mem,
    tapa::ostream<int16_vec_t>&      W1_fifo
) {
    for (int iter = 0; iter < N; iter++) {
#pragma HLS loop_tripcount min=1 max=4096
        weight_trigger_t trig = trigger_in.read();
        const int base = trig * W1_HALF_VECS;
        for (int i_req = 0, i_resp = 0; i_resp < W1_HALF_VECS;) {
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=65536 max=65536
            if ((i_req < W1_HALF_VECS) & !W1_mem.read_addr.full()) {
                W1_mem.read_addr.try_write(base + i_req); ++i_req;
            }
            if (!W1_mem.read_data.empty()) {
                int16_vec_t tmp; W1_mem.read_data.try_read(tmp);
                W1_fifo.write(tmp);
                ++i_resp;
            }
        }
    }
}

void w3_weight_loader(
    const int N,
    tapa::istream<weight_trigger_t>& trigger_in,
    tapa::async_mmap<int16_vec_t>&   W3_mem,
    tapa::ostream<int16_vec_t>&      W3_fifo
) {
    for (int iter = 0; iter < N; iter++) {
#pragma HLS loop_tripcount min=1 max=4096
        weight_trigger_t trig = trigger_in.read();
        const int base = trig * W3_HALF_VECS;
        for (int i_req = 0, i_resp = 0; i_resp < W3_HALF_VECS;) {
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=65536 max=65536
            if ((i_req < W3_HALF_VECS) & !W3_mem.read_addr.full()) {
                W3_mem.read_addr.try_write(base + i_req); ++i_req;
            }
            if (!W3_mem.read_data.empty()) {
                int16_vec_t tmp; W3_mem.read_data.try_read(tmp);
                W3_fifo.write(tmp);
                ++i_resp;
            }
        }
    }
}

void w2_weight_loader(
    const int N,
    tapa::istream<weight_trigger_t>& trigger_in,
    tapa::async_mmap<int16_vec_t>&   W2_mem,
    tapa::ostream<int16_vec_t>&      W2_fifo
) {
    for (int iter = 0; iter < N; iter++) {
#pragma HLS loop_tripcount min=1 max=4096
        weight_trigger_t trig = trigger_in.read();
        const int base = trig * W2_HALF_VECS;
        for (int i_req = 0, i_resp = 0; i_resp < W2_HALF_VECS;) {
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=229376 max=229376
            if ((i_req < W2_HALF_VECS) & !W2_mem.read_addr.full()) {
                W2_mem.read_addr.try_write(base + i_req); ++i_req;
            }
            if (!W2_mem.read_data.empty()) {
                int16_vec_t tmp; W2_mem.read_data.try_read(tmp);
                W2_fifo.write(tmp);
                ++i_resp;
            }
        }
    }
}

// ============================================================================
// Opt 2: per-lane FFN dataflow split into 6 cooperating tasks.
//
//   dequant_split  → routes the 6-float dequant chain into per-projection pairs
//   ffn_x_split    → broadcasts hdr scalars + INT16 activations to W1/W3 lanes
//   ffn_w1_proj    → W1 GEMV + dequant → MOE_INTER FP32 acc1
//   ffn_w3_proj    → W3 GEMV + dequant → MOE_INTER FP32 acc3
//   ffn_swiglu     → SiLU(acc1)*acc3, requantize → INT16 intermediate vectors
//   ffn_w2_proj    → W2 GEMV + dequant + routing-weighted partial reduction
// ============================================================================

// dequant_split: forwards all 6 dequant floats down the chain, and additionally
// routes the (scale, shift) pair per matrix to this lane's projection tasks.
//   index 0,1 → w1_dq_out  (w1_scale, w1_shift)
//   index 2,3 → w3_dq_out  (w3_scale, w3_shift)
//   index 4,5 → w2_dq_out  (w2_scale, w2_shift)
void dequant_split(
    tapa::istream<float>& dequant_in,
    tapa::ostream<float>& dequant_fwd,
    tapa::ostream<float>& w1_dq_out,
    tapa::ostream<float>& w3_dq_out,
    tapa::ostream<float>& w2_dq_out
) {
    for (int i = 0; i < DEQUANT_PARAMS_PER_SLOT; i++) {
#pragma HLS pipeline II=1
        float v = dequant_in.read();
        dequant_fwd.write(v);
        if (i == 0 || i == 1) {
            w1_dq_out.write(v);
        } else if (i == 2 || i == 3) {
            w3_dq_out.write(v);
        } else {
            w2_dq_out.write(v);
        }
    }
}

// ffn_x_split: broadcasts the dispatch header's per-token scalars to the
// downstream projection tasks and replicates the INT16 activation tensor to
// W1 and W3 lanes (W2 needs only the routing weight, which is forwarded once
// per token on a separate stream).
void ffn_x_split(
    const int N,
    tapa::istream<dispatch_hdr_t>& hdr_in,
    tapa::istream<int16_vec_t>&    x_in,
    tapa::ostream<int16_vec_t>&    x1_out,
    tapa::ostream<int16_vec_t>&    x3_out,
    tapa::ostream<float>&          xs_w1_out,
    tapa::ostream<float>&          xs_w3_out,
    tapa::ostream<float>&          w_route_out
) {
    for (int t = 0; t < N; t++) {
#pragma HLS loop_tripcount min=1 max=4096
        for (int v = 0; v < HIDDEN_VECS; v++) {
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=224 max=224
            int16_vec_t xv = x_in.read();
            x1_out.write(xv);
            x3_out.write(xv);
        }
        dispatch_hdr_t hdr = hdr_in.read();
        xs_w1_out.write(hdr.x_scale);
        xs_w3_out.write(hdr.x_scale);
        w_route_out.write(hdr.routing_weight);
    }
}

// ffn_w1_proj: W1 GEMV + dequant, 2-way output-row parallelism (Opt 3).
// Even rows from W1a_fifo and odd rows from W1b_fifo are processed together per
// inner iteration. Each output beat packs both results as float2_t (one FIFO
// write per cycle carries two FP32 values, avoiding a double-write conflict).
void ffn_w1_proj(
    const int N,
    tapa::istream<int16_vec_t>& x1_in,
    tapa::istream<int16_vec_t>& W1a_fifo,
    tapa::istream<int16_vec_t>& W1b_fifo,
    tapa::istream<float>&       xs_w1_in,
    tapa::istream<float>&       w1_dq_in,
    tapa::ostream<float2_t>&    acc1_out
) {
    float w1_scale = w1_dq_in.read();
    float w1_shift = w1_dq_in.read();

    for (int iter = 0; iter < N; iter++) {
#pragma HLS loop_tripcount min=1 max=4096

        int16_vec_t x_cur[HIDDEN_VECS];
#pragma HLS bind_storage variable=x_cur type=RAM_2P impl=BRAM
        for (int v = 0; v < HIDDEN_VECS; v++) {
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=224 max=224
            x_cur[v] = x1_in.read();
        }
        float x_scale = xs_w1_in.read();
        float scale_xw = w1_scale * x_scale;
        for (int rh = 0; rh < MOE_INTER/2; rh++) {
#pragma HLS loop_tripcount min=1024 max=1024
            ap_int<44> sum_a = 0, sum_b = 0;
            for (int v = 0; v < HIDDEN_VECS; v++) {
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=224 max=224
                int16_vec_t xv   = x_cur[v];
                int16_vec_t w1av = W1a_fifo.read();
                int16_vec_t w1bv = W1b_fifo.read();
                ap_int<37> pa = 0, pb = 0;
                for (int k = 0; k < VEC_WIDTH; k++) {
#pragma HLS unroll
                    pa += xv[k] * w1av[k];
                    pb += xv[k] * w1bv[k];
                }
                sum_a += pa;
                sum_b += pb;
            }
            float2_t pair;
            pair[0] = (float)sum_a * scale_xw + w1_shift;
            pair[1] = (float)sum_b * scale_xw + w1_shift;
            acc1_out.write(pair);
        }
    }
}

// ffn_w3_proj: symmetric to ffn_w1_proj with 2-way output-row parallelism.
void ffn_w3_proj(
    const int N,
    tapa::istream<int16_vec_t>& x3_in,
    tapa::istream<int16_vec_t>& W3a_fifo,
    tapa::istream<int16_vec_t>& W3b_fifo,
    tapa::istream<float>&       xs_w3_in,
    tapa::istream<float>&       w3_dq_in,
    tapa::ostream<float2_t>&    acc3_out
) {
    float w3_scale = w3_dq_in.read();
    float w3_shift = w3_dq_in.read();

    for (int iter = 0; iter < N; iter++) {
#pragma HLS loop_tripcount min=1 max=4096

        int16_vec_t x_cur[HIDDEN_VECS];
#pragma HLS bind_storage variable=x_cur type=RAM_2P impl=BRAM
        for (int v = 0; v < HIDDEN_VECS; v++) {
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=224 max=224
            x_cur[v] = x3_in.read();
        }

        float x_scale = xs_w3_in.read();
        float scale_xw = w3_scale * x_scale;
        for (int rh = 0; rh < MOE_INTER/2; rh++) {
#pragma HLS loop_tripcount min=1024 max=1024
            ap_int<44> sum_a = 0, sum_b = 0;
            for (int v = 0; v < HIDDEN_VECS; v++) {
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=224 max=224
                int16_vec_t xv   = x_cur[v];
                int16_vec_t w3av = W3a_fifo.read();
                int16_vec_t w3bv = W3b_fifo.read();
                ap_int<37> pa = 0, pb = 0;
                for (int k = 0; k < VEC_WIDTH; k++) {
#pragma HLS unroll
                    pa += xv[k] * w3av[k];
                    pb += xv[k] * w3bv[k];
                }
                sum_a += pa;
                sum_b += pb;
            }
            float2_t pair;
            pair[0] = (float)sum_a * scale_xw + w3_shift;
            pair[1] = (float)sum_b * scale_xw + w3_shift;
            acc3_out.write(pair);
        }
    }
}

// ffn_swiglu: SiLU gate + requantize (Opt 3: consumes float2_t pairs).
// Each acc1_in / acc3_in beat carries two FP32 values (even+odd row), so the
// gating loop runs MOE_INTER/2 iterations and fills inter_buf two elements per
// cycle without reading either input FIFO twice in the same cycle.
void ffn_swiglu(
    const int N,
    tapa::istream<float2_t>&    acc1_in,
    tapa::istream<float2_t>&    acc3_in,
    tapa::ostream<int16_vec_t>& inter_out,
    tapa::ostream<float>&       inter_scale_out
) {
    for (int iter = 0; iter < N; iter++) {
#pragma HLS loop_tripcount min=1 max=4096
        float inter_buf[MOE_INTER];
#pragma HLS bind_storage variable=inter_buf type=RAM_2P impl=BRAM
#pragma HLS array_partition variable=inter_buf cyclic factor=32

        float inter_max = 0.0f;
        for (int rh = 0; rh < MOE_INTER/2; rh++) {
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=1024 max=1024
            float2_t p1 = acc1_in.read();
            float2_t p3 = acc3_in.read();
            float va = moe_siluf(p1[0]) * p3[0];
            float vb = moe_siluf(p1[1]) * p3[1];
            inter_buf[2*rh]   = va;
            inter_buf[2*rh+1] = vb;
            float aa = (va >= 0.0f) ? va : -va;
            float ab = (vb >= 0.0f) ? vb : -vb;
            float mx = (aa > ab) ? aa : ab;
            if (mx > inter_max) inter_max = mx;
        }

        float inter_scale = (inter_max > 0.0f) ? (inter_max / 32767.0f) : 1.0f;
        float inter_inv   = 1.0f / inter_scale;

        inter_scale_out.write(inter_scale);

        for (int v = 0; v < INTER_VECS; v++) {
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=64 max=64
            int16_vec_t iv;
            for (int k = 0; k < VEC_WIDTH; k++) {
#pragma HLS unroll
                float f = inter_buf[v * VEC_WIDTH + k] * inter_inv;
                int q = (int)(f >= 0.0f ? f + 0.5f : f - 0.5f);
                if (q >  32767) q =  32767;
                if (q < -32768) q = -32768;
                iv[k] = (ap_int<16>)q;
            }
            inter_out.write(iv);
        }
    }
}

// ffn_w2_proj: W2 GEMV + dequant + partial-sum reduction, 2-way row parallelism.
// Even rows from W2a_fifo and odd rows from W2b_fifo are read per inner
// iteration; two y_buf entries are filled per outer cycle, halving W2 latency.
void ffn_w2_proj(
    const int N,
    tapa::istream<int16_vec_t>& inter_in,
    tapa::istream<float>&       inter_scale_in,
    tapa::istream<float>&       w_route_in,
    tapa::istream<int16_vec_t>& W2a_fifo,
    tapa::istream<int16_vec_t>& W2b_fifo,
    tapa::istream<float>&       w2_dq_in,
    tapa::istream<fp32_vec_t>&  partial_in,
    tapa::ostream<fp32_vec_t>&  partial_out
) {
    float w2_scale = w2_dq_in.read();
    float w2_shift = w2_dq_in.read();

    for (int iter = 0; iter < N; iter++) {
#pragma HLS loop_tripcount min=1 max=4096
        float inter_scale = inter_scale_in.read();
        float w_route     = w_route_in.read();

        int16_vec_t inter_buf[INTER_VECS];
#pragma HLS bind_storage variable=inter_buf type=RAM_2P impl=BRAM
        for (int v = 0; v < INTER_VECS; v++) {
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=64 max=64
            inter_buf[v] = inter_in.read();
        }

        float y_buf[HIDDEN];
#pragma HLS bind_storage variable=y_buf type=RAM_2P impl=BRAM
#pragma HLS array_partition variable=y_buf cyclic factor=16

        float scale_wi = w2_scale * inter_scale;
        for (int rh = 0; rh < HIDDEN/2; rh++) {
#pragma HLS loop_tripcount min=3584 max=3584
            ap_int<43> sum_a = 0, sum_b = 0;
            for (int v = 0; v < INTER_VECS; v++) {
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=64 max=64
                int16_vec_t iv   = inter_buf[v];
                int16_vec_t w2av = W2a_fifo.read();
                int16_vec_t w2bv = W2b_fifo.read();
                ap_int<37> pa = 0, pb = 0;
                for (int k = 0; k < VEC_WIDTH; k++) {
#pragma HLS unroll
                    pa += iv[k] * w2av[k];
                    pb += iv[k] * w2bv[k];
                }
                sum_a += pa;
                sum_b += pb;
            }
            y_buf[2*rh]   = (float)sum_a * scale_wi + w2_shift;
            y_buf[2*rh+1] = (float)sum_b * scale_wi + w2_shift;
        }

        for (int v = 0; v < HIDDEN_VECS_FP; v++) {
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=448 max=448
            fp32_vec_t pin = partial_in.read();
            fp32_vec_t pout;
            for (int k = 0; k < VEC_WIDTH_FP; k++) {
#pragma HLS unroll
                pout[k] = pin[k] + w_route * y_buf[v * VEC_WIDTH_FP + k];
            }
            partial_out.write(pout);
        }
    }
}

// ============================================================================
// Compute: partial_seed
//   Seeds the streaming reduction chain with N * HIDDEN_VECS_FP zero fp32
//   vectors. Lane 0 of the expert chain consumes these as `partial_in` and
//   emits its own contribution as `partial_out`; subsequent lanes read the
//   running sum and add their weighted contribution. The final lane's output
//   is the fully accumulated y, which goes straight to y_writer.
// ============================================================================
void partial_seed(
    const int N,
    tapa::ostream<fp32_vec_t>& seed_out
) {
    const int total = N * HIDDEN_VECS_FP;
    for (int i = 0; i < total; i++) {
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=448 max=1835008
        fp32_vec_t z;
        for (int k = 0; k < VEC_WIDTH_FP; k++) {
#pragma HLS unroll
            z[k] = 0.0f;
        }
        seed_out.write(z);
    }
}

// ============================================================================
// Writer: y_writer
//   Writes N * HIDDEN_VECS_FP FP32 vectors to y_mem using async_mmap write
//   with request/response overlap. No computation.
// ============================================================================
void y_writer(
    const int N,
    tapa::istream<fp32_vec_t>&      y_in,
    tapa::async_mmap<fp32_vec_t>&   y_mem
) {
    const int total = N * HIDDEN_VECS_FP;

    for (int i_req = 0, i_resp = 0; i_resp < total;) {
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=448 max=28672
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
// Top-level task graph
//
// Readers (async_mmap only)  →  Compute tasks (streams only)  →  Writer (async_mmap only)
// ============================================================================
void moe_fpga_top(
    const int   N,
    const int   local_expert_base,
    const float x_scale,
    const float w_gate_scale,
    const float w_gate_shift,
    tapa::mmap<int16_vec_t> x_mem,
    tapa::mmap<int16_vec_t> W_gate_mem,
    tapa::mmap<float>                              bias_dequant_mem,
    // Opt 3: six HBM ports per parallel lane — even/odd row sub-matrices.
    tapa::mmaps<int16_vec_t, N_PARALLEL_SLOTS>     W1_expert_mem,
    tapa::mmaps<int16_vec_t, N_PARALLEL_SLOTS>     W1b_expert_mem,
    tapa::mmaps<int16_vec_t, N_PARALLEL_SLOTS>     W3_expert_mem,
    tapa::mmaps<int16_vec_t, N_PARALLEL_SLOTS>     W3b_expert_mem,
    tapa::mmaps<int16_vec_t, N_PARALLEL_SLOTS>     W2_expert_mem,
    tapa::mmaps<int16_vec_t, N_PARALLEL_SLOTS>     W2b_expert_mem,
    tapa::mmap<fp32_vec_t>                         y_mem
) {
    // Reader → gate_compute
    tapa::stream<int16_vec_t> x_gate_fifo("x_gate_fifo");
    tapa::stream<int16_vec_t> w_gate_fifo("w_gate_fifo");
    tapa::stream<float> bias_fifo  ("bias_fifo");

    // dequant chain: bias_dequant_reader → dequant_split[0] → ... → dequant_split[K-1] → sink.
    // depth 8 comfortably holds all DEQUANT_PARAMS_PER_SLOT (=6) entries.
    tapa::streams<float, N_PARALLEL_SLOTS + 1, 8> dequant_chain("dequant_chain");

    // dequant_split per-lane outputs (2 floats per lane to each projection).
    tapa::streams<float, N_PARALLEL_SLOTS, 4> w1_dq_fifos("w1_dq");
    tapa::streams<float, N_PARALLEL_SLOTS, 4> w3_dq_fifos("w3_dq");
    tapa::streams<float, N_PARALLEL_SLOTS, 4> w2_dq_fifos("w2_dq");

    // gate_compute → dispatch_compute
    tapa::stream<route_pkt_t,   4>  route_out  ("route_out");
    tapa::stream<int16_vec_t, 4>  x_disp_fifo("x_disp_fifo");

    // dispatch_compute → ffn_x_split (one stream per parallel lane)
    tapa::streams<dispatch_hdr_t, N_PARALLEL_SLOTS, 4>   disp_hdr_fifos("disp_hdr");
    tapa::streams<int16_vec_t,    N_PARALLEL_SLOTS, 4> disp_x_fifos  ("disp_x");

    // dispatch_compute → weight loaders (one per matrix per lane).
    tapa::streams<weight_trigger_t, N_PARALLEL_SLOTS, 4> trigger1_fifos("trigger1");
    tapa::streams<weight_trigger_t, N_PARALLEL_SLOTS, 4> trigger3_fifos("trigger3");
    tapa::streams<weight_trigger_t, N_PARALLEL_SLOTS, 4> trigger2_fifos("trigger2");
    // Opt 3: triggers for odd-row sub-matrix loaders.
    tapa::streams<weight_trigger_t, N_PARALLEL_SLOTS, 4> trigger1b_fifos("trigger1b");
    tapa::streams<weight_trigger_t, N_PARALLEL_SLOTS, 4> trigger3b_fifos("trigger3b");
    tapa::streams<weight_trigger_t, N_PARALLEL_SLOTS, 4> trigger2b_fifos("trigger2b");

    // Weight loader → projection task (one per matrix per lane).
    tapa::streams<int16_vec_t, N_PARALLEL_SLOTS> W1_fifos("W1_fifo");
    tapa::streams<int16_vec_t, N_PARALLEL_SLOTS> W3_fifos("W3_fifo");
    tapa::streams<int16_vec_t, N_PARALLEL_SLOTS> W2_fifos("W2_fifo");
    // Opt 3: odd-row sub-matrix fifos.
    tapa::streams<int16_vec_t, N_PARALLEL_SLOTS> W1b_fifos("W1b_fifo");
    tapa::streams<int16_vec_t, N_PARALLEL_SLOTS> W3b_fifos("W3b_fifo");
    tapa::streams<int16_vec_t, N_PARALLEL_SLOTS> W2b_fifos("W2b_fifo");

    // ffn_x_split → ffn_w1_proj / ffn_w3_proj / ffn_w2_proj.
    tapa::streams<int16_vec_t, N_PARALLEL_SLOTS, 4> x1_fifos("x1");
    tapa::streams<int16_vec_t, N_PARALLEL_SLOTS, 4> x3_fifos("x3");
    tapa::streams<float, N_PARALLEL_SLOTS, 4> xs_w1_fifos  ("xs_w1");
    tapa::streams<float, N_PARALLEL_SLOTS, 4> xs_w3_fifos  ("xs_w3");
    tapa::streams<float, N_PARALLEL_SLOTS, 4> w_route_fifos("w_route");

    // Projection → swiglu: each beat packs (even, odd) row results as float2_t.
    tapa::streams<float2_t, N_PARALLEL_SLOTS, 4> acc1_fifos("acc1");
    tapa::streams<float2_t, N_PARALLEL_SLOTS, 4> acc3_fifos("acc3");

    // swiglu → ffn_w2_proj.
    tapa::streams<int16_vec_t, N_PARALLEL_SLOTS,  4> inter_fifos      ("inter");
    tapa::streams<float,       N_PARALLEL_SLOTS,  4> inter_scale_fifos("inter_scale");

    // Streaming reduction chain: seed → lane0 → lane1 → ... → laneK-1 → y_writer.
    tapa::streams<fp32_vec_t, N_PARALLEL_SLOTS + 1, 16> partial_chain("partial_chain");

    tapa::task()
        .invoke<tapa::join>(x_gate_reader,       N, x_mem, x_gate_fifo)
        .invoke<tapa::join>(w_gate_reader,       W_gate_mem, w_gate_fifo)
        .invoke<tapa::join>(bias_dequant_reader, bias_dequant_mem,
                            bias_fifo, dequant_chain)
        .invoke<tapa::join>(gate_compute,        N, local_expert_base,
                            x_scale, w_gate_scale, w_gate_shift,
                            x_gate_fifo, w_gate_fifo, bias_fifo,
                            route_out, x_disp_fifo)
        .invoke<tapa::join>(dispatch_compute,    N, local_expert_base, x_scale,
                            route_out, x_disp_fifo,
                            disp_hdr_fifos, disp_x_fifos,
                            trigger1_fifos, trigger3_fifos, trigger2_fifos,
                            trigger1b_fifos, trigger3b_fifos, trigger2b_fifos)
        // dequant_split chain — instance i reads dequant_chain[i] and writes
        // dequant_chain[i+1], plus per-lane (w1/w3/w2) dequant pairs.
        .invoke<tapa::join, N_PARALLEL_SLOTS>(
            dequant_split,
            dequant_chain, dequant_chain,
            w1_dq_fifos, w3_dq_fifos, w2_dq_fifos)
        .invoke<tapa::join>(dequant_sink, dequant_chain)
        .invoke<tapa::join, N_PARALLEL_SLOTS>(
            ffn_x_split,
            N, disp_hdr_fifos, disp_x_fifos,
            x1_fifos, x3_fifos, xs_w1_fifos, xs_w3_fifos, w_route_fifos)
        // Even-row loaders (same function, different ports)
        .invoke<tapa::join, N_PARALLEL_SLOTS>(
            w1_weight_loader, N, trigger1_fifos, W1_expert_mem, W1_fifos)
        .invoke<tapa::join, N_PARALLEL_SLOTS>(
            w3_weight_loader, N, trigger3_fifos, W3_expert_mem, W3_fifos)
        .invoke<tapa::join, N_PARALLEL_SLOTS>(
            w2_weight_loader, N, trigger2_fifos, W2_expert_mem, W2_fifos)
        // Odd-row loaders (same functions, b-ports)
        .invoke<tapa::join, N_PARALLEL_SLOTS>(
            w1_weight_loader, N, trigger1b_fifos, W1b_expert_mem, W1b_fifos)
        .invoke<tapa::join, N_PARALLEL_SLOTS>(
            w3_weight_loader, N, trigger3b_fifos, W3b_expert_mem, W3b_fifos)
        .invoke<tapa::join, N_PARALLEL_SLOTS>(
            w2_weight_loader, N, trigger2b_fifos, W2b_expert_mem, W2b_fifos)
        .invoke<tapa::join, N_PARALLEL_SLOTS>(
            ffn_w1_proj,
            N, x1_fifos, W1_fifos, W1b_fifos, xs_w1_fifos, w1_dq_fifos, acc1_fifos)
        .invoke<tapa::join, N_PARALLEL_SLOTS>(
            ffn_w3_proj,
            N, x3_fifos, W3_fifos, W3b_fifos, xs_w3_fifos, w3_dq_fifos, acc3_fifos)
        .invoke<tapa::join, N_PARALLEL_SLOTS>(
            ffn_swiglu,
            N, acc1_fifos, acc3_fifos, inter_fifos, inter_scale_fifos)
        .invoke<tapa::join>(partial_seed, N, partial_chain)
        .invoke<tapa::join, N_PARALLEL_SLOTS>(
            ffn_w2_proj,
            N, inter_fifos, inter_scale_fifos, w_route_fifos,
            W2_fifos, W2b_fifos, w2_dq_fifos,
            partial_chain, partial_chain)
        .invoke<tapa::join>(y_writer, N, partial_chain, y_mem);
}
