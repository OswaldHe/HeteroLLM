// ============================================================================
// MoE FPGA Kernel — DeepSeek V3.2 Routed Experts
//
// Opt 4 — Change 1 of 3 (2026-04-27):
//   Fuse W1 + W2_first_half on the same HBM port (per slot, even/odd parity).
//   Fuse W3 + W2_second_half on the same HBM port (per slot, even/odd parity).
//   Reduces expert-weight HBM ports from 24 to 16 per build.
//
// Per-expert layout on each port (parity = a or b):
//   W1_W2[a/b] : [ W1[a/b] data (W1_HALF_VECS = 65,536 beats) ]
//                [ W2_q1[a/b] data — W2 first-half [a/b] rows  (W2_QUARTER_VECS = 114,688 beats) ]
//   W3_W2[a/b] : [ W3[a/b] data (W3_HALF_VECS = 65,536 beats) ]
//                [ W2_q2[a/b] data — W2 second-half [a/b] rows (W2_QUARTER_VECS = 114,688 beats) ]
//
// Total per channel: 180,224 vectors per expert.
//
// The per-lane FFN dataflow is rebuilt around two cooperating projections:
//   ffn_w1_w2_proj — Phase 1: W1 GEMV → acc1; Phase 2: W2 first-half GEMV → y_first_half
//   ffn_w3_w2_proj — Phase 1: W3 GEMV → acc3; Phase 2: W2 second-half GEMV → y_second_half
//   ffn_swiglu     — emits TWO copies of inter / inter_scale (one per W2 half)
//   y_writer_half  — TWO instances write partial_chain_a/b directly to y_mem_a/b in parallel;
//                    no half_merge needed (two HBM output channels, each HIDDEN/2 wide per token).
// ============================================================================

#include "moe_kernel_dec_spec.h"

static inline float moe_sigmoidf(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}
static inline float moe_siluf(float x) {
    return x * moe_sigmoidf(x);
}

// ============================================================================
// Reader: x_gate_reader
// ============================================================================
void x_gate_reader(
    tapa::async_mmap<int16_vec_t>& x_mem,
    tapa::ostream<int16_vec_t>&    out
) {
    const int total = HIDDEN_VECS;
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
// Reader: w_gate_reader (Opt 3 final: half-matrix per HBM port)
//   Each instance streams W_GATE_HALF_VECS beats from its port.
// ============================================================================
void w_gate_reader(
    tapa::async_mmap<int16_vec_t>& w_gate_mem,
    tapa::ostream<int16_vec_t>&    out
) {
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

// ============================================================================
// Reader: bias_dequant_reader
// ============================================================================
void bias_dequant_reader(
    tapa::async_mmap<float>& mem,       // combined: [bias(256) | dequant(6)]
    tapa::ostream<float>&    bias_out,
    tapa::ostream<float>&    dequant_out
) {
    const int total = N_EXPERTS_TOTAL + DEQUANT_PARAMS_PER_SLOT;
    for (int i_req = 0, i_resp = 0; i_resp < total;) {
#pragma HLS pipeline II=1
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
// Compute: gate_compute (Opt 3 final — 2-way parallel GEMV across two HBM ports)
//   w_gate_in   streams experts [0..127]   (lo-half)
//   w_gate_in_b streams experts [128..255] (hi-half)
// Two independent MAC trees run in parallel for each pair (e_lo, e_hi).
// ============================================================================
void gate_compute(
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
    float bias[N_EXPERTS_TOTAL];
#pragma HLS bind_storage variable=bias type=RAM_1P impl=BRAM
#pragma HLS array_partition variable=bias block factor=2 dim=1
    for (int i = 0; i < N_EXPERTS_TOTAL; i++) {
#pragma HLS pipeline II=1
        bias[i] = bias_in.read();
    }

    // Opt 2: input-stationary x prefetch (also forwarded downstream)
    int16_vec_t x_buf[HIDDEN_VECS];
#pragma HLS bind_storage variable=x_buf type=RAM_2P impl=BRAM
    for (int v = 0; v < HIDDEN_VECS; v++) {
#pragma HLS pipeline II=1
        int16_vec_t xv = x_in.read();
        x_buf[v] = xv;
        x_out.write(xv);
    }

    const float joint_scale = x_scale * w_gate_scale;

    // Opt 3 final: 2-way parallel streaming GEMV — same accumulator widths and
    // MAC structure as Opt 2 (8 lanes x 4 INT16 elements per cycle, ap_int<37>
    // partial, ap_int<44> sum), but two MAC trees run in parallel:
    //   sum_lo over experts [0..127] from w_gate_in
    //   sum_hi over experts [128..255] from w_gate_in_b
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

// ============================================================================
// Compute: dispatch_compute (Opt 4 — 4 trigger ports per lane: W1_W2a/b, W3_W2a/b)
// ============================================================================
void dispatch_compute(
    const int   local_expert_base,
    const float x_scale,
    tapa::istream<route_pkt_t>&                          route_in,
    tapa::istream<int16_vec_t>&                          x_in,
    tapa::ostreams<dispatch_hdr_t,    LOCAL_K>&          hdr_out,
    tapa::ostreams<int16_vec_t,       LOCAL_K>&          x_out,
    tapa::ostreams<weight_trigger_t,  LOCAL_K>&          trigger1a_out,  // W1_W2a
    tapa::ostreams<weight_trigger_t,  LOCAL_K>&          trigger1b_out,  // W1_W2b
    tapa::ostreams<weight_trigger_t,  LOCAL_K>&          trigger1c_out,  // W1_W2c
    tapa::ostreams<weight_trigger_t,  LOCAL_K>&          trigger3a_out,  // W3_W2a
    tapa::ostreams<weight_trigger_t,  LOCAL_K>&          trigger3b_out,  // W3_W2b
    tapa::ostreams<weight_trigger_t,  LOCAL_K>&          trigger3c_out   // W3_W2c
) {
    int16_vec_t x_int_buf[HIDDEN_VECS];
#pragma HLS bind_storage variable=x_int_buf type=RAM_2P impl=BRAM
    for (int v = 0; v < HIDDEN_VECS; v++) {
#pragma HLS pipeline II=1
        x_int_buf[v] = x_in.read();
    }

    route_pkt_t route = route_in.read();

    for (int i = 0; i < LOCAL_K; i++) {
#pragma HLS unroll
        int eid_global = (int)route.local_expert_ids[i];
        int rel_eid    = eid_global - local_expert_base;
        dispatch_hdr_t hdr;
        hdr.token_id         = (token_id_t)0;
        hdr.global_expert_id = (expert_id_t)eid_global;
        hdr.routing_weight   = route.weights[i];
        hdr.x_scale          = x_scale;
        hdr.done             = false;
        hdr_out[i].write(hdr);
        trigger1a_out[i].write(rel_eid);
        trigger1b_out[i].write(rel_eid);
        trigger1c_out[i].write(rel_eid);
        trigger3a_out[i].write(rel_eid);
        trigger3b_out[i].write(rel_eid);
        trigger3c_out[i].write(rel_eid);
    }

    for (int v = 0; v < HIDDEN_VECS; v++) {
#pragma HLS pipeline II=1
        int16_vec_t xv = x_int_buf[v];
        for (int i = 0; i < LOCAL_K; i++) {
#pragma HLS unroll
            x_out[i].write(xv);
        }
    }
}

// ============================================================================
// Reader: w1_w2_weight_loader
//   One HBM port carries [W1_HALF (or W3_HALF) | W2_QUARTER] per expert.
//   On each trigger, the loader streams the full W1_W2_PORT_VECS contiguous
//   beats. The downstream compute task internally separates Phase 1 (W1/W3)
//   and Phase 2 (W2 first/second half) consumption.
// ============================================================================
void w1_w2_weight_loader(
    tapa::istream<weight_trigger_t>& trigger_in,
    tapa::async_mmap<int16_vec_t>&   W_mem,
    tapa::ostream<int16_vec_t>&      W_fifo
) {
    const int TOTAL = W1_W2_PORT_VECS;
    weight_trigger_t trig = trigger_in.read();
    const int base = trig * TOTAL;
    for (int i_req = 0, i_resp = 0; i_resp < TOTAL;) {
#pragma HLS pipeline II=1
        if ((i_req < TOTAL) & !W_mem.read_addr.full()) {
            W_mem.read_addr.try_write(base + i_req); ++i_req;
        }
        if (!W_mem.read_data.empty()) {
            int16_vec_t tmp; W_mem.read_data.try_read(tmp);
            W_fifo.write(tmp);
            ++i_resp;
        }
    }
}

// ============================================================================
// Reader: w1_w2c_weight_loader (Opt 5 — third channel for W1+W2_first_half)
//   Per-expert layout on the c port:
//     [ W1 c-rows  (W1_THIRD_VECS  = 683 * HIDDEN_VECS) ]
//     [ W2_first c-rows (W2_SIXTH_VECS = 1195 * INTER_VECS) ]
//   The last beat of each section may be padding (dummy row); the consumer
//   discards out-of-bound rows.
// ============================================================================
void w1_w2c_weight_loader(
    tapa::istream<weight_trigger_t>& trigger_in,
    tapa::async_mmap<int16_vec_t>&   W_mem,
    tapa::ostream<int16_vec_t>&      W_fifo
) {
    const int TOTAL = W1_W2C_PORT_VECS;
    weight_trigger_t trig = trigger_in.read();
    const int base = trig * TOTAL;
    for (int i_req = 0, i_resp = 0; i_resp < TOTAL;) {
#pragma HLS pipeline II=1
        if ((i_req < TOTAL) & !W_mem.read_addr.full()) {
            W_mem.read_addr.try_write(base + i_req); ++i_req;
        }
        if (!W_mem.read_data.empty()) {
            int16_vec_t tmp; W_mem.read_data.try_read(tmp);
            W_fifo.write(tmp);
            ++i_resp;
        }
    }
}

// ============================================================================
// Reader: w3_w2c_weight_loader (Opt 5 — third channel for W3+W2_second_half)
//   Mirror of w1_w2c_weight_loader, using W3_W2C_PORT_VECS (numerically equal).
// ============================================================================
void w3_w2c_weight_loader(
    tapa::istream<weight_trigger_t>& trigger_in,
    tapa::async_mmap<int16_vec_t>&   W_mem,
    tapa::ostream<int16_vec_t>&      W_fifo
) {
    const int TOTAL = W3_W2C_PORT_VECS;
    weight_trigger_t trig = trigger_in.read();
    const int base = trig * TOTAL;
    for (int i_req = 0, i_resp = 0; i_resp < TOTAL;) {
#pragma HLS pipeline II=1
        if ((i_req < TOTAL) & !W_mem.read_addr.full()) {
            W_mem.read_addr.try_write(base + i_req); ++i_req;
        }
        if (!W_mem.read_data.empty()) {
            int16_vec_t tmp; W_mem.read_data.try_read(tmp);
            W_fifo.write(tmp);
            ++i_resp;
        }
    }
}

// ============================================================================
// dequant_split: routes the 6-float dequant chain into per-projection pairs.
//   index 0,1 → w1_dq_out  (w1_scale, w1_shift)
//   index 2,3 → w3_dq_out  (w3_scale, w3_shift)
//   index 4,5 → w2_dq_a_out AND w2_dq_b_out  (W2 scale/shift duplicated to both halves)
// ============================================================================
void dequant_split(
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
        if (i == 0 || i == 1) {
            w1_dq_out.write(v);
        } else if (i == 2 || i == 3) {
            w3_dq_out.write(v);
        } else {
            w2_dq_a_out.write(v);
            w2_dq_b_out.write(v);
        }
    }
}

// ============================================================================
// ffn_x_split: broadcasts the dispatch header's per-token scalars to W1/W3
// projections, and replicates routing weight to BOTH W2 half projections.
// ============================================================================
void ffn_x_split(
    tapa::istream<dispatch_hdr_t>& hdr_in,
    tapa::istream<int16_vec_t>&    x_in,
    tapa::ostream<int16_vec_t>&    x1_out,
    tapa::ostream<int16_vec_t>&    x3_out,
    tapa::ostream<float>&          xs_w1_out,
    tapa::ostream<float>&          xs_w3_out,
    tapa::ostream<float>&          w_route_a_out,
    tapa::ostream<float>&          w_route_b_out
) {
    for (int v = 0; v < HIDDEN_VECS; v++) {
#pragma HLS pipeline II=1
        int16_vec_t xv = x_in.read();
        x1_out.write(xv);
        x3_out.write(xv);
    }
    dispatch_hdr_t hdr = hdr_in.read();
    xs_w1_out.write(hdr.x_scale);
    xs_w3_out.write(hdr.x_scale);
    w_route_a_out.write(hdr.routing_weight);
    w_route_b_out.write(hdr.routing_weight);
}

// ============================================================================
// ffn_w1_w2_proj: phase 1 = W1 GEMV; phase 2 = W2 first-half GEMV producing
// y[0..HIDDEN/2) contributions, added to chain_a (HIDDEN_VECS_FP/2 = 224 vecs).
// W1_W2[a/b] streams W1[a/b] then W2_q1[a/b] for each trigger.
// ============================================================================
void ffn_w1_w2_proj(
    tapa::istream<int16_vec_t>& x1_in,
    tapa::istream<int16_vec_t>& W1_W2a_fifo,
    tapa::istream<int16_vec_t>& W1_W2b_fifo,
    tapa::istream<int16_vec_t>& W1_W2c_fifo,
    tapa::istream<float>&       xs_w1_in,
    tapa::istream<float>&       w1_dq_in,
    tapa::istream<int16_vec_t>& inter_a_in,
    tapa::istream<float>&       inter_scale_a_in,
    tapa::istream<float>&       w_route_a_in,
    tapa::istream<float>&       w2_dq_a_in,
    tapa::istream<fp32_vec_t>&  partial_a_in,
    tapa::ostream<float3_t>&    acc1_out,
    tapa::ostream<fp32_vec_t>&  partial_a_out
) {
    float w1_scale = w1_dq_in.read();
    float w1_shift = w1_dq_in.read();
    float w2_scale = w2_dq_a_in.read();
    float w2_shift = w2_dq_a_in.read();

    int16_vec_t x_cur[HIDDEN_VECS];
#pragma HLS bind_storage variable=x_cur type=RAM_2P impl=BRAM
    int16_vec_t inter_buf[INTER_VECS];
#pragma HLS bind_storage variable=inter_buf type=RAM_2P impl=BRAM
    // Opt 5: oversize y_buf to fit 3*W2_SIXTH_ROWS slots (= 3585; real range
    // is HIDDEN/2 = 3584). The final c slot is a dummy padded write driven
    // by the c-port; with shift=0 and zero-padded weights it stays at 0 and
    // is never read by the partial-sum writeback loop below (which reads
    // exactly HIDDEN/2 entries).
    float y_buf[3 * W2_SIXTH_ROWS];
#pragma HLS bind_storage variable=y_buf type=RAM_2P impl=BRAM
#pragma HLS array_partition variable=y_buf cyclic factor=16

    float x_scale     = 0.0f;
    float inter_scale = 0.0f;
    float w_route     = 0.0f;

    // Opt 5: 3-way row interleaving (a/b/c).
    //   Stage 0 = W1 GEMV: 683 rh-groups, 3 rows per group; row 2048 padded.
    //   Stage 1 = W2 first-half GEMV: 1195 rh-groups, 3 rows per group;
    //             row 3584 padded into the y_buf scratch slot.
    // The padded c-rows on each port carry zero weights, so sum_c=0 and with
    // dq_shift=0 the dummy outputs are 0; host crops outputs to real ranges.
    for (int stage = 0; stage < 2; stage++) {
        if (stage == 0) {
            for (int v = 0; v < HIDDEN_VECS; v++) {
#pragma HLS pipeline II=1
                x_cur[v] = x1_in.read();
            }
            x_scale = xs_w1_in.read();
        } else {
            inter_scale = inter_scale_a_in.read();
            w_route     = w_route_a_in.read();
            for (int v = 0; v < INTER_VECS; v++) {
#pragma HLS pipeline II=1
                inter_buf[v] = inter_a_in.read();
            }
        }

        float scale_w  = (stage == 0) ? (w1_scale * x_scale) : (w2_scale * inter_scale);
        float dq_shift = (stage == 0) ? w1_shift : w2_shift;
        int   n_rows   = (stage == 0) ? W1_THIRD_ROWS : W2_SIXTH_ROWS;
        int   n_vecs   = (stage == 0) ? HIDDEN_VECS : INTER_VECS;

        // Opt 9 (rh_flatten): flatten the (rh, v) nest into a single pipelined
        // loop with manual induction. Reset accumulators at v==0 and commit at
        // v==n_vecs-1. MAC inner block is byte-for-byte identical to the
        // original to preserve numerics.
        ap_int<44> sum_a = 0, sum_b = 0, sum_c = 0;
        int rh = 0;
        int v  = 0;
        const int total_iters = n_rows * n_vecs;
        for (int it = 0; it < total_iters; it++) {
#pragma HLS loop_tripcount min=43712 max=76480
#pragma HLS pipeline II=1
            if (v == 0) {
                sum_a = 0;
                sum_b = 0;
                sum_c = 0;
            }
            int16_vec_t xv  = (stage == 0) ? x_cur[v] : inter_buf[v];
            int16_vec_t wav = W1_W2a_fifo.read();
            int16_vec_t wbv = W1_W2b_fifo.read();
            int16_vec_t wcv = W1_W2c_fifo.read();
            ap_int<37> pa = 0, pb = 0, pc = 0;
            for (int k = 0; k < VEC_WIDTH; k++) {
#pragma HLS unroll
                pa += xv[k] * wav[k];
                pb += xv[k] * wbv[k];
                pc += xv[k] * wcv[k];
            }
            sum_a += pa;
            sum_b += pb;
            sum_c += pc;
            if (v == n_vecs - 1) {
                if (stage == 0) {
                    float3_t triple;
                    triple[0] = (float)sum_a * scale_w + dq_shift;
                    triple[1] = (float)sum_b * scale_w + dq_shift;
                    triple[2] = (float)sum_c * scale_w + dq_shift;
                    acc1_out.write(triple);
                } else {
                    y_buf[3*rh]     = (float)sum_a * scale_w + dq_shift;
                    y_buf[3*rh + 1] = (float)sum_b * scale_w + dq_shift;
                    y_buf[3*rh + 2] = (float)sum_c * scale_w + dq_shift;
                }
                rh++;
                v = 0;
            } else {
                v++;
            }
        }
    }

    for (int v = 0; v < HIDDEN_VECS_FP/2; v++) {
#pragma HLS pipeline II=1
        fp32_vec_t pin = partial_a_in.read();
        fp32_vec_t pout;
        for (int k = 0; k < VEC_WIDTH_FP; k++) {
#pragma HLS unroll
            pout[k] = pin[k] + w_route * y_buf[v * VEC_WIDTH_FP + k];
        }
        partial_a_out.write(pout);
    }
}

// ============================================================================
// ffn_w3_w2_proj: phase 1 = W3 GEMV; phase 2 = W2 second-half GEMV producing
// y[HIDDEN/2..HIDDEN) contributions, added to chain_b.
// ============================================================================
void ffn_w3_w2_proj(
    tapa::istream<int16_vec_t>& x3_in,
    tapa::istream<int16_vec_t>& W3_W2a_fifo,
    tapa::istream<int16_vec_t>& W3_W2b_fifo,
    tapa::istream<int16_vec_t>& W3_W2c_fifo,
    tapa::istream<float>&       xs_w3_in,
    tapa::istream<float>&       w3_dq_in,
    tapa::istream<int16_vec_t>& inter_b_in,
    tapa::istream<float>&       inter_scale_b_in,
    tapa::istream<float>&       w_route_b_in,
    tapa::istream<float>&       w2_dq_b_in,
    tapa::istream<fp32_vec_t>&  partial_b_in,
    tapa::ostream<float3_t>&    acc3_out,
    tapa::ostream<fp32_vec_t>&  partial_b_out
) {
    float w3_scale = w3_dq_in.read();
    float w3_shift = w3_dq_in.read();
    float w2_scale = w2_dq_b_in.read();
    float w2_shift = w2_dq_b_in.read();

    int16_vec_t x_cur[HIDDEN_VECS];
#pragma HLS bind_storage variable=x_cur type=RAM_2P impl=BRAM
    int16_vec_t inter_buf[INTER_VECS];
#pragma HLS bind_storage variable=inter_buf type=RAM_2P impl=BRAM
    // Opt 5: oversize y_buf by one slot for the padded c-row (see ffn_w1_w2_proj note).
    float y_buf[3 * W2_SIXTH_ROWS];
#pragma HLS bind_storage variable=y_buf type=RAM_2P impl=BRAM
#pragma HLS array_partition variable=y_buf cyclic factor=16

    float x_scale     = 0.0f;
    float inter_scale = 0.0f;
    float w_route     = 0.0f;

    // Opt 5: 3-way row interleaving (a/b/c) — symmetric to ffn_w1_w2_proj.
    for (int stage = 0; stage < 2; stage++) {
        if (stage == 0) {
            for (int v = 0; v < HIDDEN_VECS; v++) {
#pragma HLS pipeline II=1
                x_cur[v] = x3_in.read();
            }
            x_scale = xs_w3_in.read();
        } else {
            inter_scale = inter_scale_b_in.read();
            w_route     = w_route_b_in.read();
            for (int v = 0; v < INTER_VECS; v++) {
#pragma HLS pipeline II=1
                inter_buf[v] = inter_b_in.read();
            }
        }

        float scale_w  = (stage == 0) ? (w3_scale * x_scale) : (w2_scale * inter_scale);
        float dq_shift = (stage == 0) ? w3_shift : w2_shift;
        int   n_rows   = (stage == 0) ? W1_THIRD_ROWS : W2_SIXTH_ROWS;
        int   n_vecs   = (stage == 0) ? HIDDEN_VECS : INTER_VECS;

        // Opt 9 (rh_flatten): mirror of ffn_w1_w2_proj — flatten (rh, v) nest
        // into a single pipelined loop with manual induction. Numerics are
        // bit-identical to the unflattened version.
        ap_int<44> sum_a = 0, sum_b = 0, sum_c = 0;
        int rh = 0;
        int v  = 0;
        const int total_iters = n_rows * n_vecs;
        for (int it = 0; it < total_iters; it++) {
#pragma HLS loop_tripcount min=43712 max=76480
#pragma HLS pipeline II=1
            if (v == 0) {
                sum_a = 0;
                sum_b = 0;
                sum_c = 0;
            }
            int16_vec_t xv  = (stage == 0) ? x_cur[v] : inter_buf[v];
            int16_vec_t wav = W3_W2a_fifo.read();
            int16_vec_t wbv = W3_W2b_fifo.read();
            int16_vec_t wcv = W3_W2c_fifo.read();
            ap_int<37> pa = 0, pb = 0, pc = 0;
            for (int k = 0; k < VEC_WIDTH; k++) {
#pragma HLS unroll
                pa += xv[k] * wav[k];
                pb += xv[k] * wbv[k];
                pc += xv[k] * wcv[k];
            }
            sum_a += pa;
            sum_b += pb;
            sum_c += pc;
            if (v == n_vecs - 1) {
                if (stage == 0) {
                    float3_t triple;
                    triple[0] = (float)sum_a * scale_w + dq_shift;
                    triple[1] = (float)sum_b * scale_w + dq_shift;
                    triple[2] = (float)sum_c * scale_w + dq_shift;
                    acc3_out.write(triple);
                } else {
                    y_buf[3*rh]     = (float)sum_a * scale_w + dq_shift;
                    y_buf[3*rh + 1] = (float)sum_b * scale_w + dq_shift;
                    y_buf[3*rh + 2] = (float)sum_c * scale_w + dq_shift;
                }
                rh++;
                v = 0;
            } else {
                v++;
            }
        }
    }

    for (int v = 0; v < HIDDEN_VECS_FP/2; v++) {
#pragma HLS pipeline II=1
        fp32_vec_t pin = partial_b_in.read();
        fp32_vec_t pout;
        for (int k = 0; k < VEC_WIDTH_FP; k++) {
#pragma HLS unroll
            pout[k] = pin[k] + w_route * y_buf[v * VEC_WIDTH_FP + k];
        }
        partial_b_out.write(pout);
    }
}

// ============================================================================
// ffn_swiglu (Opt 4): emits TWO copies of inter and inter_scale — one to the
// W1_W2 chain (chain_a) and one to the W3_W2 chain (chain_b). Same data.
// ============================================================================
void ffn_swiglu(
    tapa::istream<float3_t>&    acc1_in,
    tapa::istream<float3_t>&    acc3_in,
    tapa::ostream<int16_vec_t>& inter_a_out,
    tapa::ostream<int16_vec_t>& inter_b_out,
    tapa::ostream<float>&       inter_scale_a_out,
    tapa::ostream<float>&       inter_scale_b_out
) {
    // Opt 5: oversize inter_buf by one slot for the padded c-row from the
    // last 3-way group (real range is MOE_INTER = 2048; buffer is 3*683 = 2049).
    // The padded slot holds a 0 (zero weights × shift=0) and is never read by
    // the quantization loop below (which iterates exactly INTER_VECS = MOE_INTER
    // elements).
    float inter_buf[3 * W1_THIRD_ROWS];
#pragma HLS bind_storage variable=inter_buf type=RAM_2P impl=BRAM
#pragma HLS array_partition variable=inter_buf cyclic factor=32

    float inter_max = 0.0f;
    for (int rh = 0; rh < W1_THIRD_ROWS; rh++) {
#pragma HLS pipeline II=1
        float3_t p1 = acc1_in.read();
        float3_t p3 = acc3_in.read();
        float va = moe_siluf(p1[0]) * p3[0];
        float vb = moe_siluf(p1[1]) * p3[1];
        float vc = moe_siluf(p1[2]) * p3[2];
        inter_buf[3*rh]     = va;
        inter_buf[3*rh + 1] = vb;
        inter_buf[3*rh + 2] = vc;
        float aa = (va >= 0.0f) ? va : -va;
        float ab = (vb >= 0.0f) ? vb : -vb;
        float ac = (vc >= 0.0f) ? vc : -vc;
        float mx_ab = (aa > ab) ? aa : ab;
        float mx    = (mx_ab > ac) ? mx_ab : ac;
        if (mx > inter_max) inter_max = mx;
    }

    float inter_scale = (inter_max > 0.0f) ? (inter_max / 32767.0f) : 1.0f;
    float inter_inv   = 1.0f / inter_scale;

    inter_scale_a_out.write(inter_scale);
    inter_scale_b_out.write(inter_scale);

    for (int v = 0; v < INTER_VECS; v++) {
#pragma HLS pipeline II=1
        int16_vec_t iv;
        for (int k = 0; k < VEC_WIDTH; k++) {
#pragma HLS unroll
            float f = inter_buf[v * VEC_WIDTH + k] * inter_inv;
            int q = (int)(f >= 0.0f ? f + 0.5f : f - 0.5f);
            if (q >  32767) q =  32767;
            if (q < -32768) q = -32768;
            iv[k] = (ap_int<16>)q;
        }
        inter_a_out.write(iv);
        inter_b_out.write(iv);
    }
}

// ============================================================================
// partial_seed_half: seeds a half-width partial-sum chain with N * (HIDDEN_VECS_FP/2)
// zero fp32 vectors. Two instances are used (one per chain).
// ============================================================================
void partial_seed_half(
    tapa::ostream<fp32_vec_t>& seed_out
) {
    const int total = HIDDEN_VECS_FP/2;
    for (int i = 0; i < total; i++) {
#pragma HLS pipeline II=1
        fp32_vec_t z;
        for (int k = 0; k < VEC_WIDTH_FP; k++) {
#pragma HLS unroll
            z[k] = 0.0f;
        }
        seed_out.write(z);
    }
}

// ============================================================================
// Writer: y_writer_half
//   Writes one half of the output (HIDDEN_VECS_FP/2 fp32 vecs per token).
//   Two instances run in parallel: instance 0 receives partial_chain_a
//   (first HIDDEN/2 elements), instance 1 receives partial_chain_b
//   (second HIDDEN/2 elements). Each writes to its own HBM channel.
// ============================================================================
void y_writer_half(
    tapa::istream<fp32_vec_t>&    y_in,
    tapa::async_mmap<fp32_vec_t>& y_mem
) {
    const int total = HIDDEN_VECS_FP / 2;

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
// Top-level task graph (Opt 4 — Change 1, dual-channel output)
//   y_mem_a: first  HIDDEN/2 elements per token (from partial_chain_a)
//   y_mem_b: second HIDDEN/2 elements per token (from partial_chain_b)
//   Two y_writer_half instances run in parallel on separate HBM channels;
//   no half_merge task needed.
// ============================================================================
void moe_fpga_top(
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
    tapa::mmap<fp32_vec_t>                         y_mem_a,  // first  HIDDEN/2 per token
    tapa::mmap<fp32_vec_t>                         y_mem_b   // second HIDDEN/2 per token
) {
    tapa::stream<int16_vec_t> x_gate_fifo("x_gate_fifo");
    tapa::stream<int16_vec_t> w_gate_fifo  ("w_gate_fifo");
    tapa::stream<int16_vec_t> w_gate_fifo_b("w_gate_fifo_b");
    tapa::stream<float> bias_fifo  ("bias_fifo");

    tapa::streams<float, N_PARALLEL_SLOTS + 1, 2> dequant_chain("dequant_chain");

    tapa::streams<float, N_PARALLEL_SLOTS, 2> w1_dq_fifos   ("w1_dq");
    tapa::streams<float, N_PARALLEL_SLOTS, 2> w3_dq_fifos   ("w3_dq");
    tapa::streams<float, N_PARALLEL_SLOTS, 2> w2_dq_a_fifos ("w2_dq_a");
    tapa::streams<float, N_PARALLEL_SLOTS, 2> w2_dq_b_fifos ("w2_dq_b");

    tapa::stream<route_pkt_t,   2>  route_out  ("route_out");
    tapa::stream<int16_vec_t,   2>  x_disp_fifo("x_disp_fifo");

    tapa::streams<dispatch_hdr_t, N_PARALLEL_SLOTS, 2>     disp_hdr_fifos("disp_hdr");
    tapa::streams<int16_vec_t,    N_PARALLEL_SLOTS, 2> disp_x_fifos  ("disp_x");

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

    tapa::streams<int16_vec_t, N_PARALLEL_SLOTS, 2> x1_fifos("x1");
    tapa::streams<int16_vec_t, N_PARALLEL_SLOTS, 2> x3_fifos("x3");
    tapa::streams<float, N_PARALLEL_SLOTS, 2> xs_w1_fifos    ("xs_w1");
    tapa::streams<float, N_PARALLEL_SLOTS, 2> xs_w3_fifos    ("xs_w3");
    tapa::streams<float, N_PARALLEL_SLOTS, 2> w_route_a_fifos("w_route_a");
    tapa::streams<float, N_PARALLEL_SLOTS, 2> w_route_b_fifos("w_route_b");

    tapa::streams<float3_t, N_PARALLEL_SLOTS, 2> acc1_fifos("acc1");
    tapa::streams<float3_t, N_PARALLEL_SLOTS, 2> acc3_fifos("acc3");

    tapa::streams<int16_vec_t, N_PARALLEL_SLOTS, 2> inter_a_fifos      ("inter_a");
    tapa::streams<int16_vec_t, N_PARALLEL_SLOTS, 2> inter_b_fifos      ("inter_b");
    tapa::streams<float,       N_PARALLEL_SLOTS, 2> inter_scale_a_fifos("inter_scale_a");
    tapa::streams<float,       N_PARALLEL_SLOTS, 2> inter_scale_b_fifos("inter_scale_b");

    tapa::streams<fp32_vec_t, N_PARALLEL_SLOTS + 1, 2> partial_chain_a("partial_chain_a");
    tapa::streams<fp32_vec_t, N_PARALLEL_SLOTS + 1, 2> partial_chain_b("partial_chain_b");

    tapa::task()
        .invoke<tapa::join>(x_gate_reader,       x_mem, x_gate_fifo)
        .invoke<tapa::join>(w_gate_reader,       W_gate_mem,   w_gate_fifo)
        .invoke<tapa::join>(w_gate_reader,       W_gate_mem_b, w_gate_fifo_b)
        .invoke<tapa::join>(bias_dequant_reader, bias_dequant_mem,
                            bias_fifo, dequant_chain)
        .invoke<tapa::join>(gate_compute,        local_expert_base,
                            x_scale, w_gate_scale, w_gate_shift,
                            x_gate_fifo, w_gate_fifo, w_gate_fifo_b, bias_fifo,
                            route_out, x_disp_fifo)
        .invoke<tapa::join>(dispatch_compute,    local_expert_base, x_scale,
                            route_out, x_disp_fifo,
                            disp_hdr_fifos, disp_x_fifos,
                            trigger1a_fifos, trigger1b_fifos, trigger1c_fifos,
                            trigger3a_fifos, trigger3b_fifos, trigger3c_fifos)
        .invoke<tapa::join, N_PARALLEL_SLOTS>(
            dequant_split,
            dequant_chain, dequant_chain,
            w1_dq_fifos, w3_dq_fifos, w2_dq_a_fifos, w2_dq_b_fifos)
        .invoke<tapa::join>(dequant_sink, dequant_chain)
        .invoke<tapa::join, N_PARALLEL_SLOTS>(
            ffn_x_split,
            disp_hdr_fifos, disp_x_fifos,
            x1_fifos, x3_fifos,
            xs_w1_fifos, xs_w3_fifos,
            w_route_a_fifos, w_route_b_fifos)
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
        .invoke<tapa::join>(partial_seed_half, partial_chain_a)
        .invoke<tapa::join>(partial_seed_half, partial_chain_b)
        .invoke<tapa::join, N_PARALLEL_SLOTS>(
            ffn_w1_w2_proj,
            x1_fifos, W1_W2a_fifos, W1_W2b_fifos, W1_W2c_fifos,
            xs_w1_fifos, w1_dq_fifos,
            inter_a_fifos, inter_scale_a_fifos,
            w_route_a_fifos, w2_dq_a_fifos,
            partial_chain_a,
            acc1_fifos,
            partial_chain_a)
        .invoke<tapa::join, N_PARALLEL_SLOTS>(
            ffn_w3_w2_proj,
            x3_fifos, W3_W2a_fifos, W3_W2b_fifos, W3_W2c_fifos,
            xs_w3_fifos, w3_dq_fifos,
            inter_b_fifos, inter_scale_b_fifos,
            w_route_b_fifos, w2_dq_b_fifos,
            partial_chain_b,
            acc3_fifos,
            partial_chain_b)
        .invoke<tapa::join, N_PARALLEL_SLOTS>(
            ffn_swiglu,
            acc1_fifos, acc3_fifos,
            inter_a_fifos, inter_b_fifos,
            inter_scale_a_fifos, inter_scale_b_fifos)
        .invoke<tapa::join>(y_writer_half, partial_chain_a, y_mem_a)
        .invoke<tapa::join>(y_writer_half, partial_chain_b, y_mem_b);
}
