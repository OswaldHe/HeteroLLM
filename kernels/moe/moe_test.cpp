#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <random>
#include <algorithm>
#include <cassert>

#include <gflags/gflags.h>
#include <tapa.h>

#include "moe_kernel.h"

template <typename T>
using aligned_vector = std::vector<T, tapa::aligned_allocator<T>>;

DEFINE_string(bitstream, "", "path to bitstream; run csim if empty");
DEFINE_int32 (N, 4, "number of input tokens");
DEFINE_int32 (seed, 12345, "RNG seed");

static inline float ref_sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }
static inline float ref_siluf(float x)    { return x * ref_sigmoidf(x); }

// Reference model — mirrors the kernel now that gate projection is INT16.
// x and W_gate are provided already-quantized (INT16) with global scales.
static void reference_moe(
    int N, int local_expert_base,
    const std::vector<int16_t>& x_int16,
    const std::vector<int16_t>& W_gate_int16,
    const std::vector<float>&   bias,
    const std::vector<std::vector<ap_int<16>>>& W1_all,
    const std::vector<std::vector<ap_int<16>>>& W3_all,
    const std::vector<std::vector<ap_int<16>>>& W2_all,
    float x_scale,
    float w_gate_scale, float w_gate_shift,
    float w1_scale, float w1_shift,
    float w3_scale, float w3_shift,
    float w2_scale, float w2_shift,
    std::vector<float>& y_out
) {
    y_out.assign(N * HIDDEN, 0.0f);

    for (int t = 0; t < N; t++) {
        // INT16 gate scoring matches the kernel exactly.
        std::vector<float> scores(N_EXPERTS_TOTAL);
        for (int e = 0; e < N_EXPERTS_TOTAL; e++) {
            int64_t acc = 0;
            for (int d = 0; d < HIDDEN; d++) {
                acc += (int64_t)x_int16[t * HIDDEN + d]
                     * (int64_t)W_gate_int16[e * HIDDEN + d];
            }
            scores[e] = ref_sigmoidf((float)acc * x_scale * w_gate_scale + w_gate_shift)
                      + bias[e];
        }

        std::vector<int>   top_lo_ids(LOCAL_K, 0);
        std::vector<float> top_lo_w(LOCAL_K, -1e30f);
        for (int e = 0; e < LOCAL_EXPERTS; e++) {
            float s = scores[e];
            for (int i = 0; i < LOCAL_K; i++) {
                if (s > top_lo_w[i]) {
                    for (int j = LOCAL_K-1; j > i; j--) { top_lo_ids[j]=top_lo_ids[j-1]; top_lo_w[j]=top_lo_w[j-1]; }
                    top_lo_ids[i] = e; top_lo_w[i] = s; break;
                }
            }
        }

        std::vector<int>   top_hi_ids(LOCAL_K, LOCAL_EXPERTS);
        std::vector<float> top_hi_w(LOCAL_K, -1e30f);
        for (int e = LOCAL_EXPERTS; e < N_EXPERTS_TOTAL; e++) {
            float s = scores[e];
            for (int i = 0; i < LOCAL_K; i++) {
                if (s > top_hi_w[i]) {
                    for (int j = LOCAL_K-1; j > i; j--) { top_hi_ids[j]=top_hi_ids[j-1]; top_hi_w[j]=top_hi_w[j-1]; }
                    top_hi_ids[i] = e; top_hi_w[i] = s; break;
                }
            }
        }

        float wsum = 0.0f;
        for (int i = 0; i < LOCAL_K; i++) {
            if (top_lo_w[i] > -1e29f) wsum += top_lo_w[i];
            if (top_hi_w[i] > -1e29f) wsum += top_hi_w[i];
        }
        float scale_w = (wsum > 1e-12f ? ROUTE_SCALE / wsum : 0.0f);

        std::vector<int>   top_ids = (local_expert_base == 0) ? top_lo_ids : top_hi_ids;
        std::vector<float> top_w   = (local_expert_base == 0) ? top_lo_w   : top_hi_w;
        for (int i = 0; i < LOCAL_K; i++) top_w[i] *= scale_w;

        // x for expert FFN is the same INT16 buffer — no per-token requant.
        const int16_t* xq = &x_int16[t * HIDDEN];

        for (int i = 0; i < LOCAL_K; i++) {
            int eid = top_ids[i];
            int rel = eid - local_expert_base;
            if (rel < 0 || rel >= (int)W1_all.size()) continue;

            std::vector<float> inter(MOE_INTER);
            for (int r = 0; r < MOE_INTER; r++) {
                int64_t acc1=0, acc3=0;
                for (int d = 0; d < HIDDEN; d++) {
                    acc1 += (int64_t)xq[d] * (int64_t)(int16_t)W1_all[rel][r*HIDDEN+d];
                    acc3 += (int64_t)xq[d] * (int64_t)(int16_t)W3_all[rel][r*HIDDEN+d];
                }
                float f1 = (float)acc1 * w1_scale * x_scale + w1_shift;
                float f3 = (float)acc3 * w3_scale * x_scale + w3_shift;
                inter[r] = ref_siluf(f1) * f3;
            }

            float imax=0.0f;
            for (int r=0;r<MOE_INTER;r++){float a=std::fabs(inter[r]);if(a>imax)imax=a;}
            float inter_scale=(imax>0.0f?imax/32767.0f:1.0f), inv_is=1.0f/inter_scale;
            std::vector<int16_t> iq(MOE_INTER);
            for (int r=0;r<MOE_INTER;r++){float f=inter[r]*inv_is;int q=(int)std::lrintf(f);if(q>32767)q=32767;if(q<-32768)q=-32768;iq[r]=(int16_t)q;}

            for (int r=0;r<HIDDEN;r++){
                int64_t acc2=0;
                for (int c=0;c<MOE_INTER;c++) acc2+=(int64_t)iq[c]*(int64_t)(int16_t)W2_all[rel][r*MOE_INTER+c];
                float f2=(float)acc2*w2_scale*inter_scale+w2_shift;
                y_out[t*HIDDEN+r]+=top_w[i]*f2;
            }
        }
    }
}

int main(int argc, char** argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    const int N = FLAGS_N;
    const int local_expert_base = 0;
    std::cout << "=== MoE FPGA v3 Testbench (INT16 gate) ===" << std::endl;
    std::cout << "  N=" << N << "  LOCAL_K=" << LOCAL_K << std::endl;

    const float w_scale = 1.0f / 32767.0f, w_shift = 0.0f;

    std::mt19937 gen(FLAGS_seed);
    std::uniform_real_distribution<float> xdist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> wdist(-0.1f, 0.1f);
    std::uniform_real_distribution<float> bdist(-0.01f, 0.01f);
    std::uniform_int_distribution<int>    iwdist(-1000, 1000);

    std::vector<float> x_ref(N * HIDDEN);
    for (auto& v : x_ref) v = xdist(gen);

    std::vector<float> w_gate_ref(N_EXPERTS_TOTAL * HIDDEN);
    for (auto& v : w_gate_ref) v = wdist(gen);

    std::vector<float> bias_ref(N_EXPERTS_TOTAL);
    for (auto& v : bias_ref) v = bdist(gen);
    // Bias experts 0,1,2,3 to always be top-4 from lower half
    bias_ref[0]=10.0f; bias_ref[1]=9.0f; bias_ref[2]=8.0f; bias_ref[3]=7.0f;

    constexpr int N_TEST_EXPERTS = LOCAL_K;
    std::vector<std::vector<ap_int<16>>> W1_all(N_TEST_EXPERTS), W3_all(N_TEST_EXPERTS), W2_all(N_TEST_EXPERTS);
    for (int e = 0; e < N_TEST_EXPERTS; e++) {
        W1_all[e].resize(MOE_INTER * HIDDEN); for (auto& v:W1_all[e]) v=(ap_int<16>)iwdist(gen);
        W3_all[e].resize(MOE_INTER * HIDDEN); for (auto& v:W3_all[e]) v=(ap_int<16>)iwdist(gen);
        W2_all[e].resize(HIDDEN * MOE_INTER); for (auto& v:W2_all[e]) v=(ap_int<16>)iwdist(gen);
    }

    // ------------------------------------------------------------------------
    // Symmetric zero-centered quantization for x (global scale over all tokens)
    // ------------------------------------------------------------------------
    float x_max_abs = 0.0f;
    for (auto v : x_ref) { float a = std::fabs(v); if (a > x_max_abs) x_max_abs = a; }
    const float x_scale = (x_max_abs > 0.0f) ? (x_max_abs / 32767.0f) : 1.0f;
    const float inv_xs  = 1.0f / x_scale;

    std::vector<int16_t> x_int16(N * HIDDEN);
    for (int i = 0; i < N * HIDDEN; i++) {
        float f = x_ref[i] * inv_xs;
        int q = (int)std::lrintf(f);
        if (q > 32767) q = 32767; if (q < -32768) q = -32768;
        x_int16[i] = (int16_t)q;
    }

    // ------------------------------------------------------------------------
    // Symmetric zero-centered quantization for W_gate (global scale)
    // ------------------------------------------------------------------------
    float wg_max_abs = 0.0f;
    for (auto v : w_gate_ref) { float a = std::fabs(v); if (a > wg_max_abs) wg_max_abs = a; }
    const float w_gate_scale = (wg_max_abs > 0.0f) ? (wg_max_abs / 32767.0f) : 1.0f;
    const float w_gate_shift = 0.0f;
    const float inv_wgs = 1.0f / w_gate_scale;

    std::vector<int16_t> W_gate_int16(N_EXPERTS_TOTAL * HIDDEN);
    for (int i = 0; i < N_EXPERTS_TOTAL * HIDDEN; i++) {
        float f = w_gate_ref[i] * inv_wgs;
        int q = (int)std::lrintf(f);
        if (q > 32767) q = 32767; if (q < -32768) q = -32768;
        W_gate_int16[i] = (int16_t)q;
    }

    // x_mem: N * HIDDEN_VECS int16_vec_t
    aligned_vector<int16_vec_t> x_mem(N * HIDDEN_VECS);
    for (int t = 0; t < N; t++)
        for (int v = 0; v < HIDDEN_VECS; v++) {
            int16_vec_t vec;
            for (int k = 0; k < VEC_WIDTH; k++)
                vec[k] = (ap_int<16>)x_int16[t*HIDDEN + v*VEC_WIDTH + k];
            x_mem[t*HIDDEN_VECS + v] = vec;
        }

    // w_gate_mem: N_EXPERTS_TOTAL * HIDDEN_VECS int16_vec_t
    aligned_vector<int16_vec_t> w_gate_mem(N_EXPERTS_TOTAL * HIDDEN_VECS);
    for (int e = 0; e < N_EXPERTS_TOTAL; e++)
        for (int v = 0; v < HIDDEN_VECS; v++) {
            int16_vec_t vec;
            for (int k = 0; k < VEC_WIDTH; k++)
                vec[k] = (ap_int<16>)W_gate_int16[e*HIDDEN + v*VEC_WIDTH + k];
            w_gate_mem[e*HIDDEN_VECS + v] = vec;
        }

    // Combined bias + dequant buffer on a single port:
    //   [ bias[0..N_EXPERTS_TOTAL-1] | dequant[0..DEQUANT_PARAMS_PER_SLOT-1] ]
    aligned_vector<float> bias_dequant_mem(N_EXPERTS_TOTAL + DEQUANT_PARAMS_PER_SLOT);
    for (int e=0; e<N_EXPERTS_TOTAL; e++) bias_dequant_mem[e]=bias_ref[e];

    // Opt 3: split each weight matrix into even-row (a) and odd-row (b) sub-matrices.
    // Layout: expert e, half-row rh → row 2*rh (even) and 2*rh+1 (odd).
    aligned_vector<int16_vec_t> W1a_mem_src((size_t)N_TEST_EXPERTS * W1_HALF_VECS);
    aligned_vector<int16_vec_t> W1b_mem_src((size_t)N_TEST_EXPERTS * W1_HALF_VECS);
    aligned_vector<int16_vec_t> W3a_mem_src((size_t)N_TEST_EXPERTS * W3_HALF_VECS);
    aligned_vector<int16_vec_t> W3b_mem_src((size_t)N_TEST_EXPERTS * W3_HALF_VECS);
    aligned_vector<int16_vec_t> W2a_mem_src((size_t)N_TEST_EXPERTS * W2_HALF_VECS);
    aligned_vector<int16_vec_t> W2b_mem_src((size_t)N_TEST_EXPERTS * W2_HALF_VECS);
    for (int e = 0; e < N_TEST_EXPERTS; e++) {
        for (int rh = 0; rh < MOE_INTER/2; rh++)
            for (int v = 0; v < HIDDEN_VECS; v++) {
                int16_vec_t va, vb;
                for (int k = 0; k < VEC_WIDTH; k++) {
                    va[k] = W1_all[e][(2*rh)   * HIDDEN + v * VEC_WIDTH + k];
                    vb[k] = W1_all[e][(2*rh+1) * HIDDEN + v * VEC_WIDTH + k];
                }
                W1a_mem_src[(size_t)e * W1_HALF_VECS + rh * HIDDEN_VECS + v] = va;
                W1b_mem_src[(size_t)e * W1_HALF_VECS + rh * HIDDEN_VECS + v] = vb;
            }
        for (int rh = 0; rh < MOE_INTER/2; rh++)
            for (int v = 0; v < HIDDEN_VECS; v++) {
                int16_vec_t va, vb;
                for (int k = 0; k < VEC_WIDTH; k++) {
                    va[k] = W3_all[e][(2*rh)   * HIDDEN + v * VEC_WIDTH + k];
                    vb[k] = W3_all[e][(2*rh+1) * HIDDEN + v * VEC_WIDTH + k];
                }
                W3a_mem_src[(size_t)e * W3_HALF_VECS + rh * HIDDEN_VECS + v] = va;
                W3b_mem_src[(size_t)e * W3_HALF_VECS + rh * HIDDEN_VECS + v] = vb;
            }
        for (int rh = 0; rh < HIDDEN/2; rh++)
            for (int v = 0; v < INTER_VECS; v++) {
                int16_vec_t va, vb;
                for (int k = 0; k < VEC_WIDTH; k++) {
                    va[k] = W2_all[e][(2*rh)   * MOE_INTER + v * VEC_WIDTH + k];
                    vb[k] = W2_all[e][(2*rh+1) * MOE_INTER + v * VEC_WIDTH + k];
                }
                W2a_mem_src[(size_t)e * W2_HALF_VECS + rh * INTER_VECS + v] = va;
                W2b_mem_src[(size_t)e * W2_HALF_VECS + rh * INTER_VECS + v] = vb;
            }
    }
    std::vector<aligned_vector<int16_vec_t>> W1a_mem_slots(N_PARALLEL_SLOTS);
    std::vector<aligned_vector<int16_vec_t>> W1b_mem_slots(N_PARALLEL_SLOTS);
    std::vector<aligned_vector<int16_vec_t>> W3a_mem_slots(N_PARALLEL_SLOTS);
    std::vector<aligned_vector<int16_vec_t>> W3b_mem_slots(N_PARALLEL_SLOTS);
    std::vector<aligned_vector<int16_vec_t>> W2a_mem_slots(N_PARALLEL_SLOTS);
    std::vector<aligned_vector<int16_vec_t>> W2b_mem_slots(N_PARALLEL_SLOTS);
    for (int s = 0; s < N_PARALLEL_SLOTS; s++) {
        W1a_mem_slots[s] = W1a_mem_src; W1b_mem_slots[s] = W1b_mem_src;
        W3a_mem_slots[s] = W3a_mem_src; W3b_mem_slots[s] = W3b_mem_src;
        W2a_mem_slots[s] = W2a_mem_src; W2b_mem_slots[s] = W2b_mem_src;
    }

    bias_dequant_mem[N_EXPERTS_TOTAL+0]=w_scale; bias_dequant_mem[N_EXPERTS_TOTAL+1]=w_shift;
    bias_dequant_mem[N_EXPERTS_TOTAL+2]=w_scale; bias_dequant_mem[N_EXPERTS_TOTAL+3]=w_shift;
    bias_dequant_mem[N_EXPERTS_TOTAL+4]=w_scale; bias_dequant_mem[N_EXPERTS_TOTAL+5]=w_shift;

    aligned_vector<fp32_vec_t> y_mem(N * HIDDEN_VECS_FP);
    for (auto& v:y_mem) for (int k=0;k<VEC_WIDTH_FP;k++) v[k]=-999.0f;

    std::cout << "\n[1] Running kernel..." << std::endl;
    std::cout << "  x_scale=" << x_scale
              << "  w_gate_scale=" << w_gate_scale
              << "  w_gate_shift=" << w_gate_shift << std::endl;
    int64_t elapsed = tapa::invoke(
        moe_fpga_top,
        FLAGS_bitstream,
        N, local_expert_base,
        x_scale, w_gate_scale, w_gate_shift,
        tapa::read_only_mmap<int16_vec_t>(x_mem),
        tapa::read_only_mmap<int16_vec_t>(w_gate_mem),
        tapa::read_only_mmap<float>(bias_dequant_mem),
        tapa::read_only_mmaps<int16_vec_t, N_PARALLEL_SLOTS>(W1a_mem_slots),
        tapa::read_only_mmaps<int16_vec_t, N_PARALLEL_SLOTS>(W1b_mem_slots),
        tapa::read_only_mmaps<int16_vec_t, N_PARALLEL_SLOTS>(W3a_mem_slots),
        tapa::read_only_mmaps<int16_vec_t, N_PARALLEL_SLOTS>(W3b_mem_slots),
        tapa::read_only_mmaps<int16_vec_t, N_PARALLEL_SLOTS>(W2a_mem_slots),
        tapa::read_only_mmaps<int16_vec_t, N_PARALLEL_SLOTS>(W2b_mem_slots),
        tapa::write_only_mmap<fp32_vec_t>(y_mem)
    );
    std::cout << "  elapsed: " << elapsed << " ns" << std::endl;

    std::cout << "\n[2] Running reference..." << std::endl;
    std::vector<float> y_ref;
    reference_moe(N, local_expert_base, x_int16, W_gate_int16, bias_ref,
                  W1_all, W3_all, W2_all,
                  x_scale, w_gate_scale, w_gate_shift,
                  w_scale, w_shift, w_scale, w_shift, w_scale, w_shift, y_ref);

    std::cout << "\n[3] Comparing..." << std::endl;
    double max_abs_err=0.0; int n_mismatch=0;
    const float tol=0.5f;
    for (int t=0;t<N;t++) for (int v=0;v<HIDDEN_VECS_FP;v++){
        fp32_vec_t ov=y_mem[t*HIDDEN_VECS_FP+v];
        for (int k=0;k<VEC_WIDTH_FP;k++){
            float got=ov[k], exp_val=y_ref[t*HIDDEN+v*VEC_WIDTH_FP+k];
            float err=std::fabs(got-exp_val);
            if (err>max_abs_err) max_abs_err=err;
            if (err>tol){if(n_mismatch<10) std::cout<<"  MISMATCH t="<<t<<" d="<<v*VEC_WIDTH_FP+k<<" got="<<got<<" exp="<<exp_val<<" err="<<err<<"\n"; n_mismatch++;}
        }
    }

    std::cout<<"\n  max abs err: "<<max_abs_err<<"\n  mismatches: "<<n_mismatch<<"/"<<(N*HIDDEN)<<std::endl;
    std::cout<<"  y_ref[t=0,0..7]: "; for(int k=0;k<8;k++) std::cout<<y_ref[k]<<" "; std::cout<<"\n";
    std::cout<<"  y_hw [t=0,0..7]: "; for(int k=0;k<8;k++) std::cout<<y_mem[0][k]<<" "; std::cout<<"\n";

    if (n_mismatch==0){ std::cout<<"\nPASS\n"; return EXIT_SUCCESS; }
    else              { std::cout<<"\nFAIL\n"; return EXIT_FAILURE; }
}
