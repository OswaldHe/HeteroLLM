/**
 * FPGA Testbench for Engram N-gram Hash + Embedding + KV Projection
 *
 * Architecture: 8 KV weight channels, each covering [5120 x 128] columns.
 * Embedding memory split into lo (heads 0..7) and hi (heads 8..15).
 *
 * Diagnostic flags for isolating RTL vs C-sim mismatches:
 *
 *   --const_kv_weights   Set all KV weights to 1 (SW + HW).
 *                        Expected output[t][r] = sum of all 128 embedding values
 *                        for token t (same for every row r).
 *                        Pass → weight loading / URAM packing is correct.
 *                        Fail → embedding lookup or n-gram hash is wrong.
 *
 *   --const_embed        Set all embedding values to 1 (SW + HW).
 *                        Expected output[t][r] = sum of all 128 weight values for row r.
 *                        (All tokens give the same output when embeddings = 1.)
 *                        Pass → weight loading and accumulator are correct.
 *                        Fail → URAM packing / unpacking is wrong.
 *
 *   --const_kv_weights --const_embed
 *                        Both = 1.  Expected output[t][r] = 1024 for ALL t, r.
 *                        (128 cols/channel × 8 channels × 1×1 = 1024)
 *                        The simplest possible MAC test.
 *                        Pass → bare accumulator pipeline is correct; bug is in data.
 *                        Fail → fundamental pipeline / accumulator issue.
 *
 *   --tiny_mult          Use mult_0=1, mult_1=2, mult_2=3; token IDs restricted to
 *                        [0, 3].  All intermediate n-gram products fit in ~4 bits,
 *                        well inside ap_uint<40>.
 *                        Pass → n-gram hash bitwidth is NOT the problem.
 *                        Fail → n-gram hash has a bitwidth / truncation bug.
 *
 *   --L <n>              Override sequence length (default 8 for fast hardware runs).
 *                        Must be even.
 *
 * Build: make fpga_test (requires XILINX_HLS=/mnt/software/xilinx/Vitis_HLS/2023.2)
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <random>
#include <algorithm>
#include <cassert>

#include <gflags/gflags.h>
#include <tapa.h>

#include "engram.h"

template <typename T>
using aligned_vector = std::vector<T, tapa::aligned_allocator<T>>;

DEFINE_string(bitstream, "", "path to bitstream file, run csim if empty");
DEFINE_int32 (L, 8,    "sequence length; must be even (default 8 for fast hardware probing)");
DEFINE_bool  (const_kv_weights, false,
              "set all KV projection weights to 1 to isolate the embedding / hash path");
DEFINE_bool  (const_embed, false,
              "set all embedding values to 1 to isolate the weight-load / accumulator path");
DEFINE_bool  (tiny_mult, false,
              "use mult=1,2,3 and token IDs in [0,3] to probe n-gram hash bitwidth");

// ============================================================================
// Software reference: n-gram hash
// ============================================================================
void sw_ngram_hash(
    const int* token_ids, int L,
    int pad_id,
    int64_t mult_0, int64_t mult_1, int64_t mult_2,
    std::vector<std::vector<int>>& hash_ids // [L][TOTAL_HEADS]
) {
    hash_ids.resize(L, std::vector<int>(TOTAL_HEADS));

    int64_t shift_0 = pad_id;
    int64_t shift_1 = pad_id;
    int64_t shift_2 = pad_id;

    for (int t = 0; t < L; t++) {
        shift_2 = shift_1;
        shift_1 = shift_0;
        shift_0 = token_ids[t];

        int64_t mix_2gram = (shift_0 * mult_0) ^ (shift_1 * mult_1);
        int64_t mix_3gram = mix_2gram ^ (shift_2 * mult_2);

        for (int j = 0; j < NUM_HEAD; j++) {
            int64_t mod = primes_vocab_size[0][j];
            int64_t raw = mix_2gram % mod;
            if (raw < 0) raw += mod;
            hash_ids[t][j] = (int)raw;
        }
        for (int j = 0; j < NUM_HEAD; j++) {
            int64_t mod = primes_vocab_size[1][j];
            int64_t raw = mix_3gram % mod;
            if (raw < 0) raw += mod;
            hash_ids[t][NUM_HEAD + j] = (int)raw;
        }
    }
}

// ============================================================================
// Software reference: embedding lookup + flatten to [L x ENGRAM_HIDDEN]
// ============================================================================
void sw_embed_lookup(
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
            for (int d = 0; d < EMBED_DIM; d++) {
                out_ptr[offset + d] = emb[d];
            }
            offset += EMBED_DIM;
        }
    }
}

// ============================================================================
// Software reference: KV projection
// Channel h consumes head pair (h, h+8):
//   - first  64 cols of kv_weights_sw[h] multiply embedding head h
//   - second 64 cols of kv_weights_sw[h] multiply embedding head h+8
// kv_weights_sw[h]: [KV_PROJ_ROWS x KV_WEIGHT_COLS_PER_CHANNEL] row-major
// ============================================================================
void sw_kv_projection(
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
            const int lo_head_offset = h * EMBED_DIM;
            const int hi_head_offset = (h + KV_WEIGHT_CHANNELS) * EMBED_DIM;
            for (int r = 0; r < KV_PROJ_ROWS; r++) {
                float acc = 0.0f;
                const int row_base = r * KV_WEIGHT_COLS_PER_CHANNEL;

                // First half of channel weights multiply lo head (h).
                for (int c = 0; c < EMBED_DIM; c++) {
                    acc += w[row_base + c] * emb[lo_head_offset + c];
                }
                // Second half multiply hi head (h+8).
                for (int c = 0; c < EMBED_DIM; c++) {
                    acc += w[row_base + EMBED_DIM + c] * emb[hi_head_offset + c];
                }
                out[r] += acc;
            }
        }
    }
}

// ============================================================================
// Main testbench
// ============================================================================
int main(int argc, char* argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    const int L = FLAGS_L;

    // Enforce L is even (kernel processes tokens in pairs)
    if (L % 2 != 0) {
        std::cerr << "Error: L must be even (kernel pairs tokens). Got L=" << L << std::endl;
        return EXIT_FAILURE;
    }

    const int pad_id = 2;

    // Multipliers: normal or tiny (to probe n-gram hash bitwidth)
    int64_t mult_0, mult_1, mult_2;
    int token_id_max;
    if (FLAGS_tiny_mult) {
        mult_0 = 1LL;
        mult_1 = 2LL;
        mult_2 = 3LL;
        token_id_max = 3;   // products <= 9, fits in 4 bits, well inside ap_uint<40>
    } else {
        mult_0 = 123457LL;
        mult_1 = 654323LL;
        mult_2 = 314159LL;
        token_id_max = 999;
    }

    std::cout << "=== Engram FPGA Testbench (8 KV weight channels) ===" << std::endl;
    if (FLAGS_const_kv_weights) std::cout << "[PROBE] const_kv_weights=1 : all KV weights set to 1" << std::endl;
    if (FLAGS_const_embed)      std::cout << "[PROBE] const_embed=1      : all embeddings set to 1" << std::endl;
    if (FLAGS_tiny_mult)        std::cout << "[PROBE] tiny_mult          : mult=(1,2,3), token_ids in [0," << token_id_max << "]" << std::endl;
    if (FLAGS_const_kv_weights && FLAGS_const_embed)
        std::cout << "[PROBE] Both const: expected ALL outputs = " << ENGRAM_HIDDEN << std::endl;
    std::cout << "  L=" << L
              << "  TOTAL_HEADS=" << TOTAL_HEADS
              << "  EMBED_DIM=" << EMBED_DIM
              << "  KV_PROJ_ROWS=" << KV_PROJ_ROWS << std::endl;
    std::cout << "  mult=(" << mult_0 << "," << mult_1 << "," << mult_2 << ")"
              << "  token_id_max=" << token_id_max << std::endl;

    std::mt19937 gen(12345);
    std::uniform_int_distribution<int>   token_dist(0, token_id_max);
    std::uniform_int_distribution<short> val_dist_int(-100, 100);

    // ── Token IDs ──
    aligned_vector<int> token_ids(L);
    for (int i = 0; i < L; i++) token_ids[i] = token_dist(gen);

    std::cout << "  token_ids[0.." << std::min(L,8)-1 << "]:";
    for (int i = 0; i < std::min(L, 8); i++) std::cout << " " << token_ids[i];
    std::cout << std::endl;

    // ── Embedding tables: all 16 heads ──
    std::vector<std::vector<float>> embed_tables_sw(TOTAL_HEADS);
    std::vector<aligned_vector<vec16s_t>> embed_lo_hw(KV_WEIGHT_CHANNELS);
    std::vector<aligned_vector<vec16s_t>> embed_hi_hw(KV_WEIGHT_CHANNELS);

    for (int h = 0; h < TOTAL_HEADS; h++) {
        int ngram_idx = h / NUM_HEAD;
        int head_j    = h % NUM_HEAD;
        int64_t vs    = primes_vocab_size[ngram_idx][head_j];

        embed_tables_sw[h].resize(vs * EMBED_DIM);
        auto& hw_bank = (h < KV_WEIGHT_CHANNELS) ? embed_lo_hw[h] : embed_hi_hw[h - KV_WEIGHT_CHANNELS];
        hw_bank.resize(vs * EMBED_DIM_DIV_16);

        for (int64_t i = 0; i < vs; i++) {
            for (int d = 0; d < EMBED_DIM; d++) {
                short ival = FLAGS_const_embed ? (short)2 : val_dist_int(gen);
                embed_tables_sw[h][i * EMBED_DIM + d] = (float)ival;
                int vi = i * EMBED_DIM_DIV_16 + d / 16;
                int ki = d % 16;
                hw_bank[vi][ki] = (ap_int<16>)ival;
            }
        }
    }

    // ── KV weight channels ──
    std::vector<std::vector<float>> kv_weights_sw(KV_WEIGHT_CHANNELS);
    std::vector<aligned_vector<vec16s_t>> kv_weights_hw(KV_WEIGHT_CHANNELS);

    for (int h = 0; h < KV_WEIGHT_CHANNELS; h++) {
        kv_weights_sw[h].resize(KV_PROJ_ROWS * KV_WEIGHT_COLS_PER_CHANNEL);
        kv_weights_hw[h].resize(KV_WEIGHT_VECS_PER_CHANNEL);

        for (int r = 0; r < KV_PROJ_ROWS; r++) {
            for (int c = 0; c < KV_WEIGHT_COLS_PER_CHANNEL; c++) {
                short ival = FLAGS_const_kv_weights ? (short)2: val_dist_int(gen);
                kv_weights_sw[h][r * KV_WEIGHT_COLS_PER_CHANNEL + c] = (float)ival;
                int vi = r * KV_WEIGHT_COLS_PER_CHANNEL_DIV_16 + c / 16;
                int ki = c % 16;
                kv_weights_hw[h][vi][ki] = (ap_int<16>)ival;
            }
        }
    }

    // ── SW reference ──
    std::cout << "\n[1] Running SW reference..." << std::endl;

    std::vector<std::vector<int>> sw_hash_ids;
    sw_ngram_hash(token_ids.data(), L, pad_id, mult_0, mult_1, mult_2, sw_hash_ids);

    // Print n-gram hash IDs so the n-gram path can be verified independently
    for (int t = 0; t < std::min(L, 2); t++) {
        std::cout << "  SW hash_ids t=" << t << ": bigram[0..3]=";
        for (int j = 0; j < 4; j++) std::cout << " " << sw_hash_ids[t][j];
        std::cout << "  trigram[0..3]=";
        for (int j = 0; j < 4; j++) std::cout << " " << sw_hash_ids[t][NUM_HEAD + j];
        std::cout << std::endl;
    }

    std::vector<float> sw_flat;
    sw_embed_lookup(sw_hash_ids, embed_tables_sw, L, sw_flat);

    // Sanity check: with const_embed=1, sum across all 1024 dims must equal 1024
    if (FLAGS_const_embed) {
        float sum = 0;
        for (int d = 0; d < ENGRAM_HIDDEN; d++) sum += sw_flat[d];
        std::cout << "  [CHECK] flat_embed t=0 sum=" << sum
                  << " (expected " << ENGRAM_HIDDEN << ")" << std::endl;
    }

    std::vector<float> sw_kv;
    sw_kv_projection(sw_flat, kv_weights_sw, L, sw_kv);

    std::cout << "  SW KV out [t=0, r=0..7]:";
    for (int r = 0; r < std::min(KV_PROJ_ROWS, 8); r++)
        std::cout << " " << sw_kv[r];
    std::cout << std::endl;

    // With const_kv_weights=1, every row should give the same value per token
    if (FLAGS_const_kv_weights) {
        std::cout << "  [CHECK] const_kv_weights: sw_kv[t=0,r=0]=" << sw_kv[0]
                  << " sw_kv[t=0,r=1]=" << sw_kv[1]
                  << " (should be equal, = sum of 128 embed values)" << std::endl;
    }
    // With const_embed=1, every token should give the same value per row
    if (FLAGS_const_embed && L > 1) {
        std::cout << "  [CHECK] const_embed: sw_kv[t=0,r=0]=" << sw_kv[0]
                  << " sw_kv[t=1,r=0]=" << sw_kv[KV_PROJ_ROWS]
                  << " (should be equal, = sum of 128 weight values for row 0)" << std::endl;
    }

    // ── HW output buffer ──
    int output_vecs_per_ch = (L / 2) * KV_PROJ_ROWS_DIV_16;
    std::vector<aligned_vector<tapa::vec_t<float, 16>>> hw_output(2,
        aligned_vector<tapa::vec_t<float, 16>>(output_vecs_per_ch));
    for (int ch = 0; ch < 2; ch++)
        for (auto& v : hw_output[ch])
            for (int k = 0; k < 16; k++) v[k] = -999.0f;

    // ── Invoke FPGA kernel ──
    std::cout << "\n[2] Invoking engram_fpga_top..." << std::endl;

    int64_t elapsed = tapa::invoke(
        engram_fpga_top,
        FLAGS_bitstream,
        L,
        pad_id,
        (ap_int<64>)mult_0,
        (ap_int<64>)mult_1,
        (ap_int<64>)mult_2,
        tapa::read_only_mmap<int>(token_ids),
        tapa::read_only_mmaps<vec16s_t, KV_WEIGHT_CHANNELS>(embed_lo_hw),
        tapa::read_only_mmaps<vec16s_t, KV_WEIGHT_CHANNELS>(embed_hi_hw),
        tapa::read_only_mmaps<vec16s_t, KV_WEIGHT_CHANNELS>(kv_weights_hw),
        tapa::write_only_mmaps<tapa::vec_t<float, 16>, 2>(hw_output)
    );

    std::cout << "  FPGA kernel completed in " << elapsed << " ns" << std::endl;

    // ── Verify ──
    std::cout << "\n[3] Verifying KV projection output..." << std::endl;

    // Print HW output for t=0 and t=1 (covers tok=0 and tok=1 of the first pair)
    for (int t = 0; t < std::min(L, 2); t++) {
        std::cout << "  HW KV out [t=" << t << ", r=0..7]:";
        int batch = t / 2;
        int tok   = t % 2;
        auto& hw_ch = hw_output[tok];
        for (int r = 0; r < std::min(KV_PROJ_ROWS, 8); r++) {
            int rb = r / 16;
            int ki = r % 16;
            int vi = batch * KV_PROJ_ROWS_DIV_16 + rb;
            std::cout << " " << hw_ch[vi][ki];
        }
        std::cout << std::endl;
    }

    int mismatches = 0;
    float max_err = 0.0f;
    // Use a tight tolerance for const modes (integer expected values), loose otherwise
    const float tolerance = (FLAGS_const_kv_weights || FLAGS_const_embed) ? 0.5f : 1e-1f;

    // Per-token and per-row mismatch counts to reveal structural patterns
    std::vector<int> mismatch_by_tok(L, 0);
    std::vector<int> mismatch_by_row(KV_PROJ_ROWS, 0);

    for (int t = 0; t < L; t++) {
        for (int r = 0; r < KV_PROJ_ROWS; r++) {
            int rb    = r / 16;
            int ki    = r % 16;
            int batch = t / 2;
            int tok   = t % 2;
            int vi    = batch * KV_PROJ_ROWS_DIV_16 + rb;
            auto& hw_ch = hw_output[tok];
            float hw_val = hw_ch[vi][ki];
            float sw_val = sw_kv[t * KV_PROJ_ROWS + r];
            float err = std::fabs(hw_val - sw_val);
            if (err > max_err) max_err = err;
            if (err > tolerance) {
                if (mismatches < 20)
                    std::cout << "  MISMATCH t=" << t << " r=" << r
                              << " hw=" << hw_val << " sw=" << sw_val
                              << " err=" << err
                              << " [batch=" << batch << " tok=" << tok
                              << " rb=" << rb << " ki=" << ki << "]"
                              << std::endl;
                mismatches++;
                mismatch_by_tok[t]++;
                mismatch_by_row[r]++;
            }
        }
    }

    std::cout << "\n  Max absolute error : " << max_err << std::endl;
    std::cout << "  Total mismatches   : " << mismatches
              << " / " << (L * KV_PROJ_ROWS) << std::endl;

    // Structural analysis of mismatches: reveals which axis the bug lives on
    if (mismatches > 0 && mismatches < L * KV_PROJ_ROWS) {
        // By token (up to first 16)
        std::cout << "\n  --- Mismatch count by token (first " << std::min(L,16) << ") ---" << std::endl;
        for (int t = 0; t < std::min(L, 16); t++) {
            if (mismatch_by_tok[t] > 0)
                std::cout << "    t=" << t << " (tok=" << t%2 << " in pair " << t/2 << ")"
                          << " : " << mismatch_by_tok[t] << std::endl;
        }

        // By output row (first 64): if certain rb or ki values cluster, it points to
        // URAM word layout or accumulator slot bugs
        std::cout << "\n  --- Mismatch count by row (first 64 rows) ---" << std::endl;
        for (int r = 0; r < std::min(KV_PROJ_ROWS, 64); r++) {
            if (mismatch_by_row[r] > 0)
                std::cout << "    r=" << r << " [rb=" << r/16 << " ki=" << r%16 << "]"
                          << " : " << mismatch_by_row[r] << std::endl;
        }

        // Mismatch count by ki (r%16): a non-uniform distribution would point to a
        // specific int16 slot within the 64-bit URAM word (packing/extraction bug)
        int mismatch_by_ki[16] = {};
        for (int r = 0; r < KV_PROJ_ROWS; r++) mismatch_by_ki[r % 16] += mismatch_by_row[r];
        std::cout << "\n  --- Mismatch count by r%%16 (ki index within float16 vector) ---" << std::endl;
        for (int ki = 0; ki < 16; ki++)
            std::cout << "    ki=" << ki << " : " << mismatch_by_ki[ki] << std::endl;

        // Mismatch count by tok within pair: tok=0 vs tok=1
        // Asymmetry here would point to the 2-token parallel processing bug
        int tok0_err = 0, tok1_err = 0;
        for (int t = 0; t < L; t++) {
            (t % 2 == 0 ? tok0_err : tok1_err) += mismatch_by_tok[t];
        }
        std::cout << "\n  --- Mismatch by slot within 2-token pair ---" << std::endl;
        std::cout << "    tok=0 (even tokens): " << tok0_err << std::endl;
        std::cout << "    tok=1 (odd  tokens): " << tok1_err << std::endl;
    }

    if (mismatches == 0) {
        std::cout << "\nFPGA testbench PASSED!" << std::endl;
    } else {
        std::cout << "\nFPGA testbench FAILED!" << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
