#ifndef __ENGRAM_H__
#define __ENGRAM_H__

#include <tapa.h>
#include <ap_int.h>
#include <hls_math.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <cstdint>
#include <limits>

// ============================================================================
// Constants
// ============================================================================
constexpr int BATCH_SIZE = 1;
constexpr int VOCAB_SIZE = 129280;
constexpr int ENGRAM_VOCAB_SIZE = VOCAB_SIZE * 5;
constexpr int primes_vocab_size[2][8] = {
    {646403, 646411, 646421, 646423, 646433, 646453, 646519, 646523},
    {646537, 646543, 646549, 646571, 646573, 646577, 646609, 646619}
};
constexpr int MAX_SEQ_LEN = 1024;
constexpr int NUM_GROUP = 4;           // hc_mult
constexpr int NUM_HEAD = 8;            // n_head_per_ngram
constexpr int MAX_NGRAM_SIZE = 3;      // 2-gram and 3-gram
constexpr int NUM_NGRAM = MAX_NGRAM_SIZE - 1; // 2 (bigram + trigram)
constexpr int TOTAL_HEADS = NUM_NGRAM * NUM_HEAD; // 16
constexpr int EMBED_DIM = 512 / NUM_HEAD;  // 64 per head
constexpr int EMBED_DIM_DIV_16 = EMBED_DIM / 16; // 4
constexpr int ENGRAM_HIDDEN = NUM_NGRAM * 512;    // 1024 (flat embedding)
constexpr int HIDDEN_DIM = 1024;       // backbone hidden_size

// KV projection constants
// Key: NUM_GROUP separate projections [ENGRAM_HIDDEN -> HIDDEN_DIM] each
// Value: 1 shared projection [ENGRAM_HIDDEN -> HIDDEN_DIM]
// Treat as a single large matrix: (NUM_GROUP + 1) * HIDDEN_DIM rows x ENGRAM_HIDDEN cols
constexpr int KV_PROJ_ROWS = (NUM_GROUP + 1) * HIDDEN_DIM; // 5120
constexpr int KV_PROJ_COLS = ENGRAM_HIDDEN;                  // 1024
constexpr int KV_PROJ_COLS_DIV_16 = KV_PROJ_COLS / 16;       // 64
constexpr int KV_PROJ_ROWS_DIV_16 = KV_PROJ_ROWS / 16;       // 320
constexpr int HIDDEN_DIM_DIV_16 = HIDDEN_DIM / 16;            // 64

// KV weight reading channels: 8 channels, each covers 2 embedding heads
// Weight slice per channel: [KV_PROJ_ROWS x (2*EMBED_DIM)] = [5120 x 128]
constexpr int KV_WEIGHT_CHANNELS = TOTAL_HEADS / 2;                              // 8
constexpr int KV_WEIGHT_COLS_PER_CHANNEL = EMBED_DIM * 2;                        // 128
constexpr int KV_WEIGHT_COLS_PER_CHANNEL_DIV_16 = KV_WEIGHT_COLS_PER_CHANNEL / 16; // 8
constexpr int KV_WEIGHT_VECS_PER_CHANNEL = KV_PROJ_ROWS * KV_WEIGHT_COLS_PER_CHANNEL_DIV_16; // 40960
constexpr int KV_WEIGHT_URAM_WORDS = KV_WEIGHT_VECS_PER_CHANNEL * 4;  // 163840

using vec16i_t = tapa::vec_t<int, 16>;
using vec16s_t = tapa::vec_t<ap_int<16>, 16>;
// Accumulator: 16×16 multiply = 32 bits, sum of 128 products = +7 bits (39),
// sum across 8 channels = +3 bits (42). Use ap_int<42> for full precision.
using vec16acc_t = tapa::vec_t<ap_int<42>, 16>;
// Packed accumulator: 2 tokens × 16 rows = 32 elements per chain transfer
using vec32acc_t = tapa::vec_t<ap_int<42>, 32>;

// ============================================================================
// Read compressed token IDs from external memory
// ============================================================================
void read_token_ids(
    const int L,
    tapa::async_mmap<int>& token_id_mem,
    tapa::ostream<int>& token_id_fifo
) {
    for (int i_req = 0, i_resp = 0; i_resp < L;) {
        #pragma HLS pipeline II=1
        if ((i_req < L) & !token_id_mem.read_addr.full()) {
            token_id_mem.read_addr.try_write(i_req);
            ++i_req;
        }
        if (!token_id_mem.read_data.empty()) {
            int tmp;
            token_id_mem.read_data.try_read(tmp);
            token_id_fifo.write(tmp);
            ++i_resp;
        }
    }
}

// ============================================================================
// N-gram hash computation for 2-gram and 3-gram
//
// For each position t in [0, L):
//   shift_0 = token[t],  shift_1 = token[t-1] (or pad_id), shift_2 = token[t-2] (or pad_id)
//
//   For n=2 (bigram):
//     mix = (shift_0 * mult_0) XOR (shift_1 * mult_1)
//     For each head j in [0,8): hash = mix % vocab_sizes_2gram[j]
//
//   For n=3 (trigram):
//     mix = (shift_0 * mult_0) XOR (shift_1 * mult_1) XOR (shift_2 * mult_2)
//     For each head j in [0,8): hash = mix % vocab_sizes_3gram[j]
//
// Output: 16 hash IDs per token position (8 bigram + 8 trigram), packed into vec16i_t
// ============================================================================
void ngram_hash(
    const int L,
    const int pad_id,
    const ap_int<64> mult_0,
    const ap_int<64> mult_1,
    const ap_int<64> mult_2,
    tapa::istream<int>& token_id_fifo,
    tapa::ostream<vec16i_t>& ngram_id_fifo
) {
    ap_int<64> shift_0 = pad_id;
    ap_int<64> shift_1 = pad_id;
    ap_int<64> shift_2 = pad_id;

    for (int t = 0; t < L; t++) {
        #pragma HLS pipeline II=1
        const int token_val = token_id_fifo.read();
        shift_2 = shift_1;
        shift_1 = shift_0;
        shift_0 = token_val;

        // Use ap_uint<40>: token_id (≤17 bits) × multiplier (≤20 bits) ≤ 37 bits, fits in 40 bits
        // Unsigned since token values are non-negative; no correction needed
        ap_uint<40> mix_2gram = (ap_uint<40>)((shift_0 * mult_0) ^ (shift_1 * mult_1));
        ap_uint<40> mix_3gram = (ap_uint<40>)(mix_2gram ^ (shift_2 * mult_2));

        vec16i_t ngram_id_out;

        for (int j = 0; j < NUM_HEAD; j++) {
            #pragma HLS unroll
            ngram_id_out[j] = (int)(mix_2gram % (ap_uint<40>)primes_vocab_size[0][j]);
        }
        for (int j = 0; j < NUM_HEAD; j++) {
            #pragma HLS unroll
            ngram_id_out[j + NUM_HEAD] = (int)(mix_3gram % (ap_uint<40>)primes_vocab_size[1][j]);
        }
        ngram_id_fifo.write(ngram_id_out);
    }
}

// ============================================================================
// Dispatch 16 hash IDs per token to per-head FIFOs.
// Split into lo (heads 0..7) and hi (heads 8..15), each of size KV_WEIGHT_CHANNELS.
// This lets each KV projection channel h read from lo[h] and hi[h].
// ============================================================================
void dispatch_hash_ids(
    const int L,
    tapa::istream<vec16i_t>& ngram_id_fifo,
    tapa::ostreams<int, KV_WEIGHT_CHANNELS>& lo_head_id_fifos,
    tapa::ostreams<int, KV_WEIGHT_CHANNELS>& hi_head_id_fifos
) {
    for (int t = 0; t < L; t++) {
        #pragma HLS pipeline II=1
        vec16i_t ngram_ids = ngram_id_fifo.read();
        for (int h = 0; h < KV_WEIGHT_CHANNELS; h++) {
            #pragma HLS unroll
            lo_head_id_fifos[h].write(ngram_ids[h]);               // head h
            hi_head_id_fifos[h].write(ngram_ids[h + KV_WEIGHT_CHANNELS]); // head h+8
        }
    }
}

// ============================================================================
// Read embedding for a single head from HBM
// Each embedding is EMBED_DIM=64 floats = 4 x vec_t<float,16>
// ============================================================================
void read_head_embed(
    const int L,
    tapa::istream<int>& head_id_fifo,
    tapa::async_mmap<vec16s_t>& embed_mem,
    tapa::ostream<vec16s_t>& embed_fifo
) {
    for (int t = 0; t < L; t++) {
        int ngram_id = head_id_fifo.read();
        int base_addr = ngram_id * EMBED_DIM_DIV_16;

        for (int i_req = 0, i_resp = 0; i_resp < EMBED_DIM_DIV_16;) {
            #pragma HLS pipeline II=1
            if ((i_req < EMBED_DIM_DIV_16) & !embed_mem.read_addr.full()) {
                embed_mem.read_addr.try_write(base_addr + i_req);
                ++i_req;
            }
            if (!embed_mem.read_data.empty()) {
                vec16s_t tmp;
                embed_mem.read_data.try_read(tmp);
                embed_fifo.write(tmp);
                ++i_resp;
            }
        }
    }
}

// ============================================================================
// Read KV projection weights for one channel from HBM (8 channels total)
//
// Streams the entire weight slice ONCE (not L times). The downstream
// kv_projection task caches the data into URAM and reuses it across all
// L tokens.
//
// Per-channel layout:
//   Each row: KV_WEIGHT_COLS_PER_CHANNEL_DIV_16 (8) vec_t<ap_int<16>,16>
//   Total: KV_WEIGHT_VECS_PER_CHANNEL = 5120 * 8 = 40960 vectors
// ============================================================================
void read_kv_proj_weights(
    tapa::async_mmap<vec16s_t>& kv_weight_mem,
    tapa::ostream<vec16s_t>& kv_proj_weight_fifo
) {
    for (int i_req = 0, i_resp = 0; i_resp < KV_WEIGHT_VECS_PER_CHANNEL;) {
        #pragma HLS pipeline II=1
        if ((i_req < KV_WEIGHT_VECS_PER_CHANNEL) &
            !kv_weight_mem.read_addr.full()) {
            kv_weight_mem.read_addr.try_write(i_req);
            ++i_req;
        }
        if (!kv_weight_mem.read_data.empty()) {
            vec16s_t tmp;
            kv_weight_mem.read_data.try_read(tmp);
            kv_proj_weight_fifo.write(tmp);
            ++i_resp;
        }
    }
}

// ============================================================================
// Seed the reduction chain with zero partial sums
//
// Writes KV_PROJ_ROWS_DIV_16 (320) zero vectors per token for L tokens.
// This feeds into the first kv_projection channel's prev_partial_fifo.
// ============================================================================
void seed_chain_zero(
    const int L,
    tapa::ostream<vec32acc_t>& chain_out
) {
    vec32acc_t zero;
    for (int k = 0; k < 32; k++) {
        #pragma HLS unroll
        zero[k] = 0;
    }
    // One combined element per rb per 2-token batch (L/2 batches × 320 rb)
    for (int t = 0; t < L; t += 2) {
        for (int r = 0; r < KV_PROJ_ROWS_DIV_16; r++) {
            #pragma HLS pipeline II=1
            chain_out.write(zero);
        }
    }
}

// ============================================================================
// Per-channel KV projection with URAM weight caching and reduction chain
//
// Optimization (opt4): 4 groups of 4 rows each per rb iteration.
// Each group processes 4 consecutive output rows simultaneously (16 URAM reads/cycle).
//
// URAM bank analysis (cyclic=40): 4 consecutive rows r,r+1,r+2,r+3 access banks
// at offsets {0,32,24,16} mod 40, each spanning 4 consecutive banks -> always disjoint.
// 16 simultaneous URAM reads per c-iteration with no bank conflicts.
//
// Performance: 2-token parallel — shares URAM reads across 2 tokens per rb traversal.
// ~5,120 cycles/token (2x throughput vs single-token version).
// ============================================================================
void kv_projection(
    const int L,
    tapa::istream<vec16s_t>& kv_proj_weight_fifo,
    tapa::istream<vec16s_t>& embed_fifo_lo,
    tapa::istream<vec16s_t>& embed_fifo_hi,
    tapa::istream<vec32acc_t>& prev_partial_fifo,
    tapa::ostream<vec32acc_t>& next_partial_fifo
) {
    // URAM weight cache: 163,840 x 64-bit (4 int16 packed per word)
    // cyclic factor=40 -> 40 banks; 2 consecutive rows use 8 distinct banks (no conflict)
    ap_uint<64> weight_cache[KV_WEIGHT_URAM_WORDS];
    #pragma HLS bind_storage variable=weight_cache type=RAM_1P impl=URAM
    #pragma HLS array_partition variable=weight_cache cyclic factor=40

    // -- Phase 1: Load all weights from FIFO into URAM (unchanged) --
    for (int i = 0; i < KV_WEIGHT_VECS_PER_CHANNEL; i++) {
        #pragma HLS pipeline II=1
        vec16s_t w = kv_proj_weight_fifo.read();
        for (int j = 0; j < 4; j++) {
            #pragma HLS unroll
            ap_uint<64> word;
            for (int k = 0; k < 4; k++) {
                #pragma HLS unroll
                word(k * 16 + 15, k * 16) = (ap_uint<16>)w[j * 4 + k];
            }
            weight_cache[i * 4 + j] = word;
        }
    }

    // -- Embedding buffer for 2 tokens --
    ap_int<16> embed_buf[2][KV_WEIGHT_COLS_PER_CHANNEL];
    #pragma HLS array_partition variable=embed_buf complete dim=2

    // result_buf for 2 tokens, 16 rows each
    ap_int<39> result_buf[2][16];
    #pragma HLS array_partition variable=result_buf complete dim=0

    // -- Phase 2: Process pairs of tokens --
    loop_t: for (int t = 0; t < L; t += 2) {
        #pragma HLS loop_tripcount min=4 max=1024

        // Load lo embedding for token t into embed_buf[0][0..63]
        for (int i = 0; i < EMBED_DIM_DIV_16; i++) {
            #pragma HLS pipeline II=1
            vec16s_t v = embed_fifo_lo.read();
            for (int k = 0; k < 16; k++) {
                #pragma HLS unroll
                embed_buf[0][i * 16 + k] = v[k];
            }
        }
        // Load hi embedding for token t into embed_buf[0][64..127]
        for (int i = 0; i < EMBED_DIM_DIV_16; i++) {
            #pragma HLS pipeline II=1
            vec16s_t v = embed_fifo_hi.read();
            for (int k = 0; k < 16; k++) {
                #pragma HLS unroll
                embed_buf[0][EMBED_DIM + i * 16 + k] = v[k];
            }
        }
        // Load lo embedding for token t+1 into embed_buf[1][0..63]
        for (int i = 0; i < EMBED_DIM_DIV_16; i++) {
            #pragma HLS pipeline II=1
            vec16s_t v = embed_fifo_lo.read();
            for (int k = 0; k < 16; k++) {
                #pragma HLS unroll
                embed_buf[1][i * 16 + k] = v[k];
            }
        }
        // Load hi embedding for token t+1 into embed_buf[1][64..127]
        for (int i = 0; i < EMBED_DIM_DIV_16; i++) {
            #pragma HLS pipeline II=1
            vec16s_t v = embed_fifo_hi.read();
            for (int k = 0; k < 16; k++) {
                #pragma HLS unroll
                embed_buf[1][EMBED_DIM + i * 16 + k] = v[k];
            }
        }

        loop_rb: for (int rb = 0; rb < KV_PROJ_ROWS_DIV_16; rb++) {
            #pragma HLS loop_tripcount min=320 max=320

            for (int g = 0; g < 4; g++) {

                // partial[tok][row][col_partial]: 2 tokens, 4 rows, 32 col-partials
                ap_int<35> partial[2][4][32];
                #pragma HLS array_partition variable=partial complete dim=0

                for (int tok = 0; tok < 2; tok++) {
                    #pragma HLS unroll
                    for (int l = 0; l < 4; l++) {
                        #pragma HLS unroll
                        for (int k = 0; k < 32; k++) {
                            #pragma HLS unroll
                            partial[tok][l][k] = 0;
                        }
                    }
                }

                const int row0 = rb * 16 + g * 4;

                // c-loop: 8 iterations, II=1
                // Process same URAM weights for BOTH tokens simultaneously
                for (int c = 0; c < KV_WEIGHT_COLS_PER_CHANNEL_DIV_16 / 2; c++) {
                    #pragma HLS pipeline II=1

                    const int base0 = (row0 * KV_WEIGHT_COLS_PER_CHANNEL_DIV_16 + c * 2) * 4;
                    const int base1 = base0 + KV_WEIGHT_COLS_PER_CHANNEL_DIV_16 * 4;
                    const int base2 = base1 + KV_WEIGHT_COLS_PER_CHANNEL_DIV_16 * 4;
                    const int base3 = base2 + KV_WEIGHT_COLS_PER_CHANNEL_DIV_16 * 4;

                    for (int j = 0; j < 8; j++) {
                        #pragma HLS unroll
                        ap_uint<64> word0 = weight_cache[base0 + j];
                        ap_uint<64> word1 = weight_cache[base1 + j];
                        ap_uint<64> word2 = weight_cache[base2 + j];
                        ap_uint<64> word3 = weight_cache[base3 + j];
                        for (int k = 0; k < 4; k++) {
                            #pragma HLS unroll
                            ap_int<16> w0 = (ap_int<16>)word0(k * 16 + 15, k * 16);
                            ap_int<16> w1 = (ap_int<16>)word1(k * 16 + 15, k * 16);
                            ap_int<16> w2 = (ap_int<16>)word2(k * 16 + 15, k * 16);
                            ap_int<16> w3 = (ap_int<16>)word3(k * 16 + 15, k * 16);
                            const int emb_idx = c * 32 + j * 4 + k;
                            // Token 0 (t)
                            partial[0][0][j * 4 + k] += w0 * embed_buf[0][emb_idx];
                            partial[0][1][j * 4 + k] += w1 * embed_buf[0][emb_idx];
                            partial[0][2][j * 4 + k] += w2 * embed_buf[0][emb_idx];
                            partial[0][3][j * 4 + k] += w3 * embed_buf[0][emb_idx];
                            // Token 1 (t+1)
                            partial[1][0][j * 4 + k] += w0 * embed_buf[1][emb_idx];
                            partial[1][1][j * 4 + k] += w1 * embed_buf[1][emb_idx];
                            partial[1][2][j * 4 + k] += w2 * embed_buf[1][emb_idx];
                            partial[1][3][j * 4 + k] += w3 * embed_buf[1][emb_idx];
                        }
                    }
                }

                // Binary reduction for both tokens, all 4 rows
                for (int tok = 0; tok < 2; tok++) {
                    #pragma HLS unroll
                    for (int row = 0; row < 4; row++) {
                        #pragma HLS unroll
                        ap_int<36> s1[16];
                        #pragma HLS array_partition variable=s1 complete
                        for (int k = 0; k < 16; k++) {
                            #pragma HLS unroll
                            s1[k] = partial[tok][row][k] + partial[tok][row][k + 16];
                        }
                        ap_int<37> s2[8];
                        #pragma HLS array_partition variable=s2 complete
                        for (int k = 0; k < 8; k++) {
                            #pragma HLS unroll
                            s2[k] = s1[k] + s1[k + 8];
                        }
                        ap_int<38> s3[4];
                        #pragma HLS array_partition variable=s3 complete
                        for (int k = 0; k < 4; k++) {
                            #pragma HLS unroll
                            s3[k] = s2[k] + s2[k + 4];
                        }
                        ap_int<39> s4[2];
                        #pragma HLS array_partition variable=s4 complete
                        for (int k = 0; k < 2; k++) {
                            #pragma HLS unroll
                            s4[k] = s3[k] + s3[k + 2];
                        }
                        result_buf[tok][g * 4 + row] = (ap_int<39>)(s4[0] + s4[1]);
                    }
                }
            }

            // Chain reduction: single packed read+write for both tokens
            {
                vec32acc_t prev = prev_partial_fifo.read();
                vec32acc_t out;
                for (int k = 0; k < 16; k++) {
                    #pragma HLS unroll
                    out[k]      = (ap_int<42>)result_buf[0][k] + prev[k];       // token 0
                    out[k + 16] = (ap_int<42>)result_buf[1][k] + prev[k + 16];  // token 1
                }
                next_partial_fifo.write(out);
            }
        }
    }
}

// ============================================================================
// Convert ap_int<42> accumulator vectors to float before write-out.
// Separate task so the sitofp pipeline does not create II>1 in write_kv_output.
// ============================================================================
void acc_to_float(
    const int L,
    tapa::istream<vec32acc_t>& acc_in,
    tapa::ostreams<tapa::vec_t<float, 16>, 2>& float_out
) {
    // Each vec32acc_t packs 2 tokens: [0..15]=token0, [16..31]=token1.
    // Write token0 to float_out[0] and token1 to float_out[1] — different FIFOs,
    // so exactly one write per FIFO per iteration. II=1.
    const int total_pairs = L * KV_PROJ_ROWS_DIV_16 / 2;
    for (int i = 0; i < total_pairs; i++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_tripcount min=640 max=163840
        vec32acc_t acc = acc_in.read();
        tapa::vec_t<float, 16> out0, out1;
        for (int k = 0; k < 16; k++) {
            #pragma HLS unroll
            out0[k] = (float)acc[k];
            out1[k] = (float)acc[k + 16];
        }
        float_out[0].write(out0);
        float_out[1].write(out1);
    }
}

// ============================================================================
// Write final KV projection output to memory (P2P accessible by GPU)
// Each token produces KV_PROJ_ROWS_DIV_16 (320) vec_t<float,16>
// Total output: L * KV_PROJ_ROWS_DIV_16 vectors
// ============================================================================
void write_kv_output(
    const int L,
    tapa::istream<tapa::vec_t<float, 16>>& kv_output_fifo,
    tapa::async_mmap<tapa::vec_t<float, 16>>& output_mem
) {
    const int total_vecs = (L / 2) * KV_PROJ_ROWS_DIV_16;
    for (int i_req = 0, i_resp = 0; i_resp < total_vecs;) {
        #pragma HLS pipeline II=1
        if ((i_req < total_vecs) & !kv_output_fifo.empty() &
            !output_mem.write_addr.full() & !output_mem.write_data.full()) {
            tapa::vec_t<float, 16> tmp;
            kv_output_fifo.try_read(tmp);
            output_mem.write_addr.try_write(i_req);
            output_mem.write_data.try_write(tmp);
            ++i_req;
        }
        bool success = false;
        auto resp = output_mem.write_resp.read(success);
        if (success) {
            i_resp += (unsigned)(resp) + 1;
        }
    }
}

// ============================================================================
// Top-level FPGA kernel: engram_fpga_top
//
// Pipeline:
//   token IDs -> n-gram hash -> dispatch (lo/hi split) ->
//     8 lo embed lookups  +  8 hi embed lookups
//     8 KV weight channel reads
//   -> seed zeros -> 8 kv_projection channels (systolic reduction chain)
//   -> write final KV output
//
// The KV weight matrix [5120 x 1024] is split column-wise across 8 channels.
// Channel h covers columns [h*128, (h+1)*128), pairing embedding heads 2h & 2h+1.
// Reduction chain: zeros -> proj[0] -> proj[1] -> ... -> proj[7] -> output
//
// Memory layout:
//   token_id_mem:          L ints
//   embed_mem_lo[0..7]:    embedding tables for heads 0..7
//   embed_mem_hi[0..7]:    embedding tables for heads 8..15
//   kv_weight_mem[0..7]:   KV weight slices [5120 x 128] per channel
//   output_mem[0..1]:       (L/2) * KV_PROJ_ROWS_DIV_16 vec_t<float,16> each
// ============================================================================
void engram_fpga_top(
    const int L,
    const int pad_id,
    const ap_int<64> mult_0,
    const ap_int<64> mult_1,
    const ap_int<64> mult_2,
    tapa::mmap<int> token_id_mem,
    tapa::mmaps<vec16s_t, KV_WEIGHT_CHANNELS> embed_mem_lo,
    tapa::mmaps<vec16s_t, KV_WEIGHT_CHANNELS> embed_mem_hi,
    tapa::mmaps<vec16s_t, KV_WEIGHT_CHANNELS> kv_weight_mem,
    tapa::mmaps<tapa::vec_t<float, 16>, 2> output_mem
) {
    // ── Hash pipeline streams ──
    tapa::stream<int>      token_id_fifo("token_id_fifo");
    tapa::stream<vec16i_t> ngram_id_fifo("ngram_id_fifo");

    // ── Per-channel head ID dispatch (lo = heads 0..7, hi = heads 8..15) ──
    tapa::streams<int, KV_WEIGHT_CHANNELS> lo_head_id_fifos("lo_head_id_fifos");
    tapa::streams<int, KV_WEIGHT_CHANNELS> hi_head_id_fifos("hi_head_id_fifos");

    // ── Per-channel embedding streams (int16) ──
    tapa::streams<vec16s_t, KV_WEIGHT_CHANNELS>
        head_embed_lo("head_embed_lo");
    tapa::streams<vec16s_t, KV_WEIGHT_CHANNELS>
        head_embed_hi("head_embed_hi");

    // ── Per-channel KV weight streams (int16, streamed once for URAM caching) ──
    tapa::streams<vec16s_t, KV_WEIGHT_CHANNELS>
        kv_weight_fifos("kv_weight_fifos");

    // ── Reduction chain: KV_WEIGHT_CHANNELS + 1 links (int42 accumulator) ──
    // kv_chain[0]                  ← seed_chain_zero (zeros)
    // kv_chain[h+1]                ← kv_projection instance h
    // kv_chain[KV_WEIGHT_CHANNELS] → acc_to_float → write_kv_output
    tapa::streams<vec32acc_t, KV_WEIGHT_CHANNELS + 1>
        kv_chain("kv_chain");

    // ── Float streams: acc_to_float→[2]→write_kv_output (2 channels) ──
    tapa::streams<tapa::vec_t<float, 16>, 2> kv_float_tok("kv_float_tok");
    tapa::task()
        // Stage 1: Read token IDs
        .invoke<tapa::join>(
            read_token_ids, L, token_id_mem, token_id_fifo
        )
        // Stage 2: Compute n-gram hashes
        .invoke<tapa::join>(
            ngram_hash, L, pad_id, mult_0, mult_1, mult_2,
            token_id_fifo, ngram_id_fifo
        )
        // Stage 3: Dispatch — lo (heads 0..7) and hi (heads 8..15)
        .invoke<tapa::join>(
            dispatch_hash_ids, L, ngram_id_fifo,
            lo_head_id_fifos, hi_head_id_fifos
        )
        // Stage 4a: 8 lo embedding lookups (heads 0..7)
        .invoke<tapa::join, KV_WEIGHT_CHANNELS>(
            read_head_embed, L,
            lo_head_id_fifos, embed_mem_lo, head_embed_lo
        )
        // Stage 4b: 8 hi embedding lookups (heads 8..15)
        .invoke<tapa::join, KV_WEIGHT_CHANNELS>(
            read_head_embed, L,
            hi_head_id_fifos, embed_mem_hi, head_embed_hi
        )
        // Stage 5: 8 KV weight channel reads (streamed once; cached in URAM by kv_projection)
        .invoke<tapa::join, KV_WEIGHT_CHANNELS>(
            read_kv_proj_weights, kv_weight_mem, kv_weight_fifos
        )
        // Stage 6: Seed the reduction chain with zeros
        .invoke<tapa::join>(
            seed_chain_zero, L, kv_chain
        )
        // Stage 7: 8 kv_projection channels — systolic reduction chain
        // Instance h reads: kv_weight_fifos[h], head_embed_lo[h], head_embed_hi[h], kv_chain[h]
        // Instance h writes: kv_chain[h+1]
        .invoke<tapa::join, KV_WEIGHT_CHANNELS>(
            kv_projection, L,
            kv_weight_fifos, head_embed_lo, head_embed_hi,
            kv_chain, kv_chain
        )
        // Stage 8a: Convert ap_int<42> accumulator chain to float (two token streams)
        .invoke<tapa::join>(
            acc_to_float, L, kv_chain, kv_float_tok
        )
        // Stage 9: Write final KV output to HBM (2 channels: tok0 and tok1)
        .invoke<tapa::join, 2>(
            write_kv_output, L, kv_float_tok, output_mem
        );
}

#endif // __ENGRAM_H__
