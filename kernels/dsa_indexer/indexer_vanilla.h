#ifndef __INDEXER_V_H__
#define __INDEXER_V_H__

#include <tapa.h>
#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_vector.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <cstdint>
#include <limits>

constexpr int NUM_INDEX_HEAD = 16;
constexpr int HEAD_DIM = 128;
constexpr int DIM_PER_PE = HEAD_DIM / 8;
constexpr int TOP_K = 2048; // 64

// TOP_K must be a multiple of 16 (packed 16 lanes at a time) and a power of two
// (topk_parallel_cmp reduces over all TOP_K entries with a binary tree).
static_assert(TOP_K >= 16 && (TOP_K & (TOP_K - 1)) == 0,
              "TOP_K must be a power of two and at least 16");

// 8x channels
void read_qk_vec(
    const int L, // length of key vectors
    tapa::async_mmap<tapa::vec_t<ap_int<16>, 16>>& qk_vec_mem,
    tapa::ostream<tapa::vec_t<ap_int<16>, 16>>& qk_vec_fifo
) {
    const int total_size = (L + NUM_INDEX_HEAD) * (HEAD_DIM >> 7);
    for(int i_req = 0, i_resp = 0; i_resp < total_size;){
        #pragma HLS pipeline II=1
		if((i_req < total_size) & !qk_vec_mem.read_addr.full()){
            qk_vec_mem.read_addr.try_write(i_req);
            ++i_req;
		}
		if(!qk_vec_mem.read_data.empty()){
            tapa::vec_t<ap_int<16>, 16> tmp;
            qk_vec_mem.read_data.try_read(tmp);
            qk_vec_fifo.write(tmp);
            ++i_resp;
		}
	}
}

// 1x channel
void read_weight(
    tapa::async_mmap<tapa::vec_t<float, 16>>& weight_mem,
    tapa::ostream<tapa::vec_t<float, 16>>& weight_fifo
) {
    for(int i_req = 0, i_resp = 0; i_resp < 1;){
        #pragma HLS pipeline II=1
		if((i_req < 1) & !weight_mem.read_addr.full()){
            weight_mem.read_addr.try_write(i_req);
            ++i_req;
		}
		if(!weight_mem.read_data.empty()){
            tapa::vec_t<float, 16> tmp;
            weight_mem.read_data.try_read(tmp);
            weight_fifo.write(tmp);
            ++i_resp;
		}
	}
}

void gemm_pe(
    const int L,
    tapa::istream<tapa::vec_t<ap_int<16>, 16>>& qk_vec_fifo,
    tapa::istream<tapa::vec_t<ap_int<40>, 16>>& psum_in_fifo,
    tapa::ostream<tapa::vec_t<ap_int<40>, 16>>& psum_out_fifo
) {
    ap_int<16> q_vec[DIM_PER_PE][NUM_INDEX_HEAD];
    #pragma HLS array_partition variable=q_vec complete dim=2

    for(int i = 0; i < DIM_PER_PE; i++) {
        #pragma HLS pipeline II=1
        tapa::vec_t<ap_int<16>, 16> q_vec_pack = qk_vec_fifo.read();
        for(int j = 0; j < NUM_INDEX_HEAD; j++){
            #pragma HLS unroll
            q_vec[i][j] = q_vec_pack[j];
        }
    }

    for(int r = 0; r < (L >> 4); r++){

        ap_int<40> psum_reg[16][16];
        #pragma HLS array_partition variable=psum_reg complete dim=1
        #pragma HLS array_partition variable=psum_reg complete dim=2
        for(int i = 0; i < 16; i++){
            #pragma HLS unroll
            for(int j = 0; j < 16; j++){
                #pragma HLS unroll
                psum_reg[i][j] = 0;
            }
        }

        for(int i = 0; i < DIM_PER_PE; i++) {
            #pragma HLS pipeline II=1
            tapa::vec_t<ap_int<16>, 16> k_vec_pack = qk_vec_fifo.read();
            for(int j = 0; j < 16; j++){
                #pragma HLS unroll
                for(int k = 0; k < 16; k++){
                    #pragma HLS unroll
                    psum_reg[k][j] += q_vec[i][j] * k_vec_pack[k];
                }
            }
        }

        for(int i = 0; i < 16; i++) {
            #pragma HLS pipeline II=1
            auto psum_prior = psum_in_fifo.read();
            tapa::vec_t<ap_int<40>, 16> psum_out_pack;
            for(int j = 0; j < 16; j++){
                #pragma HLS unroll
                psum_out_pack[j] = psum_prior[j] + psum_reg[j][i];
            }
            psum_out_fifo.write(psum_out_pack);
        }
    }
}

void gemm_pe_head(
    const int L,
    tapa::istream<tapa::vec_t<ap_int<16>, 16>>& qk_vec_fifo,
    tapa::ostream<tapa::vec_t<ap_int<40>, 16>>& psum_out_fifo
) {
    ap_int<16> q_vec[DIM_PER_PE][NUM_INDEX_HEAD];
    #pragma HLS array_partition variable=q_vec complete dim=2

    for(int i = 0; i < DIM_PER_PE; i++) {
        #pragma HLS pipeline II=1
        tapa::vec_t<ap_int<16>, 16> q_vec_pack = qk_vec_fifo.read();
        for(int j = 0; j < NUM_INDEX_HEAD; j++){
            #pragma HLS unroll
            q_vec[i][j] = q_vec_pack[j];
        }
    }

    for(int r = 0; r < (L >> 4); r++){

        ap_int<40> psum_reg[16][16];
        #pragma HLS array_partition variable=psum_reg complete dim=1
        #pragma HLS array_partition variable=psum_reg complete dim=2
        for(int i = 0; i < 16; i++){
            #pragma HLS unroll
            for(int j = 0; j < 16; j++){
                #pragma HLS unroll
                psum_reg[i][j] = 0;
            }
        }

        for(int i = 0; i < DIM_PER_PE; i++) {
            #pragma HLS pipeline II=1
            tapa::vec_t<ap_int<16>, 16> k_vec_pack = qk_vec_fifo.read();
            for(int j = 0; j < 16; j++){
                #pragma HLS unroll
                for(int k = 0; k < 16; k++){
                    #pragma HLS unroll
                    psum_reg[k][j] += q_vec[i][j] * k_vec_pack[k];
                }
            }
        }

        for(int i = 0; i < 16; i++) {
            #pragma HLS pipeline II=1
            tapa::vec_t<ap_int<40>, 16> psum_out_pack;
            for(int j = 0; j < 16; j++){
                #pragma HLS unroll
                psum_out_pack[j] = psum_reg[j][i];
            }
            psum_out_fifo.write(psum_out_pack);
        }
    }
}

void relu(
    const int L,
    tapa::istream<tapa::vec_t<ap_int<40>, 16>>& psum_in_fifo,
    tapa::ostream<tapa::vec_t<ap_int<40>, 16>>& psum_out_fifo
) {
    for(int r = 0; r < L; r++){
        #pragma HLS pipeline II=1
        tapa::vec_t<ap_int<40>, 16> psum_in_pack = psum_in_fifo.read();
        tapa::vec_t<ap_int<40>, 16> psum_out_pack;
        for(int i = 0; i < 16; i++){
            #pragma HLS unroll
            if(psum_in_pack[i] > 0) {
                psum_out_pack[i] = psum_in_pack[i];
            } else {
                psum_out_pack[i] = 0;
            }
        }
        psum_out_fifo.write(psum_out_pack);
    }
}

void weighted_sum_mul(
    const int L,
    tapa::istream<tapa::vec_t<ap_int<40>, 16>>& psum_in_fifo,
    tapa::istream<tapa::vec_t<float, 16>>& weight_fifo,
    tapa::ostream<tapa::vec_t<float, 16>>& psum_out_fifo
) {
    auto weight_pack = weight_fifo.read();

    for(int r = 0; r < L; r++){
        #pragma HLS pipeline II=1
        tapa::vec_t<ap_int<40>, 16> psum_pack = psum_in_fifo.read();
        float weight = weight_pack[r%16];
        tapa::vec_t<float, 16> psum_out_pack;
        for(int j = 0; j < 16; j++){
            #pragma HLS unroll
            psum_out_pack[j] = weight * (float)psum_pack[j];
        }
        psum_out_fifo.write(psum_out_pack);
    }
}

void weighted_sum_add(
    const int L,
    tapa::istream<tapa::vec_t<float, 16>>& psum_in_fifo,
    tapa::ostream<tapa::vec_t<ap_uint<64>, 16>>& score_id_fifo
) {
    for(int r = 0; r < (L >> 4); r++){
        #pragma HLS dataflow

        ap_int<44> score_reg[16][2];
        #pragma HLS array_partition variable=score_reg complete dim=1
        #pragma HLS array_partition variable=score_reg complete dim=2
        for(int i = 0; i < 16; i++){
            #pragma HLS unroll
            for(int j = 0; j < 2; j++){
                #pragma HLS unroll
                score_reg[i][j] = 0;
            }
        }

        for(int i = 0; i < 16; i++) {
            #pragma HLS pipeline II=1
            tapa::vec_t<float, 16> psum_pack = psum_in_fifo.read();
            for(int j = 0; j < 16; j++){
                #pragma HLS unroll
                auto tmp = score_reg[j][i%2];
                tmp += ap_int<44>( (long long) psum_pack[j]);
                score_reg[j][i%2] = tmp;
            }
        }

        tapa::vec_t<ap_uint<64>, 16> score_id_pack;
        for(int i = 0; i < 16; i++){
            #pragma HLS unroll
            ap_uint<64> tmp;
            auto score = score_reg[i][0] + score_reg[i][1];
            float score_f = (float)score;
            tmp(31, 0) = tapa::bit_cast<ap_uint<32>>(score_f);
            // LOG(INFO) << "Score for node " << (r*16 + i) << ": " << score;
            tmp(63, 32) = ap_uint<32>(r*16 + i); // node id
            score_id_pack[i] = tmp;
        }
        score_id_fifo.write(score_id_pack);
    }
}

// Compile-time binary reduction that returns the smallest score (bits [31:0])
// among topk_node[OFF, OFF+N) together with its absolute index. Everything is
// inlined, so it lowers to the same hand-unrolled comparator tree the previous
// fixed-size (TOP_K==64) code produced, but for any power-of-two size.
template<int OFF, int N>
struct topk_argmin {
    static void run(ap_uint<64> topk_node[TOP_K], float& min_score, int& min_idx) {
        #pragma HLS inline
        float lo_score, hi_score;
        int lo_idx, hi_idx;
        topk_argmin<OFF, N / 2>::run(topk_node, lo_score, lo_idx);
        topk_argmin<OFF + N / 2, N / 2>::run(topk_node, hi_score, hi_idx);
        if(hi_score < lo_score) {
            min_score = hi_score;
            min_idx = hi_idx;
        } else {
            min_score = lo_score;
            min_idx = lo_idx;
        }
    }
};

template<int OFF>
struct topk_argmin<OFF, 1> {
    static void run(ap_uint<64> topk_node[TOP_K], float& min_score, int& min_idx) {
        #pragma HLS inline
        min_score = tapa::bit_cast<float>(ap_uint<32>(topk_node[OFF](31, 0)));
        min_idx = OFF;
    }
};

void topk_parallel_cmp(
    const int L,
    tapa::istream<tapa::vec_t<ap_uint<64>, 16>>& score_id_fifo,
    tapa::ostream<tapa::vec_t<int, 16>>& topk_id_fifo
) {
    ap_uint<64> topk_node[TOP_K];
    #pragma HLS array_partition variable=topk_node complete
    int min_idx = 0;
    float min_score = 1e15f;
    for(int r = 0; r < (TOP_K >> 4); r++){
        #pragma HLS pipeline II=1
        #pragma HLS dependence variable=min_score inter true
        #pragma HLS dependence variable=min_idx inter true
        auto score_id_pack = score_id_fifo.read();
        for(int i = 0; i < 16; i++){
            #pragma HLS unroll
            topk_node[r*16+i] = score_id_pack[i];
        }
        // find min in 16 values
        // binary reduction
        float min_score_r[8];
        int min_idx_r[8];
        #pragma HLS array_partition variable=min_score_r complete
        #pragma HLS array_partition variable=min_idx_r complete
        for(int i = 0; i < 8; i++){
            #pragma HLS unroll
            if(tapa::bit_cast<float>(ap_uint<32>(score_id_pack[i*2](31, 0))) < tapa::bit_cast<float>(ap_uint<32>(score_id_pack[i*2+1](31, 0)))) {
                min_score_r[i] = tapa::bit_cast<float>(ap_uint<32>(score_id_pack[i*2](31, 0)));
                min_idx_r[i] = i*2;
            } else {
                min_score_r[i] = tapa::bit_cast<float>(ap_uint<32>(score_id_pack[i*2+1](31, 0)));
                min_idx_r[i] = i*2+1;
            }  
        }
        float min_score_r_2[4];
        int min_idx_r_2[4];
        #pragma HLS array_partition variable=min_score_r_2 complete
        #pragma HLS array_partition variable=min_idx_r_2 complete
        for(int i = 0; i < 4; i++){
            #pragma HLS unroll
            if(min_score_r[i*2] < min_score_r[i*2+1]) {
                min_score_r_2[i] = min_score_r[i*2];
                min_idx_r_2[i] = min_idx_r[i*2];
            } else {
                min_score_r_2[i] = min_score_r[i*2+1];
                min_idx_r_2[i] = min_idx_r[i*2+1];
            }
        }

        float min_score_r_3[2];
        int min_idx_r_3[2];
        #pragma HLS array_partition variable=min_score_r_3 complete
        #pragma HLS array_partition variable=min_idx_r_3 complete
        for(int i = 0; i < 2; i++){
            #pragma HLS unroll
            if(min_score_r_2[i*2] < min_score_r_2[i*2+1]) {
                min_score_r_3[i] = min_score_r_2[i*2];
                min_idx_r_3[i] = min_idx_r_2[i*2];
            } else {
                min_score_r_3[i] = min_score_r_2[i*2+1];
                min_idx_r_3[i] = min_idx_r_2[i*2+1];
            } 
        }

        float current_min_score = 0.0f;
        int current_min_idx = 0;
        if(min_score_r_3[0] < min_score_r_3[1]) {
            current_min_score = min_score_r_3[0];
            current_min_idx = min_idx_r_3[0];
        } else {
            current_min_score = min_score_r_3[1];
            current_min_idx = min_idx_r_3[1];
        }

        if(min_score > current_min_score) {
            min_score = current_min_score;
            min_idx = r*16 + current_min_idx;
        }
    }

    for(int r = 0; r < ((L-TOP_K) >> 4); r++){
        auto score_id_pack = score_id_fifo.read();
        bool fast_check = false;
        for(int i = 0; i < 16; i++) {
            #pragma HLS unroll
            fast_check |= (tapa::bit_cast<float>(ap_uint<32>(score_id_pack[i](31, 0))) > min_score);
        }

        if(fast_check) {
            for(int i = 0; i < 16; i++){
                #pragma HLS pipeline off

                if(tapa::bit_cast<float>(ap_uint<32>(score_id_pack[i](31, 0))) > min_score) {
                    topk_node[min_idx] = score_id_pack[i];
                    // binary reduction parallel min search over all TOP_K entries
                    topk_argmin<0, TOP_K>::run(topk_node, min_score, min_idx);
                }
            }
        }
    }

    //write out topk ids
    for(int i = 0; i < (TOP_K >> 4); i++){
        #pragma HLS pipeline II=1
        tapa::vec_t<int, 16> topk_id_pack;
        for(int j = 0; j < 16; j++){
            #pragma HLS unroll
            topk_id_pack[j] = ap_int<32>(topk_node[i*16+j](63, 32)).to_int();
        }
        topk_id_fifo.write(topk_id_pack);
    }
}

void topk_minheap(
    const int L,
    tapa::istream<tapa::vec_t<ap_uint<64>, 16>>& score_id_fifo,
    tapa::ostream<tapa::vec_t<int, 16>>& topk_id_fifo
) {
    ap_uint<64> topk_node[TOP_K];
    #pragma HLS array_partition variable=topk_node complete
    for(int r = 0; r < (L >> 4); r++){
        auto score_id_pack = score_id_fifo.read();
        for(int i = 0; i < 16; i++){
            
            if(r*16+i < TOP_K) {
                // directly push to heap
                topk_node[r*16+i] = score_id_pack[i];
                int c = r*16+i;
                while(c > 0) {
                    #pragma HLS pipeline II=3
                    int p = (c - 1) >> 1;
                    if (tapa::bit_cast<float>(ap_uint<32>(topk_node[c](31, 0))) < tapa::bit_cast<float>(ap_uint<32>(topk_node[p](31, 0)))) {
                        ap_uint<64> t = topk_node[c]; topk_node[c] = topk_node[p]; topk_node[p] = t;
                        c = p;
                    } else {
                        break;
                    }
                }
            } else if (tapa::bit_cast<float>(ap_uint<32>(score_id_pack[i](31, 0))) > tapa::bit_cast<float>(ap_uint<32>(topk_node[0](31, 0)))) {
                // replace root and heapify down
                topk_node[0] = score_id_pack[i];
                int c = 0;
                int prev_c = -1;
                while(c != prev_c) {
                    #pragma HLS pipeline II=4
                    #pragma HLS dependence variable=topk_node inter true
                    #pragma HLS dependence variable=c inter true
                    #pragma HLS dependence variable=prev_c inter true
                    int l = (c << 1) + 1;
                    int r = (c << 1) + 2;
                    int s = c;
                    float l_v = (l < TOP_K) ? tapa::bit_cast<float>(ap_uint<32>(topk_node[l](31, 0))) : 1e15f;
                    float r_v = (r < TOP_K) ? tapa::bit_cast<float>(ap_uint<32>(topk_node[r](31, 0))) : 1e15f;
                    float s_v = tapa::bit_cast<float>(ap_uint<32>(topk_node[s](31, 0)));
                    prev_c = c;

                    if ((l_v < s_v) && (l_v < r_v)) {
                        s = l;
                    } else if ((r_v < s_v) && (r_v < l_v)) {
                        s = r;
                    }
                    ap_uint<64> t = topk_node[c]; topk_node[c] = topk_node[s]; topk_node[s] = t;
                    c = s;
                }
            }
        }
    }

    //write out topk ids
    for(int i = 0; i < (TOP_K >> 4); i++){
        #pragma HLS pipeline II=1
        tapa::vec_t<int, 16> topk_id_pack;
        for(int j = 0; j < 16; j++){
            #pragma HLS unroll
            topk_id_pack[j] = ap_int<32>(topk_node[i*16+j](63, 32)).to_int();
        }
        topk_id_fifo.write(topk_id_pack);
    }
}

void write_id(
    tapa::async_mmap<tapa::vec_t<int, 16>>& topk_id_mem,
    tapa::istream<tapa::vec_t<int, 16>>& topk_id_fifo
) {
    for(int i_req = 0, i_resp = 0; i_resp < (TOP_K >> 4);){
        #pragma HLS pipeline II=1 style=stp
        if((i_req < (TOP_K >> 4)) & !topk_id_fifo.empty() & !topk_id_mem.write_addr.full() & !topk_id_mem.write_data.full()){
            topk_id_mem.write_addr.try_write(i_req);
            tapa::vec_t<int, 16> tmp; topk_id_fifo.try_read(tmp);
            topk_id_mem.write_data.try_write(tmp);
            ++i_req;
        }
        bool success = false;
        auto resp = topk_id_mem.write_resp.read(success);
        if(success){
            i_resp += unsigned(resp)+1;
        }
    }
}

void indexer_top(
    const int L,
    tapa::mmaps<tapa::vec_t<ap_int<16>, 16>, 8> qk_vec_mem,
    tapa::mmap<tapa::vec_t<float, 16>> weight_mem,
    tapa::mmap<tapa::vec_t<int, 16>> topk_id_mem
) {
    tapa::streams<tapa::vec_t<ap_int<16>, 16>, 8> qk_vec_fifo("qk_vec_fifo");
    tapa::stream<tapa::vec_t<float, 16>> weight_fifo("weight_fifo");
    tapa::streams<tapa::vec_t<ap_int<40>, 16>, 9, 4> psum_fifo("psum_fifo");
    tapa::stream<tapa::vec_t<float, 16>> weight_psum_fifo("weight_psum_fifo");
    tapa::stream<tapa::vec_t<ap_uint<64>, 16>, 512> score_id_fifo("score_id_fifo");
    tapa::stream<tapa::vec_t<int, 16>> topk_id_fifo("topk_id_fifo");

    tapa::task()
        .invoke<tapa::join, 8>(
            read_qk_vec, L, qk_vec_mem, qk_vec_fifo
        )
        .invoke<tapa::join>(
            read_weight, weight_mem, weight_fifo
        )
        .invoke<tapa::join>(
            gemm_pe_head, L, qk_vec_fifo, psum_fifo
        )
        .invoke<tapa::join, 7>(
            gemm_pe, L, qk_vec_fifo, psum_fifo, psum_fifo
        )
        .invoke<tapa::join>(
            relu, L, psum_fifo, psum_fifo
        )
        .invoke<tapa::join>(
            weighted_sum_mul, L, psum_fifo, weight_fifo, weight_psum_fifo
        )
        .invoke<tapa::join>(
            weighted_sum_add, L, weight_psum_fifo, score_id_fifo
        )
        .invoke<tapa::join>(
            topk_parallel_cmp, L, score_id_fifo, topk_id_fifo
        )
        .invoke<tapa::join>(
            write_id, topk_id_mem, topk_id_fifo
        );
}

#endif // __INDEXER_V_H__