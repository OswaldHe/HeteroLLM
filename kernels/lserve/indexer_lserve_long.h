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

constexpr int NUM_INDEX_HEAD = 1;
constexpr int HEAD_DIM = 128;
constexpr int HEAD_DIM_DIV_2 = HEAD_DIM / 2;
constexpr int HEAD_DIM_DIV_4 = HEAD_DIM / 4;
constexpr int MAX_LEN = 49152;
constexpr int MAX_LEN_PER_CNL = MAX_LEN / 8;
constexpr int MAX_LEN_BRAM = 16384;
constexpr int MAX_LEN_PER_CNL_BRAM = MAX_LEN_BRAM / 8;
constexpr int MAX_LEN_URAM = MAX_LEN - MAX_LEN_BRAM;
constexpr int MAX_LEN_PER_CNL_URAM = MAX_LEN_URAM / 8;
constexpr int MAX_ONCHIP_LEN = (MAX_LEN_PER_CNL_BRAM >> 7) + (MAX_LEN_PER_CNL_URAM >> 5);
constexpr int TOP_K = 64;

// 8x channels
// order is: all past keys, q0, k0, q1, k0, ..., q7, k0, q8, k1
void read_qk_vec(
    const int L,
    const int NUM_QUERY,
    tapa::async_mmap<tapa::vec_t<ap_uint<32>, 16>>& qk_vec_mem,
    tapa::ostream<tapa::vec_t<ap_uint<32>, 16>>& qk_vec_fifo
) {
    const int total_size = (L > MAX_LEN_PER_CNL) ? 
        (((MAX_LEN_PER_CNL) * HEAD_DIM) >> 5) + (((L - MAX_LEN_PER_CNL + 2) * HEAD_DIM * NUM_QUERY) >> 5) + ((HEAD_DIM / 4) * NUM_QUERY) : 
        (((L + NUM_QUERY*2) * HEAD_DIM) >> 5);
    read: for(int i_req = 0, i_resp = 0; i_resp < total_size;){
        #pragma HLS pipeline II=1
		if((i_req < total_size) & !qk_vec_mem.read_addr.full()){
            qk_vec_mem.read_addr.try_write(i_req);
            ++i_req;
		}
		if(!qk_vec_mem.read_data.empty()){
            tapa::vec_t<ap_uint<32>, 16> tmp;
            qk_vec_mem.read_data.try_read(tmp);
            qk_vec_fifo.write(tmp);
            ++i_resp;
		}
	}
}

void gemm_pe(
    const int L,
    const int L_logical, // inflate when L is large
    const int NUM_QUERY,
    tapa::istream<tapa::vec_t<ap_uint<32>, 16>>& qk_vec_fifo,
    tapa::ostream<tapa::vec_t<float, 16>>& score_out_fifo
) {
    // read keys to bram, if not enough, use uram or read from off-chip memory
    ap_uint<32> k_vec_buffer[HEAD_DIM_DIV_2][MAX_LEN_PER_CNL_BRAM];
    #pragma HLS array_partition variable=k_vec_buffer cyclic factor=2 dim=1
    #pragma HLS array_partition variable=k_vec_buffer cyclic factor=128 dim=2
    #pragma HLS bind_storage variable=k_vec_buffer type=ram_1p impl=bram

    ap_uint<64> k_vec_uram[HEAD_DIM_DIV_4][MAX_LEN_PER_CNL_URAM];
    #pragma HLS array_partition variable=k_vec_uram cyclic factor=32 dim=2
    #pragma HLS bind_storage variable=k_vec_uram type=ram_1p impl=uram

    const int offchip_len = L - MAX_LEN_PER_CNL;
    const int remain_len = (offchip_len > 0) ? MAX_LEN_PER_CNL_URAM : L - MAX_LEN_PER_CNL_BRAM;
    const int bram_len = (remain_len > 0) ? MAX_LEN_PER_CNL_BRAM : L;

    for(int r = 0; r < (bram_len >> 4); r++) {
        for(int i = 0; i < HEAD_DIM_DIV_2; i++) {
            #pragma HLS pipeline II=1
            tapa::vec_t<ap_uint<32>, 16> k_vec_pack = qk_vec_fifo.read();
            for(int j = 0; j < 16; j++){
                #pragma HLS unroll
                k_vec_buffer[i][r*16 + j] = k_vec_pack[j];
            }
        }
    }

    if(remain_len > 0) {
        // load to uram
        for(int r = 0; r < (remain_len >> 3); r++) {
            for(int i = 0; i < HEAD_DIM_DIV_4; i++) {
                #pragma HLS pipeline II=1
                tapa::vec_t<ap_uint<32>, 16> k_vec_pack = qk_vec_fifo.read();
                for(int j = 0; j < 8; j++){
                    #pragma HLS unroll
                    k_vec_uram[i][r*8 + j] = ap_uint<64>((k_vec_pack[j*2+1], k_vec_pack[j*2]));
                }
            }
        }
    }

    // start query
    // every round, there are 1 query and 1 key vector come in
    // push key into the k buffer, then do gemm between q and all k in the buffer
    for(int r = 0; r < NUM_QUERY; r++){
        int current_logical_pos = L_logical;
        if (L > MAX_LEN_PER_CNL) current_logical_pos += 16 * (r / 8);
        else if (L > MAX_LEN_PER_CNL_BRAM) current_logical_pos += 4 * (r / 8);
        const int current_pos = L + (r / 8);

        ap_uint<32> q_vec[HEAD_DIM_DIV_2];
        #pragma HLS array_partition variable=q_vec cyclic factor=16
        for(int i = 0; i < (HEAD_DIM_DIV_2 >> 4); i++) {
            #pragma HLS pipeline II=1
            tapa::vec_t<ap_uint<32>, 16> q_vec_pack = qk_vec_fifo.read();
            for(int j = 0; j < 16; j++){
                #pragma HLS unroll
                q_vec[i*16 + j] = tapa::bit_cast<ap_uint<32>>(q_vec_pack[j]);
            }
        }
        
        // we will repeatedly read k to the same position, but it will update through time
        for(int i = 0; i < (HEAD_DIM_DIV_2 >> 4); i++) {
            tapa::vec_t<ap_uint<32>, 16> k_vec_pack = qk_vec_fifo.read();
            if(current_pos < MAX_LEN_PER_CNL_BRAM){
                for (int j = 0; j < 8; j++) {
                    #pragma HLS pipeline II=1
                    for(int k = 0; k < 2; k++) {
                        #pragma HLS unroll
                        k_vec_buffer[i*16+j*2+k][current_pos] = tapa::bit_cast<ap_uint<32>>(k_vec_pack[j*2 + k]);
                    }
                } 
            } else if (current_pos < MAX_LEN_PER_CNL) {
                for (int j = 0; j < 8; j++) {
                    #pragma HLS pipeline II=1
                    k_vec_uram[i*8+j][(current_pos - MAX_LEN_PER_CNL_BRAM)] = ap_uint<64>((k_vec_pack[j*2+1], k_vec_pack[j*2]));
                }
            }
        }

        for(int r_k = 0; r_k < ((current_logical_pos + 128) >> 7); r_k++){
            // LOG(INFO) << "query " << r << ", processing key block " << r_k;

            int mask_pos = 128;
            if (r_k < (MAX_LEN_PER_CNL_BRAM >> 7)){
                mask_pos = 128;
            } else if (r_k < MAX_ONCHIP_LEN) {
                mask_pos = 32;
            } else {
                mask_pos = 8;
            }

            ap_int<40> psum_reg[4][128];
            #pragma HLS array_partition variable=psum_reg complete dim=1
            #pragma HLS array_partition variable=psum_reg complete dim=2
            for(int i = 0; i < 4; i++){
                #pragma HLS unroll
                for(int j = 0; j < 128; j++){
                    #pragma HLS unroll
                    psum_reg[i][j] = 0;
                }
            }

            if(r_k * 128 < MAX_LEN_PER_CNL_BRAM){ 
                for(int i = 0; i < (HEAD_DIM_DIV_2 >> 1); i++) {
                    #pragma HLS pipeline II=1
                    for(int j = 0; j < 2; j++) {
                        for(int k = 0; k < 128; k++) {
                            #pragma HLS unroll
                            ap_uint<32> q_elem = q_vec[i*2 + j];
                            ap_uint<32> k_elem = k_vec_buffer[i*2 + j][r_k*128 + k];
                            ap_int<16> q_low = q_elem(15, 0);
                            ap_int<16> q_high = q_elem(31, 16);
                            ap_int<16> k_low = k_elem(15, 0);
                            ap_int<16> k_high = k_elem(31, 16);
                            psum_reg[j*2][k] += q_low * k_low;
                            psum_reg[j*2+1][k] += q_high * k_high;
                        }
                    }
                }
            } else if (r_k < MAX_ONCHIP_LEN) {
                for(int i = 0; i < (HEAD_DIM_DIV_2 >> 1); i++) {
                    #pragma HLS pipeline II=1
                    for(int k = 0; k < 32; k++) {
                        #pragma HLS unroll
                        ap_uint<32> q_elem_0 = q_vec[i*2];
                        ap_uint<32> k_elem_0 = k_vec_uram[i][((r_k-(MAX_LEN_PER_CNL_BRAM >> 7))*32+k)](31, 0);
                        ap_int<16> q_low_0 = q_elem_0(15, 0);
                        ap_int<16> q_high_0 = q_elem_0(31, 16);
                        ap_int<16> k_low_0 = k_elem_0(15, 0);
                        ap_int<16> k_high_0 = k_elem_0(31, 16);
                        psum_reg[0][k] += q_low_0 * k_low_0;
                        psum_reg[1][k] += q_high_0 * k_high_0;

                        ap_uint<32> q_elem_1 = q_vec[i*2 + 1];
                        ap_uint<32> k_elem_1 = k_vec_uram[i][((r_k-(MAX_LEN_PER_CNL_BRAM >> 7))*32+k)](63, 32);
                        ap_int<16> q_low_1 = q_elem_1(15, 0);
                        ap_int<16> q_high_1 = q_elem_1(31, 16);
                        ap_int<16> k_low_1 = k_elem_1(15, 0);
                        ap_int<16> k_high_1 = k_elem_1(31, 16);
                        psum_reg[2][k] += q_low_1 * k_low_1;
                        psum_reg[3][k] += q_high_1 * k_high_1;
                    }
                }
            } else {
                for(int i = 0; i < (HEAD_DIM_DIV_2 >> 1); i++) {
                    #pragma HLS pipeline II=1
                    auto k_pack = qk_vec_fifo.read();
                    for(int j = 0; j < 2; j++) {
                        for(int k = 0; k < 8; k++) {
                            #pragma HLS unroll
                            ap_uint<32> q_elem = q_vec[i*2 + j];
                            ap_uint<32> k_elem = k_pack[j*8 + k];
                            ap_int<16> q_low = q_elem(15, 0);
                            ap_int<16> q_high = q_elem(31, 16);
                            ap_int<16> k_low = k_elem(15, 0);
                            ap_int<16> k_high = k_elem(31, 16);
                            psum_reg[j*2][k] += q_low * k_low;
                            psum_reg[j*2+1][k] += q_high * k_high;
                        }
                    }
                }
            }

            // reduction
            for(int i = 1; i < 4; i++) {
                #pragma HLS pipeline II=1
                for(int j = 0; j < 128; j++) {
                    #pragma HLS unroll
                    psum_reg[0][j] += psum_reg[i][j];
                }
            }

            // write out score, filter out invalid positions
            for(int i = 0; i < 8; i++) {
                #pragma HLS pipeline II=1
                tapa::vec_t<float, 16> score_pack;
                for(int j = 0; j < 16; j++) {
                    #pragma HLS unroll
                    int idx = i*16 + j;
                    if(idx < ((current_logical_pos + 128) - r_k*128) && idx < mask_pos) {
                        score_pack[j] = float(psum_reg[0][idx]);
                    } else {
                        score_pack[j] = -1e10f; // pad with large negative value
                    }
                }
                score_out_fifo.write(score_pack);
            }
        }
    }
}


void min_max_page_reduction(
    tapa::istream<tapa::vec_t<float, 16>>& score_in_fifo,
    tapa::ostream<tapa::vec_t<float, 2>>& score_out_fifo
) {
    for(;;){
        for(int i = 0; i < 8; i++) {
            #pragma HLS pipeline II=1
            auto score_pack = score_in_fifo.read();
            // max reduction
            float max_level_1[8];
            #pragma HLS array_partition variable=max_level_1 complete
            for(int j = 0; j < 8; j++){
                #pragma HLS unroll
                max_level_1[j] = (score_pack[j*2] > score_pack[j*2+1]) ? score_pack[j*2] : score_pack[j*2+1];
            }
            float max_level_2[4];
            #pragma HLS array_partition variable=max_level_2 complete
            for(int j = 0; j < 4; j++){
                #pragma HLS unroll
                max_level_2[j] = (max_level_1[j*2] > max_level_1[j*2+1]) ? max_level_1[j*2] : max_level_1[j*2+1];
            }
            tapa::vec_t<float, 2> max_level_3;
            #pragma HLS array_partition variable=max_level_3 complete
            for(int j = 0; j < 2; j++){
                #pragma HLS unroll
                max_level_3[j] = (max_level_2[j*2] > max_level_2[j*2+1]) ? max_level_2[j*2] : max_level_2[j*2+1];
            }
            score_out_fifo.write(max_level_3);
        }
    }
}

void score_packing(
    tapa::istreams<tapa::vec_t<float, 2>, 8>& score_in_fifo,
    tapa::ostream<tapa::vec_t<float, 16>>& score_out_fifo
) {
    for(;;){
        bool flag = false;
        for(int i = 0; i < 8; i++){
            #pragma HLS unroll
            flag |= score_in_fifo[i].empty();
        }
        if(!flag) {
            tapa::vec_t<float, 16> score_pack;
            for(int i = 0; i < 8; i++){
                #pragma HLS unroll
                tapa::vec_t<float, 2> score; score_in_fifo[i].try_read(score);
                score_pack[i*2] = score[0];
                score_pack[i*2+1] = score[1];
            }
            score_out_fifo.write(score_pack);
        }
    }
}

void topk_parallel_cmp(
    const int L,
    const int NUM_QUERY,
    tapa::istream<tapa::vec_t<float, 16>>& score_id_fifo,
    tapa::ostream<tapa::vec_t<int, 16>>& topk_id_fifo
) {
  for(int q = 0; q < NUM_QUERY; q++){
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
            int idx = r*16 + i;
            ap_uint<64> score_id = ap_uint<64>((ap_uint<32>(idx), tapa::bit_cast<ap_uint<32>>(score_id_pack[i])));
            topk_node[r*16+i] = score_id;
        }
        // find min in 16 values
        // binary reduction
        float min_score_r[8];
        int min_idx_r[8];
        #pragma HLS array_partition variable=min_score_r complete
        #pragma HLS array_partition variable=min_idx_r complete
        for(int i = 0; i < 8; i++){
            #pragma HLS unroll
            if(score_id_pack[i*2] < score_id_pack[i*2+1]) {
                min_score_r[i] = score_id_pack[i*2];
                min_idx_r[i] = i*2;
            } else {
                min_score_r[i] = score_id_pack[i*2+1];
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

    const int remain_len = (((L + (q/8) + 128) >> 7) << 3) - (TOP_K >> 4);
    
    for(int r = 0; r < remain_len; r++){
        auto score_id_pack = score_id_fifo.read();
        bool fast_check = false;
        for(int i = 0; i < 16; i++) {
            #pragma HLS unroll
            fast_check |= (score_id_pack[i] > min_score);
        }

        if(fast_check) {
            for(int i = 0; i < 16; i++){
                #pragma HLS pipeline off

                if(score_id_pack[i] > min_score) {
                    topk_node[min_idx] = ap_uint<64>((ap_uint<32>(r*16 + i + TOP_K), tapa::bit_cast<ap_uint<32>>(score_id_pack[i])));
                    // binary reduction parallel min search
                    float min_score_b[64];
                    #pragma HLS array_partition variable=min_score_b complete
                    for(int i = 0; i < 64; i++){
                        #pragma HLS unroll
                        min_score_b[i] = tapa::bit_cast<float>(ap_uint<32>(topk_node[i](31, 0)));
                    }

                    float min_score_r[32];
                    int min_idx_r[32];
                    #pragma HLS array_partition variable=min_score_r complete
                    #pragma HLS array_partition variable=min_idx_r complete
                    for(int i = 0; i < 32; i++){
                        #pragma HLS unroll
                        if(min_score_b[i*2] < min_score_b[i*2+1]) {
                            min_score_r[i] = min_score_b[i*2];
                            min_idx_r[i] = i*2;
                        } else {
                            min_score_r[i] = min_score_b[i*2+1];
                            min_idx_r[i] = i*2+1;
                        }
                    }

                    float min_score_r_2[16];
                    int min_idx_r_2[16];
                    #pragma HLS array_partition variable=min_score_r_2 complete
                    #pragma HLS array_partition variable=min_idx_r_2 complete
                    for(int i = 0; i < 16; i++){
                        #pragma HLS unroll
                        if(min_score_r[i*2] < min_score_r[i*2+1]) {
                            min_score_r_2[i] = min_score_r[i*2];
                            min_idx_r_2[i] = min_idx_r[i*2];
                        } else {
                            min_score_r_2[i] = min_score_r[i*2+1];
                            min_idx_r_2[i] = min_idx_r[i*2+1];
                        }
                    }

                    float min_score_r_3[8];
                    int min_idx_r_3[8];
                    #pragma HLS array_partition variable=min_score_r_3 complete
                    #pragma HLS array_partition variable=min_idx_r_3 complete
                    for(int i = 0; i < 8; i++){
                        #pragma HLS unroll
                        if(min_score_r_2[i*2] < min_score_r_2[i*2+1]) {
                            min_score_r_3[i] = min_score_r_2[i*2];
                            min_idx_r_3[i] = min_idx_r_2[i*2];
                        } else {
                            min_score_r_3[i] = min_score_r_2[i*2+1];
                            min_idx_r_3[i] = min_idx_r_2[i*2+1];
                        }
                    }

                    float min_score_r_4[4];
                    int min_idx_r_4[4];
                    #pragma HLS array_partition variable=min_score_r_4 complete
                    #pragma HLS array_partition variable=min_idx_r_4 complete
                    for(int i = 0; i < 4; i++){
                        #pragma HLS unroll
                        if(min_score_r_3[i*2] < min_score_r_3[i*2+1]) {
                            min_score_r_4[i] = min_score_r_3[i*2];
                            min_idx_r_4[i] = min_idx_r_3[i*2];
                        } else {
                            min_score_r_4[i] = min_score_r_3[i*2+1];
                            min_idx_r_4[i] = min_idx_r_3[i*2+1];
                        }
                    }

                    float min_score_r_5[2];
                    int min_idx_r_5[2];
                    #pragma HLS array_partition variable=min_score_r_5 complete
                    #pragma HLS array_partition variable=min_idx_r_5 complete
                    for(int i = 0; i < 2; i++){
                        #pragma HLS unroll
                        if(min_score_r_4[i*2] < min_score_r_4[i*2+1]) {
                            min_score_r_5[i] = min_score_r_4[i*2];
                            min_idx_r_5[i] = min_idx_r_4[i*2];
                        } else {
                            min_score_r_5[i] = min_score_r_4[i*2+1];
                            min_idx_r_5[i] = min_idx_r_4[i*2+1];
                        }
                    }

                    if(min_score_r_5[0] < min_score_r_5[1]) {
                        min_score = min_score_r_5[0];
                        min_idx = min_idx_r_5[0];
                    } else {
                        min_score = min_score_r_5[1];
                        min_idx = min_idx_r_5[1];
                    }
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
        // LOG(INFO) << "Query " << q << " Topk IDs Pack " << i;
        topk_id_fifo.write(topk_id_pack);
    }
  } // end query loop
}

void write_id(
    const int NUM_QUERY,
    tapa::async_mmap<tapa::vec_t<int, 16>>& topk_id_mem,
    tapa::istream<tapa::vec_t<int, 16>>& topk_id_fifo
) {
  const int total_writes = NUM_QUERY * (TOP_K >> 4);
  for(int i_req = 0, i_resp = 0; i_resp < total_writes;){
      #pragma HLS pipeline II=1 style=stp
      if((i_req < total_writes) & !topk_id_fifo.empty() & !topk_id_mem.write_addr.full() & !topk_id_mem.write_data.full()){
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

// used to adjust bandwidth
void dummy_consumer(
    tapa::istream<tapa::vec_t<ap_uint<32>, 16>>& qk_vec_fifo
) {
    for(;;){
        tapa::vec_t<ap_uint<32>, 16> tmp = qk_vec_fifo.read();
    }
}

void dummy_producer(
    const int L,
    const int NUM_QUERY,
    tapa::ostream<tapa::vec_t<float, 2>>& score_out_fifo
) {
    for(int q = 0; q < NUM_QUERY; q++){
        const int total_reads = ((L + (q/8) + 128) >> 7) << 3;
        for(int r = 0; r < total_reads; r++){
            #pragma HLS pipeline II=1
            tapa::vec_t<float, 2> tmp;
            for(int i = 0; i < 2; i++){
                #pragma HLS unroll
                tmp[i] = -2e10f;
            }
            score_out_fifo.write(tmp);
        }
    }
}

void indexer_top(
    const int L,
    const int L_logical,
    const int NUM_QUERY,
    tapa::mmaps<tapa::vec_t<ap_uint<32>, 16>, 8> qk_vec_mem,
    tapa::mmap<tapa::vec_t<int, 16>> topk_id_mem
) {
    tapa::streams<tapa::vec_t<ap_uint<32>, 16>, 8> qk_vec_fifo("qk_vec_fifo");
    tapa::streams<tapa::vec_t<float, 16>, 5> score_out_fifo("score_out_fifo");
    tapa::streams<tapa::vec_t<float, 2>, 8> max_score_fifo("max_score_fifo");
    tapa::stream<tapa::vec_t<float, 16>> score_packing_fifo("score_packing_fifo");
    tapa::stream<tapa::vec_t<int, 16>> topk_id_fifo("topk_id_fifo");

    tapa::task()
        .invoke<tapa::join, 8>(
            read_qk_vec, L, NUM_QUERY, qk_vec_mem, qk_vec_fifo
        )
        .invoke<tapa::join, 5>(
            gemm_pe, L, L_logical, NUM_QUERY, qk_vec_fifo, score_out_fifo
        )
        .invoke<tapa::detach, 3>(
            dummy_consumer, qk_vec_fifo
        )
        .invoke<tapa::detach, 5>(
            min_max_page_reduction, score_out_fifo, max_score_fifo
        )
        .invoke<tapa::join, 3>(
            dummy_producer, L_logical, NUM_QUERY, max_score_fifo
        )
        .invoke<tapa::detach>(
            score_packing, max_score_fifo, score_packing_fifo
        )
        .invoke<tapa::join>(
            topk_parallel_cmp, L_logical, NUM_QUERY, score_packing_fifo, topk_id_fifo
        )
        .invoke<tapa::join>(
            write_id, NUM_QUERY, topk_id_mem, topk_id_fifo
        );
       
}

#endif // __INDEXER_V_H__