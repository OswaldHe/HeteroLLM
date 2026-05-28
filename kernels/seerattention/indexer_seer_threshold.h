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
constexpr int MAX_LEN = 16384;
constexpr int MAX_LEN_PER_CNL = MAX_LEN / 8;

// 8x channels
// order is: all past keys, q0, k0, q1, k0, ..., q7, k0, q8, k1
void read_qk_vec(
    const int L, // length of key vectors per channel
    const int NUM_QUERY,
    tapa::async_mmap<tapa::vec_t<ap_uint<32>, 16>>& qk_vec_mem,
    tapa::ostream<tapa::vec_t<ap_uint<32>, 16>>& qk_vec_fifo
) {
    const int total_size = ((L + NUM_QUERY*2) * HEAD_DIM) >> 5;
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
    const int NUM_QUERY,
    const int THRESHOLD,
    tapa::istream<tapa::vec_t<ap_uint<32>, 16>>& qk_vec_fifo,
    tapa::ostream<ap_uint<128>>& sparse_bitmap_fifo
) {
    // first, load all keys to uram
    ap_uint<32> k_vec_buffer[HEAD_DIM_DIV_2][MAX_LEN_PER_CNL];
    #pragma HLS array_partition variable=k_vec_buffer cyclic factor=2 dim=1
    #pragma HLS array_partition variable=k_vec_buffer cyclic factor=128 dim=2
    #pragma HLS bind_storage variable=k_vec_buffer type=ram_1p impl=bram

    for(int r = 0; r < (L >> 4); r++) {
        for(int i = 0; i < HEAD_DIM_DIV_2; i++) {
            #pragma HLS pipeline II=1
            tapa::vec_t<ap_uint<32>, 16> k_vec_pack = qk_vec_fifo.read();
            for(int j = 0; j < 16; j++){
                #pragma HLS unroll
                k_vec_buffer[i][r*16 + j] = tapa::bit_cast<ap_uint<32>>(k_vec_pack[j]);
            }
        }
    }

    // start query
    // every round, there are 1 query and 1 key vector come in
    // push key into the k buffer, then do gemm between q and all k in the buffer
    for(int r = 0; r < NUM_QUERY; r++){
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
            for (int j = 0; j < 8; j++) {
                #pragma HLS pipeline II=1
                for(int k = 0; k < 2; k++) {
                    #pragma HLS unroll
                    k_vec_buffer[i*16+j*2+k][current_pos] = tapa::bit_cast<ap_uint<32>>(k_vec_pack[j*2 + k]);
                }
            }
        }


        for(int r_k = 0; r_k < ((current_pos + 128) >> 7); r_k++){
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

            // every 32 bit element has 2 16-bit integer
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

            // reduction
            for(int i = 1; i < 4; i++) {
                #pragma HLS pipeline II=1
                for(int j = 0; j < 128; j++) {
                    #pragma HLS unroll
                    psum_reg[0][j] += psum_reg[i][j];
                }
            }

            ap_uint<128> bitmap;
            for(int i = 0; i < 128; i++) {
                #pragma HLS unroll
                if(psum_reg[0][i] > THRESHOLD) {
                    bitmap[i] = 1;
                } else {
                    bitmap[i] = 0;
                }
            }

            sparse_bitmap_fifo.write(bitmap);
        }
    }
}



void write_id(
    const int L,
    const int NUM_QUERY,
    tapa::async_mmap<ap_uint<128>>& sparse_bitmap_mem,
    tapa::istream<ap_uint<128>>& sparse_bitmap_fifo
) {
    const int total_size = L * NUM_QUERY + NUM_QUERY/1024 + NUM_QUERY*128;
    for(int i_req = 0, i_resp = 0; i_resp < (total_size >> 7);){
        #pragma HLS pipeline II=1 style=stp
        if((i_req < (total_size >> 7)) & !sparse_bitmap_fifo.empty() & !sparse_bitmap_mem.write_addr.full() & !sparse_bitmap_mem.write_data.full()){
            sparse_bitmap_mem.write_addr.try_write(i_req);
            ap_uint<128> tmp; sparse_bitmap_fifo.try_read(tmp);
            sparse_bitmap_mem.write_data.try_write(tmp);
            ++i_req;
        }
        bool success = false;
        auto resp = sparse_bitmap_mem.write_resp.read(success);
        if(success){
            i_resp += unsigned(resp)+1;
        }
    }
}

void indexer_top(
    const int L,
    const int NUM_QUERY,
    const int THRESHOLD,
    tapa::mmaps<tapa::vec_t<ap_uint<32>, 16>, 8> qk_vec_mem,
    tapa::mmaps<ap_uint<128>, 8> sparse_bitmap_mem
) {
    tapa::streams<tapa::vec_t<ap_uint<32>, 16>, 8> qk_vec_fifo("qk_vec_fifo");
    tapa::streams<ap_uint<128>, 8> sparse_bitmap_fifo("sparse_bitmap_fifo");

    tapa::task()
        .invoke<tapa::join, 8>(
            read_qk_vec, L, NUM_QUERY, qk_vec_mem, qk_vec_fifo
        )
        .invoke<tapa::join, 8>(
            gemm_pe, L, NUM_QUERY, THRESHOLD, qk_vec_fifo, sparse_bitmap_fifo
        )
        .invoke<tapa::join, 8>(
            write_id, L, NUM_QUERY, sparse_bitmap_mem, sparse_bitmap_fifo
        );
       
}

#endif // __INDEXER_V_H__