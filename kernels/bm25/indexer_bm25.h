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

constexpr int VOCAB_SIZE = 65536;
constexpr int VOCAB_SIZE_DIV_16 = VOCAB_SIZE / 16;
constexpr int VOCAB_SIZE_DIV_512 = VOCAB_SIZE / 512;
constexpr int MAX_DOCUMENT_COUNT = 20000000; 
constexpr int DOCUMENT_SIZE = 128;
constexpr int TOP_K = 64;

constexpr float K1 = 1.2f;
constexpr float K1_plus_1 = K1 + 1.0f;
constexpr float B = 0.75f;

template <typename data_t>
inline void bh(tapa::istream<data_t> & q) {
#pragma HLS inline
    for (;;) {
#pragma HLS pipeline II=1
        data_t tmp; q.try_read(tmp);
    }
}

void black_hole_int_vec(tapa::istream<tapa::vec_t<int, 16>>& q) {
    bh(q);
}

void black_hole_ap_uint512(tapa::istream<ap_uint<512>>& q) {
    bh(q);
}

void black_hold_int(tapa::istream<int>& q) {
    bh(q);
}

void read_df(
    tapa::async_mmap<tapa::vec_t<int, 16>>& df_buffer,
    tapa::ostream<tapa::vec_t<int, 16>>& df_fifo
) {
    read: for(int i_req = 0, i_resp = 0; i_resp < VOCAB_SIZE_DIV_16;){
        #pragma HLS pipeline II=1
		if((i_req < VOCAB_SIZE_DIV_16) & !df_buffer.read_addr.full()){
            df_buffer.read_addr.try_write(i_req);
            ++i_req;
		}
		if(!df_buffer.read_data.empty()){
            tapa::vec_t<int, 16> tmp;
            df_buffer.read_data.try_read(tmp);
            df_fifo.write(tmp);
            ++i_resp;
		}
	}
}

void read_query(
    tapa::async_mmap<ap_uint<512>>& query_bitmap_mem,
    tapa::ostream<ap_uint<512>>& query_fifo
) {
    read: for(int i_req = 0, i_resp = 0; i_resp < VOCAB_SIZE_DIV_512;){
        #pragma HLS pipeline II=1
        if((i_req < VOCAB_SIZE_DIV_512) & !query_bitmap_mem.read_addr.full()){
            query_bitmap_mem.read_addr.try_write(i_req);
            ++i_req;
        }
        if(!query_bitmap_mem.read_data.empty()){
            ap_uint<512> tmp;
            query_bitmap_mem.read_data.try_read(tmp);
            query_fifo.write(tmp);
            ++i_resp;
        }
    }
}

void read_inst(
    const int L,
    tapa::async_mmap<int>& inst_mem,
    tapa::ostream<int>& inst_fifo
) {
    read: for(int i_req = 0, i_resp = 0; i_resp < (L >> 6);){
        #pragma HLS pipeline II=1
        if((i_req < (L >> 6)) & !inst_mem.read_addr.full()){
            inst_mem.read_addr.try_write(i_req);
            ++i_req;
        }
        if(!inst_mem.read_data.empty()){
            int tmp;
            inst_mem.read_data.try_read(tmp);
            inst_fifo.write(tmp);
            ++i_resp;
        }
    }
}

void read_doc_dict(
    const int L_doc_total,
    tapa::async_mmap<tapa::vec_t<ap_uint<32>, 16>>& doc_mem,
    tapa::ostream<tapa::vec_t<ap_uint<32>, 16>>& doc_fifo
) {
    read: for(int i_req = 0, i_resp = 0; i_resp < L_doc_total;){
        #pragma HLS pipeline II=1
        if((i_req < L_doc_total) & !doc_mem.read_addr.full()){
            doc_mem.read_addr.try_write(i_req);
            ++i_req;
        }
        if(!doc_mem.read_data.empty()){
            tapa::vec_t<ap_uint<32>, 16> tmp;
            doc_mem.read_data.try_read(tmp);
            doc_fifo.write(tmp);
            ++i_resp;
        }
    }
}

void bm25_core(
    const int L,
    tapa::istream<tapa::vec_t<int, 16>>& df_fifo_in, // load only once
    tapa::ostream<tapa::vec_t<int, 16>>& df_fifo_out,
    tapa::istream<int>& inst_fifo,
    tapa::ostream<int>& inst_fifo_next,
    tapa::ostream<int>& inst_fifo_out,
    tapa::istream<ap_uint<512>>& query_fifo_in,
    tapa::ostream<ap_uint<512>>& query_fifo_out,
    tapa::istream<tapa::vec_t<ap_uint<32>, 16>>& doc_fifo, // 16-bit token id + 8-bit count
    tapa::ostream<tapa::vec_t<int, 16>>& df_t_fifo,
    tapa::ostream<tapa::vec_t<ap_uint<8>, 16>>& f_fifo
) {
    int df_buf[16][VOCAB_SIZE_DIV_16];
    #pragma HLS array_partition variable=df_buf complete dim=1

    load_df: for(int i = 0; i < VOCAB_SIZE_DIV_16; i++){
        #pragma HLS pipeline II=1
        auto df_pack = df_fifo_in.read();
        df_fifo_out.write(df_pack);
        for(int j = 0; j < 16; j++){
            #pragma HLS unroll
            df_buf[j][i] = df_pack[j];
        }
    }

    ap_uint<512> query_bitmap[16][VOCAB_SIZE_DIV_512];
    #pragma HLS array_partition variable=query_bitmap complete dim=1

    load_query: for(int i = 0; i < VOCAB_SIZE_DIV_512; i++){
        #pragma HLS pipeline II=1
        auto query_bit = query_fifo_in.read();
        query_fifo_out.write(query_bit);
        for(int j = 0; j < 16; j++){
            #pragma HLS unroll
            query_bitmap[j][i] = query_bit;
        }
    }

    // load document and process
    for(int r = 0; r < (L >> 6); r++) {
        const int L_doc = inst_fifo.read();
        inst_fifo_next.write(L_doc);
        inst_fifo_out.write(L_doc);

        for(int i = 0; i < L_doc; i++) {
            #pragma HLS pipeline II=1
            auto doc_pack = doc_fifo.read();
            tapa::vec_t<int, 16> df_pack;
            tapa::vec_t<ap_uint<8>, 16> f_pack;
            for(int j = 0; j < 16; j++) {
                #pragma HLS unroll
                ap_uint<32> doc_token_count = doc_pack[j];
                ap_uint<16> token_id = doc_token_count(15, 0);
                ap_uint<8> term_freq = doc_token_count(23, 16);

                // check if in query
                ap_uint<512> check_bit = query_bitmap[j][(token_id >> 9)];

                if (check_bit[ap_uint<9>(token_id(8, 0))] == 1) {
                    f_pack[j] = term_freq;
                } else {
                    f_pack[j] = 0;
                }

                df_pack[j] = df_buf[j][(token_id >> 4)];
            }
            df_t_fifo.write(df_pack);
            f_fifo.write(f_pack);
        }
    }

}


void bm25_idf(
    const int L,
    tapa::istream<tapa::vec_t<int, 16>>& df_fifo,
    tapa::ostream<tapa::vec_t<float, 16>>& idf_fifo
) {
    for(;;){
        #pragma HLS pipeline II=1
        if(!df_fifo.empty()){
            tapa::vec_t<int, 16> df_pack; df_fifo.try_read(df_pack);
            tapa::vec_t<float, 16> idf_pack;
            for(int i = 0; i < 16; i++){
                #pragma HLS unroll
                float idf_num = (float)(L - df_pack[i]) + 0.5f;
                float idf_den = (float)(df_pack[i]) + 0.5f;
                float idf_den_inv = 1.0f / idf_den;
                #pragma HLS bind_op variable=idf_den_inv op=frecip impl=fulldsp
                idf_pack[i] = logf(idf_num * idf_den_inv);
            }
            idf_fifo.write(idf_pack);
        }
    }
}

void bm25_tf_weight(
    tapa::istream<tapa::vec_t<ap_uint<8>, 16>>& f_fifo,
    tapa::ostream<tapa::vec_t<float, 16>>& tf_weight_fifo
) {
    for(;;){
        #pragma HLS pipeline II=1
        if(!f_fifo.empty()){
            tapa::vec_t<ap_uint<8>, 16> f_pack; f_fifo.try_read(f_pack);
            tapa::vec_t<float, 16> tf_weight_pack;
            for(int i = 0; i < 16; i++){
                #pragma HLS unroll
                float f_val = (float)(f_pack[i].to_int());
                float tf_weight_num = f_val * K1_plus_1;
                float tf_weight_den = f_val + K1;
                float tf_weight_den_inv = 1.0f / tf_weight_den;
                #pragma HLS bind_op variable=tf_weight_den_inv op=frecip impl=fulldsp
                tf_weight_pack[i] = tf_weight_num * tf_weight_den_inv;
            }
            tf_weight_fifo.write(tf_weight_pack);
        }
    }
}

void bm25_score_mul(
    tapa::istream<tapa::vec_t<float, 16>>& idf_fifo,
    tapa::istream<tapa::vec_t<float, 16>>& tf_weight_fifo,
    tapa::ostream<tapa::vec_t<float, 16>>& score_psum_fifo
) {
    for(;;){
        #pragma HLS pipeline II=1
        if((!idf_fifo.empty()) & (!tf_weight_fifo.empty())) {
            tapa::vec_t<float, 16> idf_pack; idf_fifo.try_read(idf_pack);
            tapa::vec_t<float, 16> tf_weight_pack; tf_weight_fifo.try_read(tf_weight_pack);
            tapa::vec_t<float, 16> score_pack;
            for(int i = 0; i < 16; i++){
                #pragma HLS unroll
                score_pack[i] = idf_pack[i] * tf_weight_pack[i];
            }
            score_psum_fifo.write(score_pack);
        }
    }
}

void bm25_score_acc(
    const int L,
    tapa::istream<int>& inst_fifo,
    tapa::istream<tapa::vec_t<float, 16>>& score_psum_fifo,
    tapa::ostream<tapa::vec_t<float, 16>>& score_out_fifo
) {
    for(int r = 0; r < (L >> 6); r++) {
        const int L_doc = inst_fifo.read();
        float score_acc[4][16];
        #pragma HLS array_partition variable=score_acc complete dim=1
        #pragma HLS array_partition variable=score_acc complete dim=2
        for(int i = 0; i < 4; i++) {
            #pragma HLS unroll
            for(int j = 0; j < 16; j++) {
                #pragma HLS unroll
                score_acc[i][j] = 0.0f;
            }
        }
        acc: for(int i = 0; i < L_doc; i++) {
            #pragma HLS pipeline II=1
            #pragma HLS dependence variable=score_acc inter distance=4 true
            auto score_pack = score_psum_fifo.read();
            for(int j = 0; j < 16; j++) {
                #pragma HLS unroll
                score_acc[i%4][j] += score_pack[j];
            }
        }

        for(int j = 0; j < 16; j++) {
            #pragma HLS unroll
            score_acc[0][j] += score_acc[1][j];
            score_acc[2][j] += score_acc[3][j];
        }


        // write out score
        tapa::vec_t<float, 16> score_pack;
        for(int i = 0; i < 16; i++) {
            #pragma HLS unroll
            score_pack[i] = score_acc[0][i] + score_acc[2][i];
        }
        
        score_out_fifo.write(score_pack);
    }
}

void merge_score(
    tapa::istreams<tapa::vec_t<float, 16>, 4>& score_in_fifos,
    tapa::ostream<tapa::vec_t<float, 16>>& score_out_fifo
) {
    for(;;){
        for(int i = 0; i < 4; i++){
            #pragma HLS pipeline II=1
            auto score_pack = score_in_fifos[i].read();
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

    const int remain_len = (L >> 4) - (TOP_K >> 4);
    
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

// NUM_QUERY = 1 for now
void indexer_top(
    const int L,
    const int L_doc_total,
    tapa::mmap<tapa::vec_t<int, 16>> df_buffer,
    tapa::mmap<ap_uint<512>> query_bitmap_mem,
    tapa::mmap<int> inst_mem,
    tapa::mmaps<tapa::vec_t<ap_uint<32>, 16>, 4> doc_mem,
    tapa::mmap<tapa::vec_t<int, 16>> topk_id_mem
) {
    tapa::streams<tapa::vec_t<int, 16>, 5> df_fifo("df_fifo");
    tapa::streams<int, 5> inst_fifo("inst_fifo");
    tapa::streams<int, 4> inst_fifo_out("inst_fifo_out");
    tapa::streams<ap_uint<512>, 5> query_fifo("query_fifo");
    tapa::streams<tapa::vec_t<ap_uint<32>, 16>, 4> doc_fifo("doc_fifo");
    tapa::streams<tapa::vec_t<int, 16>, 4> df_t_fifo("df_t_fifo");
    tapa::streams<tapa::vec_t<ap_uint<8>, 16>, 4> f_fifo("f_fifo");
    tapa::streams<tapa::vec_t<float, 16>, 4> idf_fifo("idf_fifo");
    tapa::streams<tapa::vec_t<float, 16>, 4> tf_weight_fifo("tf_weight_fifo");
    tapa::streams<tapa::vec_t<float, 16>, 4> score_psum_fifo("score_psum_fifo");
    tapa::streams<tapa::vec_t<float, 16>, 4, 4> score_out_fifo("score_out_fifo");
    tapa::stream<tapa::vec_t<float, 16>> score_merged_fifo("score_merged_fifo");
    tapa::stream<tapa::vec_t<int, 16>> topk_id_fifo("topk_id_fifo");

    tapa::task()
        .invoke<tapa::join>(
            read_df, df_buffer, df_fifo
        )
        .invoke<tapa::join>(
            read_query, query_bitmap_mem, query_fifo
        )
        .invoke<tapa::join>(
            read_inst, L, inst_mem, inst_fifo
        )
        .invoke<tapa::join, 4>(
            read_doc_dict, L_doc_total, doc_mem, doc_fifo
        )
        .invoke<tapa::join, 4>(
            bm25_core, L, df_fifo, df_fifo, inst_fifo, inst_fifo, inst_fifo_out, query_fifo, query_fifo, doc_fifo, df_t_fifo, f_fifo
        )
        .invoke<tapa::detach>(black_hole_int_vec, df_fifo)
        .invoke<tapa::detach>(black_hole_ap_uint512, query_fifo)
        .invoke<tapa::detach>(black_hold_int, inst_fifo)
        .invoke<tapa::detach, 4>(
            bm25_idf, L, df_t_fifo, idf_fifo
        )
        .invoke<tapa::detach, 4>(
            bm25_tf_weight, f_fifo, tf_weight_fifo
        )
        .invoke<tapa::detach, 4>(
            bm25_score_mul, idf_fifo, tf_weight_fifo, score_psum_fifo
        )
        .invoke<tapa::join, 4>(
            bm25_score_acc, L, inst_fifo_out, score_psum_fifo, score_out_fifo
        )
        .invoke<tapa::detach>(
            merge_score, score_out_fifo, score_merged_fifo
        )
        .invoke<tapa::join>(
            topk_parallel_cmp, L, 1, score_merged_fifo, topk_id_fifo
        )
        .invoke<tapa::join>(
            write_id, 1, topk_id_mem, topk_id_fifo
        );
       
}

#endif // __INDEXER_V_H__