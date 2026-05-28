TOOL=tapa
HIP_TOOL=hipcc
ROCM_LIB=/opt/rocm/lib
TOP=indexer_top
FLAGS=-std=c++17 -Wno-attributes -Wno-unknown-pragmas -Wno-unused-label -isystem /home/oswaldhe/.rapidstream-tapa/usr/include -isystem /mnt/software/xilinx/Vitis_HLS/2023.2/include 
LINK_OPS=-L /home/oswaldhe/.rapidstream-tapa/usr/lib -Wl,-rpath,/home/oswaldhe/.rapidstream-tapa/usr/lib -ltapa -lcontext -lfrt -lglog -lgflags -lOpenCL -ltinyxml2 -lyaml-cpp -lthread -L$(ROCM_LIB) -lrocblas


csim: indexer_vanilla.h indexer_tb.cpp
	$(TOOL) g++ -- $^ -o indexer_vanilla

csim_lut: indexer_lut.h indexer_lut_tb.cpp
	$(TOOL) g++ -- $^ -o indexer_lut

csim_seer_threshold: indexer_seer_threshold_tb.cpp
	$(TOOL) g++ -- $< -o indexer_seer_threshold

csim_seer_token_budget: indexer_seer_token_budget_tb.cpp
	$(TOOL) g++ -- $< -o indexer_seer_token_budget

csim_lserve: indexer_lserve_tb.cpp
	$(TOOL) g++ -- $< -o indexer_lserve

csim_lserve_long: indexer_lserve_long_tb.cpp
	$(TOOL) g++ -- $< -o indexer_lserve_long

csim_bm25: indexer_bm25_tb.cpp
	$(TOOL) g++ -- $< -o indexer_bm25

hip_csim: indexer_hip.cpp
	$(HIP_TOOL) $(FLAGS) $^ $(LINK_OPS) -o indexer_vanilla_hip

hip_csim_lut: indexer_lut_hip.cpp
	$(HIP_TOOL) $(FLAGS) $^ $(LINK_OPS) -o indexer_lut_hip

simple: simplist.h indexer_tb.cpp
	$(TOOL) g++ -- $^ -o simplist

hls: indexer_vanilla.h
	$(TOOL) compile --top $(TOP) --part-num xcv80-lsva4737-2MHP-e-S --clock-period 3.33 -f $< -o $(TOP).xo

hls-u55c: indexer_vanilla.h
	$(TOOL) compile --top $(TOP) --platform xilinx_u55c_gen3x16_xdma_3_202210_1 -f $< -o $(TOP).xo

hls-u55c-lut: indexer_lut.h
	$(TOOL) compile --top $(TOP) --platform xilinx_u55c_gen3x16_xdma_3_202210_1 -f $< -o $(TOP).xo

hls-u55c-seer-threshold: indexer_seer_threshold.h
	$(TOOL) compile --top $(TOP) --platform xilinx_u55c_gen3x16_xdma_3_202210_1 -f $< -o $(TOP).xo

hls-u55c-seer-token-budget: indexer_seer_token_budget.h
	$(TOOL) compile --top $(TOP) --platform xilinx_u55c_gen3x16_xdma_3_202210_1 -f $< -o $(TOP).xo

hls-u55c-lserve: indexer_lserve.h
	$(TOOL) compile --top $(TOP) --platform xilinx_u55c_gen3x16_xdma_3_202210_1 -f $< -o $(TOP).xo

hls-u55c-lserve-long: indexer_lserve_long.h
	$(TOOL) compile --top $(TOP) --platform xilinx_u55c_gen3x16_xdma_3_202210_1 -f $< -o $(TOP).xo

hls-u55c-bm25: indexer_bm25.h
	$(TOOL) compile --top $(TOP) --platform xilinx_u55c_gen3x16_xdma_3_202210_1 -f $< -o $(TOP).xo

clean:
	rm -f indexer_vanilla indexer_lut indexer_seer_threshold indexer_seer_token_budget indexer_lserve indexer_bm25 *.xo work.out