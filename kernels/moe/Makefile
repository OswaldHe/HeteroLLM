# Makefile for MoE FPGA Kernel (DeepSeek V3.2 Routed Experts)
#
# Build targets:
#   make csim           - Build and run TAPA C-simulation testbench (opt3 baseline)
#   make hls            - Synthesize kernel to XO using TAPA (opt3 baseline)
#   make csim_dec_spec  - Build and run TAPA C-simulation testbench (dec_spec variant)
#   make hls_dec_spec   - Synthesize dec_spec kernel to XO using TAPA
#   make clean          - Clean build artifacts

# ============================================================================
# Configuration
# ============================================================================

PLATFORM := xilinx_u55c_gen3x16_xdma_3_202210_1

# Tool paths
XILINX_VITIS ?= /opt/xilinx/Vitis/2024.2
TAPA_PATH ?= $(shell which tapa 2>/dev/null | xargs dirname 2>/dev/null | xargs dirname 2>/dev/null)

# Compilers
CXX := g++
TOOL := tapa

BUILD_DIR := build

# ============================================================================
# Flags
# ============================================================================

CXXFLAGS := -std=c++17

TAPA_PREFIX := $(shell which tapa 2>/dev/null | xargs dirname 2>/dev/null | xargs dirname 2>/dev/null)

# HLS include dir (for ap_int.h, hls_math.h, etc.)
HLS_INC_DIR := /mnt/software/xilinx/Vitis/2024.2/include

# ============================================================================
# Targets
# ============================================================================

.PHONY: all csim hls csim_dec_spec hls_dec_spec csim_int8_decode hls_int8_decode csim_batch_16b hls_batch_16b csim_batch_16b_2ch hls_batch_16b_2ch clean help

all: csim

help:
	@echo "MoE FPGA Kernel - Build System"
	@echo ""
	@echo "Targets:"
	@echo "  make csim           - Build and run TAPA C-simulation testbench (opt3 baseline)"
	@echo "  make hls            - Synthesize kernel to XO (opt3 baseline)"
	@echo "  make csim_dec_spec  - Build and run TAPA C-simulation testbench (dec_spec variant)"
	@echo "  make hls_dec_spec   - Synthesize dec_spec kernel to XO (TAPA compile)"
	@echo "  make clean          - Clean build artifacts"
	@echo ""
	@echo "CSIM flags (override N):"
	@echo "  ./$(BUILD_DIR)/moe_test --N=4"
	@echo "  ./$(BUILD_DIR)/moe_test_dec_spec  (decode-only: always 1 token)"

# ============================================================================
# FPGA Testbench (TAPA C-simulation)
# ============================================================================

TOP := moe_fpga_top

$(BUILD_DIR)/moe_test: moe_test.cpp moe_kernel.cpp moe_kernel.h
	@mkdir -p $(BUILD_DIR)
	$(TOOL) g++ -- moe_test.cpp moe_kernel.cpp -o $@ -isystem$(HLS_INC_DIR)

csim: $(BUILD_DIR)/moe_test
	@echo "=== Running MoE FPGA testbench (CSIM) ==="
	./$(BUILD_DIR)/moe_test --N=4

# ============================================================================
# FPGA Hardware Synthesis (TAPA compile → XO)
# ============================================================================

hls: moe_kernel.cpp moe_kernel.h
	$(TOOL) compile --top $(TOP) --platform $(PLATFORM) \
		-f moe_kernel.cpp -o $(TOP).xo

# ============================================================================
# dec_spec variant (Opt 4+: W1/W3 channel sharing with W2, further opt)
# ============================================================================

$(BUILD_DIR)/moe_test_dec_spec: moe_test_dec_spec.cpp moe_kernel_dec_spec.cpp moe_kernel_dec_spec.h
	@mkdir -p $(BUILD_DIR)
	$(TOOL) g++ -- moe_test_dec_spec.cpp moe_kernel_dec_spec.cpp -o $@ -isystem$(HLS_INC_DIR)

csim_dec_spec: $(BUILD_DIR)/moe_test_dec_spec
	@echo "=== Running MoE FPGA dec_spec testbench (CSIM) ==="
	./$(BUILD_DIR)/moe_test_dec_spec

hls_dec_spec: moe_kernel_dec_spec.cpp moe_kernel_dec_spec.h
	$(TOOL) compile --top $(TOP) --platform $(PLATFORM) \
		-f moe_kernel_dec_spec.cpp -o $(TOP)_dec_spec.xo

# ============================================================================
# int8_decode variant (Opt 10: INT8 weights, INT16 activations)
# ============================================================================

$(BUILD_DIR)/moe_test_int8_decode: moe_test_int8_decode.cpp moe_int8_decode.cpp moe_int8_decode.h
	@mkdir -p $(BUILD_DIR)
	$(TOOL) g++ -- moe_test_int8_decode.cpp moe_int8_decode.cpp -o $@ -isystem$(HLS_INC_DIR)

csim_int8_decode: $(BUILD_DIR)/moe_test_int8_decode
	@echo "=== Running MoE FPGA int8_decode testbench (CSIM) ==="
	./$(BUILD_DIR)/moe_test_int8_decode

hls_int8_decode: moe_int8_decode.cpp moe_int8_decode.h
	$(TOOL) compile --top $(TOP) --platform $(PLATFORM) \
		-f moe_int8_decode.cpp -o $(TOP)_int8.xo

# ============================================================================
# batch_16b variant (INT16 batched-token kernel, derived from dec_spec / Opt 9)
# ============================================================================

$(BUILD_DIR)/moe_test_batch_16b: moe_test_batch_16b.cpp moe_batch_16b.cpp moe_batch_16b.h
	@mkdir -p $(BUILD_DIR)
	$(TOOL) g++ -- moe_test_batch_16b.cpp moe_batch_16b.cpp -o $@ -isystem$(HLS_INC_DIR)

N ?= 8
csim_batch_16b: $(BUILD_DIR)/moe_test_batch_16b
	@echo "=== Running MoE FPGA batch_16b testbench (CSIM, N=$(N)) ==="
	./$(BUILD_DIR)/moe_test_batch_16b --N=$(N)

hls_batch_16b: moe_batch_16b.cpp moe_batch_16b.h
	$(TOOL) compile --top $(TOP) --platform $(PLATFORM) \
		-f moe_batch_16b.cpp -o $(TOP)_batch_16b.xo

# ============================================================================
# batch_16b_2ch variant (2-way row interleaving: drops W*_W2c HBM ports)
# ============================================================================
$(BUILD_DIR)/moe_test_batch_16b_2ch: moe_test_batch_16b_2ch.cpp moe_batch_16b_2ch.cpp moe_batch_16b_2ch.h
	@mkdir -p $(BUILD_DIR)
	$(TOOL) g++ -- moe_test_batch_16b_2ch.cpp moe_batch_16b_2ch.cpp -o $@ -isystem$(HLS_INC_DIR)

csim_batch_16b_2ch: $(BUILD_DIR)/moe_test_batch_16b_2ch
	@echo "=== Running MoE FPGA batch_16b_2ch testbench (CSIM, N=$(N)) ==="
	./$(BUILD_DIR)/moe_test_batch_16b_2ch --N=$(N)

hls_batch_16b_2ch: moe_batch_16b_2ch.cpp moe_batch_16b_2ch.h
	$(TOOL) compile --top $(TOP) --platform $(PLATFORM) \
		-f moe_batch_16b_2ch.cpp -o $(TOP)_batch_16b_2ch.xo

# ============================================================================
# Clean
# ============================================================================

clean:
	rm -rf $(BUILD_DIR)
	rm -f *.log *.jou $(TOP).xo
	rm -rf _x .Xil
