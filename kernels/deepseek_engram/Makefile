# Makefile for Engram GPU-FPGA Heterogeneous Implementation
#
# Build targets:
#   make fpga_test     - Build FPGA testbench (TAPA C-simulation)
#   make gpu_test      - Build GPU kernel testbench (HIP)
#   make host          - Build end-to-end host (XRT + HIP)
#   make xclbin        - Build FPGA hardware (Vitis, takes hours)
#   make clean         - Clean build artifacts

# ============================================================================
# Configuration
# ============================================================================

PLATFORM := xilinx_u55c_gen3x16_xdma_3_202210_1

# Sequence length passed to GPU test (override: make gpu_test L=1024)
L ?= 256

# Tool paths
XILINX_VITIS ?= /opt/xilinx/Vitis/2024.2
XILINX_XRT ?= /opt/xilinx/xrt
ROCM_PATH ?= /opt/rocm
TAPA_PATH ?= $(shell which tapa 2>/dev/null | xargs dirname 2>/dev/null | xargs dirname 2>/dev/null)

# Compilers
CXX := g++
TOOL := tapa
HIPCC := hipcc
VPP := $(XILINX_VITIS)/bin/v++

BUILD_DIR := build

# ============================================================================
# Flags
# ============================================================================

CXXFLAGS := -std=c++17

# XRT
XRT_INC := -I$(XILINX_XRT)/include
XRT_LIB := -L$(XILINX_XRT)/lib -lxrt_coreutil -luuid -lpthread

# HIP
HIP_INC := -I$(ROCM_PATH)/include
HIP_LIB := -L$(ROCM_PATH)/lib -lamdhip64

# TAPA — derive prefix from the binary location (bundles gflags, glog, etc.)
TAPA_PREFIX := $(shell which tapa 2>/dev/null | xargs dirname 2>/dev/null | xargs dirname 2>/dev/null)

# HLS include directory (Vitis HLS headers: ap_int.h, hls_math.h, etc.)
HLS_INC_DIR := /mnt/software/xilinx/Vitis/2024.2/include

# HIP compile + link flags (mirrors the reference hip_csim pattern)
ROCM_LIB := $(ROCM_PATH)/lib
HIP_FLAGS := -std=c++17 -Wno-attributes -Wno-unknown-pragmas -Wno-unused-label \
             -isystem $(TAPA_PREFIX)/include \
             -isystem $(HLS_INC_DIR)
HIP_LINK  := -L $(TAPA_PREFIX)/lib -Wl,-rpath,$(TAPA_PREFIX)/lib \
             -ltapa -lcontext -lfrt -lglog -lgflags \
             -lOpenCL -ltinyxml2 -lyaml-cpp -lthread \
             -L$(ROCM_LIB) -lamdhip64

# Legacy flags (kept for non-HIP targets)
TAPA_INC  := -isystem $(TAPA_PREFIX)/include
TAPA_LIB  := -L $(TAPA_PREFIX)/lib -Wl,-rpath,$(TAPA_PREFIX)/lib \
             -ltapa -lcontext -lfrt -lglog -lgflags \
             -lOpenCL -ltinyxml2 -lyaml-cpp -lthread
GFLAGS_INC :=
GFLAGS_LIB :=

# ============================================================================
# Targets
# ============================================================================

.PHONY: all fpga_test gpu_test gpu_fpga_test host xclbin clean help

all: fpga_test gpu_fpga_test

help:
	@echo "Engram GPU-FPGA Implementation - Build System"
	@echo ""
	@echo "Targets:"
	@echo "  make fpga_test      - FPGA testbench (TAPA C-simulation)"
	@echo "  make gpu_fpga_test  - End-to-end FPGA+GPU testbench"
	@echo "  make gpu_test       - GPU-only kernel testbench (HIP)"
	@echo "  make host           - End-to-end host (XRT + HIP)"
	@echo "  make xclbin         - FPGA hardware synthesis (hours)"
	@echo "  make clean          - Clean build artifacts"
	@echo ""
	@echo "Run:"
	@echo "  ./$(BUILD_DIR)/engram_tb                           # FPGA CSIM"
	@echo "  ./$(BUILD_DIR)/engram_tb --bitstream=engram.xclbin # FPGA HW"
	@echo "  ./$(BUILD_DIR)/engram_gpu_fpga_tb                  # E2E csim"
	@echo "  ./$(BUILD_DIR)/engram_gpu_fpga_tb --bitstream=engram.xclbin  # E2E HW"
	@echo "  ./$(BUILD_DIR)/engram_host --xclbin=engram.xclbin  # XRT E2E"
	@echo ""
	@echo "GPU test with custom L:"
	@echo "  make gpu_test L=1024"
	@echo "  ./$(BUILD_DIR)/engram_gpu_test 1024"
	@echo "  ./$(BUILD_DIR)/engram_gpu_test --L=1024"

# ============================================================================
# FPGA Testbench (TAPA C-simulation)
# ============================================================================

TOP := engram_fpga_top

$(BUILD_DIR)/engram_tb: engram_tb.cpp engram.h
	@mkdir -p $(BUILD_DIR)
	$(TOOL) g++ -- $< -o $@ -isystem$(HLS_INC_DIR)

fpga_test: $(BUILD_DIR)/engram_tb
	@echo "=== Running FPGA testbench (CSIM) ==="
	./$(BUILD_DIR)/engram_tb 8

# ============================================================================
# End-to-End GPU-FPGA Testbench (primary test target)
# ============================================================================

$(BUILD_DIR)/engram_gpu_fpga_tb: engram_gpu_fpga_tb.cpp engram_kernels.hip
	@mkdir -p $(BUILD_DIR)
	$(HIPCC) $(HIP_FLAGS) engram_gpu_fpga_tb.cpp engram_kernels.hip $(HIP_LINK) -o $@

gpu_fpga_test: $(BUILD_DIR)/engram_gpu_fpga_tb
	@echo "=== Running GPU-FPGA end-to-end testbench (csim) ==="
	./$(BUILD_DIR)/engram_gpu_fpga_tb

gpu_fpga_test_hw: $(BUILD_DIR)/engram_gpu_fpga_tb
	@echo "=== Running GPU-FPGA end-to-end testbench (csim) ==="
	./$(BUILD_DIR)/engram_gpu_fpga_tb --bitstream=engram_v1_fixed.xclbin

# ============================================================================
# GPU Kernel Testbench (legacy, separate GPU-only test)
# ============================================================================

$(BUILD_DIR)/engram_gpu_test: engram_gpu_test.cpp engram_kernels.hip
	@mkdir -p $(BUILD_DIR)
	$(HIPCC) -std=c++17 -O2 -g -o $@ engram_gpu_test.cpp engram_kernels.hip

gpu_test: $(BUILD_DIR)/engram_gpu_test
	@echo "=== Running GPU kernel testbench (L=$(L)) ==="
	./$(BUILD_DIR)/engram_gpu_test $(L)

# ============================================================================
# End-to-End Host (XRT + HIP)
# ============================================================================

$(BUILD_DIR)/engram_host: engram_host.cpp engram_kernels.hip
	@mkdir -p $(BUILD_DIR)
	$(HIPCC) -std=c++17 -O2 -g $(XRT_INC) $(HIP_INC) \
		-o $@ engram_host.cpp engram_kernels.hip \
		$(XRT_LIB) $(HIP_LIB)

host: $(BUILD_DIR)/engram_host

# ============================================================================
# FPGA Hardware Synthesis (Vitis)
# ============================================================================

# Step 1: Synthesize kernel to XO using TAPA
hls-engram: engram.h
	$(TOOL) compile --top $(TOP) --platform xilinx_u55c_gen3x16_xdma_3_202210_1 -f $< -o $(TOP).xo

# Step 2: Link XO to XCLBIN
$(BUILD_DIR)/engram.xclbin: $(BUILD_DIR)/engram.xo kernel_config.cfg
	$(VPP) --link --target hw \
		--platform $(PLATFORM) \
		--config kernel_config.cfg \
		--output $@ \
		$(BUILD_DIR)/engram.xo

xclbin: $(BUILD_DIR)/engram.xclbin

# ============================================================================
# Clean
# ============================================================================

clean:
	rm -rf $(BUILD_DIR)
	rm -f *.log *.jou
	rm -rf _x .Xil
