# Makefile for FPGA-GPU P2P Transfer Demo
#
# Build targets:
#   make simple        - Build simple demo (no xclbin needed) - RECOMMENDED
#   make host          - Build host application (requires xclbin)
#   make fpga_kernel   - Build FPGA xclbin (hardware, takes hours)
#   make clean         - Clean build artifacts

# ============================================================================
# Configuration
# ============================================================================

# FPGA Platform
PLATFORM := xilinx_u55c_gen3x16_xdma_3_202210_1

# Tool paths (adjust if needed)
XILINX_VITIS ?= /opt/xilinx/Vitis/2024.2
XILINX_XRT ?= /opt/xilinx/xrt
ROCM_PATH ?= /opt/rocm

# Compilers
CXX := g++
HIPCC := $(ROCM_PATH)/bin/hipcc
VPP := $(XILINX_VITIS)/bin/v++

# Build configuration
TARGET ?= hw
BUILD_DIR := build_$(TARGET)

# ============================================================================
# Flags
# ============================================================================

# Common C++ flags
CXXFLAGS := -std=c++17 -O2 -g -Wall

# XRT include/lib (with OpenCL support)
XRT_INC := -I$(XILINX_XRT)/include -I$(XILINX_XRT)/include/CL
XRT_LIB := -L$(XILINX_XRT)/lib -lxrt_coreutil -lxilinxopencl -luuid -lpthread

# ROCm/HIP include/lib
HIP_INC := -I$(ROCM_PATH)/include
HIP_LIB := -L$(ROCM_PATH)/lib -lamdhip64

# Combined flags for host
HOST_INC := $(XRT_INC) $(HIP_INC)
HOST_LIB := $(XRT_LIB) $(HIP_LIB)

# ============================================================================
# Source files
# ============================================================================

HOST_SRCS := host.cpp
GPU_SRCS := gpu_kernel.hip
FPGA_SRCS := fpga_kernel.cpp
FPGA_INDEX_SRCS := fpga_index_generator.cpp

# ============================================================================
# Output files
# ============================================================================

HOST_EXE := p2p_transfer
SIMPLE_EXE := p2p_simple
SPMV_EXE := spmv_p2p_demo
SPMV_BM25_EXE := spmv_bm25_demo
GPU_OBJ := $(BUILD_DIR)/gpu_kernel.o

# FPGA kernel outputs - one XO per kernel
XO_WRITE := $(BUILD_DIR)/fpga_write_pattern.xo
XO_READ := $(BUILD_DIR)/fpga_read_verify.xo
XO_COPY := $(BUILD_DIR)/fpga_memcpy.xo
XO_INDEX_GEN := $(BUILD_DIR)/fpga_index_generator.xo
XO_INDEX_BURST := $(BUILD_DIR)/fpga_index_generator_burst.xo
FPGA_XCLBIN := $(BUILD_DIR)/p2p_demo.xclbin
SPMV_XCLBIN := $(BUILD_DIR)/spmv_index_gen.xclbin

# ============================================================================
# Build targets
# ============================================================================

.PHONY: all host simple spmv spmv_bm25 gpu_write fpga_verify_csim fpga_verify_xo fpga_verify_cosim fpga_verify_standalone fpga_verify_hw verify_xclbin fpga_kernel fpga_kernel_emu spmv_kernel spmv_kernel_emu clean help

all: simple

help:
	@echo "FPGA-GPU P2P Transfer Demo - Build System"
	@echo ""
	@echo "Targets:"
	@echo "  make simple          - Build simple demo (no xclbin needed) [RECOMMENDED]"
	@echo "  make spmv            - Build SpMV P2P demo (CPU simulates FPGA)"
	@echo "  make spmv_bm25       - Build SpMV with BM25 indexer (uses pre-built kernel)"
	@echo "  make gpu_write       - Build GPU→FPGA P2P write demo"
	@echo "  make fpga_verify_csim       - TAPA software C-sim (needs TAPA C++ runtime)"
	@echo "  make fpga_verify_xo         - TAPA compile: synthesize kernel to XO"
	@echo "  make fpga_verify_cosim      - TAPA fast cosim: RTL sim with XO"
	@echo "  make fpga_verify_standalone  - Standalone verify (no TAPA needed)"
	@echo "  make fpga_verify_hw  - E2E: GPU writes, FPGA verifies (needs xclbin)"
	@echo "  make verify_xclbin   - Synthesize fpga_verify xclbin (hours)"
	@echo "  make spmv_kernel     - Build FPGA index generator kernel (2-4 hours)"
	@echo "  make spmv_kernel_emu - Build FPGA kernel for emulation (~30 min)"
	@echo "  make host            - Build full host application (needs xclbin)"
	@echo "  make fpga_kernel     - Build FPGA xclbin (takes 2-4 hours)"
	@echo "  make fpga_kernel_emu - Build for hardware emulation (~30 min)"
	@echo "  make clean           - Clean build artifacts"
	@echo ""
	@echo "Quick start:"
	@echo "  source env.sh && make simple && ./p2p_simple --fpga 81:00.1"
	@echo "  source env.sh && make spmv && ./spmv_p2p_demo"

# ============================================================================
# Build directory
# ============================================================================

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)
	mkdir -p $(BUILD_DIR)/temp
	mkdir -p $(BUILD_DIR)/logs
	mkdir -p $(BUILD_DIR)/reports

# ============================================================================
# Host applications
# ============================================================================

simple: $(SIMPLE_EXE)

host: $(HOST_EXE)

$(SIMPLE_EXE): p2p_simple.cpp | $(BUILD_DIR)
	$(HIPCC) $(CXXFLAGS) $(HOST_INC) $< $(HOST_LIB) -o $@
	@echo ""
	@echo "====================================================="
	@echo "Simple P2P demo built: $(SIMPLE_EXE)"
	@echo "Run with: ./$(SIMPLE_EXE) --fpga 81:00.1 --gpu 0"
	@echo "====================================================="

spmv: $(SPMV_EXE)

$(SPMV_EXE): spmv_p2p_demo.cpp gpu_spmv.hip | $(BUILD_DIR)
	$(HIPCC) $(CXXFLAGS) $(HOST_INC) spmv_p2p_demo.cpp gpu_spmv.hip $(HOST_LIB) -o $@
	@echo ""
	@echo "====================================================="
	@echo "SpMV P2P demo built: $(SPMV_EXE)"
	@echo "Run with: ./$(SPMV_EXE)"
	@echo "====================================================="

spmv_bm25: $(SPMV_BM25_EXE)

$(SPMV_BM25_EXE): spmv_bm25_demo.cpp gpu_spmv.hip | $(BUILD_DIR)
	$(HIPCC) $(CXXFLAGS) $(HOST_INC) spmv_bm25_demo.cpp gpu_spmv.hip $(HOST_LIB) -o $@
	@echo ""
	@echo "====================================================="
	@echo "SpMV BM25 demo built: $(SPMV_BM25_EXE)"
	@echo "Run with: ./$(SPMV_BM25_EXE)"
	@echo "Note: Requires indexer_bm25_fixed.xclbin in current directory"
	@echo "====================================================="

# GPU→FPGA P2P demo
GPU_WRITE_EXE := gpu_to_fpga_demo

gpu_write: $(GPU_WRITE_EXE)

$(GPU_WRITE_EXE): gpu_to_fpga_demo.cpp gpu_write_kernels.hip | $(BUILD_DIR)
	$(HIPCC) $(CXXFLAGS) $(HOST_INC) gpu_to_fpga_demo.cpp gpu_write_kernels.hip $(HOST_LIB) -o $@
	@echo ""
	@echo "====================================================="
	@echo "GPU→FPGA demo built: $(GPU_WRITE_EXE)"
	@echo "Run with: ./$(GPU_WRITE_EXE)"
	@echo "====================================================="

# --------------------------------------------------------------------------
# FPGA Verification Kernel (TAPA)
# --------------------------------------------------------------------------

# TAPA — derive prefix from the binary location (bundles gflags, glog, etc.)
TAPA_PREFIX ?= $(shell which tapa 2>/dev/null | xargs dirname 2>/dev/null | xargs dirname 2>/dev/null)
TAPA_TOOL := tapa
HLS_INC_DIR ?= $(XILINX_VITIS)/include

TAPA_INC := -isystem $(TAPA_PREFIX)/include
TAPA_LIB := -L $(TAPA_PREFIX)/lib -Wl,-rpath,$(TAPA_PREFIX)/lib \
            -ltapa -lfrt -lglog -lgflags

FPGA_VERIFY_CSIM_EXE := fpga_verify_tb
FPGA_VERIFY_STANDALONE_EXE := fpga_verify_standalone_tb
FPGA_VERIFY_HW_EXE := gpu_to_fpga_verify
FPGA_VERIFY_XO := fpga_verify_top.xo
FPGA_VERIFY_XCLBIN := fpga_verify.xclbin

# --- Stage 0: Software C-simulation (TAPA C++ runtime) ---
fpga_verify_csim: $(FPGA_VERIFY_CSIM_EXE)
	@echo "Run: ./$(FPGA_VERIFY_CSIM_EXE) --count 1024"

$(FPGA_VERIFY_CSIM_EXE): fpga_verify_tb.cpp fpga_verify_kernel.h
	g++ -std=c++17 -O2 $< -o $@ $(TAPA_INC) -isystem$(HLS_INC_DIR) $(TAPA_LIB)
	@echo "Built TAPA C-sim testbench: $(FPGA_VERIFY_CSIM_EXE)"

# --- Stage 1: Synthesize kernel to XO via TAPA ---
fpga_verify_xo: $(FPGA_VERIFY_XO)

$(FPGA_VERIFY_XO): fpga_verify_kernel.h
	$(TAPA_TOOL) compile \
		--top fpga_verify_top \
		--platform $(PLATFORM) \
		-f $< \
		-o $@
	@echo "Synthesized XO: $(FPGA_VERIFY_XO)"

# --- Stage 2: Fast RTL cosimulation (pass XO as bitstream) ---
fpga_verify_cosim: $(FPGA_VERIFY_CSIM_EXE) $(FPGA_VERIFY_XO)
	./$(FPGA_VERIFY_CSIM_EXE) --bitstream $(FPGA_VERIFY_XO) --count 1024

# --- Stage 3: Link XO to hardware xclbin (takes hours) ---
verify_xclbin: $(FPGA_VERIFY_XCLBIN)

$(FPGA_VERIFY_XCLBIN): $(FPGA_VERIFY_XO)
	$(VPP) --link --target hw --platform $(PLATFORM) \
		--output $@ $<
	@echo "Synthesized xclbin: $(FPGA_VERIFY_XCLBIN)"

# --- Standalone testbench (no TAPA required — plain g++) ---
fpga_verify_standalone: $(FPGA_VERIFY_STANDALONE_EXE)

$(FPGA_VERIFY_STANDALONE_EXE): fpga_verify_standalone_tb.cpp fpga_verify_kernel.h
	g++ -std=c++17 -O2 -o $@ $<
	@echo "Built standalone testbench: $(FPGA_VERIFY_STANDALONE_EXE)"
	@echo "Run: ./$(FPGA_VERIFY_STANDALONE_EXE) [count]"

# --- GPU→FPGA E2E verification (needs fpga_verify.xclbin) ---
fpga_verify_hw: $(FPGA_VERIFY_HW_EXE)

$(FPGA_VERIFY_HW_EXE): gpu_to_fpga_verify.cpp gpu_write_kernels.hip | $(BUILD_DIR)
	$(HIPCC) $(CXXFLAGS) $(HOST_INC) gpu_to_fpga_verify.cpp gpu_write_kernels.hip $(HOST_LIB) -o $@
	@echo "Built GPU→FPGA verify E2E: $(FPGA_VERIFY_HW_EXE)"
	@echo "Run: ./$(FPGA_VERIFY_HW_EXE) --xclbin $(FPGA_VERIFY_XCLBIN) --fpga 81:00.1"

# For the full host app, compile GPU kernels and host together (OpenCL API)
$(HOST_EXE): $(HOST_SRCS) $(GPU_SRCS) | $(BUILD_DIR)
	$(HIPCC) $(CXXFLAGS) $(HOST_INC) -DCL_HPP_TARGET_OPENCL_VERSION=120 -DCL_TARGET_OPENCL_VERSION=120 $(HOST_SRCS) $(GPU_SRCS) $(HOST_LIB) -o $@
	@echo ""
	@echo "====================================================="
	@echo "Full P2P demo built: $(HOST_EXE)"
	@echo "Run with: ./$(HOST_EXE) --fpga 81:00.1 --gpu 0 --xclbin <path/to/your.xclbin>"
	@echo "====================================================="
	@echo ""
	@echo "Host application built: $(HOST_EXE)"

# ============================================================================
# FPGA kernel (Vitis compilation)
# ============================================================================

# Vitis common flags
VPP_COMMON := --platform $(PLATFORM) --target $(TARGET) --save-temps

fpga_kernel: $(FPGA_XCLBIN)
	cp $(FPGA_XCLBIN) p2p_demo.xclbin
	@echo ""
	@echo "====================================================="
	@echo "FPGA bitstream built: p2p_demo.xclbin"
	@echo "====================================================="

fpga_kernel_emu:
	$(MAKE) TARGET=hw_emu fpga_kernel

# Compile each kernel separately (Vitis requires one -k per invocation)
$(XO_WRITE): $(FPGA_SRCS) | $(BUILD_DIR)
	$(VPP) $(VPP_COMMON) -c -k fpga_write_pattern \
		--temp_dir $(BUILD_DIR)/temp \
		--log_dir $(BUILD_DIR)/logs \
		--report_dir $(BUILD_DIR)/reports \
		-o $@ $<

$(XO_READ): $(FPGA_SRCS) | $(BUILD_DIR)
	$(VPP) $(VPP_COMMON) -c -k fpga_read_verify \
		--temp_dir $(BUILD_DIR)/temp \
		--log_dir $(BUILD_DIR)/logs \
		--report_dir $(BUILD_DIR)/reports \
		-o $@ $<

$(XO_COPY): $(FPGA_SRCS) | $(BUILD_DIR)
	$(VPP) $(VPP_COMMON) -c -k fpga_memcpy \
		--temp_dir $(BUILD_DIR)/temp \
		--log_dir $(BUILD_DIR)/logs \
		--report_dir $(BUILD_DIR)/reports \
		-o $@ $<

# Link all XO files to create XCLBIN
$(FPGA_XCLBIN): $(XO_WRITE) $(XO_READ) $(XO_COPY) connectivity.cfg
	$(VPP) $(VPP_COMMON) -l --config connectivity.cfg \
		--temp_dir $(BUILD_DIR)/temp \
		--log_dir $(BUILD_DIR)/logs \
		--report_dir $(BUILD_DIR)/reports \
		-o $@ $(XO_WRITE) $(XO_READ) $(XO_COPY)

# Generate connectivity configuration
connectivity.cfg:
	@echo "[connectivity]" > $@
	@echo "sp=fpga_write_pattern_1.out:HBM[0]" >> $@
	@echo "sp=fpga_read_verify_1.in:HBM[0]" >> $@
	@echo "sp=fpga_read_verify_1.error_count:HBM[1]" >> $@
	@echo "sp=fpga_memcpy_1.in:HBM[0]" >> $@
	@echo "sp=fpga_memcpy_1.out:HBM[1]" >> $@
	@echo "nk=fpga_write_pattern:1" >> $@
	@echo "nk=fpga_read_verify:1" >> $@
	@echo "nk=fpga_memcpy:1" >> $@

# ============================================================================
# SpMV FPGA Kernel (Index Generator)
# ============================================================================

spmv_kernel: $(SPMV_XCLBIN)
	cp $(SPMV_XCLBIN) spmv_index_gen.xclbin
	@echo ""
	@echo "====================================================="
	@echo "SpMV FPGA kernel built: spmv_index_gen.xclbin"
	@echo "====================================================="

spmv_kernel_emu:
	$(MAKE) TARGET=hw_emu spmv_kernel

# Compile index generator kernels
$(XO_INDEX_GEN): $(FPGA_INDEX_SRCS) | $(BUILD_DIR)
	$(VPP) $(VPP_COMMON) -c -k fpga_index_generator \
		--temp_dir $(BUILD_DIR)/temp \
		--log_dir $(BUILD_DIR)/logs \
		--report_dir $(BUILD_DIR)/reports \
		-o $@ $<

$(XO_INDEX_BURST): $(FPGA_INDEX_SRCS) | $(BUILD_DIR)
	$(VPP) $(VPP_COMMON) -c -k fpga_index_generator_burst \
		--temp_dir $(BUILD_DIR)/temp \
		--log_dir $(BUILD_DIR)/logs \
		--report_dir $(BUILD_DIR)/reports \
		-o $@ $<

# Link index generator kernels
$(SPMV_XCLBIN): $(XO_INDEX_GEN) $(XO_INDEX_BURST) kernel_config.cfg
	$(VPP) $(VPP_COMMON) -l --config kernel_config.cfg \
		--temp_dir $(BUILD_DIR)/temp \
		--log_dir $(BUILD_DIR)/logs \
		--report_dir $(BUILD_DIR)/reports \
		-o $@ $(XO_INDEX_GEN) $(XO_INDEX_BURST)

# ============================================================================
# Clean
# ============================================================================

clean:
	rm -rf $(HOST_EXE) $(SIMPLE_EXE) $(SPMV_EXE) $(SPMV_BM25_EXE) $(GPU_WRITE_EXE) $(FPGA_VERIFY_CSIM_EXE) $(FPGA_VERIFY_HW_EXE)
	rm -rf build_hw build_hw_emu build_sw_emu
	rm -rf connectivity.cfg emconfig.json
	rm -rf *.xclbin *.xo *.log *.info
	rm -rf _x .Xil .run *.ltx
	rm -rf *.csv *.protoinst *.wdb *.wcfg

.PHONY: clean
