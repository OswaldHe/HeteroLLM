TOOL=tapa
TOP=indexer_top

# TAPA and HLS include paths
FLAGS=-std=c++17 -O3 -Wno-attributes -Wno-unknown-pragmas -Wno-unused-label \
      -isystem /home/oswaldhe/.rapidstream-tapa/usr/include \
      -isystem /mnt/software/xilinx/Vitis_HLS/2023.2/include

# Link libraries for TAPA runtime
LINK_OPS=-L /home/oswaldhe/.rapidstream-tapa/usr/lib \
         -Wl,-rpath,/home/oswaldhe/.rapidstream-tapa/usr/lib \
         -ltapa -lcontext -lfrt -lglog -lgflags -lOpenCL -ltinyxml2 -lyaml-cpp -lpthread

# XRT paths (set XILINX_XRT environment variable or modify path below)
XILINX_XRT ?= /opt/xilinx/xrt
XRT_INCLUDE = -I$(XILINX_XRT)/include
XRT_LIB = -L$(XILINX_XRT)/lib -Wl,-rpath,$(XILINX_XRT)/lib -lxrt_coreutil -luuid

# Default target
all: bm25_loader bm25_loader_xrt

# Build BM25 loader with TAPA kernel
bm25_loader: bm25_loader.cpp indexer_bm25.h
	$(TOOL) g++ -- $< -o $@ -pthread -O3

# Build BM25 loader with XRT native APIs
bm25_loader_xrt: bm25_loader_xrt.cpp
	g++ -g -std=c++17 -O3 $(XRT_INCLUDE) $< -o $@ $(XRT_LIB) -lpthread

# C-simulation only (no FPGA bitstream) - TAPA version
csim: bm25_loader
	./bm25_loader --export_dir=./export --use_mmap=true --num_threads=8

# Run with FPGA bitstream - TAPA version
run: bm25_loader
	./bm25_loader --export_dir=./export --use_mmap=true --num_threads=8 \
		--bitstream=../indexer_bm25.xclbin --query_tokens="100,200,300"

# Run with FPGA bitstream - XRT version
run_xrt: bm25_loader_xrt
	./bm25_loader_xrt --export_dir=./export --use_mmap=true --num_threads=8 \
		--bitstream=../indexer_bm25.xclbin --query_tokens="100,200,300"

# Run with FPGA bitstream - PyXRT version (Python)
run_pyxrt:
	LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 PYTHONPATH=$(XILINX_XRT)/python:$$PYTHONPATH python3 bm25_loader_xrt.py \
		--export_dir=./export --use_mmap=true --num_threads=8 \
		--bitstream=../indexer_bm25.xclbin

# Copy indexer_bm25.h from parent directory if needed
indexer_bm25.h: ../indexer_bm25.h
	cp $< $@

clean:
	rm -f bm25_loader bm25_loader_xrt indexer_bm25.h

.PHONY: all csim run run_xrt clean
