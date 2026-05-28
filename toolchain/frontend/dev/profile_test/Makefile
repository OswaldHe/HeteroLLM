# Makefile for profile tests
# This Makefile generates profiled kernel files and compiles test files

CXX := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -O2 -g
INCLUDES := -I../.. -I../../..
PYTHON := python3

# Directory paths
STEPS_DIR := ../steps
KERNEL_DIR := $(STEPS_DIR)/example_kernel
BUILD_DIR := build
GENERATOR := generate_profile.py

# Default target: build all profile tests
.PHONY: all clean help generate

all: profile_innerproduct profile_paged_kv_index_builder profile_topk_retrieval profile_block_sparse_attention

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Generate profiled kernel from original source
# Usage: make generate KERNEL=inner_product_compute
.PHONY: generate
generate:
ifndef KERNEL
	$(error KERNEL is not set. Usage: make generate KERNEL=inner_product_compute)
endif
	$(PYTHON) $(GENERATOR) $(KERNEL_DIR)/$(KERNEL).cpp $(KERNEL_DIR)/$(KERNEL)_profile.cpp

# Generate inner_product_compute_profile.cpp
$(KERNEL_DIR)/inner_product_compute_profile.cpp: $(KERNEL_DIR)/inner_product_compute.cpp $(GENERATOR)
	$(PYTHON) $(GENERATOR) $< $@

# Build test_innerproduct_profile executable
$(BUILD_DIR)/test_innerproduct_profile: test_innerproduct.cpp $(KERNEL_DIR)/inner_product_compute_profile.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^

# Run profile test
.PHONY: run_innerproduct
run_innerproduct: $(BUILD_DIR)/test_innerproduct_profile
	@echo "========================================"
	@echo "Running Inner Product Profile Test"
	@echo "========================================"
	./$(BUILD_DIR)/test_innerproduct_profile

# Convenience target: generate and run in one step
.PHONY: profile_innerproduct
profile_innerproduct: $(BUILD_DIR)/test_innerproduct_profile
	./$(BUILD_DIR)/test_innerproduct_profile

# Generate paged_kv_index_builder_profile.cpp
$(KERNEL_DIR)/paged_kv_index_builder_profile.cpp: $(KERNEL_DIR)/paged_kv_index_builder.cpp $(GENERATOR)
	$(PYTHON) $(GENERATOR) $< $@

# Build test_paged_kv_index_builder_profile executable
$(BUILD_DIR)/test_paged_kv_index_builder_profile: test_paged_kv_index_builder.cpp $(KERNEL_DIR)/paged_kv_index_builder_profile.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^

# Run paged_kv_index_builder profile test
.PHONY: run_paged_kv_index_builder
run_paged_kv_index_builder: $(BUILD_DIR)/test_paged_kv_index_builder_profile
	@echo "========================================"
	@echo "Running Paged KV Index Builder Profile Test"
	@echo "========================================"
	./$(BUILD_DIR)/test_paged_kv_index_builder_profile

# Convenience target: generate and run in one step
.PHONY: profile_paged_kv_index_builder
profile_paged_kv_index_builder: $(BUILD_DIR)/test_paged_kv_index_builder_profile
	./$(BUILD_DIR)/test_paged_kv_index_builder_profile

# Generate topk_retrieval_profile.cpp
$(KERNEL_DIR)/topk_retrieval_profile.cpp: $(KERNEL_DIR)/topk_retrieval.cpp $(GENERATOR)
	$(PYTHON) $(GENERATOR) $< $@

# Build test_topk_retrieval_profile executable
$(BUILD_DIR)/test_topk_retrieval_profile: test_topk_retrieval.cpp $(KERNEL_DIR)/topk_retrieval_profile.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^

# Run topk_retrieval profile test
.PHONY: run_topk_retrieval
run_topk_retrieval: $(BUILD_DIR)/test_topk_retrieval_profile
	@echo "========================================"
	@echo "Running Top-K Retrieval Profile Test"
	@echo "========================================"
	./$(BUILD_DIR)/test_topk_retrieval_profile

# Convenience target: generate and run in one step
.PHONY: profile_topk_retrieval
profile_topk_retrieval: $(BUILD_DIR)/test_topk_retrieval_profile
	./$(BUILD_DIR)/test_topk_retrieval_profile

# Generate block_sparse_attention_profile.cpp
$(KERNEL_DIR)/block_sparse_attention_profile.cpp: $(KERNEL_DIR)/block_sparse_attention.cpp $(GENERATOR)
	$(PYTHON) $(GENERATOR) $< $@

# Build test_block_sparse_attention_profile executable
$(BUILD_DIR)/test_block_sparse_attention_profile: test_block_sparse_attention.cpp $(KERNEL_DIR)/block_sparse_attention_profile.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^

# Run block_sparse_attention profile test
.PHONY: run_block_sparse_attention
run_block_sparse_attention: $(BUILD_DIR)/test_block_sparse_attention_profile
	@echo "========================================"
	@echo "Running Block Sparse Attention Profile Test"
	@echo "========================================"
	./$(BUILD_DIR)/test_block_sparse_attention_profile

# Convenience target: generate and run in one step
.PHONY: profile_block_sparse_attention
profile_block_sparse_attention: $(BUILD_DIR)/test_block_sparse_attention_profile
	./$(BUILD_DIR)/test_block_sparse_attention_profile

# Generic targets
clean:
	rm -rf $(BUILD_DIR)
	rm -f $(KERNEL_DIR)/*_profile.cpp

help:
	@echo "Profile Test Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  all                            - Build all profile tests (generates profile files as needed)"
	@echo "  generate KERNEL=<name>         - Generate *_profile.cpp for a specific kernel"
	@echo "  run_innerproduct               - Build and run inner product profile test"
	@echo "  profile_innerproduct           - Generate, build, and run inner product profile"
	@echo "  run_paged_kv_index_builder     - Build and run paged KV index builder profile test"
	@echo "  profile_paged_kv_index_builder - Generate, build, and run paged KV index builder profile"
	@echo "  run_topk_retrieval             - Build and run top-k retrieval profile test"
	@echo "  profile_topk_retrieval         - Generate, build, and run top-k retrieval profile"
	@echo "  run_block_sparse_attention     - Build and run block sparse attention profile test"
	@echo "  profile_block_sparse_attention - Generate, build, and run block sparse attention profile"
	@echo "  clean                          - Remove build directory and generated profile files"
	@echo "  help                           - Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make generate KERNEL=inner_product_compute"
	@echo "  make generate KERNEL=paged_kv_index_builder"
	@echo "  make generate KERNEL=topk_retrieval"
	@echo "  make generate KERNEL=block_sparse_attention"
	@echo "  make run_innerproduct"
	@echo "  make run_paged_kv_index_builder"
	@echo "  make run_topk_retrieval"
	@echo "  make run_block_sparse_attention"
	@echo "  make profile_innerproduct"
	@echo "  make profile_paged_kv_index_builder"
	@echo "  make profile_topk_retrieval"
	@echo "  make profile_block_sparse_attention"
	@echo ""
	@echo "To add a new profile test:"
	@echo "  1. Create a test_*.cpp file in profile_test/"
	@echo "  2. Add corresponding rules in this Makefile"
