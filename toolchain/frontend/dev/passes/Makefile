# ============================================================================
# Makefile for PythonDispatchPass
#
# Builds the pass driver, then optionally runs it on the RAG example_kernel
# sources to generate a Python pipeline script.
#
# Targets:
#   make            — build the driver binary
#   make run        — build + run the pass on RAG kernel sources
#   make clean      — remove build artifacts and generated output
#   make help       — print this help
# ============================================================================

CXX       ?= g++
CXXFLAGS  := -std=c++17 -Wall -Wextra -O2
LDFLAGS   := -lstdc++fs

# Paths (relative to this directory)
PASS_DIR      := .
STEPS_DIR     := ../steps
DEPLOY_DIR    := ../../deploy
EXAMPLE_DIR   := $(STEPS_DIR)/example_kernel
OUTPUT_DIR    := ./generated

# Source / target
DRIVER_SRC := python_dispatch_pass_driver.cpp
PASS_HDR   := python_dispatch_pass.h
BINARY     := python_dispatch_pass

# Example kernel .cpp files to scan (RAG pipeline)
RAG_SOURCES := \
	example_kernel/bm25_dataset_builder.cpp \
	example_kernel/bm25_fpga_retrieval.cpp \
	example_kernel/rag_apply_memory.cpp

# Deploy headers to scan for MemoryManager API structure
DEPLOY_HEADERS := \
	memory_manager.h \
	simple_rag.h

# ---- Targets ---------------------------------------------------------------

.PHONY: all run clean help

all: $(BINARY)

$(BINARY): $(DRIVER_SRC) $(PASS_HDR)
	$(CXX) $(CXXFLAGS) -o $@ $(DRIVER_SRC) $(LDFLAGS)

run: $(BINARY)
	@mkdir -p $(OUTPUT_DIR)
	./$(BINARY) \
		--base-dir $(STEPS_DIR) \
		$(foreach src,$(RAG_SOURCES),--source $(src)) \
		--deploy-dir $(DEPLOY_DIR) \
		$(foreach hdr,$(DEPLOY_HEADERS),--deploy $(hdr)) \
		--output-dir $(OUTPUT_DIR) \
		--verbose
	@echo ""
	@echo "Generated: $(OUTPUT_DIR)/heteromm_pipeline.py"

clean:
	rm -f $(BINARY)
	rm -rf $(OUTPUT_DIR)

help:
	@echo "PythonDispatchPass Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all   (default)  Build the pass driver binary"
	@echo "  run              Build + run on RAG example_kernel sources"
	@echo "  clean            Remove binary and generated/ directory"
	@echo "  help             Show this message"
	@echo ""
	@echo "Variables:"
	@echo "  CXX=$(CXX)"
	@echo "  CXXFLAGS=$(CXXFLAGS)"
	@echo "  OUTPUT_DIR=$(OUTPUT_DIR)"
