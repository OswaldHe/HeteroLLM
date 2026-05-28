# Makefile for Design Space Explorer
# Builds the design space exploration testing framework

CXX := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -O2 -g
INCLUDES := -I../.. -I../../frontend

# Directory paths
BUILD_DIR := build

# Source files
SRC_DSE := paged_attention_dse.cpp

# Kernel implementation sources
KERNEL_DIR := ../../frontend/dev/steps/example_kernel
KERNEL_SRCS := $(KERNEL_DIR)/inner_product_compute.cpp \
               $(KERNEL_DIR)/paged_kv_index_builder.cpp \
               $(KERNEL_DIR)/topk_retrieval.cpp \
               $(KERNEL_DIR)/block_sparse_attention.cpp

# Targets
TARGET_DSE := $(BUILD_DIR)/paged_attention_dse

.PHONY: all clean run run_custom help

all: $(TARGET_DSE)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Build paged attention design space explorer
$(TARGET_DSE): $(SRC_DSE) $(KERNEL_SRCS) realsys_profile.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $(SRC_DSE) $(KERNEL_SRCS)

# Run the design space exploration with default config
run: $(TARGET_DSE)
	@echo "========================================"
	@echo "Running Paged Attention Design Space Exploration"
	@echo "========================================"
	./$(TARGET_DSE) design_space.json

# Run with custom design space file
# Usage: make run_custom DESIGN_SPACE=path/to/custom.json
run_custom: $(TARGET_DSE)
ifndef DESIGN_SPACE
	$(error DESIGN_SPACE is not set. Usage: make run_custom DESIGN_SPACE=path/to/design_space.json)
endif
	./$(TARGET_DSE) $(DESIGN_SPACE)

clean:
	rm -rf $(BUILD_DIR)

help:
	@echo "Design Space Explorer Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all          - Build the paged attention design space explorer (default)"
	@echo "  run          - Build and run with default design_space.json"
	@echo "  run_custom   - Run with custom design space file"
	@echo "                 Usage: make run_custom DESIGN_SPACE=path/to/file.json"
	@echo "  clean        - Remove build artifacts"
	@echo "  help         - Show this help message"
	@echo ""
	@echo "Example:"
	@echo "  make run"
	@echo "  make run_custom DESIGN_SPACE=my_config.json"
