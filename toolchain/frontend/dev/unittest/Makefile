# Makefile for unit tests

CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -I..

# Kernel source files
INNER_PRODUCT_KERNEL = ../steps/example_kernel/inner_product_compute.cpp
TOPK_KERNEL = ../steps/example_kernel/topk_retrieval.cpp
THRESHOLD_KERNEL = ../steps/example_kernel/threshold_retrieval.cpp
PAGED_KV_INDEX_KERNEL = ../steps/example_kernel/paged_kv_index_builder.cpp
BLOCK_SPARSE_ATTENTION_KERNEL = ../steps/example_kernel/block_sparse_attention.cpp

# Test executables
TEST_INNERPRODUCT = test_innerproduct
TEST_TOPK = test_topk_retrieval
TEST_THRESHOLD = test_threshold_retrieval
TEST_PAGED_KV = test_paged_kv_index_builder
TEST_BLOCK_SPARSE_ATTENTION = test_block_sparse_attention

ALL_TESTS = $(TEST_INNERPRODUCT) $(TEST_TOPK) $(TEST_THRESHOLD) $(TEST_PAGED_KV) $(TEST_BLOCK_SPARSE_ATTENTION)

.PHONY: all clean test test-verbose

all: $(ALL_TESTS)

$(TEST_INNERPRODUCT): test_innerproduct.cpp $(INNER_PRODUCT_KERNEL)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(TEST_TOPK): test_topk_retrieval.cpp $(TOPK_KERNEL)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(TEST_THRESHOLD): test_threshold_retrieval.cpp $(THRESHOLD_KERNEL)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(TEST_PAGED_KV): test_paged_kv_index_builder.cpp $(PAGED_KV_INDEX_KERNEL)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(TEST_BLOCK_SPARSE_ATTENTION): test_block_sparse_attention.cpp $(BLOCK_SPARSE_ATTENTION_KERNEL)
	$(CXX) $(CXXFLAGS) -o $@ $^

test: $(ALL_TESTS)
	@echo "=== Running InnerProductCompute tests ==="
	./$(TEST_INNERPRODUCT)
	@echo ""
	@echo "=== Running TopKRetrieval tests ==="
	./$(TEST_TOPK)
	@echo ""
	@echo "=== Running ThresholdRetrieval tests ==="
	./$(TEST_THRESHOLD)
	@echo ""
	@echo "=== Running PagedKVIndexBuilder tests ==="
	./$(TEST_PAGED_KV)
	@echo ""
	@echo "=== Running BlockSparseAttention tests ==="
	./$(TEST_BLOCK_SPARSE_ATTENTION)
	@echo ""
	@echo "=== All tests completed ==="
	$(MAKE) clean

clean:
	rm -f $(ALL_TESTS)

# Run tests with verbose output
test-verbose: $(ALL_TESTS)
	@echo "=== Running InnerProductCompute tests ==="
	./$(TEST_INNERPRODUCT) --success
	@echo ""
	@echo "=== Running TopKRetrieval tests ==="
	./$(TEST_TOPK) --success
	@echo ""
	@echo "=== Running ThresholdRetrieval tests ==="
	./$(TEST_THRESHOLD) --success
	@echo ""
	@echo "=== Running PagedKVIndexBuilder tests ==="
	./$(TEST_PAGED_KV) --success
	@echo ""
	@echo "=== Running BlockSparseAttention tests ==="
	./$(TEST_BLOCK_SPARSE_ATTENTION) --success
	@echo ""
	@echo "=== All tests completed ==="
	$(MAKE) clean
