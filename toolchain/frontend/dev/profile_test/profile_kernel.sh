#!/bin/bash
# profile_kernel.sh - Script to profile kernel operations
#
# This script automates the profiling workflow:
# 1. Generates a *_profile.cpp from the kernel source using generate_profile.py
# 2. Compiles the profiled kernel with the test file
# 3. Runs the compiled executable to get operation counts
#
# Usage:
#   ./profile_kernel.sh <kernel_name>
#
# Examples:
#   ./profile_kernel.sh inner_product_compute
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNEL_DIR="$SCRIPT_DIR/../steps/example_kernel"
BUILD_DIR="$SCRIPT_DIR/build"
GENERATOR="$SCRIPT_DIR/generate_profile.py"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_usage() {
    echo "Usage: $0 <kernel_name>"
    echo ""
    echo "Arguments:"
    echo "  kernel_name   Name of the kernel (e.g., inner_product_compute)"
    echo ""
    echo "Examples:"
    echo "  $0 inner_product_compute"
}

# Check arguments
if [ $# -lt 1 ]; then
    print_usage
    exit 1
fi

KERNEL_NAME="$1"

# Derive file names
KERNEL_FILE="$KERNEL_DIR/${KERNEL_NAME}.cpp"
PROFILE_FILE="$KERNEL_DIR/${KERNEL_NAME}_profile.cpp"

# Map kernel names to test files
# Add new mappings here as needed
case "$KERNEL_NAME" in
    inner_product_compute)
        TEST_FILE="$SCRIPT_DIR/test_innerproduct.cpp"
        ;;
    threshold_retrieval)
        TEST_FILE="$SCRIPT_DIR/test_threshold.cpp"
        ;;
    topk_retrieval)
        TEST_FILE="$SCRIPT_DIR/test_topk.cpp"
        ;;
    *)
        TEST_FILE="$SCRIPT_DIR/test_${KERNEL_NAME}.cpp"
        ;;
esac

# Check if kernel file exists
if [ ! -f "$KERNEL_FILE" ]; then
    echo -e "${RED}Error: Kernel file not found: $KERNEL_FILE${NC}"
    exit 1
fi

# Check if test file exists
if [ ! -f "$TEST_FILE" ]; then
    echo -e "${RED}Error: Test file not found: $TEST_FILE${NC}"
    echo -e "${YELLOW}Please create a test file for this kernel.${NC}"
    exit 1
fi

# Check if generator exists
if [ ! -f "$GENERATOR" ]; then
    echo -e "${RED}Error: Generator script not found: $GENERATOR${NC}"
    exit 1
fi

# Generate profile file
echo -e "${GREEN}Generating profile file...${NC}"
python3 "$GENERATOR" "$KERNEL_FILE" "$PROFILE_FILE"

# Create build directory
mkdir -p "$BUILD_DIR"

# Compile
EXECUTABLE="$BUILD_DIR/${KERNEL_NAME}_profile"
echo -e "${GREEN}Compiling: $EXECUTABLE${NC}"
g++ -std=c++17 -Wall -Wextra -O2 -g \
    -I"$SCRIPT_DIR/../.." -I"$SCRIPT_DIR/../../.." \
    -o "$EXECUTABLE" \
    "$TEST_FILE" "$PROFILE_FILE"

echo -e "${GREEN}Compilation successful!${NC}"

# Run
echo ""
echo -e "${GREEN}Running profiled kernel...${NC}"
echo "========================================"
"$EXECUTABLE"
echo "========================================"
echo -e "${GREEN}Profiling complete!${NC}"
