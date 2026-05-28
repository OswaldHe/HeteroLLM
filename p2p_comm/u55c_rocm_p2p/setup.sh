#!/bin/bash
# Setup script for FPGA-GPU P2P Transfer Demo
# Run this once before building and running the demo

set -e

echo "============================================"
echo "FPGA-GPU P2P Demo Setup"
echo "============================================"

# Check for required tools
echo ""
echo "Checking for required tools..."

if ! command -v hipcc &> /dev/null; then
    echo "ERROR: hipcc not found. Please source ROCm environment:"
    echo "  source /opt/rocm/bin/rocmvars.sh"
    echo "  or add /opt/rocm/bin to PATH"
    exit 1
fi
echo "  ✓ hipcc found: $(which hipcc)"

if ! command -v xbutil &> /dev/null; then
    echo "ERROR: xbutil not found. Please source XRT environment:"
    echo "  source /opt/xilinx/xrt/setup.sh"
    exit 1
fi
echo "  ✓ xbutil found: $(which xbutil)"

# Check GPU
echo ""
echo "Checking GPU..."
GPU_COUNT=$(rocm-smi --showbus 2>/dev/null | grep -c "GPU" || true)
if [ "$GPU_COUNT" -eq 0 ]; then
    echo "ERROR: No AMD GPU found"
    exit 1
fi
echo "  ✓ Found $GPU_COUNT AMD GPU(s)"
rocm-smi --showbus 2>/dev/null | grep "GPU"

# Check FPGA
echo ""
echo "Checking FPGA..."
FPGA_LIST=$(xbutil examine 2>/dev/null | grep -i "u55c" || true)
if [ -z "$FPGA_LIST" ]; then
    echo "WARNING: No U55C FPGA found. Checking all devices..."
    xbutil examine 2>/dev/null | head -20
fi
echo "  ✓ FPGA devices found"

# Check P2P status on FPGA
echo ""
echo "Checking FPGA P2P status..."
FPGA_BDF=${1:-"81:00.1"}
echo "  Checking device $FPGA_BDF..."

P2P_STATUS=$(xbutil examine -d "$FPGA_BDF" --report platform 2>/dev/null | grep -i "p2p" || true)
if [ -z "$P2P_STATUS" ]; then
    echo "  Could not determine P2P status"
else
    echo "  $P2P_STATUS"
fi

# Check IOMMU
echo ""
echo "Checking IOMMU status..."
IOMMU_STATUS=$(dmesg 2>/dev/null | grep -i "iommu" | head -3 || true)
if [ -z "$IOMMU_STATUS" ]; then
    echo "  No IOMMU messages found (may need sudo)"
else
    echo "$IOMMU_STATUS" | while read line; do echo "  $line"; done
fi

# Check PCIe topology
echo ""
echo "PCIe Topology (relevant devices):"
lspci -tv 2>/dev/null | grep -E "Xilinx|AMD.*MI|Aldebaran" | head -10 || true

# Environment variables
echo ""
echo "============================================"
echo "Environment Setup"
echo "============================================"
echo ""
echo "Add these to your shell or run before building:"
echo ""
echo "  # ROCm"
echo "  export PATH=/opt/rocm/bin:\$PATH"
echo "  export LD_LIBRARY_PATH=/opt/rocm/lib:\$LD_LIBRARY_PATH"
echo ""
echo "  # XRT"
echo "  source /opt/xilinx/xrt/setup.sh"
echo ""
echo "  # Vitis (for FPGA kernel build only)"
echo "  source /opt/xilinx/Vitis/2023.2/settings64.sh"
echo ""

# Create environment script
cat > env.sh << 'EOF'
#!/bin/bash
# Source this file before building or running

# ROCm 6.2
export ROCM_PATH=/opt/rocm
export PATH=$ROCM_PATH/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH

# XRT 2.14
export XILINX_XRT=/opt/xilinx/xrt
source $XILINX_XRT/setup.sh

# Vitis 2023.2 (uncomment for FPGA kernel build)
# source /opt/xilinx/Vitis/2023.2/settings64.sh

echo "Environment configured for FPGA-GPU P2P demo"
echo "  ROCm: $ROCM_PATH"
echo "  XRT:  $XILINX_XRT"
EOF
chmod +x env.sh
echo "Created env.sh - source this file before building"

echo ""
echo "============================================"
echo "Setup complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. source env.sh"
echo "  2. make host              # Build host application"
echo "  3. make fpga_kernel_emu   # Build FPGA emulation (optional)"
echo "  4. ./p2p_transfer --help  # Run with --xclbin option"
echo ""
