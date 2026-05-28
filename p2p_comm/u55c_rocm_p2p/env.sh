#!/bin/bash
# Source this file before building or running the P2P demo
# Usage: source env.sh

# ROCm 6.2
export ROCM_PATH=/opt/rocm
export PATH=$ROCM_PATH/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH
export HIP_PLATFORM=amd

# XRT 2.14
export XILINX_XRT=/opt/xilinx/xrt
if [ -f "$XILINX_XRT/setup.sh" ]; then
    source $XILINX_XRT/setup.sh
else
    export PATH=$XILINX_XRT/bin:$PATH
    export LD_LIBRARY_PATH=$XILINX_XRT/lib:$LD_LIBRARY_PATH
fi

# Vitis 2023.2 (uncomment for FPGA kernel build)
# export XILINX_VITIS=/opt/xilinx/Vitis/2023.2
# if [ -f "$XILINX_VITIS/settings64.sh" ]; then
#     source $XILINX_VITIS/settings64.sh
# fi

echo "============================================"
echo "FPGA-GPU P2P Demo Environment"
echo "============================================"
echo "ROCm:  $ROCM_PATH"
echo "XRT:   $XILINX_XRT"
echo "hipcc: $(which hipcc 2>/dev/null || echo 'not found')"
echo "xbutil: $(which xbutil 2>/dev/null || echo 'not found')"
echo "============================================"
