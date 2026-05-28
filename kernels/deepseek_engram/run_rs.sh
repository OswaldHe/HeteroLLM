#!/bin/bash

rapidstream-tapaopt \
    -j 6 \
    --run-impl \
    --work-dir ./build \
    --tapa-xo-path engram_fpga_top.xo \
    --device-config u55c_device.json \
    --implementation-config impl_config.json \
    --floorplan-config floorplan_config.json \
    --connectivity-ini kernel_config.ini 