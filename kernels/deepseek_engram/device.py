from rapidstream import get_u55c_vitis_device_factory

# Set the Vitis platform name
factory = get_u55c_vitis_device_factory("xilinx_u55c_gen3x16_xdma_3_202210_1")
# Generate the virtual device in JSON format
factory.generate_virtual_device("u55c_device.json")