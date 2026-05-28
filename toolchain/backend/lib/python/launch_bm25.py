from bm25_loader_xrt import *
import logging
import pyxrt

logger = logging.getLogger(__name__)

def fpga_retriever_setup(
    bitstream: str = "../indexer_bm25.xclbin",
    export_dir: str = "./export",
):
    
    # Load document frequency
    logger.info("Loading doc_freq.bin...")
    doc_freq = load_document_frequency_mmap(os.path.join(export_dir, "doc_freq.bin"))
    
    if doc_freq is None:
        logger.error("Failed to load doc_freq.bin")
        return None
    
    # Load term frequencies
    logger.info("Loading term_freq.bin...")
    term_freq = load_term_frequencies_mmap(os.path.join(export_dir, "term_freq.bin"))

    if term_freq is None:
        logger.error("Failed to load term_freq.bin")
        return None
    
    # Pack documents for hardware
    packed = pack_documents_for_hw(term_freq, 8)
    
    # Compute L and L_doc_total
    L = packed.num_docs
    L = ((L + 63) // 64) * 64  # Round up to multiple of 64
    L_doc_total = packed.vectors_per_channel()
    
    logger.info("\n======================================")
    logger.info("KERNEL LAUNCH PARAMETERS")
    logger.info("======================================")
    logger.info(f"L (num docs, padded): {L}")
    logger.info(f"L_doc_total (vectors per channel): {L_doc_total}")
    logger.info(f"Num super-batches: {packed.num_super_batches}")
    
    # ===============================
    # PyXRT Device and Kernel Initialization
    # ===============================
    
    logger.info("\n======================================")
    logger.info("PYXRT DEVICE AND KERNEL INITIALIZATION")
    logger.info("======================================")
    
    result = find_working_device(bitstream, -1)
    if result is None:
        logger.error(f" No working device found for XCLBIN: {bitstream}")
        logger.info("Please check:")
        logger.info("  1. FPGA devices are properly installed and visible")
        logger.info("  2. The XCLBIN file exists and is compatible with the device")
        logger.info("  3. XRT runtime is properly installed")
        return None
    
    device, xclbin_uuid, selected_device = result
    logger.info(f"Device {selected_device} opened and XCLBIN loaded")
    
    # Create kernel object
    logger.info("Creating kernel object...")
    try:
        kernel = pyxrt.kernel(device, xclbin_uuid, "indexer_top")
    except Exception as e:
        logger.error(f" Failed to create kernel object: {e}")
        return None
    logger.info(f"Kernel created.")
    
    # ===============================
    # Buffer Allocation
    # ===============================
    
    logger.info("\n======================================")
    logger.info("BUFFER ALLOCATION")
    logger.info("======================================")
    
    # Calculate buffer sizes
    df_buffer_size = VOCAB_SIZE_DIV_16 * 16 * 4  # 16 ints per vector
    query_bitmap_size = VOCAB_SIZE_DIV_512 * 64  # 512 bits = 64 bytes per chunk
    inst_mem_size = packed.num_super_batches * 4
    doc_mem_size = L_doc_total * 16 * 4  # 16 uint32 per vector
    output_size = (TOP_K + 15) // 16
    topk_id_size = output_size * 16 * 4
    
    logger.info("Buffer sizes:")
    logger.info(f"  df_buffer: {df_buffer_size / 1024:.2f} KB")
    logger.info(f"  query_bitmap: {query_bitmap_size / 1024:.2f} KB")
    logger.info(f"  inst_mem: {inst_mem_size / 1024:.2f} KB")
    logger.info(f"  doc_mem (per channel): {doc_mem_size / 1024 / 1024:.2f} MB")
    logger.info(f"  topk_id: {topk_id_size} bytes")
    
    # Allocate buffers using kernel.group_id() to get memory bank assignment
    # Argument order: L(0), L_doc_total(1), df_buffer(2), query_bitmap(3), inst_mem(4), 
    #                 doc_mem[0-3](5-8), topk_id(9)
    
    # Initialize with zeros like the Xilinx example
    zeros_df = bytearray(df_buffer_size)
    zeros_query = bytearray(query_bitmap_size)
    zeros_inst = bytearray(inst_mem_size)
    zeros_doc = bytearray(doc_mem_size)
    zeros_topk = bytearray(topk_id_size)
    
    logger.info("Allocate and initialize buffers")
    
    # Allocate df_buffer
    bo_df_buffer = pyxrt.bo(device, df_buffer_size, pyxrt.bo.normal, kernel.group_id(2))
    bo_df_buffer.write(zeros_df, 0)
    buf_df = bo_df_buffer.map()
    
    # Allocate query_bitmap
    bo_query_bitmap = pyxrt.bo(device, query_bitmap_size, pyxrt.bo.normal, kernel.group_id(3))
    bo_query_bitmap.write(zeros_query, 0)
    buf_query = bo_query_bitmap.map()
    
    # Allocate inst_mem
    bo_inst_mem = pyxrt.bo(device, inst_mem_size, pyxrt.bo.normal, kernel.group_id(4))
    bo_inst_mem.write(zeros_inst, 0)
    buf_inst = bo_inst_mem.map()
    
    # Allocate doc_mem channels
    bo_doc_mem_0 = pyxrt.bo(device, doc_mem_size, pyxrt.bo.normal, kernel.group_id(5))
    bo_doc_mem_0.write(zeros_doc, 0)
    buf_doc_0 = bo_doc_mem_0.map()
    
    bo_doc_mem_1 = pyxrt.bo(device, doc_mem_size, pyxrt.bo.normal, kernel.group_id(6))
    bo_doc_mem_1.write(zeros_doc, 0)
    buf_doc_1 = bo_doc_mem_1.map()
    
    bo_doc_mem_2 = pyxrt.bo(device, doc_mem_size, pyxrt.bo.normal, kernel.group_id(7))
    bo_doc_mem_2.write(zeros_doc, 0)
    buf_doc_2 = bo_doc_mem_2.map()
    
    bo_doc_mem_3 = pyxrt.bo(device, doc_mem_size, pyxrt.bo.normal, kernel.group_id(8))
    bo_doc_mem_3.write(zeros_doc, 0)
    buf_doc_3 = bo_doc_mem_3.map()
    
    # Allocate topk_id output buffer
    bo_topk_id = pyxrt.bo(device, topk_id_size, pyxrt.bo.normal, kernel.group_id(9))
    bo_topk_id.write(zeros_topk, 0)
    buf_topk = bo_topk_id.map()
    
    logger.info(f"Buffers allocated")
    
    # ===============================
    # Prepare Data and Write to Buffers
    # ===============================
    
    logger.info("\n======================================")
    logger.info("DATA PREPARATION AND TRANSFER")
    logger.info("======================================")
    
    # Prepare and write df_buffer data
    logger.info("Writing df_buffer data...")
    df_buffer_data = np.zeros(VOCAB_SIZE_DIV_16 * 16, dtype=np.int32)
    for i in range(VOCAB_SIZE_DIV_16):
        for j in range(16):
            df_buffer_data[i * 16 + j] = int(doc_freq[i * 16 + j])
    # Write using bo.write() method
    bo_df_buffer.write(df_buffer_data.tobytes(), 0)
    
    # Prepare and write inst_mem
    logger.info("Writing inst_mem data...")
    inst_mem_data = np.array(packed.inst_mem, dtype=np.int32)
    bo_inst_mem.write(inst_mem_data.tobytes(), 0)
    
    # Prepare and write doc_mem for each channel
    logger.info("Writing doc_mem data for 4 channels...")
    doc_mem_buffers = [buf_doc_0, buf_doc_1, buf_doc_2, buf_doc_3]
    doc_mem_bos = [bo_doc_mem_0, bo_doc_mem_1, bo_doc_mem_2, bo_doc_mem_3]
    
    for channel in range(4):
        channel_data = np.zeros(L_doc_total * 16, dtype=np.uint32)
        for vec_idx, vec in enumerate(packed.doc_mem[channel]):
            for j in range(16):
                channel_data[vec_idx * 16 + j] = vec[j]
        doc_mem_bos[channel].write(channel_data.tobytes(), 0)
    
    # Initialize output buffer to -1
    logger.info("Initializing topk_id output buffer...")
    topk_id_data = np.full(output_size * 16, -1, dtype=np.int32)
    bo_topk_id.write(topk_id_data.tobytes(), 0)
    
    logger.info(f"  output_size: {output_size}, topk_id_size: {topk_id_size} bytes")
    
    # Sync buffers to device
    logger.info("Syncing buffers to device...")
    
    bo_df_buffer.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, df_buffer_size, 0)
    bo_inst_mem.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, inst_mem_size, 0)
    bo_doc_mem_0.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, doc_mem_size, 0)
    bo_doc_mem_1.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, doc_mem_size, 0)
    bo_doc_mem_2.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, doc_mem_size, 0)
    bo_doc_mem_3.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, doc_mem_size, 0)

    return kernel, L, L_doc_total, bo_query_bitmap, bo_df_buffer, bo_inst_mem, bo_doc_mem_0, bo_doc_mem_1, bo_doc_mem_2, bo_doc_mem_3, bo_topk_id, buf_topk

def fpga_retriver_launch(
    kernel, L, L_doc_total, bo_query_bitmap, bo_df_buffer, bo_inst_mem, bo_doc_mem_0, bo_doc_mem_1, bo_doc_mem_2, bo_doc_mem_3, bo_topk_id, buf_topk, query_token_list
):
    query_tokens = parse_query_tokens(query_token_list)
    query_bitmap_size = VOCAB_SIZE_DIV_512 * 64  # 512 bits = 64 bytes per chunk
    output_size = (64 + 15) // 16
    topk_id_size = output_size * 16 * 4
    logger.info(f"\nParsed {len(query_tokens)} query tokens from input")
    # Prepare query bitmap
    logger.info("Preparing query bitmap...")
    query_bitmap = np.zeros(VOCAB_SIZE_DIV_512 * 64, dtype=np.uint8)
    for token_id in query_tokens:
        tid = int(token_id)
        if 0 <= tid < VOCAB_SIZE:
            chunk_idx = tid // 512
            bit_idx = tid % 512
            byte_idx = bit_idx // 8
            bit_in_byte = bit_idx % 8
            query_bitmap[chunk_idx * 64 + byte_idx] |= (1 << bit_in_byte)
    
    # Write query bitmap to buffer
    bo_query_bitmap.write(query_bitmap.tobytes(), 0)
    
    # Sync query bitmap to device
    bo_query_bitmap.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, query_bitmap_size, 0)
    
    # Launch kernel
    logger.info("Launching kernel...")
    start_time = time.time()
    
    run = kernel(
        L,
        L_doc_total,
        bo_df_buffer,
        bo_query_bitmap,
        bo_inst_mem,
        bo_doc_mem_0,
        bo_doc_mem_1,
        bo_doc_mem_2,
        bo_doc_mem_3,
        bo_topk_id
    )
    
    logger.info("Now wait for the kernel to finish")
    state = run.wait()
    
    kernel_time = (time.time() - start_time) * 1000  # in ms
    logger.info(f"Kernel execution completed in {kernel_time:.2f} ms")
    logger.info(f"  Kernel state: {state}")

    if state != pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
        logger.warning(f" Kernel did not complete successfully! State: {state}")
    
    
    # Sync output buffer from device
    bo_topk_id.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, topk_id_size, 0)
    
    # Read top-k document IDs
    hw_topk_indices = []
    topk_bytes = bytes(buf_topk[:topk_id_size])
    topk_result = np.frombuffer(topk_bytes, dtype=np.int32)

    for i in range(output_size):
        for j in range(16):
            if i * 16 + j < 64:
                hw_topk_indices.append(int(topk_result[i * 16 + j]))
    
    return hw_topk_indices, kernel_time