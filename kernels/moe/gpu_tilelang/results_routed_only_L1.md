# DeepSeek V3.2 MoE — Routed-Only Decode Latency (L=1)

GPU: AMD Instinct MI210 (gfx90a) | iters: 2000 (50 warmup) | tuned `BLOCK_N=8 reduce_threads=64`

| Implementation         | dtype | Min     | P50     | P90     | Mean    | Throughput | Power |
|------------------------|-------|---------|---------|---------|---------|------------|-------|
| PyTorch reference      | fp16  | 3.525 ms | 3.593 ms | 3.640 ms | 4.086 ms | 245 tok/s  | 99 W  |
| TileLang decode        | fp16  | 1.027 ms | 1.072 ms | 1.081 ms | 1.210 ms | 827 tok/s  | 150 W |
| TileLang decode (int8) | int8  | 0.741 ms | 0.755 ms | 0.767 ms | 0.858 ms | 1166 tok/s | 171 W |
| **Speedup vs ref (fp16)** | | **3.43×** | **3.35×** | **3.37×** | **3.38×** | **3.37×**  |       |
| **Speedup vs ref (int8)** | | **4.76×** | **4.76×** | **4.75×** | **4.76×** | **4.76×**  |       |

Int8 quantization: per-tensor symmetric on activations; per-expert symmetric on each routed weight matrix (`W_gate`, `W_up`, `W_down`). Max abs err vs fp16 reference at L=1: 0.0052 (1.5% relative).
