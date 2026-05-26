# Throughput Benchmark Results

## Qwen3.6-27B-FP8 on NVIDIA DGX Spark (GB10)

**Runtime:** sglang 0.5.12 | **Speculative Decoding:** NEXTN (5 steps, 9 draft tokens)  
**Date:** May 2026

---

## Official sglang bench_serving Results

All tests run with `--num-prompts 100 --request-rate 10`:

| Profile | Input Tokens | Output Tokens | Throughput (t/s) |
|---------|-------------|---------------|-------------------|
| decode_only | 0 | 512 | **152** |
| few_shot | 512 | 512 | **102** |
| code | 2,048 | 256 | **72** |
| long_output | 512 | 4,096 | **61** |
| long_context | 8,192 | 128 | **52** |
| mixed | 1,024 (avg) | 512 | **35** |

### Key Observations

- **Decode-heavy workloads** (pure generation) achieve the highest throughput at 152 t/s
- **Mixed workloads** with moderate I/O show 102 t/s (few_shot)
- **Long-context** workloads are bottlenecked by KV cache attention at 52 t/s
- **Long-output** generation (4K tokens) maintains 61 t/s sustained throughput
- **Mixed** benchmark represents realistic production traffic; 35 t/s accounts for varying input/output ratios

---

## Custom Chat Benchmark

**Method:** 40 diverse questions across math, physics, reasoning, and code generation  
**Concurrency:** 8 async workers  
**Temperature:** 0.0 (deterministic)

| Metric | Value |
|--------|-------|
| Total requests | 40 |
| Total completion tokens | 10,240 |
| Total time | 116.2 s |
| **Throughput** | **88.15 tokens/s** |

### Workload Mix
- Mathematics (calculation, algebra): 6 questions
- Physics/science explanations: 4 questions  
- Logical reasoning (step-by-step): 6 questions
- Code generation (Python, algorithms): 4 questions
- Computer science fundamentals: 12 questions
- System design / distributed systems: 8 questions

---

## LiveBench Generation Throughput

| Metric | Value |
|--------|-------|
| Generation throughput | **~90 tokens/s** |
| Speculative decoding acceptance rate | ~55% |
| Average accepted tokens per step | 3.8 |

---

## Methodology Notes

1. **Server configuration:** Single GPU, tp-size=1, mem-fraction=0.75, page-size=1
2. **Speculative decoding:** NEXTN with topk=1, 5 steps, 9 draft tokens per step
3. **Environment variables:** `SGLANG_ENABLE_SPEC_V2=1`, `SGLANG_DISABLE_DEEP_GEMM=1`
4. **All computation in VRAM** — no CPU offloading
5. Results are reproducible with the provided `scripts/benchmark.sh`

---

## Raw Output Files

- `decode_only_results.txt`
- `few_shot_results.txt`
- `code_results.txt`
- `long_context_results.txt`
- `long_output_results.txt`
- `mixed_results.txt`
- `custom_chat_bench.txt`
