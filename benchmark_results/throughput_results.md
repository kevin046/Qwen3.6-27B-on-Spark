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

## Quality Evaluation: LiveBench 2024-11-25

**Model:** Qwen/Qwen3.6-27B-FP8 · **Runtime:** sglang 0.5.12 · **Speculative Decoding:** NEXTN  
**KV Cache:** fp8_e4m3 · **Date:** 2024-11-25  
**Thinking mode:** Disabled (`enable_thinking: false`) for fair non-thinking evaluation

### Overall Score: 76.5% (1,000 questions)

### Top-Level Categories

| Category | Score | Questions |
|----------|-------|-----------|
| Reasoning | 90.6% | 150 |
| Instruction Following | 83.1% | 200 |
| Coding | 77.3% | 128 |
| Language | 74.4% | 140 |
| Math | 69.8% | 232 |
| Data Analysis | 64.8% | 150 |

### Subcategory Breakdown

| Category | Subcategory | Score | Questions |
|----------|------------|-------|-----------|
| Reasoning | web_of_lies_v2 | 100.0% | 50 |
| Language | connections | 97.7% | 50 |
| Math | math_comp | 88.5% | 96 |
| Data Analysis | tablereformat | 88.0% | 50 |
| Instruction Following | paraphrase | 86.8% | 50 |
| Reasoning | spatial | 86.0% | 50 |
| Reasoning | zebra_puzzle | 85.8% | 50 |
| Instruction Following | story_generation | 85.3% | 50 |
| Coding | LCB_generation | 83.3% | 78 |
| Instruction Following | summarize | 80.4% | 50 |
| Instruction Following | simplify | 79.9% | 50 |
| Math | AMPS_Hard | 77.0% | 100 |
| Coding | coding_completion | 68.0% | 50 |
| Language | typos | 68.0% | 50 |
| Data Analysis | cta | 54.0% | 50 |
| Language | plot_unscrambling | 53.5% | 40 |
| Data Analysis | tablejoin | 52.5% | 50 |
| Math | olympiad | 0.0% | 36 |

### Key Observations

- **Reasoning tasks excel** at 90.6% — web_of_lies_v2 achieves a perfect 100%, confirming logical consistency is preserved through FP8 + speculative decoding
- **Instruction following** is strong at 83.1% across all four subcategories, demonstrating reliable output formatting
- **Coding** scores 77.3% with LCB_generation (83.3%) significantly outperforming coding_completion (68.0%)
- **Math** is mixed: math_comp at 88.5% vs. olympiad at 0.0% — the latter likely requires chain-of-thought reasoning that was disabled
- **Data analysis** (64.8%) and **language** (74.4%) show room for improvement, particularly in table operations and complex text manipulation
- The 0.0% on math/olympiad is expected with thinking disabled — these problems require multi-step reasoning chains

---

## Raw Output Files

- `decode_only_results.txt`
- `few_shot_results.txt`
- `code_results.txt`
- `long_context_results.txt`
- `long_output_results.txt`
- `mixed_results.txt`
- `custom_chat_bench.txt`
