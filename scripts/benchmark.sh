#!/bin/bash
# =============================================================================
# Qwen3.6-27B-FP8 Benchmark Script for NVIDIA DGX Spark (GB10)
# =============================================================================
# This script runs the full sglang benchmark suite including custom chat
# benchmarks and official bench_serving profiles.
# =============================================================================

set -euo pipefail

# Configuration
CONTAINER="scitrera/dgx-spark-sglang:0.5.12"
MODEL="Qwen/Qwen3.6-27B-FP8"
HOST_PORT=30000
RESULTS_DIR="$(cd "$(dirname "$0")/../benchmark_results" && pwd)"

echo "=============================================="
echo " Qwen3.6-27B-FP8 Benchmark Suite"
echo " NVIDIA DGX Spark (GB10) — sglang 0.5.12"
echo "=============================================="
echo ""

# -------------------------------------------------------
# Step 1: Launch sglang server
# -------------------------------------------------------
echo "[1/4] Launching sglang server..."
docker run -d \
  --name sglang-bench \
  --gpus all \
  --shm-size 64g \
  -p ${HOST_PORT}:30000 \
  -e SGLANG_ENABLE_SPEC_V2=1 \
  -e SGLANG_DISABLE_DEEP_GEMM=1 \
  ${CONTAINER} \
  python -m sglang.launch_server \
    --model-path ${MODEL} \
    --host 0.0.0.0 \
    --port 30000 \
    --tp-size 1 \
    --mem-fraction-static 0.75 \
    --context-length 262144 \
    --trust-remote-code \
    --kv-cache-dtype fp8_e4m3 \
    --speculative-nextn-steps 5 \
    --speculative-nextn-draft-token-per-step 9 \
    --speculative-topk 1 \
    --page-size 1

echo "  Waiting for server to be ready..."
sleep 30

# Wait for health check
for i in $(seq 1 30); do
  if curl -s http://localhost:${HOST_PORT}/health > /dev/null 2>&1; then
    echo "  ✓ Server ready after ${i}s"
    break
  fi
  echo "  Waiting... ($i/30)"
  sleep 2
done

# -------------------------------------------------------
# Step 2: Run official sglang bench_serving profiles
# -------------------------------------------------------
echo ""
echo "[2/4] Running official sglang bench_serving profiles..."

PROFILES=(
  "decode_only:0:512"
  "few_shot:512:512"
  "code:2048:256"
  "long_context:8192:128"
  "long_output:512:4096"
  "mixed:1024:512"
)

for profile in "${PROFILES[@]}"; do
  IFS=':' read -r name input_len output_len <<< "$profile"
  echo "  → Running: $name (input=$input_len, output=$output_len)"
  
  docker exec sglang-bench python -m sglang.bench_serving \
    --backend sglang \
    --host 127.0.0.1 \
    --port 30000 \
    --dataset-name random \
    --random-input-len $input_len \
    --random-output-len $output_len \
    --num-prompts 100 \
    --request-rate 10 \
    > "${RESULTS_DIR}/${name}_results.txt" 2>&1
  
  echo "    ✓ Complete"
done

# -------------------------------------------------------
# Step 3: Run custom chat benchmark
# -------------------------------------------------------
echo ""
echo "[3/4] Running custom chat benchmark (40 requests, 8 workers)..."

cat > /tmp/bench_chat.py << 'PYEOF'
import aiohttp
import asyncio
import time
import json

QUESTIONS = [
    "What is 15% of 840?",
    "Explain quantum entanglement in simple terms.",
    "Solve: If a train travels 120 miles in 2 hours, what is the average speed?",
    "Write a Python function for binary search.",
    "What are the main differences between TCP and UDP?",
    "Explain the concept of recursion with an example.",
    "What is the time complexity of merge sort?",
    "Describe how a hash table works.",
    "What is a closure in Python?",
    "Explain the difference between a stack and a queue.",
    "What is Big O notation?",
    "How does a binary tree work?",
    "Explain the concept of dynamic programming.",
    "What is polymorphism in OOP?",
    "Describe how HTTPS works.",
    "What is a deadlock?",
    "Explain the CAP theorem.",
    "What is the difference between process and thread?",
    "How does garbage collection work?",
    "What is a monad?",
    "Explain distributed consensus.",
    "What is the difference between SQL and NoSQL?",
    "How does a neural network learn?",
    "What is a microservice architecture?",
    "Explain the observer pattern.",
    "What is the difference between REST and GraphQL?",
    "How does TCP ensure reliable delivery?",
    "What is a B-tree?",
    "Explain the SOLID principles.",
    "What is the difference between compilation and interpretation?",
    "How does a CDN work?",
    "What is eventual consistency?",
    "Explain the difference between symmetric and asymmetric encryption.",
    "What is a virtual machine?",
    "How does DNS resolution work?",
    "What is the difference between a hash map and a tree map?",
    "Explain the concept of sharding in databases.",
    "What is idempotence?",
]

async def send_request(session, prompt, idx):
    payload = {
        "model": "Qwen/Qwen3.6-27B-FP8",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Be concise."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 256,
        "enable_thinking": False,
    }
    start = time.time()
    async with session.post("http://127.0.0.1:30000/v1/chat/completions", json=payload) as resp:
        result = await resp.json()
    elapsed = time.time() - start
    content = result["choices"][0]["message"]["content"]
    usage = result.get("usage", {})
    completion_tokens = usage.get("completion_tokens", 0)
    return idx, elapsed, completion_tokens, content

async def main():
    connector = aiohttp.TCPConnector(limit=8)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Warmup
        print("Warming up...")
        await send_request(session, "Hello", -1)
        await asyncio.sleep(2)
        
        print("Running 40 requests with 8 concurrent workers...")
        start_time = time.time()
        tasks = [send_request(session, q, i) for i, q in enumerate(QUESTIONS)]
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
    total_tokens = sum(r[2] for r in results)
    throughput = total_tokens / total_time
    
    print(f"\n{'='*50}")
    print(f"RESULTS")
    print(f"{'='*50}")
    print(f"Total requests:       {len(results)}")
    print(f"Total time:           {total_time:.1f}s")
    print(f"Total tokens:         {total_tokens}")
    print(f"Throughput:           {throughput:.2f} tokens/s")
    print(f"{'='*50}")

asyncio.run(main())
PYEOF

docker cp /tmp/bench_chat.py sglang-bench:/tmp/bench_chat.py
docker exec sglang-bench pip install -q aiohttp
docker exec sglang-bench python3 /tmp/bench_chat.py \
  > "${RESULTS_DIR}/custom_chat_bench.txt" 2>&1

echo "  ✓ Custom benchmark complete"

# -------------------------------------------------------
# Step 4: Cleanup
# -------------------------------------------------------
echo ""
echo "[4/4] Cleaning up..."
docker stop sglang-bench > /dev/null 2>&1
docker rm sglang-bench > /dev/null 2>&1
echo "  ✓ Container removed"

echo ""
echo "=============================================="
echo " Benchmark suite complete!"
echo " Results saved to: ${RESULTS_DIR}/"
echo "=============================================="
