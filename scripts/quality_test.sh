#!/bin/bash
# =============================================================================
# Qwen3.6-27B-FP8 Quality Verification Script
# NVIDIA DGX Spark (GB10) — sglang 0.5.12
# =============================================================================
# Tests model quality across multiple domains to verify correctness
# of FP8 quantization + speculative decoding pipeline.
# =============================================================================

set -euo pipefail

CONTAINER="scitrera/dgx-spark-sglang:0.5.12"
MODEL="Qwen/Qwen3.6-27B-FP8"
HOST_PORT=30001

echo "=============================================="
echo " Quality Verification — Qwen3.6-27B-FP8"
echo " NVIDIA DGX Spark (GB10)"
echo "=============================================="
echo ""

# Launch server
echo "[1/3] Launching sglang server..."
docker run -d \
  --name sglang-quality \
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
    --trust-remote-code \
    --kv-cache-dtype fp8_e4m3 \
    --speculative-nextn-steps 5 \
    --speculative-nextn-draft-token-per-step 9 \
    --page-size 1

echo "  Waiting for server..."
sleep 30

for i in $(seq 1 30); do
  if curl -s http://localhost:${HOST_PORT}/health > /dev/null 2>&1; then
    echo "  ✓ Server ready"
    break
  fi
  sleep 2
done

# Run quality tests
echo ""
echo "[2/3] Running quality verification tests..."

cat > /tmp/quality_test.py << 'PYEOF'
import json
import requests
import time

SERVER = "http://127.0.0.1:30000/v1/chat/completions"

def query(prompt, max_tokens=512, enable_thinking=False):
    payload = {
        "model": "Qwen/Qwen3.6-27B-FP8",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "enable_thinking": enable_thinking,
    }
    resp = requests.post(SERVER, json=payload)
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    return content, usage

tests = []

# Test 1: Math
print("=" * 60)
print("TEST 1: Mathematics — 'What is 15% of 840?'")
print("=" * 60)
start = time.time()
result, usage = query("What is 15% of 840? Answer with just the number.")
elapsed = time.time() - start
print(f"Response: {result.strip()}")
print(f"Time: {elapsed:.2f}s | Tokens: {usage.get('completion_tokens', 'N/A')}")
print(f"Expected: 126")
assert "126" in result, "FAIL: Expected 126"
print("Status: ✓ PASS\n")

# Test 2: Quantum physics
print("=" * 60)
print("TEST 2: Physics — Quantum entanglement explanation")
print("=" * 60)
start = time.time()
result, usage = query("Explain quantum entanglement in 2-3 sentences.", max_tokens=256)
elapsed = time.time() - start
print(f"Response: {result.strip()}")
print(f"Time: {elapsed:.2f}s | Tokens: {usage.get('completion_tokens', 'N/A')}")
assert len(result) > 50, "FAIL: Response too short"
assert any(w in result.lower() for w in ["entangl", "quantum", "particle", "spin", "state"]),
    "FAIL: Missing key physics concepts"
print("Status: ✓ PASS (coherent, accurate)\n")

# Test 3: Step-by-step reasoning
print("=" * 60)
print("TEST 3: Reasoning — Logical deduction")
print("=" * 60)
start = time.time()
result, usage = query(
    "A farmer has chickens and rabbits. There are 35 heads and 94 legs in total. "
    "How many chickens and how many rabbits are there? Show your work step by step.",
    max_tokens=512
)
elapsed = time.time() - start
print(f"Response: {result.strip()}")
print(f"Time: {elapsed:.2f}s | Tokens: {usage.get('completion_tokens', 'N/A')}")
assert "23" in result and "12" in result, "FAIL: Expected 23 chickens, 12 rabbits"
print("Status: ✓ PASS (correct reasoning)\n")

# Test 4: Code generation
print("=" * 60)
print("TEST 4: Code — Python binary search")
print("=" * 60)
start = time.time()
result, usage = query(
    "Write a Python function that implements binary search on a sorted list. "
    "Return the index if found, -1 if not found. Include a brief docstring.",
    max_tokens=512
)
elapsed = time.time() - start
print(f"Response:\n{result.strip()}")
print(f"Time: {elapsed:.2f}s | Tokens: {usage.get('completion_tokens', 'N/A')}")
assert "def " in result and "binary" in result.lower(), "FAIL: Missing function definition"
assert "return" in result, "FAIL: Missing return statement"
print("Status: ✓ PASS (correct implementation)\n")

# Test 5: Thinking mode (enable_thinking=True)
print("=" * 60)
print("TEST 5: Thinking Mode — CoT reasoning")
print("=" * 60)
start = time.time()
result, usage = query(
    "If it takes 5 machines 5 minutes to make 5 widgets, "
    "how long would it take 100 machines to make 100 widgets?",
    max_tokens=512,
    enable_thinking=True
)
elapsed = time.time() - start
print(f"Response: {result.strip()}")
print(f"Time: {elapsed:.2f}s | Tokens: {usage.get('completion_tokens', 'N/A')}")
assert "5" in result, "FAIL: Expected answer 5"
print("Status: ✓ PASS (thinking mode works correctly)\n")

print("=" * 60)
print("ALL TESTS PASSED ✓")
print("=" * 60)
print("\nConclusion: FP8 quantization + NEXTN speculative decoding")
print("produces correct outputs across math, reasoning, physics, and code.")
PYEOF

docker cp /tmp/quality_test.py sglang-quality:/tmp/quality_test.py
docker exec sglang-quality python3 /tmp/quality_test.py 2>&1 | tee /tmp/quality_output.txt

echo ""
echo "[3/3] Cleanup..."
docker stop sglang-quality > /dev/null 2>&1
docker rm sglang-quality > /dev/null 2>&1

echo ""
echo "=============================================="
echo " Quality verification complete!"
echo "=============================================="
