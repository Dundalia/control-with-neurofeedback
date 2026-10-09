#!/usr/bin/env bash
# Serve OLMo-3.1-32B-Instruct as an OpenAI-compatible endpoint, for its role as a writer in
# Experiment 1 (it is not on OpenRouter). The subjects reach it at $OLMO_URL, e.g.
#   CUDA_VISIBLE_DEVICES=3 bash scripts/serve_olmo.sh &
#   export OLMO_URL=http://localhost:8000/v1
set -euo pipefail
PORT="${PORT:-8000}"
export VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_USE_FLASHINFER_SAMPLER=0
exec vllm serve allenai/Olmo-3.1-32B-Instruct --host 0.0.0.0 --port "$PORT" \
  --served-model-name allenai/Olmo-3.1-32B-Instruct \
  --max-model-len 16384 --gpu-memory-utilization 0.9 --enforce-eager
