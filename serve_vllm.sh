#!/usr/bin/env bash
# Serve Qwen3.8-Flash-Next-FP8 on 8x H100 80GB with an OpenAI-compatible API.
# Requires vLLM >= 0.30.0. Flags follow the vLLM recipe's 8x H100 FP8 configuration
# (TEP8 with the Triton MoE backend; the n-gram embedding is CPU-offloaded by default).
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOCAL=${MODEL_LOCAL_DIR:-"$HERE/models/Qwen3.8-Flash-Next-FP8"}
PYTHON="$HERE/.venv-agent-rl/bin/python"
[ -x "$PYTHON" ] || { echo "Run 'source setup.sh' first." >&2; exit 1; }

# Prefix caching and MTP speculative decoding are off by default (PREFIX_CACHING=1, SPEC_TOKENS=3 enable
# them): on this hybrid model each request keeps one linear-attention state per speculative position,
# which caps concurrency; without them a TP4 server sustains ~10K tok/s at 300+ requests vs ~2.5K.
extra=()
[ "${PREFIX_CACHING:-0}" = 1 ] && extra+=(--enable-prefix-caching) || extra+=(--no-enable-prefix-caching)
[ "${SPEC_TOKENS:-0}" -gt 0 ] && extra+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${SPEC_TOKENS:-0}}")

[ -f "$LOCAL/config.json" ] || { echo "Set MODEL_LOCAL_DIR to a local FP8 checkpoint." >&2; exit 1; }

exec "$HERE/.venv-agent-rl/bin/vllm" serve "$LOCAL" \
  --served-model-name Qwen3.8-Flash-Next \
  --tensor-parallel-size "${TP:-8}" \
  --enable-expert-parallel \
  --moe-backend triton \
  --max-num-seqs "${MAX_NUM_SEQS:-512}" \
  --kv-cache-dtype fp8 \
  --attention-config.indexer_kv_dtype fp8 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --host 127.0.0.1 --port "${VLLM_PORT:-8000}" "${extra[@]}" "$@"
