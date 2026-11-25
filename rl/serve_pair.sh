#!/usr/bin/env bash
# Two TP4 vLLM servers for RL (GPUs 0-3 on :8000, 4-7 on :8001) with LoRA hot-loading and sleep mode,
# so the trainer can take the GPUs between rollouts. Logs go to runs/vllm-rl-<port>.log.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p runs
for spec in "0,1,2,3 8000" "4,5,6,7 8001"; do
    set -- $spec
    # Keep the previous log (a restart would otherwise erase why the last server died).
    [ -s "runs/vllm-rl-$2.log" ] && mv "runs/vllm-rl-$2.log" "runs/vllm-rl-$2.log.prev"
    CUDA_VISIBLE_DEVICES=$1 TP=4 VLLM_PORT=$2 VLLM_SERVER_DEV_MODE=1 VLLM_ALLOW_RUNTIME_LORA_UPDATING=True \
        setsid nohup bash serve_vllm.sh --enable-lora --max-lora-rank 16 --max-loras 4 --enable-sleep-mode --gpu-memory-utilization 0.85 \
        > "runs/vllm-rl-$2.log" 2>&1 &
done
echo "Started; wait for 'Application startup complete' in runs/vllm-rl-800{0,1}.log"
