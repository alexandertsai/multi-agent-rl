#!/usr/bin/env bash
# Set up the Python environment and run checks on an 8×H100 host.
# Run from the repository root with: source setup.sh
# The environment lives in .venv-agent-rl. Supply local weights separately.

# Install in a subshell so failures cannot change the caller's shell options.
_agent_rl_setup() (
    set -euo pipefail
    cd "$(dirname "${BASH_SOURCE[0]}")"
    export PATH="$HOME/.local/bin:$PATH"
    venv=.venv-agent-rl

    echo "[agent-rl] Checking the machine..."
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sort | uniq -c
        gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | grep -c H100 || true)
        [ "$gpus" -eq 8 ] || echo "[agent-rl] WARNING: expected 8 H100s, found $gpus; serve_vllm.sh assumes 8."
        # The CUDA 12.9 build needs a CUDA 12.x driver (R525.60+) via minor-version compatibility.
        cuda=$(nvidia-smi | grep -o 'CUDA Version: [0-9.]*' | grep -o '[0-9.]*$')
        if [ "${cuda%%.*}" -lt 12 ]; then
            echo "[agent-rl] The NVIDIA driver supports CUDA $cuda; this environment needs CUDA 12.x." >&2
            exit 1
        fi
    else
        echo "[agent-rl] WARNING: nvidia-smi not found; vLLM serving will not work on this machine."
    fi
    if ! command -v uv >/dev/null 2>&1; then
        echo "[agent-rl] Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi
    if [ ! -x "$venv/bin/python" ]; then
        uv venv --python 3.12 "$venv"
    fi
    echo "[agent-rl] Installing vLLM and rollout dependencies from requirements.lock..."
    # vLLM 0.32.0's CUDA 12.9 wheel; torch must be the matching cu129 build.
    uv pip sync --python "$venv/bin/python" --torch-backend cu129 requirements.lock

    echo "[agent-rl] Checking the environment..."
    "$venv/bin/python" - <<'EOF'
import chess, openai, torch, vllm
print(f'vllm {vllm.__version__}, torch {torch.__version__} (CUDA {torch.version.cuda}), '
      f'{torch.cuda.device_count()} GPUs visible, openai {openai.__version__}, chess {chess.__version__}')
EOF
    "$venv/bin/python" check_gpus.py
    "$venv/bin/python" -m unittest tests.test_chess_env
)

_agent_rl_setup
_agent_rl_setup_status=$?
unset -f _agent_rl_setup
if [ "$_agent_rl_setup_status" -ne 0 ]; then
    unset _agent_rl_setup_status
    echo "[agent-rl] Setup failed. Fix the error above and rerun." >&2
    return 1 2>/dev/null || exit 1
fi
unset _agent_rl_setup_status

export PATH="$HOME/.local/bin:$PATH"
source "$(dirname "${BASH_SOURCE[0]}")/.venv-agent-rl/bin/activate"
echo "[agent-rl] Ready. Start the server with: bash serve_vllm.sh"
