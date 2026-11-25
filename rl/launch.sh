#!/usr/bin/env bash
# Launch the 8-GPU trainer against running rl/serve_pair.sh servers. Arguments go to rl.train.
#   bash rl/launch.sh --preset team4 --out runs/rl-team4
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Ask the OS for a free port; fixed or random ports on this VM are often already taken. Probe the wildcard
# address, as TCPStore listens on all addresses (a 127.0.0.1 probe once returned a port it could not bind).
port=$(.venv-agent-rl/bin/python -c 'import socket; s = socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')
exec .venv-agent-rl/bin/torchrun --nproc-per-node 8 --master-addr 127.0.0.1 --master-port "$port" -m rl.train "$@"
