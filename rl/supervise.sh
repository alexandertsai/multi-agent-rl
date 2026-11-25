#!/usr/bin/env bash
# Keep an RL run going unattended: restart dead vLLM servers, relaunch the trainer after a failure
# (it resumes from the last checkpoint, saved every --ckpt-every steps), give up after MAX_RESTARTS.
#   setsid nohup bash rl/supervise.sh runs/rl-team4 --preset team4 --steps 115 > runs/supervise.log 2>&1 &
# If a trainer for this run is already running, it is waited for first.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
OUT=$1; shift
LOG=$OUT.log
MAX_RESTARTS=${MAX_RESTARTS:-6}
vllm_dir=$(pwd -P)/.venv-agent-rl/bin
VLLM="^$(printf '%s' "$vllm_dir" | sed 's/[][\.^$*+?(){}|]/\\&/g')/python .*/vllm serve"
say() { echo "[$(date '+%F %T')] $*"; }

servers_ok() {
    for port in 8000 8001; do
        curl -sf -m 10 "http://127.0.0.1:$port/v1/models" >/dev/null || return 1
    done
    ! grep -q EngineDeadError runs/vllm-rl-8000.log runs/vllm-rl-8001.log
}

restart_servers() {
    say "restarting vLLM servers"
    for pid in $(pgrep -f "$VLLM"); do kill "$pid"; done
    for _ in $(seq 60); do pgrep -f "$VLLM" >/dev/null || break; sleep 5; done
    for pid in $(pgrep -f "$VLLM"); do kill -9 "$pid"; done
    sleep 10
    bash rl/serve_pair.sh
    for _ in $(seq 180); do servers_ok && return 0; sleep 5; done
    say "servers did not come up"; return 1
}

trainer_running() { pgrep -f "rl.train .*--out $OUT( |$)" >/dev/null; }

restarts=0
while true; do
    if trainer_running; then
        say "waiting for the running trainer"
        while trainer_running; do sleep 30; done
    fi
    code=$(grep -oE '^exit [0-9]+' "$LOG" 2>/dev/null | tail -1 | cut -d' ' -f2)
    if [ "${code:-1}" = 0 ] && grep -q "eval step" "$LOG"; then
        say "run finished"; exit 0
    fi
    if [ "$restarts" -ge "$MAX_RESTARTS" ]; then
        say "giving up after $restarts restarts"; exit 1
    fi
    restarts=$((restarts + 1))
    say "trainer exited (${code:-unknown}); restart $restarts/$MAX_RESTARTS"
    sleep 30
    servers_ok || restart_servers || { sleep 300; continue; }
    echo "=== restart $restarts at $(date '+%F %T') ===" >> "$LOG"
    bash rl/launch.sh --out "$OUT" "$@" >> "$LOG" 2>&1
    echo "exit $?" >> "$LOG"
done
