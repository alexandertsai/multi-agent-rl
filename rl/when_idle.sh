#!/usr/bin/env bash
# Run a command once every GPU has been idle for IDLE_MINUTES in a row, then exit.
#   setsid nohup bash rl/when_idle.sh bash runs/resume-team4.sh > runs/when-idle.log 2>&1 &
# Idle: every GPU at or below THRESHOLD % utilisation, sampled every INTERVAL seconds; any busy
# sample restarts the clock. If the vLLM servers are down, GPUs must also be nearly empty
# (another program holding memory would make the resumed run fail with out-of-memory).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
IDLE_MINUTES=${IDLE_MINUTES:-40}
THRESHOLD=${THRESHOLD:-5}
INTERVAL=${INTERVAL:-10}
FREE_MIB=${FREE_MIB:-2048}
say() { echo "[$(date '+%F %T')] $*"; }

servers_up() {
    for port in 8000 8001; do curl -sf -m 5 "http://127.0.0.1:$port/v1/models" >/dev/null || return 1; done
}

gpus_idle() {
    local util mem
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits) || return 1
    [ "$(echo "$util" | wc -l)" -ge 1 ] || return 1
    echo "$util" | awk -v t="$THRESHOLD" '$1 > t {busy = 1} END {exit busy}' || return 1
    servers_up && return 0
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) || return 1
    echo "$mem" | awk -v m="$FREE_MIB" '$1 > m {used = 1} END {exit used}'
}

say "waiting until all GPUs are <= ${THRESHOLD}% for ${IDLE_MINUTES} min, then: $*"
since=$(date +%s); last_report=0
while true; do
    now=$(date +%s)
    gpus_idle || since=$now
    idle=$(( (now - since) / 60 ))
    if [ $((now - since)) -ge $((IDLE_MINUTES * 60)) ]; then
        say "GPUs idle for ${IDLE_MINUTES} min; starting"
        exec "$@"
    fi
    if [ $((now - last_report)) -ge 600 ]; then say "idle for ${idle} min"; last_report=$now; fi
    sleep "$INTERVAL"
done
