#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 [num_agents] <entity/project/sweep_id>"
  echo "Example: CUDA_VISIBLE_DEVICES=0 $0 3 qensemble/qensemble/abc123"
  echo "Example: CUDA_VISIBLE_DEVICES=0 $0 qensemble/qensemble/abc123"
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage
  exit 1
fi

if [[ $# -eq 1 ]]; then
  num_agents="1"
  sweep_id="$1"
else
  num_agents="$1"
  sweep_id="$2"
fi

if ! [[ "$num_agents" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: <num_agents> must be a positive integer"
  usage
  exit 1
fi

gpu_setting="${CUDA_VISIBLE_DEVICES:-0}"
echo "Starting ${num_agents} agent(s) for sweep '${sweep_id}'"
echo "CUDA_VISIBLE_DEVICES=${gpu_setting}"

pids=()

cleanup() {
  local code=$?
  if [[ ${#pids[@]} -gt 0 ]]; then
    echo "Stopping running agents..."
    kill "${pids[@]}" 2>/dev/null || true
  fi
  exit "$code"
}

trap cleanup INT TERM

for ((i = 1; i <= num_agents; i++)); do
  echo "Launching agent ${i}/${num_agents}"
  CUDA_VISIBLE_DEVICES="$gpu_setting" wandb agent --count 1 "$sweep_id" &
  pids+=("$!")
done

failures=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failures=$((failures + 1))
  fi
done

if [[ "$failures" -gt 0 ]]; then
  echo "${failures} agent(s) exited with an error"
  exit 1
fi

echo "All agents completed successfully"