#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 [num_agents] [--count N] <entity/project/sweep_id>"
  echo "Example: CUDA_VISIBLE_DEVICES=0 $0 3 qensemble/qensemble/abc123"
  echo "Example: CUDA_VISIBLE_DEVICES=0 $0 qensemble/qensemble/abc123"
  echo "Example: CUDA_VISIBLE_DEVICES=0 $0 3 --count 1 qensemble/qensemble/abc123"
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

num_agents="1"
agent_count=""

if [[ "$1" =~ ^[1-9][0-9]*$ ]]; then
  num_agents="$1"
  shift
fi

if [[ $# -eq 0 ]]; then
  usage
  exit 1
fi

if [[ "$1" == "--count" ]]; then
  if [[ $# -ne 3 ]]; then
    usage
    exit 1
  fi
  agent_count="$2"
  sweep_id="$3"
else
  sweep_id="$1"
fi

if ! [[ "$num_agents" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: <num_agents> must be a positive integer"
  usage
  exit 1
fi

gpu_setting="${CUDA_VISIBLE_DEVICES:-5}"
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
  if [[ -n "$agent_count" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu_setting" wandb agent --count "$agent_count" "$sweep_id" &
  else
    CUDA_VISIBLE_DEVICES="$gpu_setting" wandb agent "$sweep_id" &
  fi
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