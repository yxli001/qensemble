#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 <base_config> [run_prefix]"
  echo "Example: $0 configs/mnist/base.yaml"
  echo "Example: $0 configs/cifar10/base.yaml cifar10"
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage
  exit 1
fi

config_path="$1"
if [[ ! -f "$config_path" ]]; then
  echo "Error: base config not found: $config_path"
  exit 1
fi

if [[ $# -eq 2 ]]; then
  run_prefix="$2"
else
  run_prefix="$(python - "$config_path" <<'PY'
import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1])
with config_path.open() as f:
    config = yaml.safe_load(f) or {}

run_name = config.get("run", {}).get("name")
if not run_name:
    raise SystemExit(f"Error: run.name is missing from {config_path}")

print(run_name)
PY
)"
fi

seeds=(0 1 42)

combos=(
  "w8a8-single|8|2|8|3|1"
  "w4a4-ens2|4|1|4|2|2"
  "w2a2-ens4|2|1|2|1|4"
)

for combo in "${combos[@]}"; do
  IFS='|' read -r combo_name weight_total_bits weight_int_bits activation_total_bits activation_int_bits ensemble_size <<< "$combo"

  for seed in "${seeds[@]}"; do
    run_name="${run_prefix}-${combo_name}-seed${seed}"
    echo "Running ${run_name}"

    python -m qensemble.main \
      --config "$config_path" \
      --set run.name="$run_name" \
      --set run.seed="$seed" \
      --set quant.weight_total_bits="$weight_total_bits" \
      --set quant.weight_int_bits="$weight_int_bits" \
      --set quant.activation_total_bits="$activation_total_bits" \
      --set quant.activation_int_bits="$activation_int_bits" \
      --set ensemble.size="$ensemble_size"
  done
done