#!/usr/bin/env bash
set -euo pipefail

SWEEP_CONFIG="${1}"

if [[ ! -f "$SWEEP_CONFIG" ]]; then
	echo "Sweep config not found: $SWEEP_CONFIG" >&2
	echo "Usage: bash scripts/sweep_create.sh [path/to/sweep.yaml]" >&2
	exit 1
fi

wandb sweep "$SWEEP_CONFIG"
