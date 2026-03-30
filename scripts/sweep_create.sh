#!/usr/bin/env bash
set -euo pipefail

wandb sweep configs/cifar10/sweep_phase1_single.yaml
