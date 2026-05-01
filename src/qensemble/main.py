import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
tf.compat.v1.enable_eager_execution()

import argparse
from typing import Any

import yaml

from qensemble.config import (
    AppConfig,
    apply_dotted_overrides,
    load_config,
    merge_wandb_overrides,
)
from qensemble.wandb.setup import init_wandb


def _parse_set_overrides(values: list[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Invalid --set value '{item}'. Expected key=value")
        key, raw = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --set value '{item}'. Key cannot be empty")
        overrides[key] = yaml.safe_load(raw)
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(description="QEnsemble training entrypoint")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override config values using dotted keys, e.g. model.variant=large",
    )
    args = parser.parse_args()

    cfg: AppConfig = load_config(args.config)
    cfg = apply_dotted_overrides(cfg, _parse_set_overrides(args.set))

    wandb_run = init_wandb(cfg)
    cfg = merge_wandb_overrides(cfg, wandb_run)

    from qensemble.runners import (
        infer_training_mode,
        run_train_dependent,
        run_train_independent,
        run_train_single,
    )

    mode = infer_training_mode(cfg)
    if mode == "train_single":
        metrics = run_train_single(cfg, wandb_run=wandb_run)
    elif mode == "train_independent":
        metrics = run_train_independent(cfg, wandb_run=wandb_run)
    else:
        metrics = run_train_dependent(cfg, wandb_run=wandb_run)

    print("Final metrics:")
    for key, value in metrics.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
