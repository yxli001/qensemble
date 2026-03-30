import json
from pathlib import Path
from typing import Any

import yaml

from qensemble.config import AppConfig


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _write_config(path: Path, cfg: AppConfig) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(cfg.model_dump(), f, sort_keys=False)


def save_single_bundle(
    bundle_dir: str, cfg: AppConfig, model: Any, metrics: dict[str, float]
) -> None:
    bundle_path = Path(bundle_dir)
    bundle_path.mkdir(parents=True, exist_ok=True)

    _write_config(bundle_path / "config.yaml", cfg)
    model.save_weights(str(bundle_path / "model.weights.h5"))
    _write_json(bundle_path / "metrics.json", metrics)


def save_dependent_bundle(
    bundle_dir: str,
    cfg: AppConfig,
    qensemble: Any,
    metrics: dict[str, float],
) -> None:
    bundle_path = Path(bundle_dir)
    bundle_path.mkdir(parents=True, exist_ok=True)

    _write_config(bundle_path / "config.yaml", cfg)
    qensemble.save_member_weights(str(bundle_path), prefix="model")

    meta = {"ensemble_size": int(qensemble.size), "format_version": 1}
    _write_json(bundle_path / "ensemble_meta.json", meta)
    _write_json(bundle_path / "metrics.json", metrics)
