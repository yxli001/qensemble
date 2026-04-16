import json
from pathlib import Path
from typing import Any

import yaml

import wandb
from qensemble.config import AppConfig


def log_bundle_as_artifact(
    run: Any | None,
    bundle_dir: str,
    name: str,
    artifact_type: str,
    aliases: list[str] | None = None,
) -> None:
    if run is None:
        return

    artifact = wandb.Artifact(name=name, type=artifact_type)
    artifact.add_dir(bundle_dir)
    run.log_artifact(artifact, aliases=aliases or [])


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
    model.save(str(bundle_path / "model.keras"))
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

    ensemble_functional = qensemble.to_functional_model(name="qensemble")
    ensemble_functional.save(str(bundle_path / "ensemble.keras"))

    qensemble.save_member_models(str(bundle_path), prefix="model")

    _write_json(bundle_path / "metrics.json", metrics)
