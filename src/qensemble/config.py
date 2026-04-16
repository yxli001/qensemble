from copy import deepcopy
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    seed: int = 0
    out_root: str = "outputs"
    name: str


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    source: str
    dataset_name: str
    batch_size: int = 128
    augment: bool | None = None
    val_split: float = 0.1
    # OpenML loader settings.
    test_split: float = 0.2
    random_state: int = 42
    cache: bool = False


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str


class QuantConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    activation_total_bits: int
    activation_int_bits: int
    weight_total_bits: int
    weight_int_bits: int


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    epochs: int = 1
    optimizer: str = "adam"
    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-7
    weight_decay: float = 0.0
    loss: str = "sparse_ce"
    metrics: list[str] = Field(default_factory=lambda: ["sparse_acc"])


class EarlyStoppingConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    enabled: bool = False
    patience: int = 10


class CheckpointConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    enabled: bool = False
    monitor: str = "val/sparse_acc"


class CallbackWandbConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    enabled: bool = False
    log_freq: str = "epoch"


class CallbacksConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    early_stopping: EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    wandb: CallbackWandbConfig = Field(default_factory=CallbackWandbConfig)


class EnsembleConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    size: int = 1


class WandbConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    enabled: bool = False
    project: str = "qensemble"
    entity: str | None = None
    group: str | None = None
    tags: list[str] = Field(default_factory=list)


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    run: RunConfig
    data: DataConfig
    model: ModelConfig
    quant: QuantConfig
    train: TrainConfig = Field(default_factory=TrainConfig)
    callbacks: CallbacksConfig = Field(default_factory=CallbacksConfig)
    ensemble: EnsembleConfig = Field(default_factory=EnsembleConfig)
    wandb: WandbConfig


def load_yaml(path: str) -> dict[str, Any]:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        msg = f"Config root must be a mapping, got {type(data).__name__}"
        raise ValueError(msg)
    return data


def load_config(path: str) -> AppConfig:
    return AppConfig.model_validate(load_yaml(path))


def _set_dotted_key(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    current = cfg
    for part in parts[:-1]:
        next_value = current.get(part)
        if next_value is None:
            next_value = {}
            current[part] = next_value
        if not isinstance(next_value, dict):
            msg = f"Cannot assign '{dotted_key}': '{part}' is not a mapping"
            raise ValueError(msg)
        current = next_value
    current[parts[-1]] = value


def apply_dotted_overrides(
    cfg: AppConfig | dict[str, Any], overrides: dict[str, Any] | None
) -> AppConfig:
    if not overrides:
        if isinstance(cfg, AppConfig):
            return cfg
        return AppConfig.model_validate(cfg)

    out = deepcopy(cfg.model_dump() if isinstance(cfg, AppConfig) else cfg)
    for key, value in overrides.items():
        _set_dotted_key(out, key, value)
    return AppConfig.model_validate(out)


def merge_wandb_overrides(
    cfg: AppConfig | dict[str, Any], wandb_run: Any | None
) -> AppConfig:
    if wandb_run is None:
        if isinstance(cfg, AppConfig):
            return cfg
        return AppConfig.model_validate(cfg)
    # W&B sweep parameters are typically flat keys like "model.variant".
    overrides = {k: v for k, v in dict(wandb_run.config).items() if "." in k}
    return apply_dotted_overrides(cfg, overrides)
