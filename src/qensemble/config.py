from copy import deepcopy
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    seed: int = 0
    out_root: str = "outputs"
    name: str
    name_prefix: str | None = None
    name_fields: list[str] = Field(default_factory=list)


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
    independent: bool = False


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


def _format_run_name_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        raise ValueError("Run name fields cannot resolve to None")
    if isinstance(value, dict):
        items = [
            f"{key}={_format_run_name_value(val)}" for key, val in sorted(value.items())
        ]
        return ",".join(items)
    if isinstance(value, (list | tuple)):
        parts = [_format_run_name_value(item) for item in value]
        return parts[0] if len(parts) == 1 else "x".join(parts)
    text = str(value)
    return text.replace("/", "-").replace(" ", "-")


def resolve_run_name(cfg: AppConfig) -> AppConfig:
    name_fields = list(cfg.run.name_fields)
    if not name_fields:
        return cfg

    base_name = str(cfg.run.name_prefix or cfg.run.name).strip()
    if not base_name:
        raise ValueError(
            "run.name or run.name_prefix must be set before composing a derived run name"
        )

    cfg_dump = cfg.model_dump()
    suffix_parts: list[str] = []
    for field_name in name_fields:
        current: Any = cfg_dump
        for part in field_name.split("."):
            if not isinstance(current, dict) or part not in current:
                raise ValueError(
                    f"Cannot compose run name: field '{field_name}' was not found"
                )
            current = current[part]
        if field_name == "quant" and isinstance(current, dict):
            suffix_parts.append(
                "quant="
                f"w{current['weight_total_bits']}a{current['activation_total_bits']}"
            )
        else:
            suffix_parts.append(f"{field_name}={_format_run_name_value(current)}")

    resolved_name = "-".join([base_name, *suffix_parts])
    return cfg.model_copy(
        update={"run": cfg.run.model_copy(update={"name": resolved_name})}
    )


def merge_wandb_overrides(
    cfg: AppConfig | dict[str, Any], wandb_run: Any | None
) -> AppConfig:
    cfg_dict = cfg.model_dump() if isinstance(cfg, AppConfig) else cfg

    if wandb_run is None:
        if isinstance(cfg, AppConfig):
            return resolve_run_name(cfg)
        return resolve_run_name(AppConfig.model_validate(cfg))

    valid_roots = set(cfg_dict.keys())
    # Keep sweep-provided config keys (including top-level objects like "quant")
    # and ignore W&B runtime metadata keys like "system.*".
    overrides = {
        k: v
        for k, v in dict(wandb_run.config).items()
        if k.split(".", 1)[0] in valid_roots
    }

    return resolve_run_name(apply_dotted_overrides(cfg, overrides))
