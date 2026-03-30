import socket

import wandb
from qensemble.config import AppConfig
from qensemble.utils.env import get_env
from qensemble.utils.gitinfo import get_git_sha


def _resolve_wandb_name(cfg: AppConfig) -> str | None:
    wcfg = cfg.wandb
    if wcfg.name:
        return wcfg.name.replace("${run.mode}", cfg.run.mode).replace(
            "${run.name}", cfg.run.name
        )
    if cfg.run.name != "auto":
        return cfg.run.name
    return None


def init_wandb(cfg: AppConfig) -> object | None:
    wcfg = cfg.wandb
    if not wcfg.enabled:
        return None

    group = wcfg.group
    if group == "${run.mode}":
        group = cfg.run.mode

    run = wandb.init(
        project=wcfg.project,
        name=_resolve_wandb_name(cfg),
        entity=wcfg.entity,
        group=group,
        tags=wcfg.tags,
        config=cfg.model_dump(),
    )
    run.config.update(
        {
            "system.git_sha": get_git_sha(),
            "system.hostname": socket.gethostname(),
            "system.cuda_visible_devices": get_env().CUDA_VISIBLE_DEVICES,
        },
        allow_val_change=True,
    )
    return run
