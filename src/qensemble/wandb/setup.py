import socket

import wandb
from qensemble.config import AppConfig
from qensemble.utils.env import get_env
from qensemble.utils.gitinfo import get_git_sha


def init_wandb(cfg: AppConfig) -> object | None:
    wcfg = cfg.wandb
    if not wcfg.enabled:
        return None

    group = wcfg.group

    run = wandb.init(
        project=wcfg.project,
        name=cfg.run.name,
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
