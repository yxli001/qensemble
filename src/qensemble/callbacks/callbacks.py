from typing import Any, Literal, cast

import tensorflow as tf

from qensemble.config import CallbacksConfig


def build_callbacks(
    cfg_callbacks: CallbacksConfig, run_dir: str, wandb_run: Any | None
) -> list[tf.keras.callbacks.Callback]:
    callbacks: list[tf.keras.callbacks.Callback] = []

    if cfg_callbacks.early_stopping.enabled:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=cfg_callbacks.early_stopping.patience,
                restore_best_weights=True,
            )
        )

    if cfg_callbacks.checkpoint.enabled:
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f"{run_dir}/best.weights.h5",
                monitor=cfg_callbacks.checkpoint.monitor.replace("/", "_"),
                save_best_only=True,
                save_weights_only=True,
                mode="max",
            )
        )

    if cfg_callbacks.wandb.enabled and wandb_run is not None:
        from wandb.integration.keras import WandbMetricsLogger

        raw_log_freq: str | int = cfg_callbacks.wandb.log_freq
        if isinstance(raw_log_freq, str):
            if raw_log_freq in {"epoch", "batch"}:
                safe_log_freq = cast(Literal["epoch", "batch"], raw_log_freq)
            else:
                safe_log_freq = "epoch"
            callbacks.append(WandbMetricsLogger(log_freq=safe_log_freq))
        else:
            callbacks.append(WandbMetricsLogger(log_freq=cast(int, raw_log_freq)))

    return callbacks
