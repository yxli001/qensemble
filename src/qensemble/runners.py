import datetime as dt
from pathlib import Path
from typing import Any

import tensorflow as tf

from qensemble.callbacks.callbacks import build_callbacks
from qensemble.config import AppConfig, merge_wandb_overrides
from qensemble.ensemble.qensemble_model import QEnsemble
from qensemble.factories import build_dataset, build_model
from qensemble.io.bundles import save_dependent_bundle, save_single_bundle
from qensemble.optim.optimizers import build_optimizer
from qensemble.utils.seed import set_seed
from qensemble.utils.tf_gpu import configure_gpu_memory_growth, log_visible_devices
from qensemble.wandb.artifacts import log_bundle_as_artifact
from qensemble.wandb.setup import init_wandb


def _new_run_dir(cfg: AppConfig) -> Path:
    out_root = Path(cfg.run.out_root)
    name = cfg.run.name
    if name == "auto":
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        name = f"{cfg.run.mode}-{stamp}"
    run_dir = out_root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _compile_model(model: tf.keras.Model, cfg: AppConfig) -> None:
    optimizer = build_optimizer(cfg.train)
    loss_name = cfg.train.loss
    metric_names = cfg.train.metrics

    if loss_name == "sparse_ce":
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    elif loss_name == "ce":
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    else:
        raise ValueError(f"Unsupported loss '{loss_name}'")

    metrics: list[tf.keras.metrics.Metric] = []
    for metric_name in metric_names:
        if metric_name == "sparse_acc":
            metrics.append(
                tf.keras.metrics.SparseCategoricalAccuracy(name="sparse_acc")
            )
        elif metric_name == "acc":
            metrics.append(tf.keras.metrics.CategoricalAccuracy(name="acc"))
        else:
            raise ValueError(f"Unsupported metric '{metric_name}'")

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)


def _fit_and_eval(
    model: tf.keras.Model,
    cfg: AppConfig,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    test_ds: tf.data.Dataset,
    run_dir: Path,
    wandb_run: Any | None,
) -> dict[str, float]:
    callbacks = build_callbacks(cfg.callbacks, str(run_dir), wandb_run)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.train.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    eval_values = model.evaluate(test_ds, return_dict=True, verbose=1)
    if isinstance(eval_values, dict):
        return {f"test/{k}": float(v) for k, v in eval_values.items()}

    metric_names = ["loss", *model.metrics_names[1:]]
    values = eval_values if isinstance(eval_values, list) else [eval_values]
    return {f"test/{k}": float(v) for k, v in zip(metric_names, values, strict=False)}


def run_train_single(cfg: AppConfig, wandb_run: Any | None = None) -> dict[str, float]:
    cfg = cfg.model_copy(deep=True)
    set_seed(cfg.run.seed)
    configure_gpu_memory_growth()
    log_visible_devices()

    if wandb_run is None:
        wandb_run = init_wandb(cfg)
    cfg = merge_wandb_overrides(cfg, wandb_run)

    run_dir = _new_run_dir(cfg)
    train_ds, val_ds, test_ds, info = build_dataset(cfg.data)
    model = build_model(cfg.model, cfg.quant, info)

    _compile_model(model, cfg)
    metrics = _fit_and_eval(model, cfg, train_ds, val_ds, test_ds, run_dir, wandb_run)

    bundle_dir = run_dir / "bundle"
    save_single_bundle(str(bundle_dir), cfg, model, metrics)
    log_bundle_as_artifact(
        wandb_run,
        bundle_dir=str(bundle_dir),
        name=f"{run_dir.name}-single",
        artifact_type="model",
        aliases=["latest"],
    )

    if wandb_run is not None:
        wandb_run.log(metrics)
        wandb_run.finish()

    return metrics


def run_train_dependent(
    cfg: AppConfig, wandb_run: Any | None = None
) -> dict[str, float]:
    cfg = cfg.model_copy(deep=True)
    set_seed(cfg.run.seed)
    configure_gpu_memory_growth()
    log_visible_devices()

    if wandb_run is None:
        wandb_run = init_wandb(cfg)
    cfg = merge_wandb_overrides(cfg, wandb_run)

    run_dir = _new_run_dir(cfg)
    train_ds, val_ds, test_ds, info = build_dataset(cfg.data)

    member_kwargs = {
        "cfg_model": cfg.model,
        "cfg_quant": cfg.quant,
        "info": info,
    }

    def _member_builder(**kwargs: Any) -> tf.keras.Model:
        return build_model(kwargs["cfg_model"], kwargs["cfg_quant"], kwargs["info"])

    qensemble = QEnsemble(
        member_builder=_member_builder,
        size=cfg.ensemble.size,
        member_kwargs=member_kwargs,
    )

    _compile_model(qensemble, cfg)
    metrics = _fit_and_eval(
        qensemble, cfg, train_ds, val_ds, test_ds, run_dir, wandb_run
    )

    bundle_dir = run_dir / "bundle"
    save_dependent_bundle(str(bundle_dir), cfg, qensemble, metrics)
    log_bundle_as_artifact(
        wandb_run,
        bundle_dir=str(bundle_dir),
        name=f"{run_dir.name}-dependent",
        artifact_type="dependent_ensemble",
        aliases=["latest"],
    )

    if wandb_run is not None:
        wandb_run.log(metrics)
        wandb_run.finish()

    return metrics
