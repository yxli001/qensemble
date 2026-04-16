import io
from pathlib import Path
from typing import Any

import tensorflow as tf

from qensemble.callbacks.callbacks import build_callbacks
from qensemble.config import AppConfig, merge_wandb_overrides
from qensemble.datasets.openml import build_openml
from qensemble.datasets.tf_keras import build_tf_keras
from qensemble.ensemble.qensemble import QEnsemble
from qensemble.models.cnn_resnet import build_cnn_resnet
from qensemble.models.mlp import build_mlp
from qensemble.optim.optimizers import build_optimizer
from qensemble.utils.seed import set_seed
from qensemble.utils.tf_gpu import configure_gpu_memory_growth, log_visible_devices
from qensemble.wandb.artifacts import (
    log_bundle_as_artifact,
    save_dependent_bundle,
    save_single_bundle,
)
from qensemble.wandb.setup import init_wandb


def build_dataset(cfg_data: Any) -> tuple[Any, Any, Any, dict[str, Any]]:
    source = str(cfg_data.source).lower()
    if source == "tf_keras":
        return build_tf_keras(cfg_data)
    if source == "openml":
        return build_openml(cfg_data)
    raise ValueError(f"Unsupported data.source '{cfg_data.source}'")


def build_model(cfg_model: Any, cfg_quant: Any, info: dict[str, Any]) -> tf.keras.Model:
    name = str(cfg_model.name).lower()
    if name == "mlp":
        return build_mlp(cfg_model, cfg_quant, info)
    if name == "cnn_resnet":
        return build_cnn_resnet(cfg_model, cfg_quant, info)
    raise ValueError(f"Unsupported model.name '{cfg_model.name}'")


def _model_param_metrics(
    model: tf.keras.Model, weight_total_bits: int
) -> dict[str, int]:
    num_params = int(model.count_params())
    return {
        "model/num_params": num_params,
        "model/param_bits_total": num_params * int(weight_total_bits),
    }


def _ensemble_param_metrics(
    qensemble: tf.keras.Model, weight_total_bits: int
) -> dict[str, int]:
    member_num_params = 0
    members = getattr(qensemble, "members", [])
    if members:
        member_num_params = int(members[0].count_params())

    total_num_params = int(qensemble.count_params())

    return {
        "model/num_params": total_num_params,
        "model/param_bits_total": total_num_params * int(weight_total_bits),
        "model/member_num_params": member_num_params,
        "model/member_param_bits_total": member_num_params * int(weight_total_bits),
        "model/ensemble_size": int(qensemble.size),
    }


def infer_training_mode(cfg: AppConfig) -> str:
    return "train_dependent" if int(cfg.ensemble.size) > 1 else "train_single"


def _new_run_dir(cfg: AppConfig) -> Path:
    out_root = Path(cfg.run.out_root)
    name = cfg.run.name
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


def _log_model_summary(
    model: tf.keras.Model,
    run_dir: Path,
) -> None:
    try:
        buffer = io.StringIO()

        def _write_line(line: str) -> None:
            buffer.write(f"{line}\n")

        model.summary(print_fn=_write_line)
        summary_text = buffer.getvalue().rstrip()
        print(summary_text)

        summary_name = (
            "ensemble_architecture.txt"
            if isinstance(model, QEnsemble)
            else "model_architecture.txt"
        )
        summary_path = run_dir / "bundle" / summary_name
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(f"{summary_text}\n")
    except Exception as exc:
        print(f"[warn] Could not log model summary: {exc}")


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
    _log_model_summary(model, run_dir)
    metrics = _fit_and_eval(model, cfg, train_ds, val_ds, test_ds, run_dir, wandb_run)
    metrics.update(_model_param_metrics(model, int(cfg.quant.weight_total_bits)))

    bundle_dir = run_dir / "bundle"
    save_single_bundle(str(bundle_dir), cfg, model, metrics)
    log_bundle_as_artifact(
        wandb_run,
        bundle_dir=str(bundle_dir),
        name=run_dir.name,
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

    members = [
        build_model(cfg.model, cfg.quant, info) for _ in range(int(cfg.ensemble.size))
    ]
    qensemble = QEnsemble(members=members)

    _compile_model(qensemble, cfg)
    metrics = _fit_and_eval(
        qensemble, cfg, train_ds, val_ds, test_ds, run_dir, wandb_run
    )
    _log_model_summary(qensemble, run_dir)
    metrics.update(_ensemble_param_metrics(qensemble, int(cfg.quant.weight_total_bits)))

    bundle_dir = run_dir / "bundle"
    save_dependent_bundle(str(bundle_dir), cfg, qensemble, metrics)
    log_bundle_as_artifact(
        wandb_run,
        bundle_dir=str(bundle_dir),
        name=run_dir.name,
        artifact_type="dependent_ensemble",
        aliases=["latest"],
    )

    if wandb_run is not None:
        wandb_run.log(metrics)
        wandb_run.finish()

    return metrics
