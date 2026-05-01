import io
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from qensemble.callbacks.callbacks import build_callbacks
from qensemble.config import AppConfig, merge_wandb_overrides
from qensemble.datasets.openml import build_openml
from qensemble.datasets.tf_keras import build_tf_keras
from qensemble.ensemble.qensemble import QEnsemble
from qensemble.models.cnn_resnet import build_cnn_resnet
from qensemble.models.mlp import build_mlp
from qensemble.optim.optimizers import build_optimizer
from qensemble.utils.graphs import save_pairwise_disagreement_heatmap
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
        "model/ensemble_size": 1,
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
    if int(cfg.ensemble.size) <= 1:
        return "train_single"
    if bool(getattr(cfg.ensemble, "independent", False)):
        return "train_independent"
    return "train_dependent"


def _new_run_dir(cfg: AppConfig) -> Path:
    out_root = Path(cfg.run.out_root)
    name = cfg.run.name
    run_dir = out_root / name
    if run_dir.exists():
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = out_root / f"{name}-{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _artifact_name_with_timestamp(run_name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_run_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", run_name).strip("-._")
    if not safe_run_name:
        safe_run_name = "run"
    return f"{safe_run_name}-{timestamp}"


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


def _evaluate_model(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
) -> dict[str, float]:
    eval_values = model.evaluate(test_ds, return_dict=True, verbose=1)
    if isinstance(eval_values, dict):
        return {f"test/{k}": float(v) for k, v in eval_values.items()}

    metric_names = ["loss", *model.metrics_names[1:]]
    values = eval_values if isinstance(eval_values, list) else [eval_values]
    return {f"test/{k}": float(v) for k, v in zip(metric_names, values, strict=False)}


def _collect_member_predicted_classes_and_labels(
    members: list[tf.keras.Model], dataset: tf.data.Dataset
) -> tuple[np.ndarray, np.ndarray]:
    pred_chunks: list[list[np.ndarray]] = [[] for _ in members]
    label_chunks: list[np.ndarray] = []

    for batch in dataset:
        if not isinstance(batch, tuple | list) or len(batch) < 2:
            msg = "Dataset batches must contain both inputs and labels"
            raise ValueError(msg)

        x_batch = batch[0]
        y_batch = batch[1]

        y_np = np.asarray(y_batch.numpy())
        if y_np.ndim > 1:
            y_np = np.argmax(y_np, axis=-1)
        label_chunks.append(y_np.reshape(-1).astype(np.int32, copy=False))

        for idx, member in enumerate(members):
            logits = member(x_batch, training=False)
            pred = tf.argmax(logits, axis=-1, output_type=tf.int32)
            pred_chunks[idx].append(np.asarray(pred.numpy()).reshape(-1))

    if not pred_chunks or not pred_chunks[0] or not label_chunks:
        raise ValueError("No predictions or labels available from dataset")

    member_pred_classes = np.stack(
        [np.concatenate(chunks) for chunks in pred_chunks], axis=0
    )
    true_labels = np.concatenate(label_chunks)

    return member_pred_classes, true_labels


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
        name=_artifact_name_with_timestamp(cfg.run.name),
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

    member_pred_classes, true_labels = _collect_member_predicted_classes_and_labels(
        qensemble.members, test_ds
    )

    member_test_accuracies: list[float] = []
    for idx, pred_classes in enumerate(member_pred_classes):
        member_test_acc = float(np.mean(pred_classes == true_labels))
        metrics[f"member_{idx}_test_acc"] = member_test_acc
        member_test_accuracies.append(member_test_acc)

    metrics["test/member_test_acc_mean"] = float(np.mean(member_test_accuracies))

    save_pairwise_disagreement_heatmap(
        member_pred_classes=member_pred_classes,
        output_path=bundle_dir / "pairwise_disagreement.png",
    )

    save_dependent_bundle(str(bundle_dir), cfg, qensemble, metrics)
    log_bundle_as_artifact(
        wandb_run,
        bundle_dir=str(bundle_dir),
        name=_artifact_name_with_timestamp(cfg.run.name),
        artifact_type="dependent_ensemble",
        aliases=["latest"],
    )

    if wandb_run is not None:
        wandb_run.log(metrics)
        wandb_run.finish()

    return metrics


def run_train_independent(
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

    metrics: dict[str, float] = {}
    for idx, member in enumerate(members):
        member_run_dir = run_dir / f"member_{idx}"
        _compile_model(member, cfg)
        _fit_and_eval(
            member,
            cfg,
            train_ds,
            val_ds,
            test_ds,
            member_run_dir,
            wandb_run,
        )

    qensemble = QEnsemble(members=members)
    _compile_model(qensemble, cfg)
    ensemble_metrics = _evaluate_model(qensemble, test_ds)
    metrics.update(ensemble_metrics)
    _log_model_summary(qensemble, run_dir)
    metrics.update(_ensemble_param_metrics(qensemble, int(cfg.quant.weight_total_bits)))

    bundle_dir = run_dir / "bundle"

    member_pred_classes, true_labels = _collect_member_predicted_classes_and_labels(
        qensemble.members, test_ds
    )

    member_test_accuracies: list[float] = []
    for idx, pred_classes in enumerate(member_pred_classes):
        member_test_acc = float(np.mean(pred_classes == true_labels))
        metrics[f"member_{idx}_test_acc"] = member_test_acc
        member_test_accuracies.append(member_test_acc)

    metrics["test/member_test_acc_mean"] = float(np.mean(member_test_accuracies))

    save_pairwise_disagreement_heatmap(
        member_pred_classes=member_pred_classes,
        output_path=bundle_dir / "pairwise_disagreement.png",
    )

    save_dependent_bundle(str(bundle_dir), cfg, qensemble, metrics)
    log_bundle_as_artifact(
        wandb_run,
        bundle_dir=str(bundle_dir),
        name=_artifact_name_with_timestamp(cfg.run.name),
        artifact_type="independent_ensemble",
        aliases=["latest"],
    )

    if wandb_run is not None:
        wandb_run.log(metrics)
        wandb_run.finish()

    return metrics
