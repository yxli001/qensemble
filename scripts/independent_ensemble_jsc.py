import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import argparse

import numpy as np
import tensorflow as tf

from qensemble.config import AppConfig, load_config
from qensemble.runners import _compile_model, build_dataset, build_model
from qensemble.utils.seed import set_seed
from qensemble.utils.tf_gpu import configure_gpu_memory_growth, log_visible_devices


def _normalize_labels(y: tf.Tensor) -> np.ndarray:
    y_np = np.asarray(y.numpy())
    if y_np.ndim > 1:
        y_np = np.argmax(y_np, axis=-1)
    return y_np.reshape(-1).astype(np.int32, copy=False)


def _collect_test_arrays(test_ds: tf.data.Dataset) -> tuple[np.ndarray, np.ndarray]:
    x_chunks: list[np.ndarray] = []
    y_chunks: list[np.ndarray] = []
    for x_batch, y_batch in test_ds:
        x_chunks.append(np.asarray(x_batch.numpy()))
        y_chunks.append(_normalize_labels(y_batch))
    x_test = np.concatenate(x_chunks, axis=0)
    y_test = np.concatenate(y_chunks, axis=0)
    return x_test, y_test


def _train_seeded_model(
    base_cfg: AppConfig, ensemble_size: int, seed: int
) -> tuple[tf.keras.Model, float]:
    cfg = base_cfg.model_copy(deep=True)
    cfg.run.seed = seed
    cfg.run.name = f"{base_cfg.run.name}-independent-n{ensemble_size}-seed{seed}"
    cfg.ensemble.size = 1
    cfg.wandb.enabled = False
    cfg.callbacks.wandb.enabled = False

    set_seed(cfg.run.seed)
    configure_gpu_memory_growth()
    log_visible_devices()

    train_ds, val_ds, test_ds, info = build_dataset(cfg.data)
    model = build_model(cfg.model, cfg.quant, info)
    _compile_model(model, cfg)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.train.epochs,
        callbacks=[],
        verbose=1,
    )

    eval_dict = model.evaluate(test_ds, return_dict=True, verbose=1)
    test_acc = float(eval_dict["sparse_acc"])
    return model, test_acc


def _evaluate_independent_ensemble(
    models: list[tf.keras.Model], x_test: np.ndarray, y_test: np.ndarray
) -> tuple[list[float], float]:
    member_logits: list[np.ndarray] = []
    member_accs: list[float] = []

    for model in models:
        logits = model.predict(x_test, verbose=0)
        preds = np.argmax(logits, axis=-1)
        acc = float(np.mean(preds == y_test))
        member_logits.append(np.asarray(logits))
        member_accs.append(acc)

    ensemble_logits = np.mean(np.stack(member_logits, axis=0), axis=0)
    ensemble_preds = np.argmax(ensemble_logits, axis=-1)
    ensemble_acc = float(np.mean(ensemble_preds == y_test))

    return member_accs, ensemble_acc


def run_independent_experiment(config_path: str, ensemble_size: int) -> None:
    base_cfg = load_config(config_path)

    # Build once to get the canonical test split used for all model comparisons.
    _, _, test_ds, _ = build_dataset(base_cfg.data)
    x_test, y_test = _collect_test_arrays(test_ds)

    print(f"\n=== Independent Ensemble Experiment: N={ensemble_size} ===")
    models: list[tf.keras.Model] = []
    standalone_accs: list[float] = []
    for seed in range(ensemble_size):
        print(f"Training member seed={seed} ...")
        model, test_acc = _train_seeded_model(
            base_cfg, ensemble_size=ensemble_size, seed=seed
        )
        models.append(model)
        standalone_accs.append(test_acc)
        print(f"- member_{seed}_seed_{seed}_test/sparse_acc: {test_acc:.6f}")

    member_accs, ensemble_acc = _evaluate_independent_ensemble(models, x_test, y_test)

    print(f"\nResults for N={ensemble_size}")
    for idx, acc in enumerate(member_accs):
        print(f"- member_{idx}_seed_{idx}_test/sparse_acc: {acc:.6f}")
    print(f"- independent_ensemble_test/sparse_acc: {ensemble_acc:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train independently-seeded models and ensemble logits at test time."
    )
    parser.add_argument(
        "--config",
        default="configs/jsc/base-256.yaml",
        help="Base config used for each independently-trained member.",
    )
    parser.add_argument(
        "--ensemble-sizes",
        nargs="+",
        type=int,
        default=[2, 4],
        help="List of independent ensemble sizes to run (default: 2 4).",
    )
    args = parser.parse_args()

    for size in args.ensemble_sizes:
        if size < 2:
            raise ValueError("Each ensemble size must be >= 2 for ensembling.")
        run_independent_experiment(args.config, size)


if __name__ == "__main__":
    main()
