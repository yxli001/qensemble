from typing import Any

import numpy as np
import tensorflow as tf

from qensemble.config import DataConfig


def _extra(cfg: DataConfig, key: str, default: Any) -> Any:
    extras = cfg.model_extra or {}
    return extras.get(key, default)


def build_tf_keras(
    cfg_data: DataConfig,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, dict[str, Any]]:
    dataset_name = cfg_data.dataset_name
    batch_size = cfg_data.batch_size
    val_split = cfg_data.val_split
    normalize = bool(_extra(cfg_data, "normalize", True))
    shuffle_buffer = int(_extra(cfg_data, "shuffle_buffer", 10000))

    dataset_mod = getattr(tf.keras.datasets, dataset_name, None)
    if dataset_mod is None or not hasattr(dataset_mod, "load_data"):
        raise KeyError(f"Unsupported tf.keras dataset '{dataset_name}'")

    (x_train, y_train), (x_test, y_test) = dataset_mod.load_data()
    x_train = np.asarray(x_train, dtype="float32")
    x_test = np.asarray(x_test, dtype="float32")
    y_train = np.asarray(y_train).squeeze().astype("int32")
    y_test = np.asarray(y_test).squeeze().astype("int32")

    if normalize:
        x_train /= 255.0
        x_test /= 255.0

    # Add a channel axis for grayscale images such as MNIST.
    if x_train.ndim == 3:
        x_train = x_train[..., None]
        x_test = x_test[..., None]

    n_val = int(len(x_train) * val_split)
    x_val, y_val = x_train[:n_val], y_train[:n_val]
    x_train, y_train = x_train[n_val:], y_train[n_val:]

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(shuffle_buffer)
        .batch(batch_size)
    )
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

    input_shape = tuple(int(x) for x in x_train.shape[1:])
    num_classes = int(np.max(np.concatenate((y_train, y_test)))) + 1
    info = {
        "input_shape": input_shape,
        "num_classes": num_classes,
    }
    return (
        train_ds.prefetch(tf.data.AUTOTUNE),
        val_ds.prefetch(tf.data.AUTOTUNE),
        test_ds.prefetch(tf.data.AUTOTUNE),
        info,
    )
