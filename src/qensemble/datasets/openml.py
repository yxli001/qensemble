import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from qensemble.config import DataConfig


def build_openml(
    cfg_data: DataConfig,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, dict[str, object]]:
    dataset_name = cfg_data.dataset_name
    test_split = cfg_data.test_split
    batch_size = cfg_data.batch_size
    val_split = cfg_data.val_split
    random_state = cfg_data.random_state
    cache = cfg_data.cache

    # Force liac-arff parser to avoid optional pandas dependency in sklearn's
    # default parser path for dense datasets.
    data = fetch_openml(
        dataset_name,
        as_frame=False,
        cache=cache,
        parser="liac-arff",
    )
    x_all = np.asarray(data.data, dtype="float32")
    y_raw = data.target

    # OpenML targets may be strings; normalize to contiguous int labels.
    encoder = LabelEncoder()
    y_all = np.asarray(encoder.fit_transform(y_raw), dtype="int32")

    x_train, x_test, y_train, y_test = train_test_split(
        x_all,
        y_all,
        test_size=test_split,
        random_state=random_state,
        stratify=y_all,
    )

    n_val = int(len(x_train) * val_split)
    x_val, y_val = x_train[:n_val], y_train[:n_val]
    x_train, y_train = x_train[n_val:], y_train[n_val:]

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(10000)
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
