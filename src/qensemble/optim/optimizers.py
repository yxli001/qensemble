import tensorflow as tf

from qensemble.config import TrainConfig


def build_optimizer(cfg_train: TrainConfig) -> tf.keras.optimizers.Optimizer:
    name = cfg_train.optimizer.lower()
    lr = cfg_train.lr
    weight_decay = cfg_train.weight_decay

    if name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=lr)
    if name == "adamw":
        return tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay)
    if name == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)

    raise ValueError(f"Unsupported optimizer '{name}'")
