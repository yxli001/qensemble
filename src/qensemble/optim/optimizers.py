import tensorflow as tf

from qensemble.config import TrainConfig


def build_optimizer(cfg_train: TrainConfig) -> tf.keras.optimizers.Optimizer:
    name = cfg_train.optimizer.lower()
    lr = cfg_train.lr
    beta1 = cfg_train.beta1
    beta2 = cfg_train.beta2
    epsilon = cfg_train.epsilon
    weight_decay = cfg_train.weight_decay

    if name == "adam":
        return tf.keras.optimizers.AdamW(
            learning_rate=lr,
            weight_decay=weight_decay,
            beta_1=beta1,
            beta_2=beta2,
            epsilon=epsilon,
        )

    if name == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)

    raise ValueError(f"Unsupported optimizer '{name}'")
