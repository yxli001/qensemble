from collections.abc import Callable
from typing import Any

import tensorflow as tf


class QEnsemble(tf.keras.Model):
    def __init__(
        self,
        member_builder: Callable[..., tf.keras.Model],
        size: int,
        member_kwargs: dict[str, Any],
        name: str = "qensemble",
    ):
        super().__init__(name=name)
        self.size = int(size)
        self.members = [member_builder(**member_kwargs) for _ in range(self.size)]

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        preds = [m(x, training=training) for m in self.members]
        stacked = tf.stack(preds, axis=0)
        return tf.reduce_mean(stacked, axis=0)

    def save_member_weights(self, out_dir: str, prefix: str = "model") -> None:
        tf.io.gfile.makedirs(out_dir)
        for i, member in enumerate(self.members):
            member.save_weights(f"{out_dir}/{prefix}_{i}.weights.h5")

    def load_member_weights(self, out_dir: str, prefix: str = "model") -> None:
        for i, member in enumerate(self.members):
            member.load_weights(f"{out_dir}/{prefix}_{i}.weights.h5")
