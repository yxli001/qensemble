from typing import Any

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="qensemble")
class QEnsemble(tf.keras.Model):
    def __init__(
        self,
        members: list[tf.keras.Model],
        name: str = "qensemble",
        **kwargs: Any,
    ):
        if len(members) < 2:
            raise ValueError("QEnsemble size must be >= 2")

        resolved_members: list[tf.keras.Model] = []
        for i, member in enumerate(members):
            member._name = f"member_{i}_{member.name}"
            resolved_members.append(member)

        input_shape = resolved_members[0].input_shape
        if isinstance(input_shape, list):
            raise ValueError("QEnsemble currently supports single-input members only")
        if input_shape is None:
            raise ValueError("Member model is not built; input_shape is unavailable")

        inputs = tf.keras.Input(shape=tuple(input_shape[1:]), name="input")
        member_outputs = [member(inputs, training=False) for member in resolved_members]

        outputs = tf.keras.layers.Average(name="ensemble_average")(member_outputs)

        super().__init__(inputs=inputs, outputs=outputs, name=name, **kwargs)

        self.members = resolved_members
        self.size = len(resolved_members)

    def save_member_models(self, out_dir: str, prefix: str = "model") -> None:
        tf.io.gfile.makedirs(out_dir)
        for i, member in enumerate(self.members):
            member.save(f"{out_dir}/{prefix}_{i}.keras")

    def load_member_models(self, out_dir: str, prefix: str = "model") -> None:
        self.members = [
            tf.keras.models.load_model(f"{out_dir}/{prefix}_{i}.keras")
            for i in range(self.size)
        ]

    def save_member_weights(self, out_dir: str, prefix: str = "model") -> None:
        tf.io.gfile.makedirs(out_dir)
        for i, member in enumerate(self.members):
            member.save_weights(f"{out_dir}/{prefix}_{i}.weights.h5")

    def load_member_weights(self, out_dir: str, prefix: str = "model") -> None:
        for i, member in enumerate(self.members):
            member.load_weights(f"{out_dir}/{prefix}_{i}.weights.h5")

    def to_functional_model(
        self, name: str = "qensemble_functional_export"
    ) -> tf.keras.Model:
        return tf.keras.Model(inputs=self.inputs, outputs=self.outputs, name=name)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "size": int(self.size),
                "members": [
                    tf.keras.utils.serialize_keras_object(member)
                    for member in self.members
                ],
            }
        )
        return config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "QEnsemble":
        members_cfg = config.pop("members", None)
        if members_cfg is None:
            raise ValueError("Serialized QEnsemble config is missing 'members'")

        members = [
            tf.keras.utils.deserialize_keras_object(member_cfg)
            for member_cfg in members_cfg
        ]
        return cls(members=members, **config)
