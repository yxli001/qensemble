import tensorflow as tf

from qensemble.config import ModelConfig, QuantConfig


def _extra(cfg: ModelConfig, key: str, default: object) -> object:
    extras = cfg.model_extra or {}
    return extras.get(key, default)


def _as_int_list(value: object, field_name: str) -> list[int]:
    if not isinstance(value, list | tuple):
        raise TypeError(f"model.{field_name} must be a list of integers")
    out = [_as_int(v, field_name) for v in value]
    if not out:
        raise ValueError(f"model.{field_name} must not be empty")
    return out


def _as_int(value: object, field_name: str) -> int:
    if not isinstance(value, int | float | str):
        raise TypeError(f"model.{field_name} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"model.{field_name} must be an integer") from exc


def _as_float(value: object, field_name: str) -> float:
    if not isinstance(value, int | float | str):
        raise TypeError(f"model.{field_name} must be a float")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"model.{field_name} must be a float") from exc


def build_mlp(
    cfg_model: ModelConfig, cfg_quant: QuantConfig, info: dict[str, object]
) -> tf.keras.Model:
    del cfg_quant
    input_shape_raw = info.get("input_shape")
    if not isinstance(input_shape_raw, list | tuple):
        raise TypeError("Model info must provide iterable 'input_shape'")
    input_shape = tuple(int(x) for x in input_shape_raw)

    num_classes_raw = _extra(cfg_model, "num_classes", info.get("num_classes", 10))
    num_classes = _as_int(num_classes_raw, "num_classes")

    # Single config key: `model.width` can be an int or a list of ints.
    width_cfg = _extra(cfg_model, "width", 128)
    if isinstance(width_cfg, list | tuple):
        hidden = _as_int_list(width_cfg, "width")
    else:
        hidden = [_as_int(width_cfg, "width")]

    dropout = _as_float(_extra(cfg_model, "dropout", 0.0), "dropout")

    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Flatten()(inputs)
    for width in hidden:
        x = tf.keras.layers.Dense(width, activation="relu")(x)
        if dropout > 0:
            x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(num_classes)(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="mlp")
