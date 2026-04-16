from typing import Any

import qkeras
import tensorflow as tf
from qkeras.qlayers import QActivation, QDense

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


def _build_quant_api(cfg_quant: QuantConfig) -> dict[str, Any]:
    activation_total_bits = int(cfg_quant.activation_total_bits)
    activation_int_bits = int(cfg_quant.activation_int_bits)
    weight_total_bits = int(cfg_quant.weight_total_bits)
    weight_int_bits = int(cfg_quant.weight_int_bits)

    activation_quantizer = qkeras.quantizers.quantized_relu(
        activation_total_bits, activation_int_bits
    )
    weight_quantizer = qkeras.quantizers.quantized_bits(
        weight_total_bits, weight_int_bits
    )

    return {
        "Dense": lambda units: QDense(
            units,
            kernel_quantizer=weight_quantizer,
            bias_quantizer=weight_quantizer,
        ),
        "Act": lambda: QActivation(activation_quantizer),
    }


def build_mlp(
    cfg_model: ModelConfig, cfg_quant: QuantConfig, info: dict[str, object]
) -> tf.keras.Model:
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
    quant_api: dict[str, Any] = _build_quant_api(cfg_quant)

    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Flatten()(inputs)
    for width in hidden:
        x = quant_api["Dense"](width)(x)
        x = quant_api["Act"]()(x)
        if dropout > 0:
            x = tf.keras.layers.Dropout(dropout)(x)
    outputs = quant_api["Dense"](num_classes)(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="mlp")
