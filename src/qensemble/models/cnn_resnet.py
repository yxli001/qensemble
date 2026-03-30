from typing import Any

import qkeras
import tensorflow as tf
from qkeras.qconvolutional import QConv2D
from qkeras.qlayers import QActivation, QDense

from qensemble.config import ModelConfig, QuantConfig


def _extra(cfg: ModelConfig, key: str, default: Any) -> Any:
    extras = cfg.model_extra or {}
    return extras.get(key, default)


def _variant_filters(variant: str) -> tuple[int, int, int]:
    variants = {
        "small": (8, 16, 32),
        "medium": (12, 24, 48),
        "large": (16, 32, 64),
    }
    if variant not in variants:
        raise ValueError(f"Unsupported model.variant '{variant}'")
    return variants[variant]


def _build_filter_config(cfg_model: ModelConfig) -> tuple[list[int], list[int], str]:
    num_filters_raw = _extra(cfg_model, "num_filters", None)
    if num_filters_raw is None:
        variant = str(_extra(cfg_model, "variant", "large"))
        f1, f2, f3 = _variant_filters(variant)
        num_filters = [f1, f1, f2, f2, f3, f3]
    else:
        num_filters = [int(v) for v in num_filters_raw]

    if len(num_filters) < 6:
        raise ValueError("model.num_filters must include at least 6 entries")

    kernel_sizes_raw = _extra(cfg_model, "kernel_sizes", None)
    if kernel_sizes_raw is None:
        kernel_sizes = [3, 3, 3, 3, 3, 3]
    else:
        kernel_sizes = [int(v) for v in kernel_sizes_raw]

    if len(kernel_sizes) < 6:
        raise ValueError("model.kernel_sizes must include at least 6 entries")

    strides_str = str(_extra(cfg_model, "strides", "111212212"))
    if len(strides_str) < 9:
        raise ValueError("model.strides must have at least 9 digits")

    return num_filters, kernel_sizes, strides_str


def _try_build_quant_layers(cfg_quant: QuantConfig) -> dict[str, Any] | None:
    if not cfg_quant.enabled:
        return None

    quant_extra = cfg_quant.model_extra or {}
    activation_total_bits = int(quant_extra.get("activation_total_bits", 4))
    logits_total_bits = int(quant_extra.get("logits_total_bits", 4))
    activation_int_bits = 0 if activation_total_bits <= 2 else 1
    logits_int_bits = 0 if logits_total_bits <= 2 else 1

    activation_quantizer = qkeras.quantizers.quantized_relu(
        activation_total_bits, activation_int_bits
    )
    logits_quantizer = qkeras.quantizers.quantized_bits(
        logits_total_bits, logits_int_bits
    )

    return {
        "Conv": lambda filters, k, s=1: QConv2D(
            filters,
            k,
            strides=s,
            padding="same",
            kernel_quantizer=logits_quantizer,
            bias_quantizer=logits_quantizer,
        ),
        "Dense": lambda units: QDense(
            units,
            kernel_quantizer=logits_quantizer,
            bias_quantizer=logits_quantizer,
        ),
        "Act": lambda: QActivation(activation_quantizer),
    }


def _res_block(
    x: tf.Tensor,
    filters: int,
    kernel_size: int,
    stride: int,
    api: dict[str, Any],
) -> tf.Tensor:
    shortcut = x

    y = api["Conv"](filters, kernel_size, s=stride)(x)
    y = tf.keras.layers.BatchNormalization()(y)
    y = api["Act"]()(y)
    y = api["Conv"](filters, kernel_size)(y)
    y = tf.keras.layers.BatchNormalization()(y)

    channel_dim = shortcut.shape[-1]
    if stride != 1 or channel_dim is None or channel_dim != filters:
        shortcut = api["Conv"](filters, 1, s=stride)(shortcut)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)

    x = tf.keras.layers.Add()([shortcut, y])
    return api["Act"]()(x)


def build_cnn_resnet(
    cfg_model: ModelConfig,
    cfg_quant: QuantConfig,
    info: dict[str, Any],
) -> tf.keras.Model:
    input_shape = tuple(info.get("input_shape", (32, 32, 3)))
    if len(input_shape) != 3:
        raise ValueError(
            f"cnn_resnet expects 3D input shape (H,W,C), got {input_shape}. "
            "Choose model.name=mlp for non-image tasks."
        )

    num_classes = int(_extra(cfg_model, "num_classes", info.get("num_classes", 10)))
    num_filters, kernel_sizes, strides_str = _build_filter_config(cfg_model)
    strides = [
        [int(strides_str[0]), int(strides_str[1]), int(strides_str[2])],
        [int(strides_str[3]), int(strides_str[4]), int(strides_str[5])],
        [int(strides_str[6]), int(strides_str[7]), int(strides_str[8])],
    ]

    quant_api = _try_build_quant_layers(cfg_quant)
    if quant_api is None:
        quant_api = {
            "Conv": lambda filters, k, s=1: tf.keras.layers.Conv2D(
                filters,
                k,
                strides=s,
                padding="same",
                use_bias=True,
                kernel_initializer="he_normal",
            ),
            "Dense": lambda units: tf.keras.layers.Dense(
                units,
                kernel_initializer="he_normal",
            ),
            "Act": lambda: tf.keras.layers.ReLU(),
        }

    inputs = tf.keras.layers.Input(shape=input_shape)
    x = quant_api["Conv"](num_filters[0], kernel_sizes[0], s=strides[0][0])(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = quant_api["Act"]()(x)

    x = _res_block(x, num_filters[1], kernel_sizes[1], strides[0][1], quant_api)
    x = _res_block(x, num_filters[3], kernel_sizes[3], strides[1][0], quant_api)
    x = _res_block(x, num_filters[5], kernel_sizes[5], strides[2][0], quant_api)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = quant_api["Dense"](num_classes)(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="cnn_resnet")
