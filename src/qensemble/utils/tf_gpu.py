import tensorflow as tf


def configure_gpu_memory_growth() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def log_visible_devices() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    cpus = tf.config.list_physical_devices("CPU")
    print(f"Visible GPUs: {len(gpus)}")
    print(f"Visible CPUs: {len(cpus)}")
