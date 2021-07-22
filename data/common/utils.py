from dataclasses import dataclass
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE


@dataclass
class DatasetMeta:
    data_path: str
    feature_description: dict
    input_size: list
