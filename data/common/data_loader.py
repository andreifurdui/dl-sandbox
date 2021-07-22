from abc import ABC, abstractmethod
import tensorflow as tf

from data.common.utils import AUTOTUNE
from data.common.dataset_meta import DatasetMeta


class BaseDataLoader(ABC):
    def __init__(self, dataset_meta: DatasetMeta):
        self.dataset_meta = dataset_meta
        self.files = tf.io.gfile.glob(self.dataset_meta.data_path + "/*.tfrec")

    def __call__(self):
        dataset = tf.data.TFRecordDataset(self.files)
        dataset = dataset.map(self._decode, num_parallel_calls=AUTOTUNE)
        dataset = dataset.map(self._preprocess, num_parallel_calls=AUTOTUNE)

        return dataset

    def _decode(self, example):
        return tf.io.parse_single_example(example, features=self.dataset_meta.feature_description)

    @abstractmethod
    def _preprocess(self, example):
        pass
