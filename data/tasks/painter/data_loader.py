import tensorflow as tf
import matplotlib.pyplot as plt
from data.common.data_loader import BaseDataLoader
from data.common.dataset_meta import DatasetMeta

PAINTER_MONET_META = DatasetMeta(
    data_path="data/tasks/painter/dataset/monet_tfrec",
    feature_description={
        "image_name": tf.io.FixedLenFeature([], tf.string),
        "image": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.FixedLenFeature([], tf.string)
    },
    input_size=[256, 256]
)

PAINTER_PHOTO_META = DatasetMeta(
    data_path="data/tasks/painter/dataset/photo_tfrec",
    feature_description={
        "image_name": tf.io.FixedLenFeature([], tf.string),
        "image": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.FixedLenFeature([], tf.string)
    },
    input_size=[256, 256]
)


class DataLoader(BaseDataLoader):
    def __init__(self, dataset_meta: DatasetMeta):
        super().__init__(dataset_meta)

    def _preprocess(self, example):
        image = example["image"]
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.cast(image, tf.float32) / 127.5 - 1
        image = tf.reshape(image, [*self.dataset_meta.input_size, 3])
        return image

    def __call__(self):
        return super().__call__().batch(1)


if __name__ == '__main__':
    monet_dataset_factory = DataLoader(PAINTER_MONET_META)
    photo_dataset_factory = DataLoader(PAINTER_PHOTO_META)

    monet_dataset = monet_dataset_factory()
    photo_dataset = photo_dataset_factory()

    example_monet = next(iter(monet_dataset))
    example_photo = next(iter(photo_dataset))

    plt.subplot(121)
    plt.title('Photo')
    plt.imshow(example_photo[0] * 0.5 + 0.5)

    plt.subplot(122)
    plt.title('Monet')
    plt.imshow(example_monet[0] * 0.5 + 0.5)

    plt.show()
