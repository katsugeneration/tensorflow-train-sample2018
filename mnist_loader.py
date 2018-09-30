import gzip
import pathlib
import requests
import numpy as np
import tensorflow as tf


class Converter(object):
    def __init__(self):
        pass

    def _filter(self, images, labels):
        return True

    def filter(self, images, labels):
        predicts = tf.py_func(
                    self._filter,
                    [images, labels],
                    [tf.bool])[0]
        return predicts

    def _convert(self, images, labels):
        images = images.reshape((28, 28, 1))
        images = images.astype(np.float32)

        labels = labels.astype(np.uint8)
        labels = labels.reshape((1, ))

        return images, labels

    def convert(self, images, labels):
        images = tf.decode_raw(images, tf.uint8)
        labels = tf.decode_raw(labels, tf.uint8)
        images, labels = tf.py_func(
                    self._convert,
                    [images, labels],
                    [tf.float32, tf.uint8])
        images.set_shape((28, 28, 1))
        labels.set_shape((1, ))
        return images, labels


def load(train_data,
         converter,
         batch_size=32,
         is_training=True,
         buffer_size=2000,
         threads=10):
    """read dataset operation
    """
    root_path = pathlib.Path(train_data)
    if not root_path.exists():
        root_path.mkdir()

    image_path = root_path.joinpath("train_images")
    label_path = root_path.joinpath("train_labels")

    def get_data(path, url):
        gz_path = path.with_suffix(".gz")
        if not gz_path.exists():
            with requests.get(url, stream=True) as res:
                with gz_path.open("wb") as f:
                    f.writelines(res.iter_content(chunk_size=1024))

        if not path.exists():
            with gzip.GzipFile(fileobj=gz_path.open("rb")) as f:
                with path.open("wb") as w:
                    w.writelines(f)

    # get remote mnist dataset to local path
    get_data(image_path, "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")
    get_data(label_path, "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")

    image_dataset = tf.data.FixedLengthRecordDataset(str(image_path), record_bytes=28*28, header_bytes=16)
    label_dataset = tf.data.FixedLengthRecordDataset(str(label_path), record_bytes=1, header_bytes=8)
    dataset = tf.data.Dataset.zip((image_dataset, label_dataset))
    dataset = (dataset
               .filter(converter.filter)
               .map(converter.convert, num_parallel_calls=threads))

    if is_training:
        dataset = dataset.shuffle(buffer_size=buffer_size).repeat()

    dataset = dataset.padded_batch(batch_size, dataset.output_shapes)

    return dataset
