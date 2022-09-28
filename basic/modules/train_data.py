from typing import List

import tensorflow as tf

from .utils import INFO
from .common import IMAGE_TFREC_KEY, IMAGE_SHAPE_TFREC_KEY, LABEL_TFREC_KEY
from .common import IMAGE_MODEL_KEY, LABEL_MODEL_KEY
from .hyperparams import BATCH_SIZE


def _parse_tfr(proto):
    feature_description = {
        IMAGE_TFREC_KEY:        tf.io.VarLenFeature(tf.float32),
        IMAGE_SHAPE_TFREC_KEY:  tf.io.VarLenFeature(tf.int64),
        LABEL_TFREC_KEY:        tf.io.VarLenFeature(tf.int64),
    }

    rec = tf.io.parse_single_example(proto, feature_description)

    image_shape = tf.sparse.to_dense(rec[IMAGE_SHAPE_TFREC_KEY])
    image = tf.reshape(tf.sparse.to_dense(rec[IMAGE_TFREC_KEY]), image_shape)

    label = tf.sparse.to_dense(rec[LABEL_TFREC_KEY])

    return {IMAGE_MODEL_KEY: image, LABEL_MODEL_KEY: label}


def _preprocess(example_batch):
    images = example_batch[IMAGE_MODEL_KEY]
    images = tf.transpose(
        images, perm=[0, 1, 2, 3]
    )  # (batch_size, height, width, num_channels)
    images = tf.image.resize(images, (224, 224))
    images = tf.transpose(images, perm=[0, 3, 1, 2])

    labels = example_batch[LABEL_MODEL_KEY]
    labels = tf.transpose(labels, perm=[0, 1])  # So, that TF can evaluation the shapes.

    return {IMAGE_MODEL_KEY: images, LABEL_MODEL_KEY: labels}


def input_fn(
    file_pattern: List[str],
    batch_size: int = BATCH_SIZE,
    is_train: bool = False,
) -> tf.data.Dataset:
    INFO(f"Reading data from: {file_pattern}")

    dataset = tf.data.TFRecordDataset(
        tf.io.gfile.glob(file_pattern[0] + ".gz"),
        num_parallel_reads=tf.data.AUTOTUNE,
        compression_type="GZIP",
    ).map(_parse_tfr, num_parallel_calls=tf.data.AUTOTUNE)

    if is_train:
        dataset = dataset.shuffle(batch_size * 2)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    dataset = dataset.map(_preprocess)
    return dataset
