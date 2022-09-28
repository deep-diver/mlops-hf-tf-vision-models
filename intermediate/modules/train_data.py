from typing import List

import tensorflow as tf
import tensorflow_transform as tft

from tfx_bsl.tfxio import dataset_options
from tfx.components.trainer.fn_args_utils import DataAccessor

from .utils import INFO
from .common import LABEL_MODEL_KEY
from .hyperparams import BATCH_SIZE


def input_fn(
    file_pattern: List[str],
    data_accessor: DataAccessor,
    tf_transform_output: tft.TFTransformOutput,
    is_train: bool = False,
    batch_size: int = BATCH_SIZE,
) -> tf.data.Dataset:
    INFO(f"Reading data from: {file_pattern}")

    dataset = data_accessor.tf_dataset_factory(
        file_pattern,
        dataset_options.TensorFlowDatasetOptions(
            batch_size=batch_size, label_key=LABEL_MODEL_KEY
        ),
        tf_transform_output.transformed_metadata.schema,
    )

    return dataset
