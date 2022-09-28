import tensorflow as tf
from .common import IMAGE_TFREC_KEY, LABEL_TFREC_KEY
from .common import IMAGE_MODEL_KEY, LABEL_MODEL_KEY
from .hyperparams import INPUT_IMG_SIZE


def preprocessing_fn(inputs):
    """tf.transform's callback function for preprocessing inputs.
    Args:
      inputs: map from feature keys to raw not-yet-transformed features.
    Returns:
      Map from string feature key to transformed feature operations.
    """
    # print(inputs)
    outputs = {}

    inputs[IMAGE_TFREC_KEY] = tf.image.resize(
        inputs[IMAGE_TFREC_KEY], [INPUT_IMG_SIZE, INPUT_IMG_SIZE]
    )

    inputs[IMAGE_TFREC_KEY] = inputs[IMAGE_TFREC_KEY] / 255.0
    inputs[IMAGE_TFREC_KEY] = tf.transpose(inputs[IMAGE_TFREC_KEY], [0, 3, 1, 2])

    outputs[IMAGE_MODEL_KEY] = inputs[IMAGE_TFREC_KEY]
    outputs[LABEL_MODEL_KEY] = inputs[LABEL_TFREC_KEY]

    return outputs
