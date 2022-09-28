import tensorflow as tf
from transformers import TFViTForImageClassification

from .common import LABELS
from .common import PRETRAIN_CHECKPOINT
from .utils import INFO


def build_model():
    id2label = {str(i): c for i, c in enumerate(LABELS)}
    label2id = {c: str(i) for i, c in enumerate(LABELS)}

    model = TFViTForImageClassification.from_pretrained(
        PRETRAIN_CHECKPOINT,
        num_labels=len(LABELS),
        label2id=label2id,
        id2label=id2label,
    )

    model.layers[0].trainable = False

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])

    INFO(model.summary())

    return model
