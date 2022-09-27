import tensorflow as tf
from transformers import ViTFeatureExtractor

from .common import PRETRAIN_CHECKPOINT
from .common import CONCRETE_INPUT

feature_extractor = ViTFeatureExtractor.from_pretrained(PRETRAIN_CHECKPOINT)

def _normalize_img(
    img, mean=feature_extractor.image_mean, std=feature_extractor.image_std
):
    img = img / 255
    mean = tf.constant(mean)
    std = tf.constant(std)
    return (img - mean) / std


def _preprocess_serving(string_input):
    decoded_input = tf.io.decode_base64(string_input)
    decoded = tf.io.decode_jpeg(decoded_input, channels=3)
    resized = tf.image.resize(decoded, size=(224, 224))
    normalized = _normalize_img(resized)
    normalized = tf.transpose(
        normalized, (2, 0, 1)
    )  # Since HF models are channel-first.
    return normalized


@tf.function(input_signature=[tf.TensorSpec([None], tf.string)])
def _preprocess_fn(string_input):
    decoded_images = tf.map_fn(
        _preprocess_serving, string_input, dtype=tf.float32, back_prop=False
    )
    return {CONCRETE_INPUT: decoded_images}


def model_exporter(model: tf.keras.Model):
    m_call = tf.function(model.call).get_concrete_function(
        tf.TensorSpec(
            shape=[None, 3, 224, 224], dtype=tf.float32, name=CONCRETE_INPUT
        )
    )

    @tf.function(input_signature=[tf.TensorSpec([None], tf.string)])
    def serving_fn(string_input):
        labels = tf.constant(list(model.config.id2label.values()), dtype=tf.string)
        images = _preprocess_fn(string_input)

        predictions = m_call(**images)
        indices = tf.argmax(predictions.logits, axis=1)
        pred_source = tf.gather(params=labels, indices=indices)
        probs = tf.nn.softmax(predictions.logits, axis=1)
        pred_confidence = tf.reduce_max(probs, axis=1)
        return {"label": pred_source, "confidence": pred_confidence}

    return serving_fn