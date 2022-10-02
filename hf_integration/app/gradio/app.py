import numpy as np
from PIL import Image
import tensorflow as tf
from transformers import ViTFeatureExtractor
from huggingface_hub import from_pretrained_keras

PRETRAIN_CHECKPOINT = "google/vit-base-patch16-224-in21k"
feature_extractor = ViTFeatureExtractor.from_pretrained(PRETRAIN_CHECKPOINT)

# $MODEL_REPO_ID should be like chansung/test-vit
# $MODEL_VERSION 
MODEL_CKPT = "$MODEL_REPO_ID@$MODEL_VERSION"
MODEL = from_pretrained_keras(MODEL_CKPT)

RESOLTUION = 224

labels = []

with open(r"labels.txt", "r") as fp:
    for line in fp:
        labels.append(line[:-1])

def normalize_img(
    img, mean=feature_extractor.image_mean, std=feature_extractor.image_std
):
    img = img / 255
    mean = tf.constant(mean)
    std = tf.constant(std)
    return (img - mean) / std

def preprocess_input(image: Image) -> tf.Tensor:
    image = np.array(image)
    image = tf.convert_to_tensor(image)

    image = tf.image.resize(image, (RESOLTUION, RESOLTUION))
    image = normalize_img(image)

    image = tf.transpose(
        image, (2, 0, 1)
    )  # Since HF models are channel-first.

    return {
        "pixel_values": tf.expand_dims(image, 0)
    }

def get_predictions(image: Image) -> tf.Tensor:
    preprocessed_image = preprocess_input(image)
    prediction = MODEL.predict(preprocessed_image)
    probs = tf.nn.softmax(prediction['logits'], axis=1)

    confidences = {labels[i]: float(probs[0][i]) for i in range(3)}
    return confidences