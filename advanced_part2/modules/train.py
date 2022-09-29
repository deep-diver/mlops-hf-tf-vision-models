import keras_tuner
import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import FnArgs

from .train_data import input_fn
from .ViT import build_model
from .signatures import (
    model_exporter,
    transform_features_signature,
    tf_examples_serving_signature,
)

from .hyperparams import TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE
from .hyperparams import TRAIN_LENGTH, EVAL_LENGTH
from .hyperparams import EPOCHS

from .utils import INFO


def run_fn(fn_args: FnArgs):
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = input_fn(
        fn_args.train_files,
        fn_args.data_accessor,
        tf_transform_output,
        is_train=True,
        batch_size=TRAIN_BATCH_SIZE,
    )

    eval_dataset = input_fn(
        fn_args.eval_files,
        fn_args.data_accessor,
        tf_transform_output,
        is_train=False,
        batch_size=EVAL_BATCH_SIZE,
    )

    hparams = keras_tuner.HyperParameters.from_config(fn_args.hyperparameters)
    INFO(f"HyperParameters for training: {hparams.get_config()}")    
    model = build_model(hparams)

    model.fit(
        train_dataset,
        steps_per_epoch=TRAIN_LENGTH // TRAIN_BATCH_SIZE,
        validation_data=eval_dataset,
        validation_steps=EVAL_LENGTH // TRAIN_BATCH_SIZE,
        epochs=EPOCHS,
    )

    model.save(
        fn_args.serving_model_dir,
        save_format="tf",
        signatures={
            "serving_default": model_exporter(model),
            "transform_features": transform_features_signature(
                model, tf_transform_output
            ),
            "from_examples": tf_examples_serving_signature(model, tf_transform_output),
        },
    )
