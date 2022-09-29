import keras_tuner
import tensorflow_transform as tft
from tensorflow_cloud import CloudTuner

from tfx.components.trainer.fn_args_utils import FnArgs
from tfx.v1.components import TunerFnResult

from .train_data import input_fn
from .ViT import build_model

from .hyperparams import TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE
from .hyperparams import TRAIN_LENGTH, EVAL_LENGTH
from .hyperparams import get_hyperparameters

import tfx.extensions.google_cloud_ai_platform.constants as vertex_const
import tfx.extensions.google_cloud_ai_platform.trainer.executor as vertex_training_const
import tfx.extensions.google_cloud_ai_platform.tuner.executor as vertex_tuner_const

def cloud_tuner_fn(fn_args: FnArgs) -> TunerFnResult:
    TUNING_ARGS_KEY = vertex_tuner_const.TUNING_ARGS_KEY
    VERTEX_PROJECT_KEY = "project"
    VERTEX_REGION_KEY = "region"

    tuner = CloudTuner(
        build_model,
        max_trials=6,
        hyperparameters=get_hyperparameters(),
        project_id=fn_args.custom_config[TUNING_ARGS_KEY][VERTEX_PROJECT_KEY],
        region=fn_args.custom_config[TUNING_ARGS_KEY][VERTEX_REGION_KEY],
        objective="val_sparse_categorical_accuracy",
        directory=fn_args.working_dir,
    )

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

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

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            "x": train_dataset,
            "validation_data": eval_dataset,
            "steps_per_epoch": TRAIN_LENGTH // TRAIN_BATCH_SIZE,
            "validation_steps": EVAL_LENGTH // EVAL_BATCH_SIZE,
        },
    )