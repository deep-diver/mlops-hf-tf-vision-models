import keras_tuner
import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import FnArgs
from tfx.v1.components import TunerFnResult

from .train_data import input_fn
from .ViT import build_model

from .hyperparams import TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE
from .hyperparams import TRAIN_LENGTH, EVAL_LENGTH
from .hyperparams import get_hyperparameters

def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
    tuner = keras_tuner.RandomSearch(
        build_model,
        max_trials=6,
        hyperparameters=get_hyperparameters(),
        allow_new_entries=False,
        objective=keras_tuner.Objective("val_accuracy", "max"),
        directory=fn_args.working_dir,
        project_name="img_classification_tuning",
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
