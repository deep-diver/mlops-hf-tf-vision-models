from tfx.components.trainer.fn_args_utils import FnArgs

from .train_data import input_fn
from .ViT import build_model
from .signatures import model_exporter

from .hyperparams import TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE
from .hyperparams import TRAIN_LENGTH, EVAL_LENGTH
from .hyperparams import EPOCHS


def run_fn(fn_args: FnArgs):
    train_dataset = input_fn(
        fn_args.train_files,
        is_train=True,
        batch_size=TRAIN_BATCH_SIZE,
    )

    eval_dataset = input_fn(
        fn_args.eval_files,
        is_train=False,
        batch_size=EVAL_BATCH_SIZE,
    )

    model = build_model()

    model.fit(
        train_dataset,
        steps_per_epoch=TRAIN_LENGTH // TRAIN_BATCH_SIZE,
        validation_data=eval_dataset,
        validation_steps=EVAL_LENGTH // TRAIN_BATCH_SIZE,
        epochs=EPOCHS,
    )

    model.save(
        fn_args.serving_model_dir, save_format="tf", signatures=model_exporter(model)
    )
