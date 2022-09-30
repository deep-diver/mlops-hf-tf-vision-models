import os
from absl import logging

from tfx import v1 as tfx
from pipeline import configs
from pipeline import local_pipeline

OUTPUT_DIR = "."

PIPELINE_ROOT = os.path.join(OUTPUT_DIR, "tfx_pipeline_output", configs.PIPELINE_NAME)
METADATA_PATH = os.path.join(
    OUTPUT_DIR, "tfx_metadata", configs.PIPELINE_NAME, "metadata.db"
)
SERVING_MODEL_DIR = os.path.join(PIPELINE_ROOT, "serving_model")


def run():
    tfx.orchestration.LocalDagRunner().run(
        local_pipeline.create_pipeline(
            pipeline_name=configs.PIPELINE_NAME,
            pipeline_root=PIPELINE_ROOT,
            data_path=configs.DATA_PATH,
            schema_path=configs.SCHEMA_PATH,
            modules={
                "training_fn": configs.TRAINING_FN,
                "preprocessing_fn": configs.PREPROCESSING_FN,
                "tuner_fn": configs.TUNER_FN,
            },
            hyperparameters=configs.HYPER_PARAMETERS,
            eval_configs=configs.EVAL_CONFIGS,
            serving_model_dir=SERVING_MODEL_DIR,
            metadata_connection_config=tfx.orchestration.metadata.sqlite_metadata_connection_config(
                METADATA_PATH
            ),
        )
    )


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    run()
