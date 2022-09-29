from absl import logging

from tfx import v1 as tfx
from tfx.orchestration.kubeflow.v2 import kubeflow_v2_dag_runner as runner
from tfx.proto import tuner_pb2

from pipeline import configs
from pipeline import kubeflow_pipeline


def run():
    runner_config = runner.KubeflowV2DagRunnerConfig(
        default_image=configs.PIPELINE_IMAGE
    )

    runner.KubeflowV2DagRunner(
        config=runner_config,
        output_filename=configs.PIPELINE_NAME + "_pipeline.json",
    ).run(
        kubeflow_pipeline.create_pipeline(
            pipeline_name=configs.PIPELINE_NAME,
            pipeline_root=configs.PIPELINE_ROOT,
            data_path=configs.DATA_PATH,
            schema_path=configs.SCHEMA_PATH,
            modules={
                "training_fn": configs.TRAINING_FN,
                "preprocessing_fn": configs.PREPROCESSING_FN,
                "tuner_fn": configs.TUNER_FN,
            },
            eval_configs=configs.EVAL_CONFIGS,
            ai_platform_training_args=configs.GCP_AI_PLATFORM_TRAINING_ARGS,
            ai_platform_tuner_args=configs.GCP_AI_PLATFORM_TUNER_ARGS,
            tuner_args=tuner_pb2.TuneArgs(
                num_parallel_trials=configs.NUM_PARALLEL_TRIALS
            ),
            ai_platform_serving_args=configs.GCP_AI_PLATFORM_SERVING_ARGS,
            example_gen_beam_args=configs.EXAMPLE_GEN_BEAM_ARGS,
            transform_beam_args=configs.TRANSFORM_BEAM_ARGS,
        )
    )


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    run()
