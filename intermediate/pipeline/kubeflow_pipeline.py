from typing import Any, Dict, List, Optional, Text

from tfx import v1 as tfx

from ml_metadata.proto import metadata_store_pb2
from tfx.proto import example_gen_pb2

from tfx.components import ImportExampleGen
from tfx.components import StatisticsGen
from tfx.components import ExampleValidator
from tfx.components import Transform
from tfx.extensions.google_cloud_ai_platform.trainer.component import (
    Trainer as VertexTrainer,
)
from tfx.extensions.google_cloud_ai_platform.pusher.component import (
    Pusher as VertexPusher,
)
from tfx.orchestration import pipeline
from tfx.proto import example_gen_pb2


def create_pipeline(
    pipeline_name: Text,
    pipeline_root: Text,
    data_path: Text,
    schema_path: Text,
    modules: Dict[Text, Text],
    metadata_connection_config: Optional[metadata_store_pb2.ConnectionConfig] = None,
    ai_platform_training_args: Optional[Dict[Text, Text]] = None,
    ai_platform_serving_args: Optional[Dict[Text, Any]] = None,
    example_gen_beam_args: Optional[List] = None,
    transform_beam_args: Optional[List] = None,
) -> tfx.dsl.Pipeline:
    components = []

    input_config = example_gen_pb2.Input(
        splits=[
            example_gen_pb2.Input.Split(name="train", pattern="train-*.tfrec"),
            example_gen_pb2.Input.Split(name="eval", pattern="val-*.tfrec"),
        ]
    )
    example_gen = ImportExampleGen(input_base=data_path, input_config=input_config)
    if example_gen_beam_args is not None:
        example_gen.with_beam_pipeline_args(example_gen_beam_args)
    components.append(example_gen)

    statistics_gen = StatisticsGen(examples=example_gen.outputs["examples"])
    components.append(statistics_gen)

    schema_gen = tfx.components.ImportSchemaGen(schema_file=schema_path)
    components.append(schema_gen)

    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs["statistics"],
        schema=schema_gen.outputs["schema"],
    )
    components.append(example_validator)

    transform_args = {
        "examples": example_gen.outputs["examples"],
        "schema": schema_gen.outputs["schema"],
        "preprocessing_fn": modules["preprocessing_fn"],
    }
    transform = Transform(**transform_args)
    if transform_beam_args is not None:
        transform.with_beam_pipeline_args(transform_beam_args)
    components.append(transform)

    trainer_args = {
        "run_fn": modules["training_fn"],
        "transformed_examples": transform.outputs["transformed_examples"],
        "transform_graph": transform.outputs["transform_graph"],
        "schema": schema_gen.outputs["schema"],
        "custom_config": ai_platform_training_args,
    }
    trainer = VertexTrainer(**trainer_args)
    components.append(trainer)

    pusher_args = {
        "model": trainer.outputs["model"],
        "custom_config": ai_platform_serving_args,
    }
    pusher = VertexPusher(**pusher_args)  # pylint: disable=unused-variable
    components.append(pusher)

    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=True,
        metadata_connection_config=metadata_connection_config,
    )
