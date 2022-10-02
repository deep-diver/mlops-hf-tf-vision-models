from typing import Dict, Optional, Text

import tensorflow_model_analysis as tfma

from tfx import v1 as tfx

from ml_metadata.proto import metadata_store_pb2
from tfx.proto import example_gen_pb2

from tfx.components import ImportExampleGen
from tfx.components import StatisticsGen
from tfx.components import ExampleValidator
from tfx.components import Transform
from tfx.components import Tuner
from tfx.components import Trainer
from tfx.components import Evaluator
from tfx.components import Pusher
from tfx.orchestration import pipeline
from tfx.proto import example_gen_pb2

from tfx.types import Channel
from tfx.types.standard_artifacts import Model
from tfx.types.standard_artifacts import ModelBlessing
from tfx.dsl.components.common import resolver
from tfx.dsl.experimental.latest_blessed_model_resolver import (
    LatestBlessedModelResolver,
)


def create_pipeline(
    pipeline_name: Text,
    pipeline_root: Text,
    data_path: Text,
    schema_path: Text,
    modules: Dict[Text, Text],
    hyperparameters: Dict[Text, Text],
    eval_configs: tfma.EvalConfig,
    serving_model_dir: Text,
    metadata_connection_config: Optional[metadata_store_pb2.ConnectionConfig] = None,
) -> tfx.dsl.Pipeline:
    components = []

    input_config = example_gen_pb2.Input(
        splits=[
            example_gen_pb2.Input.Split(name="train", pattern="train-00-*.tfrec"),
            example_gen_pb2.Input.Split(name="eval", pattern="val-00-*.tfrec"),
        ]
    )
    example_gen = ImportExampleGen(input_base=data_path, input_config=input_config)
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
    components.append(transform)

    tuner = Tuner(
        tuner_fn=modules["tuner_fn"],
        examples=transform.outputs["transformed_examples"],
        schema=schema_gen.outputs["schema"],
        transform_graph=transform.outputs["transform_graph"],
        custom_config={"hyperparameters": hyperparameters},
    )
    components.append(tuner)

    trainer_args = {
        "run_fn": modules["training_fn"],
        "transformed_examples": transform.outputs["transformed_examples"],
        "transform_graph": transform.outputs["transform_graph"],
        "schema": schema_gen.outputs["schema"],
        "hyperparameters": tuner.outputs["best_hyperparameters"],
        "custom_config": {"is_local": True},
    }
    trainer = Trainer(**trainer_args)
    components.append(trainer)

    model_resolver = resolver.Resolver(
        strategy_class=LatestBlessedModelResolver,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing),
    ).with_id("latest_blessed_model_resolver")
    components.append(model_resolver)

    evaluator = Evaluator(
        examples=example_gen.outputs["examples"],
        model=trainer.outputs["model"],
        baseline_model=model_resolver.outputs["model"],
        eval_config=eval_configs,
    )
    components.append(evaluator)

    pusher_args = {
        "model": trainer.outputs["model"],
        "model_blessing": evaluator.outputs["blessing"],
        "push_destination": tfx.proto.PushDestination(
            filesystem=tfx.proto.PushDestination.Filesystem(
                base_directory=serving_model_dir
            )
        ),
    }
    pusher = Pusher(**pusher_args)  # pylint: disable=unused-variable
    components.append(pusher)

    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=False,
        metadata_connection_config=metadata_connection_config,
    )
