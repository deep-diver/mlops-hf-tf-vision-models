from typing import Any, Dict, Optional

from typing import Dict, List, Optional

from tfx import types
from tfx.dsl.components.base import base_component, executor_spec
from tfx.types import standard_artifacts
from tfx.types.component_spec import ChannelParameter, ExecutionParameter

from pipeline.components.pusher.HFModelPusher import executor

# HFSpaceConfig(
#     app_path: str,
#     repo_name: Optional[str],
#     space_sdk: Optional[str] = "gradio",
#     placeholders: Optional[Dict] = None
# )

# # default placeholders
# # the keys should be used as is. the values can be 
# # changed as needed. if so, make sure there are the
# # same strings in the files under `app_path`
# {
#     "model_repo": "$HF_MODEL_REPO_NAME",
#     "model_branch": "$HF_MODEL_REPO_BRANCH"
# }

MODEL_KEY = "model"
PUSHED_MODEL_KEY = "pushed_model"
MODEL_BLESSING_KEY = "model_blessing"

class HFSpaceConfig:
    def __init__(
        self,
        app_path: str,
        repo_name: str,
        space_sdk: Optional[str] = "gradio",
        placeholders: Optional[Dict] = {
            "model_repo": "$HF_MODEL_REPO_NAME",
            "model_branch": "$HF_MODEL_REPO_BRANCH"
        },
    ):
        self.app_path = app_path
        self.space_sdk = space_sdk
        self.placeholders = placeholders
        self.repo_name = repo_name

class HFPusherSpec(types.ComponentSpec):
  """ComponentSpec for TFX FirebasePublisher Component."""

  PARAMETERS = {
      "username": ExecutionParameter(type=str),
      "hf_access_token": ExecutionParameter(type=str),
      "repo_name": ExecutionParameter(type=str),
      "hf_spce_config": ExecutionParameter(type=HFSpaceConfig, optional=True),
  }
  INPUTS = {
      MODEL_KEY:
      ChannelParameter(type=standard_artifacts.Model),
      MODEL_BLESSING_KEY:
      ChannelParameter(type=standard_artifacts.ModelBlessing, optional=True),
  }
  OUTPUTS = {
      PUSHED_MODEL_KEY: ChannelParameter(type=standard_artifacts.PushedModel),
  }


class HFPusher(base_component.BaseComponent):
    """Component for pushing model to Cloud AI Platform serving."""

    EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.Executor)

    def __init__(
        self,
        username: str,
        hf_access_token: str,
        repo_name: str,
        model: types.Channel,
        hf_space_config: Optional[HFSpaceConfig] = None,
        model_blessing: Optional[types.Channel] = None,
    ):
        """Construct a Pusher component.
        Args:
          model: An optional Channel of type `standard_artifacts.Model`, usually
            produced by a Trainer component, representing the model used for
            training.
          model_blessing: An optional Channel of type
            `standard_artifacts.ModelBlessing`, usually produced from an Evaluator
            component, containing the blessing model.
          infra_blessing: An optional Channel of type
            `standard_artifacts.InfraBlessing`, usually produced from an
            InfraValidator component, containing the validation result.
          custom_config: A dict which contains the deployment job parameters to be
            passed to Cloud platforms.
        """

        pushed_model = types.Channel(type=standard_artifacts.PushedModel)

        spec = HFPusherSpec(username=username,
                            hf_access_token=hf_access_token,
                            repo_name=repo_name,
                            hf_spce_config=hf_space_config,
                            model=model,
                            model_blessing=model_blessing,
                            pushed_model=pushed_model)

        super().__init__(spec=spec)