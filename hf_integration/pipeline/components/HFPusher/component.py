from typing import Any, Dict, Optional

from typing import Dict, List, Optional

from tfx import types
from tfx.dsl.components.base import base_component, executor_spec
from tfx.types import standard_artifacts
from tfx.types.component_spec import ChannelParameter, ExecutionParameter

from pipeline.components.pusher.HFModelPusher import executor

MODEL_KEY = "model"
PUSHED_MODEL_KEY = "pushed_model"
MODEL_BLESSING_KEY = "model_blessing"

class HFSpaceConfig:
    def __init__(
        self,
        app_path: str,
        repo_name: str,
        space_sdk: Optional[str] = "gradio",
        placeholders: Optional[Dict] = None
        # {
        #     "model_repo": "$HF_MODEL_REPO_NAME",
        #     "model_branch": "$HF_MODEL_REPO_BRANCH"
        # },
    ):
        self.app_path = app_path
        self.space_sdk = space_sdk
        self.placeholders = placeholders
        self.repo_name = repo_name

class HFPusherSpec(types.ComponentSpec):
  """ComponentSpec for TFX HFPusher Component."""

  PARAMETERS = {
      "username": ExecutionParameter(type=str),
      "access_token": ExecutionParameter(type=str),
      "repo_name": ExecutionParameter(type=str),
      "space_config": ExecutionParameter(type=HFSpaceConfig, optional=True),
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
        access_token: str,
        repo_name: str,
        model: types.Channel,
        space_config: Optional[HFSpaceConfig] = None,
        model_blessing: Optional[types.Channel] = None,
    ):
        """Construct a Pusher component.

        """

        pushed_model = types.Channel(type=standard_artifacts.PushedModel)

        spec = HFPusherSpec(username=username,
                            access_token=access_token,
                            repo_name=repo_name,
                            space_config=space_config,
                            model=model,
                            model_blessing=model_blessing,
                            pushed_model=pushed_model)

        super().__init__(spec=spec)