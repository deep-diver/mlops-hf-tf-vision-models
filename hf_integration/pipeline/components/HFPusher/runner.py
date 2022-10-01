# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Firebase Publisher runner module.
This module handles the workflow to publish
machine learning model to Firebase ML.
"""
import tempfile
from typing import Dict, Optional

import tensorflow as tf
from absl import logging
from tfx.utils import io_utils

from huggingface_hub import Repository
from huggingface_hub import HfApi
from requests.exceptions import HTTPError

from pipeline.components.pusher.HFModelPusher.components import HFSpaceConfig

_MODEL_REPO_KEY = "MODEL_REPO_ID"
_MODEL_URL_KEY = "MODEL_REPO_URL"
_MODEL_VERSION_KEY = "MODEL_VERSION"

_DEFAULT_MODEL_REPO_PLACEHOLDER_KEY = "$MODEL_REPO_ID"
_DEFAULT_MODEL_URL_PLACEHOLDER_KEY = "$MODEL_REPO_URL"
_DEFAULT_MODEL_VERSION_PLACEHOLDER_KEY = "$MODEL_VERSION"

def _replace_placeholders_in_files(
    root_dir: str, placeholder_to_replace: Dict[str, str]
):
    files = tf.io.gfile.listdir(root_dir)
    for file in files:
        path = tf.io.gfile.join(root_dir, file)

        if tf.io.gfile.isdir(path):
            _replace_placeholders_in_files(path, placeholder_to_replace)
        else:
            _replace_placeholders_in_file(path, placeholder_to_replace)


def _replace_placeholders_in_file(filepath: str, placeholder_to_replace: Dict[str, str]):
    with tf.io.gfile.GFile(filepath, "r") as f:
        source_code = f.read()

    for placeholder in placeholder_to_replace:
        source_code = source_code.replace(
            placeholder, placeholder_to_replace[placeholder]
        )

    with tf.io.gfile.GFile(filepath, "w") as f:
        f.write(source_code)

def _replace_placeholders(target_dir: str, placeholders: Dict[str, str], 
                          model_repo_id: str, model_repo_url: str, model_version: str):
        if placeholders is None:
            placeholders = {
                _DEFAULT_MODEL_REPO_PLACEHOLDER_KEY: _MODEL_REPO_KEY,
                _DEFAULT_MODEL_URL_PLACEHOLDER_KEY: _MODEL_URL_KEY,
                _DEFAULT_MODEL_VERSION_PLACEHOLDER_KEY: _MODEL_VERSION_KEY
            }

        placeholder_to_replace = {
            placeholders[_DEFAULT_MODEL_REPO_PLACEHOLDER_KEY]: model_repo_id,
            placeholders[_DEFAULT_MODEL_URL_PLACEHOLDER_KEY]: model_repo_url,
            placeholders[_DEFAULT_MODEL_VERSION_PLACEHOLDER_KEY]: model_version,
        }
        _replace_placeholders_in_files(target_dir, placeholder_to_replace)

def _create_remote_repo(access_token: str, repo_id: str, repo_type: str = "model", space_sdk: str = None):
    try:
        HfApi().create_repo(token=access_token, repo_id=repo_id, repo_type=repo_type, space_sdk=space_sdk)
    except HTTPError:
        logging.warning(f"this warning is expected if {repo_id} repository already exists")

def _push_to_remote_repo(repo: Repository, commit_msg: str, branch: str = "main"):
    repo.git_add(pattern=".", auto_lfs_track=True)
    repo.git_commit(commit_message=commit_msg)
    repo.git_push(upstream=f"origin {branch}")

def deploy_model_for_hf_hub(
    username: str,
    access_token: str,
    repo_name: str,
    model_path: str,
    model_version: str,
    space_config: Optional[HFSpaceConfig] = None,
) -> Dict[str, str]:
    outputs = {}

    repo_url_prefix = "https://huggingface.co"
    repo_id = f"{username}/{repo_name}"
    repo_url = f"{repo_url_prefix}/{repo_id}"

    _create_remote_repo(access_token=access_token,
                        repo_id=repo_id)
    logging.info(f"remote repository at {repo_url} is prepared")

    tmp_dir = tempfile.gettempdir()
    repository = Repository(
        local_dir=tmp_dir, clone_from=repo_url, use_auth_token=access_token
    )
    repository.git_checkout(revision=model_version, create_branch_ok=True)
    logging.info(f"remote repository is cloned, and new branch {model_version} is created")

    io_utils.copy_dir(model_path, tmp_dir)
    logging.info("current version of the model is copied to the cloned local repository")

    _push_to_remote_repo(
        repo=repository, 
        commit_msg=f"updload new version({model_version})",
        branch=model_version,
    )
    logging.info("updates are pushed to the remote repository")
    outputs["repo_id"] = repo_id
    outputs["branch"] = model_version
    outputs["commit_id"] = f"{repository.git_head_hash()}"
    outputs["repo_url"] = repo_url

    if space_config is not None:
        model_repo_id = repo_id
        model_repo_url = repo_url
        
        repo_url = f"{repo_url_prefix}/spaces/{repo_id}"

        app_path = space_config.app_path
        app_path = app_path.replace(".", "/")

        _create_remote_repo(access_token=access_token,
                        repo_id=repo_id,
                        repo_type="space",
                        space_sdk=space_config.space_sdk)

        tmp_dir = tmp_dir + "_space"
        repository = Repository(
            local_dir=tmp_dir, clone_from=repo_url, use_auth_token=access_token
        )

        io_utils.copy_dir(app_path, tmp_dir)

        _replace_placeholders(target_dir=tmp_dir, placeholders=space_config.placeholders,
                              model_repo_id=model_repo_id, model_repo_url=model_repo_url, model_version=model_version)

        _push_to_remote_repo(
            repo=repository,
            commit_msg=f"updload {model_version} model",
        )

        outputs["space_url"] = repo_url

    return outputs