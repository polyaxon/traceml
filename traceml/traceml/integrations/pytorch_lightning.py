#!/usr/bin/python
#
# Copyright 2018-2023 Polyaxon, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import uuid

from argparse import Namespace
from typing import Any, Dict, List, Optional, Union

import packaging

from polyaxon.client import RunClient
from traceml import tracking
from traceml.exceptions import TracemlException

try:
    import pytorch_lightning as pl

    NEW_PL_VERSION = packaging.version.parse(pl.__version__) < packaging.version.parse(
        "1.7"
    )

    if NEW_PL_VERSION:
        from pytorch_lightning.loggers.base import LightningLoggerBase as Logger
        from pytorch_lightning.loggers.base import rank_zero_experiment
    else:
        from pytorch_lightning.loggers.logger import (
            Logger,
            rank_zero_experiment,
        )

    from pytorch_lightning.utilities.logger import (
        _add_prefix,
        _convert_params,
        _flatten_dict,
        _sanitize_callable_params,
    )
    from pytorch_lightning.utilities.model_summary import ModelSummary
    from pytorch_lightning.utilities.rank_zero import rank_zero_only, rank_zero_warn
except ImportError:
    raise TracemlException("PytorchLightning is required to use the tracking Callback")


class Callback(Logger):
    LOGGER_JOIN_CHAR = "_"

    def __init__(
        self,
        owner: str = None,
        project: str = None,
        run_uuid: str = None,
        client: RunClient = None,
        track_code: bool = True,
        track_env: bool = True,
        refresh_data: bool = False,
        artifacts_path: str = None,
        collect_artifacts: str = None,
        collect_resources: str = None,
        is_offline: bool = None,
        is_new: bool = None,
        name: str = None,
        description: str = None,
        tags: List[str] = None,
        end_on_finalize: bool = False,
        prefix: str = "",
    ):
        super().__init__()
        self._owner = owner
        self._project = project
        self._run_uuid = run_uuid
        self._client = client
        self._track_code = track_code
        self._track_env = track_env
        self._refresh_data = refresh_data
        self._artifacts_path = artifacts_path
        self._collect_artifacts = collect_artifacts
        self._collect_resources = collect_resources
        self._is_offline = is_offline
        self._is_new = is_new
        self._name = name
        self._description = description
        self._tags = tags
        self._end_on_finalize = end_on_finalize
        self._prefix = prefix
        self._experiment = None

    @property
    @rank_zero_experiment
    def experiment(self) -> tracking.Run:
        if self._experiment:
            return self._experiment
        tracking.init(
            owner=self._owner,
            project=self._project,
            run_uuid=self._run_uuid,
            client=self._client,
            track_code=self._track_code,
            track_env=self._track_env,
            refresh_data=self._refresh_data,
            artifacts_path=self._artifacts_path,
            collect_artifacts=self._collect_artifacts,
            collect_resources=self._collect_resources,
            is_offline=self._is_offline,
            is_new=self._is_new,
            name=self._name,
            description=self._description,
            tags=self._tags,
        )
        self._experiment = tracking.TRACKING_RUN
        return self._experiment

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        params = _convert_params(params)
        params = _flatten_dict(params)
        params = _sanitize_callable_params(params)
        self.experiment.log_inputs(**params)

    @rank_zero_only
    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        assert rank_zero_only.rank == 0, "experiment tried to log from global_rank != 0"
        metrics = _add_prefix(metrics, self._prefix, self.LOGGER_JOIN_CHAR)
        self.experiment.log_metrics(**metrics, step=step)

    @rank_zero_only
    def log_model_summary(
        self, model: "pl.LightningModule", max_depth: int = -1
    ) -> None:
        summary = str(ModelSummary(model=model, max_depth=max_depth))
        rel_path = self.run.get_outputs_path("model_summary.txt")
        with open(rel_path, "w") as f:
            f.write(summary)
        self.run.log_file_ref(path=rel_path, name="model_summary", is_input=False)

    @property
    def save_dir(self) -> Optional[str]:
        return self.experiment.get_outputs_path()

    @rank_zero_only
    def finalize(self, status: str) -> None:
        if self._end_on_finalize:
            self.experiment.end()
            self._experiment = None

    @property
    def name(self) -> str:
        if self._experiment is not None and self._experiment.run_data.name is not None:
            return self.experiment.run_data.name
        if self._name:
            return self._name
        return "default"

    @property
    def version(self) -> str:
        if self._experiment is not None and self._experiment.run_data.uuid is not None:
            return self.experiment.run_data.uuid
        if self._run_uuid:
            return self._run_uuid
        return uuid.uuid4().hex
