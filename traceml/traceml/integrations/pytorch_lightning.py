import os
import uuid

from argparse import Namespace
from typing import Any, Dict, List, Optional, Union

import packaging

from polyaxon._env_vars.keys import ENV_KEYS_RUN_INSTANCE
from polyaxon.client import RunClient
from traceml import tracking
from traceml.exceptions import TracemlException

try:
    import pytorch_lightning as pl

    NEW_PL_VERSION = packaging.version.parse(pl.__version__)

    if NEW_PL_VERSION < packaging.version.parse("1.7"):
        from pytorch_lightning.loggers.base import LightningLoggerBase as Logger
        from pytorch_lightning.loggers.base import rank_zero_experiment
    else:
        from pytorch_lightning.loggers.logger import Logger, rank_zero_experiment

    if NEW_PL_VERSION < packaging.version.parse("1.9"):
        from pytorch_lightning.utilities.logger import (
            _add_prefix,
            _convert_params,
            _flatten_dict,
            _sanitize_callable_params,
        )
    else:
        from lightning_fabric.utilities.logger import (
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
        owner: Optional[str] = None,
        project: Optional[str] = None,
        run_uuid: Optional[str] = None,
        client: RunClient = None,
        track_code: bool = True,
        track_env: bool = True,
        refresh_data: bool = False,
        artifacts_path: Optional[str] = None,
        collect_artifacts: Optional[str] = None,
        collect_resources: Optional[str] = None,
        is_offline: Optional[bool] = None,
        is_new: Optional[bool] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
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
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]):
        params = _convert_params(params)
        params = _flatten_dict(params)
        params = _sanitize_callable_params(params)
        self.experiment.log_inputs(**params)

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        assert rank_zero_only.rank == 0, "experiment tried to log from global_rank != 0"
        metrics = _add_prefix(metrics, self._prefix, self.LOGGER_JOIN_CHAR)
        self.experiment.log_metrics(**metrics, step=step)

    @rank_zero_only
    def log_model_summary(self, model: "pl.LightningModule", max_depth: int = -1):
        summary = str(ModelSummary(model=model, max_depth=max_depth))
        rel_path = self.experiment.get_outputs_path("model_summary.txt")
        with open(rel_path, "w") as f:
            f.write(summary)
        self.experiment.log_file_ref(
            path=rel_path, name="model_summary", is_input=False
        )

    @property
    def save_dir(self) -> Optional[str]:
        return self.experiment.get_outputs_path()

    @rank_zero_only
    def finalize(self, status: str):
        if self._end_on_finalize:
            self.experiment.end()
            self._experiment = None

    def _set_run_instance_from_env_vars(self, force: bool = False):
        """Tries to extract run info from canonical env vars"""
        run_instance = os.getenv(ENV_KEYS_RUN_INSTANCE)
        if not run_instance:
            return

        parts = run_instance.split(".")
        if len(parts) != 4:
            return

        if not self._name or force:
            self._name = parts[2]
        if not self._run_uuid or force:
            self._run_uuid = parts[-1]

    @property
    def name(self) -> str:
        if self._experiment is not None and self._experiment.run_data.name is not None:
            return self.experiment.run_data.name

        if not self._name:
            self._set_run_instance_from_env_vars()

        if self._name:
            return self._name

        return "default"

    @property
    def version(self) -> str:
        if self._experiment is not None and self._experiment.run_data.uuid is not None:
            return self.experiment.run_data.uuid

        if not self._run_uuid:
            self._set_run_instance_from_env_vars()

        if self._run_uuid:
            return self._run_uuid

        return uuid.uuid4().hex
