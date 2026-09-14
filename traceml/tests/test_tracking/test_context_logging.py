import logging
from pathlib import Path
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock

from polyaxon import settings
from polyaxon._env_vars.keys import ENV_KEYS_HAS_PROCESS_SIDECAR, ENV_KEYS_RUN_INSTANCE
from polyaxon._managers.project import ProjectConfigManager
from polyaxon._managers.run import RunConfigManager
from polyaxon._schemas.client import ClientConfig
from polyaxon._sdk.schemas.v1_project import V1Project
from polyaxon._sdk.schemas.v1_run import V1Run
from polyaxon.exceptions import PolyaxonClientException
from traceml.tracking.run import Run


pytestmark = pytest.mark.tracking_mark

CACHED_UUID = "11111111111111111111111111111111"


@pytest.fixture
def make_run(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="polyaxon.cli")
    monkeypatch.chdir(tmp_path)
    for manager in (ProjectConfigManager, RunConfigManager):
        monkeypatch.setattr(manager, "CONFIG_PATH", str(tmp_path / "global"))
    monkeypatch.setattr(settings, "CLIENT_CONFIG", ClientConfig(host="http://polyaxon"))
    monkeypatch.delenv(ENV_KEYS_RUN_INSTANCE, raising=False)
    monkeypatch.delenv(ENV_KEYS_HAS_PROCESS_SIDECAR, raising=False)
    ProjectConfigManager.set_config(
        V1Project(owner="owner", name="project"), visibility="local"
    )
    RunConfigManager.set_config(
        V1Run(uuid=CACHED_UUID, owner="owner", project="project"), visibility="local"
    )

    def make(**kwargs):
        return Run(
            artifacts_path=str(tmp_path / "artifacts"),
            track_code=False,
            track_env=False,
            track_logs=False,
            collect_artifacts=False,
            collect_resources=False,
            auto_create=False,
            **kwargs,
        )

    return make


@pytest.mark.parametrize(
    "log_context",
    [
        pytest.param(None, id="default"),
        pytest.param(False, id="disabled"),
        pytest.param(True, id="enabled"),
    ],
)
def test_context_logging_is_opt_in(log_context, make_run, caplog, capsys):
    kwargs = {} if log_context is None else {"log_context": log_context}
    run = make_run(**kwargs)
    project_path = str(
        Path(ProjectConfigManager.get_config_filepath(create=False)).resolve()
    )
    run_path = str(Path(RunConfigManager.get_config_filepath(create=False)).resolve())

    assert run.log_context is (log_context is True)
    assert (run.owner, run.project) == ("owner", "project")
    expected = []
    if log_context:
        expected.append(
            (
                "polyaxon.cli",
                logging.INFO,
                f"Using cached project `owner/project` from `{project_path}`.",
            )
        )
    assert caplog.record_tuples == expected

    assert run.run_uuid == CACHED_UUID
    assert run.run_uuid == CACHED_UUID
    if log_context:
        expected.append(
            (
                "polyaxon.cli",
                logging.INFO,
                f"Using cached run `{CACHED_UUID}` from `{run_path}`.",
            )
        )
    assert caplog.record_tuples == expected
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


@pytest.mark.parametrize("log_context", [False, True])
def test_cached_conflict_blocks_run_request(log_context, make_run, caplog):
    sdk = SimpleNamespace(is_async=False, config=None, runs_v1=MagicMock())
    run = make_run(
        owner="owner", project="other-project", client=sdk, log_context=log_context
    )

    with pytest.raises(PolyaxonClientException, match="conflicts with project"):
        run.refresh_data()

    sdk.runs_v1.get_run.assert_not_called()
    assert caplog.record_tuples == []


def test_explicit_uuid_overrides_cached_conflict(make_run, caplog):
    run_uuid = "22222222222222222222222222222222"
    run = make_run(
        owner="owner", project="other-project", run_uuid=run_uuid, log_context=True
    )

    assert run.run_uuid == run_uuid
    assert caplog.record_tuples == []
