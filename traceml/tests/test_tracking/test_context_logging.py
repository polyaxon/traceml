from io import StringIO
import os
import pytest
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from polyaxon._contexts import paths as ctx_paths
from polyaxon._env_vars.keys import ENV_KEYS_HAS_PROCESS_SIDECAR, ENV_KEYS_RUN_INSTANCE
from polyaxon._managers.project import ProjectConfigManager
from polyaxon._managers.run import RunConfigManager
from polyaxon._sdk.schemas.v1_project import V1Project
from polyaxon._sdk.schemas.v1_run import V1Run
from polyaxon._utils.test_utils import BaseTestCase
from polyaxon.exceptions import PolyaxonClientException
from traceml.tracking.run import Run


CACHED_UUID = "11111111111111111111111111111111"


@pytest.mark.tracking_mark
class TestRunContextLogging(BaseTestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        local_cache = os.path.join(directory.name, "local", ".polyaxon")
        global_cache = os.path.join(directory.name, "global", ".polyaxon")
        for patcher in (
            patch.object(ctx_paths, "CONTEXT_USER_POLYAXON_PATH", global_cache),
            patch.object(Run, "_set_exit_handler"),
            patch.dict(os.environ),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        for manager in (ProjectConfigManager, RunConfigManager):
            patcher = patch.multiple(
                manager,
                CONFIG_PATH=None,
                _PROJECT=local_cache,
                _PROJECT_PATH=global_cache,
            )
            patcher.start()
            self.addCleanup(patcher.stop)
        super().setUp()
        os.environ.pop(ENV_KEYS_RUN_INSTANCE, None)
        os.environ.pop(ENV_KEYS_HAS_PROCESS_SIDECAR, None)

        ProjectConfigManager.set_config(
            V1Project(owner="owner", name="project"), visibility="local"
        )
        RunConfigManager.set_config(
            V1Run(uuid=CACHED_UUID, owner="owner", project="project"),
            visibility="local",
        )
        self.project_cache_path = os.path.abspath(
            ProjectConfigManager.get_config_filepath(create=False)
        )
        self.run_cache_path = os.path.abspath(
            RunConfigManager.get_config_filepath(create=False)
        )
        self.run_kwargs = {
            "artifacts_path": os.path.join(directory.name, "artifacts"),
            "track_code": False,
            "track_env": False,
            "track_logs": False,
            "collect_artifacts": False,
            "collect_resources": False,
            "auto_create": False,
        }

    @patch("sys.stderr", new_callable=StringIO)
    @patch("sys.stdout", new_callable=StringIO)
    @patch("polyaxon.logger.logger.warning")
    @patch("polyaxon.logger.logger.info")
    def test_context_logging_defaults_to_quiet(
        self, log_info, log_warning, stdout, stderr
    ):
        run = Run(**self.run_kwargs)

        assert run.log_context is False
        assert (run.owner, run.project) == ("owner", "project")
        log_info.assert_not_called()
        assert run.run_uuid == CACHED_UUID
        assert run.run_uuid == CACHED_UUID
        log_info.assert_not_called()
        log_warning.assert_not_called()
        assert stdout.getvalue() == stderr.getvalue() == ""

    @patch("sys.stderr", new_callable=StringIO)
    @patch("sys.stdout", new_callable=StringIO)
    @patch("polyaxon.logger.logger.warning")
    @patch("polyaxon.logger.logger.info")
    def test_context_logging_can_be_disabled(
        self, log_info, log_warning, stdout, stderr
    ):
        run = Run(**self.run_kwargs, log_context=False)

        assert run.log_context is False
        assert (run.owner, run.project) == ("owner", "project")
        log_info.assert_not_called()
        assert run.run_uuid == CACHED_UUID
        assert run.run_uuid == CACHED_UUID
        log_info.assert_not_called()
        log_warning.assert_not_called()
        assert stdout.getvalue() == stderr.getvalue() == ""

    @patch("sys.stderr", new_callable=StringIO)
    @patch("sys.stdout", new_callable=StringIO)
    def test_context_logging_can_be_enabled(self, stdout, stderr):
        with self.assertLogs("polyaxon.cli", level="INFO") as logs:
            run = Run(**self.run_kwargs, log_context=True)

            assert run.log_context is True
            assert (run.owner, run.project) == ("owner", "project")
            project_notice = (
                "INFO:polyaxon.cli:Using cached project `owner/project` "
                f"from `{self.project_cache_path}`."
            )
            assert logs.output == [project_notice]
            assert run.run_uuid == CACHED_UUID
            assert run.run_uuid == CACHED_UUID

        assert logs.output == [
            project_notice,
            f"INFO:polyaxon.cli:Using cached run `{CACHED_UUID}` "
            f"from `{self.run_cache_path}`.",
        ]
        assert stdout.getvalue() == stderr.getvalue() == ""

    @patch("polyaxon.logger.logger.warning")
    @patch("polyaxon.logger.logger.info")
    def test_cached_conflict_blocks_run_request(self, log_info, log_warning):
        for log_context in (False, True):
            with self.subTest(log_context=log_context):
                sdk = SimpleNamespace(is_async=False, config=None, runs_v1=MagicMock())
                run = Run(
                    **self.run_kwargs,
                    owner="owner",
                    project="other-project",
                    client=sdk,
                    log_context=log_context,
                )

                with self.assertRaisesRegex(
                    PolyaxonClientException, "conflicts with project"
                ):
                    run.refresh_data()

                sdk.runs_v1.get_run.assert_not_called()

        log_info.assert_not_called()
        log_warning.assert_not_called()

    @patch("polyaxon.logger.logger.warning")
    @patch("polyaxon.logger.logger.info")
    def test_explicit_uuid_overrides_cached_conflict(self, log_info, log_warning):
        run_uuid = "22222222222222222222222222222222"
        run = Run(
            **self.run_kwargs,
            owner="owner",
            project="other-project",
            run_uuid=run_uuid,
            log_context=True,
        )

        assert run.run_uuid == run_uuid
        log_info.assert_not_called()
        log_warning.assert_not_called()
