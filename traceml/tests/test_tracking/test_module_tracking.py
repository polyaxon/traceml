import os
import pytest
import tempfile
from unittest.mock import patch

from polyaxon._contexts import paths as ctx_paths
from polyaxon._managers.user import UserConfigManager
from polyaxon._sdk.schemas.v1_user import V1User
from polyaxon._utils.test_utils import BaseTestCase
from traceml import tracking


@pytest.mark.tracking_mark
class TestTrackingModule(BaseTestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        for patcher in (
            patch.object(
                ctx_paths,
                "CONTEXT_USER_POLYAXON_PATH",
                os.path.join(directory.name, ".polyaxon"),
            ),
            patch.object(UserConfigManager, "CONFIG_PATH", directory.name),
            patch.object(tracking, "TRACKING_RUN", None),
            patch.object(tracking.Run, "_set_exit_handler"),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        super().setUp()

        UserConfigManager.set_config(V1User(organization="owner/team"))
        self.owner_cache_path = os.path.abspath(
            UserConfigManager.get_config_filepath(create=False)
        )
        self.init_kwargs = {
            "project": "project",
            "run_uuid": "11111111111111111111111111111111",
            "artifacts_path": os.path.join(directory.name, "artifacts"),
            "track_code": False,
            "track_env": False,
            "track_logs": False,
            "collect_artifacts": False,
            "collect_resources": False,
        }

    @patch("polyaxon.logger.logger.info")
    def test_init_context_logging_defaults_to_quiet(self, log_info):
        run = tracking.init(**self.init_kwargs)

        assert run.log_context is False
        assert tracking.TRACKING_RUN is run
        log_info.assert_not_called()

    @patch("polyaxon.logger.logger.info")
    def test_init_context_logging_can_be_disabled(self, log_info):
        run = tracking.init(**self.init_kwargs, log_context=False)

        assert run.log_context is False
        assert tracking.TRACKING_RUN is run
        log_info.assert_not_called()

    def test_init_context_logging_can_be_enabled(self):
        with self.assertLogs("polyaxon.cli", level="INFO") as logs:
            run = tracking.init(**self.init_kwargs, log_context=True)

        assert run.log_context is True
        assert (run.owner, run.team) == ("owner", "team")
        assert tracking.TRACKING_RUN is run
        assert tracking.get_or_create_run() is run
        assert logs.output == [
            "INFO:polyaxon.cli:Using cached owner `owner/team` "
            f"from `{self.owner_cache_path}`."
        ]
