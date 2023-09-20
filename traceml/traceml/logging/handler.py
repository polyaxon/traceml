import logging
import os
import socket

from clipped.utils.dates import to_datetime
from clipped.utils.env import get_user

from polyaxon import settings
from polyaxon._env_vars.keys import ENV_KEYS_K8S_NODE_NAME, ENV_KEYS_K8S_POD_ID
from traceml.logging.schemas import V1Log


class LogStreamHandler(logging.Handler):
    def __init__(self, add_logs, **kwargs):
        self._add_logs = add_logs
        self._container = socket.gethostname()
        self._node = os.environ.get(ENV_KEYS_K8S_NODE_NAME, "local")
        self._pod = os.environ.get(ENV_KEYS_K8S_POD_ID, get_user())
        log_level = settings.CLIENT_CONFIG.log_level
        if log_level and isinstance(log_level, str):
            log_level = log_level.upper()
        super().__init__(
            level=kwargs.get("level", log_level or logging.NOTSET),
        )

    def set_add_logs(self, add_logs):
        self._add_logs = add_logs

    def can_record(self, record):
        return not (
            record.name == "polyaxon"
            or record.name == "traceml"
            or record.name == "polyaxon.cli"
            or record.name.startswith("polyaxon")
            or record.name.startswith("traceml")
        )

    def format_record(self, record):
        message = ""
        if record.msg:
            message = record.msg
        if record.args:
            message %= record.args
        return V1Log.process_log_line(
            value=message,
            timestamp=to_datetime(record.created),
            node=self._node,
            pod=self._pod,
            container=self._container,
        )

    def emit(self, record):  # pylint:disable=inconsistent-return-statements
        if not self.can_record(record):
            return
        try:
            return self._add_logs(self.format_record(record))
        except Exception:  # noqa
            pass


class LogStreamWriter:
    def __init__(self, logger, log_level, channel):
        self._logger = logger
        self._log_level = log_level
        self._channel = channel

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            if line != "\n":
                self._logger.log(self._log_level, line.rstrip())

    def flush(self):
        self._channel.flush()
