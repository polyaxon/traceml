from typing import TYPE_CHECKING, Any

from traceml import tracking
from traceml.exceptions import TracemlException
from traceml.integrations.tensorboard import Logger

try:
    import tensorflow as tf
except ImportError:
    raise TracemlException("tensorflow is required to use the tracking Callback")

try:
    from tensorflow.train import SessionRunHook  # noqa
except ImportError:
    raise TracemlException("tensorflow is required to use the tracking Callback")


if TYPE_CHECKING:
    from traceml.tracking import Run


class Callback(SessionRunHook):
    def __init__(
        self,
        summary_op: Any = None,
        steps_per_log: int = 1000,
        run: "Run" = None,
        log_image: bool = False,
        log_histo: bool = False,
        log_tensor: bool = False,
    ):
        self._summary_op = summary_op
        self._steps_per_log = steps_per_log
        self.run = tracking.get_or_create_run(run)
        self._log_image = log_image
        self._log_histo = log_histo
        self._log_tensor = log_tensor

    def begin(self):
        if self._summary_op is None:
            self._summary_op = tf.summary.merge_all()
        self._step = -1

    def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs({"summary": self._summary_op})

    def after_run(self, run_context, run_values):
        if self._step % self._steps_per_log == 0:
            Logger.process_summary(
                run_values.results["summary"],
                run=self.run,
                log_image=self._log_image,
                log_histo=self._log_histo,
                log_tensor=self._log_tensor,
            )
