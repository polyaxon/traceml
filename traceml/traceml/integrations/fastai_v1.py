from traceml import tracking
from traceml.exceptions import TracemlException

try:
    from fastai.callbacks import TrackerCallback
except ImportError:
    raise TracemlException("Fastai is required to use the tracking Callback")


class Callback(TrackerCallback):
    def __init__(self, learn, run=None, monitor="auto", mode="auto"):
        super().__init__(learn, monitor=monitor, mode=mode)
        if monitor is None:
            # use default TrackerCallback monitor value
            super().__init__(learn, mode=mode)
        self.run = tracking.get_or_create_run(run)

    def on_epoch_end(self, epoch, smooth_loss, last_metrics, **kwargs):
        if not self.run:
            return
        metrics = {
            name: stat
            for name, stat in list(
                zip(self.learn.recorder.names, [epoch, smooth_loss] + last_metrics)
            )[1:]
        }

        self.run.log_metrics(**metrics)
