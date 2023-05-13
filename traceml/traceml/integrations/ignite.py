from traceml.exceptions import TracemlException
from traceml.run import Run

try:
    from ignite.contrib.handlers.polyaxon_logger import PolyaxonLogger
except ImportError:
    raise TracemlException("ignite is required to use the tracking Logger")


class Logger(PolyaxonLogger):
    def __init__(self, *args, **kwargs):
        self.experiment = kwargs.get("run", Run(*args, **kwargs))
