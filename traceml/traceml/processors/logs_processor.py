import logging
import sys

from traceml.logging.handler import LogStreamHandler, LogStreamWriter

EXCLUDE_DEFAULT_LOGGERS = ("polyaxon.client", "polyaxon.cli", "traceml")


def start_log_processor(add_logs, exclude=EXCLUDE_DEFAULT_LOGGERS):
    plx_logger = logging.getLogger("__plx__")
    plx_logger.setLevel(logging.INFO)
    if LogStreamHandler in map(type, plx_logger.handlers):
        for handler in plx_logger.handlers:
            if isinstance(handler, LogStreamHandler):
                handler.set_add_logs(add_logs=add_logs)
    else:
        handler = LogStreamHandler(add_logs=add_logs)
        plx_logger.addHandler(handler)

    exclude = ("__plx__",) + (exclude or ())
    for logger_name in exclude:
        logger = logging.getLogger(logger_name)
        if logging.StreamHandler not in map(type, logger.handlers):
            logger.addHandler(logging.StreamHandler())
            logger.propagate = False

    sys.stdout = LogStreamWriter(plx_logger, logging.INFO, sys.__stdout__)
    sys.stderr = LogStreamWriter(plx_logger, logging.ERROR, sys.__stderr__)


def end_log_processor():
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
