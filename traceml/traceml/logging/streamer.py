import os

from collections import deque
from typing import Callable

from clipped.formatting import Printer
from clipped.utils.json import orjson_loads
from clipped.utils.tz import local_datetime

from polyaxon import settings
from polyaxon._containers.names import MAIN_CONTAINER_NAMES
from traceml.logging.schemas import V1Log, V1Logs


def get_logs_streamer(
    show_timestamp: bool = True, all_containers: bool = False, all_info: bool = False
) -> Callable:
    colors = deque(Printer.COLORS)
    job_to_color = {}
    if all_info:
        all_containers = True

    def handle_log_line(log: V1Log):
        log_dict = log.to_dict()
        log_line = ""
        if log.timestamp and show_timestamp:
            date_value = local_datetime(
                log_dict.get("timestamp"), tz=settings.CLIENT_CONFIG.timezone
            )
            log_line = Printer.add_log_color(date_value, "white") + " | "

        def get_container_info():
            if container_info in job_to_color:
                color = job_to_color[container_info]
            else:
                color = colors[0]
                colors.rotate(-1)
                job_to_color[container_info] = color
            return Printer.add_log_color(container_info, color) + " | "

        if not all_containers and log.container not in MAIN_CONTAINER_NAMES:
            return log_line

        if all_info:
            container_info = ""
            if log.node:
                log_line += Printer.add_log_color(log_dict.get("node"), "white") + " | "
            if log.pod:
                log_line += Printer.add_log_color(log_dict.get("pod"), "white") + " | "
            if log.container:
                container_info = log_dict.get("container")

            log_line += get_container_info()

        log_line += log_dict.get("value")
        Printer.log(log_line, nl=True)

    def handle_log_lines(logs: V1Logs):
        for log in logs.logs:
            if log:
                handle_log_line(log=log)

    return handle_log_lines


def load_logs_from_path(
    logs_path: str,
    hide_time: bool = False,
    all_containers: bool = True,
    all_info: bool = True,
):
    for file_logs in sorted(os.listdir(logs_path)):
        with open(os.path.join(logs_path, file_logs)) as f:
            logs_data = orjson_loads(f.read()).get("logs", [])
            logs_stream = V1Logs(logs=logs_data)
            get_logs_streamer(
                show_timestamp=not hide_time,
                all_containers=all_containers,
                all_info=all_info,
            )(logs_stream)
