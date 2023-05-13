from typing import Dict, List

from traceml.logger import logger
from traceml.processors.events_processors import metrics_dict_to_list
from traceml.vendor import pynvml

try:
    import psutil
except ImportError:
    psutil = None


def can_log_gpu_resources():
    if pynvml is None:
        return False

    try:
        pynvml.nvmlInit()
        return True
    except pynvml.NVMLError:
        return False


def query_gpu(handle_idx: int, handle: any) -> Dict:
    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)  # in Bytes
    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)

    return {
        "gpu_{}_memory_free".format(handle_idx): int(memory.free),
        "gpu_{}_memory_used".format(handle_idx): int(memory.used),
        "gpu_{}_utilization".format(handle_idx): utilization.gpu,
    }


def get_gpu_metrics() -> List:
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        results = []

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            results += metrics_dict_to_list(query_gpu(i, handle))
        return results
    except pynvml.NVMLError:
        logger.debug("Failed to collect gpu resources", exc_info=True)
        return []
