from typing import Dict, List

from traceml.processors.events_processors import metrics_dict_to_list

try:
    import psutil
except ImportError:
    psutil = None


def can_log_psutil_resources():
    return psutil is not None


def query_psutil() -> Dict:
    results = {}
    try:
        # psutil <= 5.6.2 did not have getloadavg:
        if hasattr(psutil, "getloadavg"):
            results["load"] = psutil.getloadavg()[0]
        else:
            # Do not log an empty metric
            pass
    except OSError:
        pass
    vm = psutil.virtual_memory()
    results["cpu"] = psutil.cpu_percent(interval=None)
    results["memory"] = vm.percent
    return results


def get_psutils_metrics() -> List:
    return metrics_dict_to_list(query_psutil())
