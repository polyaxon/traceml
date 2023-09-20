import pytest

from polyaxon._utils.test_utils import BaseTestCase
from traceml.processors.events_processors import metrics_dict_to_list


@pytest.mark.processors_mark
class TestEventWriter(BaseTestCase):
    def test_gpu_resources_to_metrics(self):
        resources = {
            "gpu_0_memory_free": 1000,
            "gpu_0_memory_used": 8388608000,
            "gpu_0_utilization": 76,
        }

        events = metrics_dict_to_list(resources)
        assert len(events) == 3
        assert [e.event.metric for e in events] == [1000, 8388608000, 76]

    def test_psutil_resources_to_metrics(self):
        resources = {
            "cpu_percent_avg": 1000,
            "cpu_percent_1": 0.3,
            "cpu_percent_2": 0.5,
            "getloadavg": 76,
            "memory_total": 12883853312,
            "memory_used": 8388608000,
        }

        events = metrics_dict_to_list(resources)
        assert len(events) == 6
        assert [e.event.metric for e in events] == [
            1000,
            0.3,
            0.5,
            76,
            12883853312,
            8388608000,
        ]
