import numpy as np
import pytest

from traceml.processors.events_processors import metric


@pytest.mark.processors_mark
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, 0.0),
        (np.int64(1), 1.0),
        (np.float32(0.25), 0.25),
        (np.array(2), 2.0),
        (np.array([3]), 3.0),
        (np.array([[4]]), 4.0),
    ],
)
def test_metric_converts_scalar_values(value, expected):
    result = metric(value)

    assert result == expected
    assert isinstance(result, float)


@pytest.mark.processors_mark
def test_metric_rejects_non_scalar_values():
    with pytest.raises(AssertionError, match="scalar should be 0D"):
        metric(np.array([1, 2]))
