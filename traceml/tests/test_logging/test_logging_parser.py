import pytest

from clipped.utils.dates import parse_datetime

from polyaxon._utils.test_utils import BaseTestCase
from traceml.logging.parser import (
    DATETIME_REGEX,
    ISO_DATETIME_REGEX,
    timestamp_search_regex,
)


@pytest.mark.logging_mark
class TestLoggingUtils(BaseTestCase):
    def test_has_timestamp(self):
        log_line = "2018-12-11 10:24:57 UTC"
        log_value, ts = timestamp_search_regex(DATETIME_REGEX, log_line)
        assert ts == parse_datetime("2018-12-11 10:24:57 UTC")
        assert log_value == ""

    def test_log_line_has_datetime(self):
        log_line = "2018-12-11 10:24:57 UTC foo"
        log_value, ts = timestamp_search_regex(DATETIME_REGEX, log_line)

        assert ts == parse_datetime("2018-12-11 10:24:57 UTC")
        assert log_value == "foo"

    def test_log_line_has_iso_datetime(self):
        log_line = "2018-12-11T08:49:07.163495183Z foo"

        log_value, ts = timestamp_search_regex(ISO_DATETIME_REGEX, log_line)

        assert ts == parse_datetime("2018-12-11T08:49:07.163495183Z")
        assert log_value == "foo"
