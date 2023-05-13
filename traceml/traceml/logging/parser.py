import re

from clipped.utils.dates import parse_datetime

# pylint:disable=anomalous-backslash-in-string

DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S %Z"  # noqa
ISO_DATETIME_REGEX = re.compile(  # noqa
    r"([0-9]+)-(0[1-9]|1[012])-(0[1-9]|[12][0-9]|3[01])[Tt]"
    r"([01][0-9]|2[0-3]):([0-5][0-9]):([0-5][0-9]|60)(\.[0-9]+)?"
    r"(([Zz])|([\+|\-]([01][0-9]|2[0-3]):[0-5][0-9]))\s?"
)
DATETIME_REGEX = re.compile(  # noqa
    r"\d{2}(?:\d{2})?-\d{1,2}-\d{1,2}\s\d{1,2}:\d{1,2}:\d{1,2}\s\w+\s?"
)


def timestamp_search_regex(regex, log_line):
    log_search = regex.search(log_line)
    if not log_search:
        return log_line, None

    ts = log_search.group()
    ts = parse_datetime(ts)

    return re.sub(regex, "", log_line), ts
