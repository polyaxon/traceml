import os

from typing import List, Union

from clipped.utils.enums import get_enum_value
from clipped.utils.paths import check_or_create_path, set_permissions

from traceml.artifacts import V1ArtifactKind
from traceml.logging import V1Log, V1Logs
from traceml.events import (
    LoggedEventListSpec,
    LoggedEventSpec,
    get_event_path,
    get_logs_path,
    get_resource_path,
)


class BaseWriter:
    """Base class for writing events and resources to files."""

    def __init__(self, run_path: str):
        self._run_path = run_path
        self._closed = False

    def flush(self):
        raise NotImplementedError("Subclasses must implement this method.")

    def close(self):
        self.flush()
        self._closed = True

    @property
    def closed(self):
        return self._closed

    def write(self, events: List[LoggedEventSpec]):
        raise NotImplementedError("Subclasses must implement this method.")


class EventWriter(BaseWriter):
    EVENTS_BACKEND = "events"
    RESOURCES_BACKEND = "resources"

    def __init__(self, run_path: str, backend: str):
        super().__init__(run_path)
        self._events_backend = backend
        self._files = {}  # type: dict[str, LoggedEventListSpec]

    def _get_event_path(self, kind: str, name: str) -> str:
        if self._events_backend == self.EVENTS_BACKEND:
            return get_event_path(
                run_path=self._run_path,
                kind=kind,
                name=name,
            )
        if self._events_backend == self.RESOURCES_BACKEND:
            return get_resource_path(
                run_path=self._run_path,
                kind=kind,
                name=name,
            )
        raise ValueError(
            "Unrecognized backend {}".format(get_enum_value(self._events_backend))
        )

    def _init_events(self, events_spec: LoggedEventListSpec):
        event_path = self._get_event_path(kind=events_spec.kind, name=events_spec.name)
        # Check if the file exists otherwise initialize
        if not os.path.exists(event_path):
            check_or_create_path(event_path, is_dir=False)
            with open(event_path, "w") as event_file:
                if V1ArtifactKind.is_jsonl_file_event(events_spec.kind):
                    event_file.write("")
                else:
                    event_file.write(events_spec.get_csv_header())
            set_permissions(event_path)

    def _append_events(self, events_spec: LoggedEventListSpec):
        event_path = self._get_event_path(kind=events_spec.kind, name=events_spec.name)
        with open(event_path, "a") as event_file:
            if V1ArtifactKind.is_jsonl_file_event(events_spec.kind):
                event_file.write(events_spec.get_jsonl_events())
            else:
                event_file.write(events_spec.get_csv_events())

    def _events_to_files(self, events: List[LoggedEventSpec]):
        for event in events:
            file_name = "{}.{}".format(event.kind, event.name)
            if file_name in self._files:
                self._files[file_name].events.append(event.event)
            else:
                self._files[file_name] = LoggedEventListSpec(
                    kind=event.kind, name=event.name, events=[event.event]
                )
                self._init_events(self._files[file_name])

    def write(self, events: List[LoggedEventSpec]):
        if not events:
            return
        if isinstance(events, LoggedEventSpec):
            events = [events]
        self._events_to_files(events)

    def flush(self):
        for file_name in self._files:
            events_spec = self._files[file_name]
            if events_spec.events:
                self._append_events(events_spec)
            self._files[file_name].empty_events()


class LogWriter(BaseWriter):
    def __init__(self, run_path: str):
        super().__init__(run_path)
        self._file = None  # type: V1Logs | None

    def _get_event_path(self) -> str:
        return get_logs_path(run_path=self._run_path)

    def _init_events(self, events_spec: V1Logs):
        event_path = self._get_event_path()
        # Check if the file exists otherwise initialize
        if not os.path.exists(event_path):
            check_or_create_path(event_path, is_dir=False)
            with open(event_path, "w") as event_file:
                event_file.write("")
            set_permissions(event_path)

    def _append_events(self, events_spec: V1Logs):
        event_path = self._get_event_path()
        with open(event_path, "a") as event_file:
            event_file.write(events_spec.get_jsonl_events())

    def _events_to_files(self, logs: List[V1Log]):
        if self._file:
            self._file.logs += logs
        else:
            self._file = V1Logs(logs=logs)
            self._init_events(self._file)

    def write(self, events: List[V1Log]):
        if not events:
            return
        if isinstance(events, V1Log):
            events = [events]
        self._events_to_files(events)

    def flush(self):
        events_spec = self._file
        if events_spec.logs:
            self._append_events(events_spec)
        self._file = V1Logs(logs=[])


class BaseFileWriter:
    """Writes `LoggedEventSpec` to event files.

    The `EventFileWriter` class creates a event files in the run path,
    and asynchronously writes Events to the files.
    """

    def __init__(self, run_path: str):
        self._run_path = run_path
        check_or_create_path(run_path, is_dir=True)

    @property
    def run_path(self):
        return self._run_path

    def add_event(self, event: Union[LoggedEventSpec, V1Log]):
        if not isinstance(event, LoggedEventSpec) and not isinstance(event, V1Log):
            raise TypeError("Expected an LoggedEventSpec,  but got %s" % type(event))
        self._async_writer.write(event)

    def add_events(self, events: List[Union[LoggedEventSpec, V1Log]]):
        for e in events:
            if not isinstance(e, LoggedEventSpec) and not isinstance(e, V1Log):
                raise TypeError("Expected an LoggedEventSpec,  but got %s" % type(e))
        self._async_writer.write(events)

    def flush(self):
        """Flushes the event files to disk.

        Call this method to make sure that all pending events have been
        written to disk.
        """
        self._async_writer.flush()

    def close(self):
        """Performs a final flush of the event files to disk, stops the
        write/flush worker and closes the files.

        Call this method when you do not need the writer anymore.
        """
        self._async_writer.close()
