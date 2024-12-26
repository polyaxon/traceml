import datetime
import json

from collections import namedtuple
from typing import Any, ClassVar, Dict, List, Mapping, Optional, Union

from clipped.compact.pydantic import (
    StrictStr,
    model_validator,
    validation_after,
    validation_before,
)
from clipped.config.parser import ConfigParser
from clipped.config.schema import skip_partial
from clipped.types.uuids import UUIDStr
from clipped.utils.dates import parse_datetime
from clipped.utils.enums import PEnum
from clipped.utils.np import sanitize_np_types
from clipped.utils.strings import validate_file_or_buffer
from clipped.utils.tz import now

from polyaxon._schemas.base import BaseSchemaModel
from polyaxon._schemas.lifecycle import V1StatusCondition, V1Statuses
from traceml.artifacts.enums import V1ArtifactKind
from traceml.logger import logger


class SearchView(str, PEnum):
    ANY = "any"
    RUNS = "runs"
    SELECTION = "selection"
    ANALYTICS = "analytics"
    COMPONENTS = "components"
    MODELS = "models"
    ARTIFACTS = "artifacts"
    PROJECTS = "projects"


class V1EventSpanKind(str, PEnum):
    LLM = "llm"
    CHAIN = "chain"
    AGENT = "agent"
    TOOL = "tool"
    EMBEDDING = "embedding"
    RETRIEVER = "retriever"


class V1EventSpan(BaseSchemaModel):
    uuid: Optional[UUIDStr] = None
    name: Optional[StrictStr] = None
    tags: Optional[List[StrictStr]] = None
    started_at: Optional[datetime.datetime] = None
    finished_at: Optional[datetime.datetime] = None
    status: Optional[V1Statuses] = None
    status_conditions: Optional[List[V1StatusCondition]] = None
    metadata: Optional[Dict[str, Any]] = None
    inputs: Optional[Dict[str, Any]] = None
    outputs: Optional[Dict[str, Any]] = None
    children: Optional[List["V1EventSpan"]] = None
    kind: Optional[V1EventSpanKind] = None

    def add_metadata(self, key: str, value: Any) -> None:
        if self.metadata is None:
            self.metadata = {}
        self.metadata[key] = value

    def add_named_result(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        self.inputs = inputs
        self.outputs = outputs

    def add_child_span(self, span: "V1EventSpan") -> None:
        if self.children is None:
            self.children = []
        self.children.append(span)


class V1EventImage(BaseSchemaModel):
    _IDENTIFIER = V1ArtifactKind.IMAGE

    height: Optional[int] = None
    width: Optional[int] = None
    colorspace: Optional[int] = None
    path: Optional[StrictStr] = None


class V1EventVideo(BaseSchemaModel):
    _IDENTIFIER = V1ArtifactKind.VIDEO

    height: Optional[int] = None
    width: Optional[int] = None
    colorspace: Optional[int] = None
    path: Optional[StrictStr] = None
    content_type: Optional[StrictStr] = None


class V1EventDataframe(BaseSchemaModel):
    _IDENTIFIER = V1ArtifactKind.DATAFRAME

    path: Optional[StrictStr] = None
    content_type: Optional[StrictStr] = None


class V1EventHistogram(BaseSchemaModel):
    _IDENTIFIER = V1ArtifactKind.HISTOGRAM

    values: Optional[List[float]] = None
    counts: Optional[List[float]] = None


class V1EventAudio(BaseSchemaModel):
    _IDENTIFIER = V1ArtifactKind.AUDIO

    sample_rate: Optional[float] = None
    num_channels: Optional[int] = None
    length_frames: Optional[int] = None
    path: Optional[StrictStr] = None
    content_type: Optional[StrictStr] = None


class V1EventChartKind(str, PEnum):
    PLOTLY = "plotly"
    BOKEH = "bokeh"
    VEGA = "vega"


class V1EventChart(BaseSchemaModel):
    _IDENTIFIER = V1ArtifactKind.CHART

    kind: Optional[V1EventChartKind] = None
    figure: Optional[Dict] = None

    def to_json(self, humanize_values=False, include_kind=False, include_version=False):
        if self.kind == V1EventChartKind.PLOTLY:
            import plotly.tools

            obj = self.to_dict(
                humanize_values=humanize_values,
                include_kind=include_kind,
                include_version=include_version,
            )
            return json.dumps(obj, cls=plotly.utils.PlotlyJSONEncoder)
        # Resume normal serialization
        return super().to_json(
            humanize_values=humanize_values,
            include_kind=include_kind,
            include_version=include_version,
        )


class V1EventCurveKind(str, PEnum):
    ROC = "roc"
    PR = "pr"
    CUSTOM = "custom"


class V1EventCurve(BaseSchemaModel):
    _IDENTIFIER = V1ArtifactKind.CURVE

    kind: Optional[V1EventCurveKind] = None
    x: Optional[List[float]] = None
    y: Optional[List[float]] = None
    annotation: Optional[StrictStr] = None


class V1EventConfusionMatrix(BaseSchemaModel):
    _IDENTIFIER = V1ArtifactKind.CONFUSION

    x: Optional[List] = None
    y: Optional[List] = None
    z: Optional[List] = None


class V1EventArtifact(BaseSchemaModel):
    _IDENTIFIER = "artifact"

    kind: Optional[V1ArtifactKind] = None
    path: Optional[StrictStr] = None


class V1EventModel(BaseSchemaModel):
    _IDENTIFIER = V1ArtifactKind.MODEL

    framework: Optional[StrictStr] = None
    path: Optional[StrictStr] = None
    spec: Optional[Dict] = None


class V1Event(BaseSchemaModel):
    _SEPARATOR: ClassVar = "|"
    _IDENTIFIER = "event"

    timestamp: Optional[datetime.datetime] = None
    step: Optional[int] = None
    metric: Optional[float] = None
    image: Optional[V1EventImage] = None
    histogram: Optional[V1EventHistogram] = None
    audio: Optional[V1EventAudio] = None
    video: Optional[V1EventVideo] = None
    html: Optional[str] = None
    text: Optional[str] = None
    chart: Optional[V1EventChart] = None
    curve: Optional[V1EventCurve] = None
    confusion: Optional[V1EventConfusionMatrix] = None
    artifact: Optional[V1EventArtifact] = None
    model: Optional[V1EventModel] = None
    dataframe: Optional[V1EventDataframe] = None
    span: Optional[V1EventSpan] = None

    @model_validator(**validation_before)
    @skip_partial
    def pre_validate(cls, values):
        v = values.get(V1ArtifactKind.IMAGE)
        get_dict = ConfigParser.parse(Dict)
        if v is not None and not isinstance(v, BaseSchemaModel):
            values[V1ArtifactKind.IMAGE] = get_dict(
                key=V1ArtifactKind.IMAGE,
                value=v,
            )
        v = values.get(V1ArtifactKind.HISTOGRAM)
        if v is not None and not isinstance(v, BaseSchemaModel):
            values[V1ArtifactKind.HISTOGRAM] = get_dict(
                key=V1ArtifactKind.HISTOGRAM,
                value=v,
            )
        v = values.get(V1ArtifactKind.AUDIO)
        if v is not None and not isinstance(v, BaseSchemaModel):
            values[V1ArtifactKind.AUDIO] = get_dict(
                key=V1ArtifactKind.AUDIO,
                value=v,
            )
        v = values.get(V1ArtifactKind.VIDEO)
        if v is not None and not isinstance(v, BaseSchemaModel):
            values[V1ArtifactKind.VIDEO] = get_dict(
                key=V1ArtifactKind.VIDEO,
                value=v,
            )
        v = values.get(V1ArtifactKind.CHART)
        if v is not None and not isinstance(v, BaseSchemaModel):
            values[V1ArtifactKind.CHART] = get_dict(
                key=V1ArtifactKind.CHART,
                value=v,
            )
        v = values.get(V1ArtifactKind.CURVE)
        if v is not None and not isinstance(v, BaseSchemaModel):
            values[V1ArtifactKind.CURVE] = get_dict(
                key=V1ArtifactKind.CURVE,
                value=v,
            )
        v = values.get(V1ArtifactKind.CONFUSION)
        if v is not None and not isinstance(v, BaseSchemaModel):
            values[V1ArtifactKind.CONFUSION] = get_dict(
                key=V1ArtifactKind.CONFUSION,
                value=v,
            )
        v = values.get(V1ArtifactKind.ARTIFACT)
        if v is not None and not isinstance(v, BaseSchemaModel):
            values[V1ArtifactKind.ARTIFACT] = get_dict(
                key=V1ArtifactKind.ARTIFACT,
                value=v,
            )
        v = values.get(V1ArtifactKind.MODEL)
        if v is not None and not isinstance(v, BaseSchemaModel):
            values[V1ArtifactKind.MODEL] = get_dict(
                key=V1ArtifactKind.MODEL,
                value=v,
            )
        v = values.get(V1ArtifactKind.DATAFRAME)
        if v is not None and not isinstance(v, BaseSchemaModel):
            values[V1ArtifactKind.DATAFRAME] = get_dict(
                key=V1ArtifactKind.DATAFRAME,
                value=v,
            )
        v = values.get(V1ArtifactKind.SPAN)
        if v is not None and not isinstance(v, BaseSchemaModel):
            values[V1ArtifactKind.SPAN] = get_dict(
                key=V1ArtifactKind.SPAN,
                value=v,
            )

        return values

    @model_validator(**validation_after)
    @skip_partial
    def validate_event(cls, values):
        count = 0

        def increment(c):
            c += 1
            if c > 1:
                raise ValueError(
                    "An event should have one and only one primitive, found {}.".format(
                        c
                    )
                )
            return c

        if cls.get_value_for_key(V1ArtifactKind.METRIC, values) is not None:
            count = increment(count)
        if cls.get_value_for_key(V1ArtifactKind.IMAGE, values) is not None:
            count = increment(count)
        if cls.get_value_for_key(V1ArtifactKind.HISTOGRAM, values) is not None:
            count = increment(count)
        if cls.get_value_for_key(V1ArtifactKind.AUDIO, values) is not None:
            count = increment(count)
        if cls.get_value_for_key(V1ArtifactKind.VIDEO, values) is not None:
            count = increment(count)
        if cls.get_value_for_key(V1ArtifactKind.HTML, values) is not None:
            count = increment(count)
        if cls.get_value_for_key(V1ArtifactKind.TEXT, values) is not None:
            count = increment(count)
        if cls.get_value_for_key(V1ArtifactKind.CHART, values) is not None:
            count = increment(count)
        if cls.get_value_for_key(V1ArtifactKind.CURVE, values) is not None:
            count = increment(count)
        if cls.get_value_for_key(V1ArtifactKind.CONFUSION, values) is not None:
            count = increment(count)
        if cls.get_value_for_key(V1ArtifactKind.ARTIFACT, values) is not None:
            count = increment(count)
        if cls.get_value_for_key(V1ArtifactKind.MODEL, values) is not None:
            count = increment(count)
        if cls.get_value_for_key(V1ArtifactKind.DATAFRAME, values) is not None:
            count = increment(count)
        if cls.get_value_for_key(V1ArtifactKind.SPAN, values) is not None:
            count = increment(count)

        if count != 1:
            raise ValueError(
                "An event should have one and only one primitive, found {}.".format(
                    count
                )
            )

        return values

    @classmethod
    def make(
        cls,
        step: Optional[int] = None,
        timestamp=None,
        metric: float = None,
        image: V1EventImage = None,
        histogram: V1EventHistogram = None,
        audio: V1EventAudio = None,
        video: V1EventVideo = None,
        html: Optional[str] = None,
        text: Optional[str] = None,
        chart: V1EventChart = None,
        curve: V1EventCurve = None,
        confusion: V1EventConfusionMatrix = None,
        artifact: V1EventArtifact = None,
        model: V1EventModel = None,
        dataframe: V1EventDataframe = None,
        span: V1EventSpan = None,
    ) -> "V1Event":
        if isinstance(timestamp, str):
            try:
                timestamp = parse_datetime(timestamp)
            except Exception as e:
                raise ValueError("Received an invalid timestamp") from e

        return cls(
            timestamp=timestamp if timestamp else now(tzinfo=True),
            step=step,
            metric=metric,
            image=image,
            histogram=histogram,
            audio=audio,
            video=video,
            html=html,
            text=text,
            chart=chart,
            curve=curve,
            confusion=confusion,
            artifact=artifact,
            model=model,
            dataframe=dataframe,
            span=span,
        )

    def get_value(self, dump=True):
        if self.metric is not None:
            return str(self.metric) if dump else self.metric
        if self.image is not None:
            return self.image.to_json() if dump else self.image
        if self.histogram is not None:
            return self.histogram.to_json() if dump else self.histogram
        if self.audio is not None:
            return self.audio.to_json() if dump else self.audio
        if self.video is not None:
            return self.video.to_json() if dump else self.video
        if self.html is not None:
            return self.html
        if self.text is not None:
            return self.text
        if self.chart is not None:
            return self.chart.to_json() if dump else self.chart
        if self.curve is not None:
            return self.curve.to_json() if dump else self.curve
        if self.confusion is not None:
            return self.confusion.to_json() if dump else self.confusion
        if self.artifact is not None:
            return self.artifact.to_json() if dump else self.artifact
        if self.model is not None:
            return self.model.to_json() if dump else self.model
        if self.dataframe is not None:
            return self.dataframe.to_json() if dump else self.dataframe
        if self.span is not None:
            return self.span.to_json() if dump else self.span

    def to_json(
        self,
        humanize_values: bool = False,
        include_kind: bool = False,
        include_version: bool = False,
        kind: Optional[str] = None,
    ) -> str:
        if not kind:
            kind = self._IDENTIFIER
        values = {
            "timestamp": str(self.timestamp),
            "step": self.step,
            kind: self.get_value(dump=True),
        }

        return json.dumps(values)

    def to_csv(self) -> str:
        values = [
            str(self.step) if self.step is not None else "",
            str(self.timestamp) if self.timestamp is not None else "",
            self.get_value(dump=True),
        ]

        return self._SEPARATOR.join(values)


class V1Events:
    ORIENT_CSV = "csv"
    ORIENT_DICT = "dict"

    def __init__(self, kind, name, df):
        self.kind = kind
        self.name = name
        self.df = df

    @staticmethod
    def _read_csv(
        data: str,
        parse_dates: Optional[bool] = True,
        engine: Optional[str] = None,
    ) -> "pandas.DataFrame":
        import pandas as pd

        data = validate_file_or_buffer(data)
        if parse_dates:
            return pd.read_csv(
                data,
                sep=V1Event._SEPARATOR,
                parse_dates=["timestamp"],
                engine=engine,
            )
        else:
            df = pd.read_csv(
                data,
                sep=V1Event._SEPARATOR,
                engine=engine,
            )
            # Pyarrow automatically converts timestamp fields
            if "timestamp" in df.columns:
                df["timestamp"] = df["timestamp"].astype(str)
            return df

    @staticmethod
    def _read_jsonl(
        data: str,
        parse_dates: Optional[bool] = True,
    ) -> "pandas.DataFrame":
        import pandas as pd

        data = validate_file_or_buffer(data)
        if parse_dates:
            df = pd.read_json(
                data,
                lines=True,
            )
        else:
            df = pd.read_json(
                data,
                lines=True,
            )
            # Pyarrow automatically converts timestamp fields
            if "timestamp" in df.columns:
                df["timestamp"] = df["timestamp"].astype(str)
        if not hasattr(df, "step"):
            df["step"] = None
        return df

    @classmethod
    def read(
        cls,
        kind: str,
        name: str,
        data: Union[str, Dict],
        parse_dates: Optional[bool] = True,
        engine: Optional[str] = None,
    ) -> "V1Events":
        import pandas as pd

        if isinstance(data, str):
            error = None
            if V1ArtifactKind.is_jsonl_file_event(kind):
                try:
                    df = cls._read_jsonl(data=data, parse_dates=parse_dates)
                except Exception as e:
                    error = e
            if not V1ArtifactKind.is_jsonl_file_event(kind) or error:
                df = cls._read_csv(data=data, parse_dates=parse_dates, engine=engine)
        elif isinstance(data, dict):
            df = pd.DataFrame.from_dict(data)
        else:
            raise ValueError(
                "V1Events received an unsupported value type: {}".format(type(data))
            )

        return cls(name=name, kind=kind, df=df)

    def to_dict(self, orient: str = "list") -> Dict:
        import numpy as np

        return self.df.replace({np.nan: None}).to_dict(orient=orient)

    def get_event_at(self, index):
        event = self.df.iloc[index].to_dict()
        event["timestamp"] = event["timestamp"].isoformat()
        event["step"] = sanitize_np_types(event["step"])
        return V1Event.from_dict(event)

    def _get_step_summary(self) -> Optional[Dict]:
        _count = self.df.step.count()
        if _count == 0:
            return None

        return {
            "count": sanitize_np_types(_count),
            "min": sanitize_np_types(self.df.step.iloc[0]),
            "max": sanitize_np_types(self.df.step.iloc[-1]),
        }

    def _get_ts_summary(self) -> Optional[Dict]:
        _count = self.df.timestamp.count()
        if _count == 0:
            return None

        try:
            return {
                "min": self.df.timestamp.iloc[0].isoformat(),
                "max": self.df.timestamp.iloc[-1].isoformat(),
            }
        except Exception as e:
            logger.debug("Received an error while parsing timestamps: %s", e)
            return None

    def get_summary(self) -> Dict:
        summary = {"is_event": True}
        step_summary = self._get_step_summary()
        if step_summary:
            summary["step"] = step_summary

        ts_summary = self._get_ts_summary()
        if ts_summary:
            summary["timestamp"] = ts_summary

        if self.kind == V1ArtifactKind.METRIC:
            summary[self.kind] = {
                k: sanitize_np_types(v)
                for k, v in self.df.metric.describe().to_dict().items()
            }
            summary[self.kind]["last"] = sanitize_np_types(self.df.metric.iloc[-1])

        return summary


class LoggedEventSpec(namedtuple("LoggedEventSpec", "name kind event")):
    pass


class LoggedEventListSpec(namedtuple("LoggedEventListSpec", "name kind events")):
    def get_csv_header(self) -> str:
        return V1Event._SEPARATOR.join(["step", "timestamp", self.kind])

    def get_csv_events(self) -> str:
        events = ["\n{}".format(e.to_csv()) for e in self.events]
        return "".join(events)

    def get_jsonl_events(self) -> str:
        events = ["\n{}".format(e.to_json(kind=self.kind)) for e in self.events]
        return "".join(events)

    def empty_events(self):
        self.events[:] = []

    def to_dict(self):
        return {
            "name": self.name,
            "kind": self.kind,
            "events": [e.to_dict() for e in self.events],
        }

    @classmethod
    def from_dict(cls, value: Mapping) -> "LoggedEventListSpec":
        return cls(
            name=value.get("name"),
            kind=value.get("kind"),
            events=[V1Event.from_dict(e) for e in value.get("events", [])],
        )
