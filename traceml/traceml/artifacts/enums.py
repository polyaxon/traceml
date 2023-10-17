from typing import Optional, Union

from clipped.utils.enums import PEnum


class V1ArtifactKind(str, PEnum):
    MODEL = "model"
    AUDIO = "audio"
    VIDEO = "video"
    HISTOGRAM = "histogram"
    IMAGE = "image"
    TENSOR = "tensor"
    DATAFRAME = "dataframe"
    CHART = "chart"
    CSV = "csv"
    TSV = "tsv"
    PSV = "psv"
    SSV = "ssv"
    METRIC = "metric"
    ENV = "env"
    HTML = "html"
    TEXT = "text"
    FILE = "file"
    DIR = "dir"
    DOCKERFILE = "dockerfile"
    DOCKER_IMAGE = "docker_image"
    DATA = "data"
    CODEREF = "coderef"
    TABLE = "table"
    TENSORBOARD = "tensorboard"
    CURVE = "curve"
    CONFUSION = "confusion"
    ANALYSIS = "analysis"
    ITERATION = "iteration"
    MARKDOWN = "markdown"
    SYSTEM = "system"
    ARTIFACT = "artifact"
    SPAN = "span"

    @classmethod
    def is_jsonl_file_event(cls, kind: Optional[Union["V1ArtifactKind", str]]) -> bool:
        return kind in {
            V1ArtifactKind.HTML,
            V1ArtifactKind.TEXT,
            V1ArtifactKind.CHART,
            V1ArtifactKind.SPAN,
        }

    @classmethod
    def is_single_file_event(cls, kind: Optional[Union["V1ArtifactKind", str]]) -> bool:
        return kind in {
            V1ArtifactKind.HTML,
            V1ArtifactKind.TEXT,
            V1ArtifactKind.HISTOGRAM,
            V1ArtifactKind.CHART,
            V1ArtifactKind.CONFUSION,
            V1ArtifactKind.CURVE,
            V1ArtifactKind.METRIC,
            V1ArtifactKind.SYSTEM,
            V1ArtifactKind.SPAN,
        }

    @classmethod
    def is_single_or_multi_file_event(
        cls, kind: Optional[Union["V1ArtifactKind", str]]
    ) -> bool:
        return kind in {
            V1ArtifactKind.MODEL,
            V1ArtifactKind.DATAFRAME,
            V1ArtifactKind.AUDIO,
            V1ArtifactKind.VIDEO,
            V1ArtifactKind.IMAGE,
            V1ArtifactKind.CSV,
            V1ArtifactKind.TSV,
            V1ArtifactKind.PSV,
            V1ArtifactKind.SSV,
        }

    @classmethod
    def is_dir(cls, kind: Optional[Union["V1ArtifactKind", str]]) -> bool:
        return kind in {
            V1ArtifactKind.TENSORBOARD,
            V1ArtifactKind.DIR,
        }

    @classmethod
    def is_file(cls, kind: Optional[Union["V1ArtifactKind", str]]) -> bool:
        return kind in {
            V1ArtifactKind.DOCKERFILE,
            V1ArtifactKind.FILE,
            V1ArtifactKind.ENV,
        }

    @classmethod
    def is_file_or_dir(cls, kind: Optional[Union["V1ArtifactKind", str]]) -> bool:
        return kind in {
            V1ArtifactKind.DATA,
            V1ArtifactKind.MODEL,
        }
