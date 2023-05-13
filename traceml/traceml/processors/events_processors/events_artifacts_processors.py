from typing import Optional

from clipped.utils.paths import copy_file_or_dir_path

from traceml.events import V1EventArtifact

try:
    import numpy as np
except ImportError:
    np = None


def artifact_path(
    from_path: str, asset_path: str, kind: str, asset_rel_path: Optional[str] = None
) -> V1EventArtifact:
    copy_file_or_dir_path(from_path, asset_path)
    return V1EventArtifact(kind=kind, path=asset_rel_path or asset_path)
