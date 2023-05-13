from typing import Optional

from clipped.utils.paths import copy_file_path

from traceml.events import V1EventDataframe

try:
    import numpy as np
except ImportError:
    np = None


def dataframe_path(
    from_path: str,
    asset_path: str,
    content_type: Optional[str] = None,
    asset_rel_path: Optional[str] = None,
) -> V1EventDataframe:
    copy_file_path(from_path, asset_path)
    return V1EventDataframe(
        path=asset_rel_path or asset_path, content_type=content_type
    )
