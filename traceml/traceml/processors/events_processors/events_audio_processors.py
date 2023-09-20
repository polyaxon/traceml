from typing import Optional

from clipped.utils.np import to_np
from clipped.utils.paths import check_or_create_path, copy_file_path

from polyaxon._constants.globals import UNKNOWN
from traceml.events import V1EventAudio
from traceml.logger import logger
from traceml.processors.errors import NUMPY_ERROR_MESSAGE

try:
    import numpy as np
except ImportError:
    np = None


def audio_path(
    from_path: str,
    asset_path: str,
    content_type=None,
    asset_rel_path: Optional[str] = None,
) -> V1EventAudio:
    copy_file_path(from_path, asset_path)
    return V1EventAudio(path=asset_rel_path or asset_path, content_type=content_type)


def audio(
    asset_path: str, tensor, sample_rate=44100, asset_rel_path: Optional[str] = None
):
    if not np:
        logger.warning(NUMPY_ERROR_MESSAGE)
        return UNKNOWN

    tensor = to_np(tensor)
    tensor = tensor.squeeze()
    if abs(tensor).max() > 1:
        print("warning: audio amplitude out of range, auto clipped.")
        tensor = tensor.clip(-1, 1)
    assert tensor.ndim == 1, "input tensor should be 1 dimensional."

    tensor_list = [int(32767.0 * x) for x in tensor]

    import struct
    import wave

    check_or_create_path(asset_path, is_dir=False)

    wave_write = wave.open(asset_path, "wb")
    wave_write.setnchannels(1)
    wave_write.setsampwidth(2)
    wave_write.setframerate(sample_rate)
    tensor_enc = b""
    for v in tensor_list:
        tensor_enc += struct.pack("<h", v)

    wave_write.writeframes(tensor_enc)
    wave_write.close()
    return V1EventAudio(
        sample_rate=sample_rate,
        num_channels=1,
        length_frames=len(tensor_list),
        path=asset_rel_path or asset_path,
        content_type="audio/wav",
    )
