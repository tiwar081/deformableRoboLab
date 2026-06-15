"""Streaming H.264 video writer, following RoboLab's video pipeline.

Same encoder settings as RoboLab's ``robolab/core/utils/video_utils.py``:
imageio-ffmpeg piping RGB frames into libx264/yuv420p with ``-preset
ultrafast`` (low CPU per stream) and ``+faststart`` (browser-playable MP4).
Memory stays bounded by the encoder lookahead regardless of video length.
"""

from __future__ import annotations

import atexit

import numpy as np


class VideoWriter:
    def __init__(self, video_path: str, fps: float):
        self.video_path = str(video_path)
        self.fps = fps
        self._writer = None
        atexit.register(self.release)

    def write(self, frame: np.ndarray) -> None:
        """Append one RGB (H, W, 3) uint8 frame."""
        if frame is None:
            return
        if self._writer is None:
            import imageio.v2 as imageio

            # macro_block_size=2 pads odd dimensions for yuv420p (even-only).
            self._writer = imageio.get_writer(
                self.video_path,
                fps=self.fps,
                codec="libx264",
                pixelformat="yuv420p",
                macro_block_size=2,
                ffmpeg_params=[
                    "-preset", "ultrafast",
                    "-threads", "1",
                    "-crf", "23",
                    "-movflags", "+faststart",
                ],
            )
        self._writer.append_data(frame)

    def release(self) -> None:
        if self._writer is not None:
            try:
                self._writer.close()
            finally:
                self._writer = None
