import cv2
import numpy as np
from pathlib import Path

from typing import Union, Tuple

"""
Class to help simplify creating a video. 
Create an object of this class in a with-block (or use open()/close() functions) and 
give frames to the add_frame() method. 
"""

# TODO: needs work
class VideoCreator:
    def __init__(
        self,
        file_name: Union[str, Path],
        image_size: Union[None, Tuple[int, int]] = None,
        fps: float = 30.0,
        input_frame_color_mode: str = "bgr",
    ):
        self.file_name = str(file_name)
        self.forcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.fps = fps
        self.image_size = image_size
        self.color_mode = input_frame_color_mode.lower()
        assert self.color_mode in {"rgb", "bgr", "gray"}

        self.vid_writer = None
        if image_size is not None:
            self.vid_writer = cv2.VideoWriter(
                self.file_name, self.forcc, self.fps, self.image_size
            )

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def open(self):
        pass

    def close(self):
        self.vid_writer.release()

    def add_frame(self, frame):
        if self.vid_writer is None:
            self.image_size = (frame.shape[1], frame.shape[0])
            self.vid_writer = cv2.VideoWriter(
                self.file_name, self.forcc, self.fps, self.image_size
            )

        if self.color_mode == "rgb":
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        elif self.color_mode == "gray":
            r = np.zeros(frame.shape + (3,), dtype=frame.dtype)
            r[:, :, 0], r[:, :, 1], r[:, :, 2] = frame, frame, frame
            frame = r
        frame = cv2.resize(frame, self.image_size)
        self.vid_writer.write(frame)
