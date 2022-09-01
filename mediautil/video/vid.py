from multiprocessing.sharedctypes import Value
from typing import Tuple, Union, List, Callable

import cv2
import numpy as np
from pathlib import Path

from mediautil.image.img import Img


class Vid:
    def __init__(
        self,
        path: Union[str, Path, int],
        color_mode: str = "bgr",
        size: Tuple[int, int] = None,
        as_img_objects: bool = True,
        with_timestamps: bool = False,
        rotation: int = 0
        # TODO: implement initialization from directory of frames
    ):
        """
        Wrapper from video-file reading.
        Object is iterable and calls to next(img) returns sequential frames
        :param path: Path to video file or a int giving mounted usb camera index
        :param color_mode: Desired color mode of returned images
        :param size: Desired size of returned images
        :param as_img_objects: Returned image is Img object if True, numpy array if False
        :param with_timestamps: When iterating, setting this flag will also include the timestamp of the frame in seconds.
        """
        self._usb_cam = False
        if type(path) == int:
            self._usb_cam = True
        elif path.startswith("rtsp"):
            pass
        elif not Path(path).exists():
            raise FileNotFoundError("{} not found".format(path))
        if color_mode not in {"rgb", "bgr"}:
            raise ValueError(
                "Unexpected color mode. {} not in (rgb, bgr)".format(color_mode)
            )

        self.rotation_degrees = rotation

        self.path = str(path) if not self._usb_cam else path
        self._color_mode = color_mode
        self.as_img_objects = as_img_objects
        self.with_timestamps = with_timestamps

        self._cap = cv2.VideoCapture(self.path)
        self._num_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._fps = float(self._cap.get(cv2.CAP_PROP_FPS))

        self._prev_time = 0

        self.output_size = size if size is not None else (self._h, self._w)

        self._fps_buffer = []

    def __next__(self) -> Union[Img, np.ndarray, Tuple[float, Union[Img, np.ndarray]]]:
        if self._cap.isOpened():
            ret, frame = self._cap.read()

            t = self._cap.get(cv2.CAP_PROP_POS_MSEC)
            if np.isnan(t):
                t = 0
            # Some codecs returns 0 for the last few frames.
            # The code below estimates the time for the last frames if they are 0
            frame_nr = self._cap.get(cv2.CAP_PROP_POS_FRAMES)
            if t == 0 and frame_nr != 1.0:
                t = self._prev_time + np.mean(np.diff(self._fps_buffer))
            else:
                self._fps_buffer.append(t)

            self._prev_time = t

            if ret:
                image = Img(
                    frame, numpy_color_mode="bgr", rotation=self.rotation_degrees
                )
                if image.hw != self.output_size:
                    image.resize(*self.output_size)
                if self._color_mode == "rgb":
                    _ = image.rgb
                elif self._color_mode == "bgr":
                    _ = image.bgr
                else:
                    raise NotImplementedError("Only supports rgb and bgr")
                if self.as_img_objects:
                    if self.with_timestamps:
                        return (t / 1000, image)
                    else:
                        return image
                if self.with_timestamps:
                    return (t / 1000, image.get())
                else:
                    return image.get()

        raise StopIteration

    def __iter__(self) -> "Vid":
        if self._usb_cam:
            return self
        self._cap = cv2.VideoCapture(self.path)
        return self

    def __len__(self) -> int:
        if self._usb_cam:
            raise ValueError("Cannot call len() on usb camera")
        return self._num_frames

    def __str__(self) -> str:
        return f"Vid: {len(self)} {self._w}x{self._h} {self._fps}fps"

    @property
    def wh(self):
        h, w = self.output_size
        return w, h

    @property
    def hw(self):
        return self.output_size

    @property
    def color_mode(self):
        return self._color_mode

    @property
    def fps(self):
        return self._fps

    def show(self, fs=False, fps=None):
        """
        Show video using cv2.
        Function uses preset colormode and image size.

        use 'q' to stop video

        :param fs: Show in fullscreen
        :param fps: overwrite video FPS for visualization. 0 if step through frame by frame
        """
        fps = self.fps if fps is None else fps
        wait_time = int(1000 / fps) if fps != 0 else 0
        name = ""
        if fs:
            cv2.namedWindow(name, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        for image in self:
            cv2.imshow(name, image.get())
            if ord("q") == cv2.waitKey(wait_time):
                break
        cv2.destroyAllWindows()

    def save_imgs(self, path, num_frames=None):
        if self._usb_cam and num_frames:
            raise ValueError(
                "Cannot store all images from camera. Provide number of frames"
            )
        Path.mkdir(Path(path), parents=True, exist_ok=True)

        i = 0
        for frame in self:
            frame.save(path + "{0:05d}.jpg".format(i))
            i += 1
            if num_frames is not None and i >= num_frames:
                break

    def get_frame_selection(
        self, frame_indices: List[int] = None, frame_filter: Callable = None
    ) -> List[Union[Img, np.ndarray]]:
        """
        Returns a list of frames from the video
        :param frame_indices: If not None, returns frames with these indicecs
        :param frame_filter: Functions which takes a Img or np.ndarray and returns True if the frame is to be returned
        :return: List of np.ndarrays or Img-objects
        """
        if self._usb_cam:
            raise ValueError("Cannont get frame indices from live stream")
        frames = []
        if frame_indices is None:
            frame_indices = []
        if frame_filter is None:
            frame_filter = lambda x: False

        frame_indices = sorted(frame_indices)
        for i, frame in enumerate(self):
            if len(frame_indices) == 0:
                break
            if frame_indices[0] == i:
                frames.append(frame)
                frame_indices.pop(0)
            elif frame_filter(frame):
                frames.append(frame)

        return frames


if __name__ == "__main__":
    v = Vid("/home/jonas/Videos/002_serve.MOV")

    v.save_imgs("/home/jonas/Videos/002/")
