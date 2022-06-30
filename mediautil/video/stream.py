import warnings
from typing import Tuple, Union

import cv2
import numpy as np

from mediautil.image.img import Img


class Stream:
    def __init__(
        self,
        stream: Union[str, int] = None,
        color_mode: str = "bgr",
        size: Tuple[int, int] = None,
        as_img_objects: bool = True,
        buffer: bool = False,
    ):
        warnings.warn(
            "Deprecated warning. Will be removed when stream functionality is added to Vid"
        )
        """
        Wrapper from video-stream reading.
        Object is iterable and calls to a cv2.VideoCaptures read() function which is yielded.
        Object expects to botleneck of process.
        :param stream: stream to subscribe to. Can be int index of camera device. Defaults to 0
        :param color_mode: Desired color mode of returned images
        :param size: Desired size of returned images
        :param as_img_objects: Returned image is Img object if True, numpy array if False
        :param buffer: Buffers yielded frames during iteration. NB: No buffer flushing is implemented!
        """
        self.stream = stream
        self._cap = cv2.VideoCapture(stream)
        if color_mode not in {"rgb", "bgr"}:
            raise ValueError(
                "Unexpected color mode. {} not in (rgb, bgr)".format(color_mode)
            )

        self._color_mode = color_mode
        self.as_img_objects = as_img_objects

        self.buffer = [] if buffer else None

        self._w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._fps = float(self._cap.get(cv2.CAP_PROP_FPS))

        self.output_size = size if size is not None else (self._h, self._w)

    def __next__(self) -> Union[Img, np.ndarray]:
        if self._cap.isOpened():
            ret, frame = self._cap.read()
            if ret:
                image = Img(frame, numpy_color_mode="bgr")
                if image.hw != self.output_size:
                    image.resize(*self.output_size)
                if self._color_mode == "rgb":
                    _ = image.rgb
                elif self._color_mode == "bgr":
                    _ = image.bgr
                else:
                    raise NotImplementedError("Only supports rgb and bgr")
                if self.as_img_objects:
                    if type(self.buffer) is not None:
                        self.buffer.append(image)
                    return image
                if type(self.buffer) is not None:
                    self.buffer.append(image.get())
                return image.get()

        raise StopIteration

    def __iter__(self) -> "Vid":
        self._cap = cv2.VideoCapture(self.stream)
        return self

    def __bool__(self) -> bool:
        return self._cap.isOpened()

    @property
    def fps(self):
        return self._fps

    @property
    def wh(self):
        return self._w, self._h

    @property
    def hw(self):
        return self._h, self._w

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

        while self:
            image = self.__next__()
            cv2.imshow(name, image.get())
            if ord("q") == cv2.waitKey(wait_time):
                break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    s = Stream("0")
    s.show()
