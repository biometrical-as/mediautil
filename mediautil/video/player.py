import traceback
from typing import List, Union
from threading import Thread, Event

import cv2
import numpy as np
from pathlib import Path

from mediautil import Img
from mediautil.video.video_creator import VideoCreator


"""
Player object usefull for inspecting frames in video.
Initially intended to run in own thread, but opencv does not like imshow to be 
outside main thread. Add threading back if this is fixed at some point. 

Use by creating a player (add output video path if needed. It is saved on Q)
and adding frames with player.add_frame. When all frames are added, call 
player.wait() to wait for player to be closed. 

Provides simple UI for playback: 
    q: quit player
    <space>: pause playback
    <left arrow>: rewind
    <right arrow>: move forward (1 frame if paused, 5 if playing)
    +: increase playback FPS
    -: decrease playback FPS
    e: jump to End of buffer
    b: jump to Begining of buffer
    m: jump to Middle of buffer

"""


class Player:
    def __init__(
        self,
        fps: Union[float, int] = 30.0,
        name: str = "Player",
        output_path: Union[str, Path] = None,
        force_writer: bool = False,
    ):
        self._fps = fps
        self._name = name
        self._force_writer = force_writer

        self._buffer = []
        self._image_index = 0
        self._pause = False

        self._running_event = Event()
        self._running_event.set()

        self._init_writer(output_path)

        # self._thread = Thread(target=self._run)
        # self._thread.start()

    # def close(self):
    #     self._running_event.clear()
    #     self._thread.join()

    # def wait(self):
    #     self._thread.join()

    def _init_writer(self, output_path: Union[str, Path]):
        self._writer = None
        if output_path is not None:
            output_path = Path(output_path)
            if output_path.is_dir():
                raise ValueError(f"Output path must be file: {output_path}")

            if output_path.exists() and not self._force_writer:
                raise ValueError(f"File exists: {output_path}")

            self._writer = VideoCreator(output_path, fps=self._fps)
            self._writer.open()

    def add_frame(self, frame: Union[np.ndarray, Img]):
        if type(frame) == Img:
            frame = frame.bgr
        self._buffer.append(frame)
        if self._writer is not None:
            self._writer.add_frame(frame)

        if self.is_running():
            self._step()
        self._process_key

    def is_running(self) -> bool:
        return self._running_event.is_set()

    def _step(self):
        if self._image_index >= len(self._buffer):
            return
        cv2.imshow(self._name, self._buffer[self._image_index])
        if self._pause:
            key = cv2.waitKey()
        else:
            key = cv2.waitKey(1000 // self._fps)
        self._process_key(key)

    def wait(self):
        while self._running_event.is_set():
            self._step()
        self._running_event.clear()
        cv2.destroyWindow(self._name)

    # def _run(self):
    #     try:
    #         while self._running_event.is_set():
    #             if self._image_index >= len(self._buffer):
    #                 continue

    #             cv2.imshow(self._name, self._buffer[self._image_index])
    #             if self._pause:
    #                 key = cv2.waitKey()
    #             else:
    #                 key = cv2.waitKey(1000 // self._fps)
    #             self._process_key(key)

    #         self._running_event.clear()
    #         if self._writer is not None:
    #             self._writer.close()
    #         cv2.destroyWindow(self._name)
    #     except Exception as e:
    #         print("Exception in Player:", e)
    #         traceback.print_exc()

    def _process_key(self, key: int):
        index_change = 0
        if key == ord("q"):
            if self._writer is not None:
                self._writer.close()
            self._running_event.clear()
        elif key == ord(" "):
            self._pause = not self._pause
        elif key == ord("+"):
            self._fps += 1
        elif key == ord("-"):
            self._fps -= 1
            self._fps = max(self._fps, 1)
        elif key == 81:
            # left arrow
            index_change = -1
            self._pause = True
        elif key == 83:
            # right arrow
            index_change = 1 if self._pause else 5
        elif key == ord("b"):
            self._image_index = 0
        elif key == ord("e"):
            self._image_index = len(self._buffer) - 1
        elif key == ord("m"):
            self._image_index = len(self._buffer) // 2
        else:
            index_change = 1
        self._image_index += index_change
        self._image_index = max(0, min(self._image_index, len(self._buffer) - 1))


if __name__ == "__main__":
    from mediautil import Vid

    image = np.zeros((500, 500, 3), dtype=np.uint8)
    fps = 30

    player = Player(fps, name="Mediautil player")
    for i in range(500):
        frame = cv2.putText(
            image.copy(),
            str(i).rjust(3, " "),
            (210, 250),
            cv2.FONT_HERSHEY_PLAIN,
            3,
            (255, 255, 255),
            3,
        )
        player.add_frame(frame)
    print("Vid finished. Waiting for player")
    player.wait()
