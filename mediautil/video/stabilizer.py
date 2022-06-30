import logging
from time import time

import cv2
import numpy as np
from pathlib import Path

from mediautil.video.vid import Vid
from mediautil.image.img import Img


class Stabilizer:
    def __init__(
        self,
        vid: Vid,
        downscaling_factor: float = 2.0,
    ):
        """
        Video stabilizer.
        Takes a Vid object and calculates translation/rotation transform for each frame.

        After stabilization, the frames returned will be post processed with the following strategy
            1. The transform is shifted to be centered around median camera "orientation"
            2. The image will be cropped (up to 20%) to minimize the amount of black borders
            3. The image will be resized back to original image size
        """
        self.vid = vid
        self.path = vid.path
        vid.as_img_objects = True
        self.as_img_objects = True

        original_output = vid.output_size
        vid.output_size = (
            int(original_output[0] / downscaling_factor),
            int(original_output[1] // downscaling_factor),
        )
        self.transform = self._calculate_transform()
        self.transform[:, :2] = self.transform[:, :2] * downscaling_factor
        vid.output_size = original_output

        self.transform = self.transform - np.median(self.transform, axis=0)
        self.post_process = self._get_post_process_func()

        # self.xmin, self.xmax = self.transform[:, 0].min(), self.transform[:, 0].max()
        # self.ymin, self.ymax = self.transform[:, 1].min(), self.transform[:, 1].max()
        # self.rmin, self.rmax = self.transform[:, 2].min(), self.transform[:, 2].max()

        # self.warp_shape = (int(self.vid._w + (self.xmax - self.xmin)), int(self.vid._h + (self.ymax - self.ymin)))

        # x = self.transform[:, 0]
        # y = self.transform[:, 1]
        # r = np.degrees(self.transform[:, 2])
        # from matplotlib import pyplot as plt
        # plt.plot(x, label='x')
        # plt.plot(y, label='y')
        # plt.plot(r, label='rotation')
        # plt.hlines(self.median_transform[0], xmin=0, xmax=len(self.vid), label='median x')
        # plt.hlines(self.median_transform[1], xmin=0, xmax=len(self.vid), label='median y')
        # plt.hlines(self.median_transform[2], xmin=0, xmax=len(self.vid), label='median y')
        # plt.legend()
        # plt.show()

        self._frame_iterator = iter(self.vid)
        self._transform_iterator = iter(self.transform)

    def _get_post_process_func(self):
        def f(image):
            s = image.shape
            T = cv2.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, 1.1)
            return cv2.warpAffine(image, T, (s[1], s[0]))

        return f

    def _calculate_transform(self):
        transform = np.zeros((len(self.vid) - 1, 3))

        iterator = iter(self.vid)
        previous = next(iterator)

        t = time()
        for i in range(transform.shape[0] - 1):
            # Detect feature points in previous frame
            prev_pts = cv2.goodFeaturesToTrack(
                previous.gray[..., 0],
                maxCorners=200,
                qualityLevel=0.01,
                minDistance=30,
                blockSize=3,
            )
            current = next(iterator)

            # Calculate optical flow (i.e. track feature points)
            curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
                previous.gray[..., 0], current.gray[..., 0], prev_pts, None
            )

            # Sanity check
            assert prev_pts.shape == curr_pts.shape

            # Filter only valid points
            idx = np.where(status == 1)[0]
            prev_pts = prev_pts[idx]
            curr_pts = curr_pts[idx]

            # Find transformation matrix
            m = cv2.estimateAffinePartial2D(prev_pts, curr_pts)[0]

            # Extract traslation
            dx = m[0, 2]
            dy = m[1, 2]

            # Extract rotation angle
            dr = np.arctan2(m[1, 0], m[0, 0])

            transform[i] = [dx, dy, dr]
            previous = current

        trajectory = np.cumsum(-transform, axis=0)
        smoothed_trajectory = np.array(trajectory)
        # Moving average of transform as smoothing strategy
        r = 2  # Smoothing radius
        window_size = 2 * r + 1
        for i in range(3):
            f = np.ones(window_size) / window_size
            padded = np.lib.pad(trajectory[:, i], (r, r), "edge")
            smooth_padded = np.convolve(padded, f, mode="same")
            smoothed_trajectory[:, i] = smooth_padded[r:-r]

        # from matplotlib import pyplot as plt
        # plt.plot(smoothed_trajectory[:, 0], label='x')
        # plt.plot(smoothed_trajectory[:, 1], label='y')

        # plt.plot(trajectory[:, 0], linestyle='dashed', label='original x')
        # plt.plot(trajectory[:, 1], linestyle='dashed', label='original y')
        # plt.show()

        logging.debug(f"calculated transform\t{time()-t}s")
        return smoothed_trajectory

    def __iter__(self) -> "Stabilizer":
        self._frame_iterator = iter(self.vid)
        self._transform_iterator = iter(self.transform)
        return self

    def __next__(self):
        image = next(self._frame_iterator).get()
        dx, dy, dr = next(self._transform_iterator)

        m = np.array([[np.cos(dr), -np.sin(dr), dx], [np.sin(dr), np.cos(dr), dy]])

        image_stab = cv2.warpAffine(image, m, self.vid.wh)

        output = self.post_process(image_stab)
        if self.as_img_objects:
            return Img(output, numpy_color_mode=self.vid.color_mode)
        return output

    def __len__(self):
        return len(self.vid)

    def __str__(self):
        return f"Stabilized {self.vid}"

    def show(self, fs=False, fps=None):
        """
        Show video using cv2.
        Function uses preset colormode and image size.

        use 'q' to stop video

        :param fs: Show in fullscreen
        :param fps: overwrite video FPS for visualization. 0 if step through frame by frame
        """
        fps = self.vid.fps if fps is None else fps
        wait_time = int(1000 / fps) if fps != 0 else 0
        name = ""
        if fs:
            cv2.namedWindow(name, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        for image in self:
            cv2.imshow(name, image)
            if ord("q") == cv2.waitKey(wait_time):
                break
        cv2.destroyAllWindows()

    @property
    def fps(self):
        return self.vid._fps

    @property
    def wh(self):
        return self.vid.wh

    @property
    def color_mode(self):
        return self.vid.color_mode

    @property
    def hw(self):
        return self.vid.hw


if __name__ == "__main__":
    # Stabilizer(Vid('3.webm'), downscaling_factor=4).show()

    for p in Path("/home/martin/repos/BIO/demo/video").glob("*"):
        v = Stabilizer(Vid(p), downscaling_factor=3).show()

    # from matplotlib import pyplot as plt

    # logging.getLogger().setLevel(logging.DEBUG)
    # dsfs = [1., 1.5, 2., 3., 4.]
    # timespv = []
    # for p in Path('/home/martin/repos/BIO/demo/video').glob('*'):
    #     times = []
    #     for dsf in dsfs:
    #         times.append(time())
    #         v = Stabilizer(Vid(p), downscaling_factor=dsf)
    #         times[-1] = time() - times[-1]

    #     timespv.append(times)

    # plt.boxplot(np.array(timespv), labels=[f'Factor: {d}' for d in dsfs])
    # plt.ylabel('time')
    # plt.show()
