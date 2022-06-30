import re
import warnings
from typing import Union, Tuple, Iterable, List

import cv2
import magic
import numpy as np
from pathlib import Path

from mediautil.draw import draw_keypoints, draw_bbox, draw_polygon


class Img:
    def __init__(
        self,
        data: Union[str, Path, np.ndarray],
        lazy: bool = False,
        numpy_color_mode: str = "rgb",
        rotation: int = 0,
    ):
        """
        QOL class for handling images.
        :param data: Path to image file or numpy-matrix
        :param lazy: Skip loading image file at initialization. Only reads meta data
        :param numpy_color_mode: If data is a numpy-matrix, set this as the color mode rgb|bgr
        """
        self.rotation = (
            (lambda x: np.rot90(x, k=rotation // 90))
            if rotation >= 90
            else (lambda x: x)
        )

        if type(data) == np.ndarray:
            self.path = "NOT DEFINED"
            self.image = self.rotation(np.array(data))
            self._h, self._w = self.image.shape[:2]
            self._color_mode = numpy_color_mode
            self.lazy = lazy
        else:
            if not Path(data).exists():
                raise FileNotFoundError("{} not found".format(data))

            self.path = str(data)
            self.lazy = lazy

            if not lazy:
                self._load()
            else:
                self.image = None
                self._color_mode = None
                t = magic.from_file(self.path).split("baseline")[-1]
                self._h, self._w = re.search("(\\d+) ?x ?(\\d+)", t).groups()
                self._h, self._w = int(self.w), int(self.h)

    def _load(self):
        """
        Loads image into memory using cv2.
        """
        self.image = self.rotation(cv2.imread(self.path))
        self._h, self._w = self.image.shape[:2]
        self._color_mode = "bgr"

    def _cond_warn_and_load(self):
        if self.image is None:
            warnings.warn("Forced image loading. Init Img-object with lazy=False")
            self._load()

    @property
    def shape(self) -> Tuple[int, int, int]:
        """
        Gets image numpy-shape. If lazy Img object, the image is loaded first
        :return: Image shape
        """
        self._cond_warn_and_load()
        return self.image.shape

    @property
    def h(self) -> int:
        """
        Get height
        :return: height
        """
        return self._h

    @property
    def w(self):
        """
        get width
        :return: width
        """
        return self._w

    @property
    def hw(self) -> Tuple[int, int]:
        """
        Gets image height and width
        :return: (image height, image width)
        """
        return self._h, self._w

    @property
    def wh(self) -> Tuple[int, int]:
        """
        Gets image width and height
        :return: (image width, image height)
        """
        return self._w, self._h

    @property
    def color(self) -> str:
        """
        Gets image color mode as a string
        :return: color mode
        """
        return self._color_mode

    @property
    def rgb(self) -> np.ndarray:
        self._cond_warn_and_load()
        if self._color_mode == "rgb":
            pass
        elif self._color_mode == "bgr":
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self._color_mode = "rgb"
        else:
            raise NotImplementedError(
                "Cannot change colormode to rgb from anything other than rgb|bgr ({})".format(
                    self._color_mode
                )
            )
        return self.image

    @property
    def bgr(self) -> np.ndarray:
        self._cond_warn_and_load()
        if self._color_mode == "bgr":
            pass
        elif self._color_mode == "rgb":
            self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
            self._color_mode = "bgr"
        else:
            raise NotImplementedError(
                "Cannot change colormode to bgr from anything other than rgb|bgr ({})".format(
                    self._color_mode
                )
            )
        return self.image

    def gray(self, channels: int = 0) -> np.ndarray:
        self._cond_warn_and_load()
        if self._color_mode == "bgr":
            c2g = cv2.COLOR_BGR2GRAY
        elif self._color_mode == "rgb":
            c2g = cv2.COLOR_RGB2GRAY
        else:
            raise NotImplementedError(
                "Cannot change colormode to bgr from anything other than rgb|bgr ({})".format(
                    self._color_mode
                )
            )
        gray = cv2.cvtColor(self.image, c2g)
        if channels <= 0:
            return gray
        return np.repeat(gray[..., np.newaxis], channels, axis=2)

    def get(self):
        return self.image

    def resize(self, *args, **kwargs):
        """
        Resizes image to given shape inplace.
        Shapes can be provided in the following formats:
            - As named arguments (h and w, H and W, height and width)
            - As 2 unnamed arguments (height, width)
            - As 1 unnamed argument (scale)

        Integers will be treated as desired pixel size, while floats will be used to scale one or both dimensions

        :param args: 0, 1 or 2 unnamed arguments
        :param kwargs:  0, 1 or 2 named arguments
        """
        self._cond_warn_and_load()

        if len(args) > 0:
            assert len(kwargs) == 0, "Only named or unnamed arguments, not both"
        if len(kwargs) > 0:
            assert len(args) == 0, "Only named or unnamed arguments, not both"
        assert len(kwargs) <= 2 or len(args) <= 2, "Ambiguous new image shape"

        h, w = None, None

        for kw in ("h", "H", "height"):
            h = kwargs[kw] if kw in kwargs else h
        for kw in ("w", "W", "width"):
            w = kwargs[kw] if kw in kwargs else w

        if h is None or w is None:
            if len(args) == 2:
                h, w = args
            elif len(args) == 1:
                h, w = args[0], args[0]

        pixel_h = int(self._h * h) if type(h) == float else h
        pixel_w = int(self._w * w) if type(w) == float else w

        pixel_h = self._h if pixel_h == 1 or pixel_h is None else pixel_h
        pixel_w = self._w if pixel_w == 1 or pixel_w is None else pixel_w

        self.image = cv2.resize(self.image, (pixel_w, pixel_h))
        self._h, self._w = self.image.shape[:2]

    def add_watermark(self, text: str, image: np.ndarray = None):
        image = np.array(self.image) if image is None else image
        return cv2.putText(
            image, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), thickness=2
        )

    def draw_on_image(
        self,
        bboxes: Union[Iterable, None] = None,
        keypoints: Union[Iterable, None] = None,
        polygons: Union[Iterable, None] = None,
        image: Union[np.ndarray, None] = None,
        as_numpy: bool = True,
    ) -> np.ndarray:
        """
        Draws bboxes and keypoints on image
        :param bboxes: Iterable of bounding box dicts.
            Each dict must contain 'box', 'label', 'confidence' keywords
        :param keypoints: Iterable of keypoints as np.ndarray.
            Can optionally be a iterable of tuples where element 0 is keypoint
            and element 1 is dictionary of named arguments to draw_keypoints
        :param image: If not None, draw on this input, else create new np.array of self
        :return : returns nparray of image in current state with keypoints/bboxes drawn
        """

        image = np.array(self.image) if image is None else image

        if bboxes is not None:
            for b in bboxes:
                image = draw_bbox(image, b, show_label=True, show_confidence=True)
        if keypoints is not None:
            if type(keypoints) == tuple:
                kps, kwargs = keypoints
                for kp in kps:
                    image = draw_keypoints(
                        image=image, keypoints=[kp], limb_only=False, **kwargs
                    )
            else:
                for kp in keypoints:
                    if type(kp) == tuple:

                        kp, kwargs = kp
                        image = draw_keypoints(image=image, keypoints=[kp], **kwargs)
                    else:
                        image = draw_keypoints(
                            image=image, keypoints=[kp], limb_only=False
                        )

        if polygons is not None:
            for polygon in polygons:
                image = draw_polygon(image=image, polygon=polygon, color=None)

        if as_numpy:
            return image
        else:
            return Img(data=image, numpy_color_mode=self._color_mode)

    def show(
        self,
        fs: bool = False,
        bboxes: Union[Iterable, None] = None,
        keypoints: Union[Iterable, None] = None,
    ):
        """
        Shows image using cv2
        :param fs: Show image in fullscreen
        :param bboxes: Iterable of bounding box dicts.
            Each dict must contain 'box', 'label', 'confidence' keywords
        :param keypoints: Iterable of keypoints as np.ndarray.
            Can optionally be a iterable of tuples where element 0 is keypoint
            and element 1 is dictionary of named arguments to draw_keypoints
        """

        orig_color = self.color
        name = ""
        if fs:
            cv2.namedWindow(name, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        image = self.draw_on_image(bboxes, keypoints)

        cv2.imshow(name, image)
        cv2.waitKey()
        cv2.destroyAllWindows()

        if orig_color == "rgb":
            _ = self.rgb
        elif orig_color == "bgr":
            _ = self.bgr

    def save(self, path: Union[str, Path]):
        cv2.imwrite(str(path), self.bgr)

    @staticmethod
    def glob_images(
        root: Union[str, Path],
        recursively: bool = True,
        filetypes: Union[str, List[str]] = "jpg",
        lazy: bool = True,
    ) -> List["Img"]:
        """
        Globs all images in a directory as Img objects with lazy loading
        :param root: root to glob
        :param recursively: if true, golbs recursively
        :param filetypes: single or list of filetypes to glob. default is jpg
        :param lazy: Create "lazy" objects if false
        :return: list of Img objects
        """
        root = Path(root)
        filetypes = [filetypes] if type(filetypes) == str else filetypes

        get_paths = root.rglob if recursively else root.glob

        imgs = []
        for ft in filetypes:
            for p in get_paths("*.{}".format(ft)):
                imgs.append(Img(p, lazy=lazy))
        return imgs
