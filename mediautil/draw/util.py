from typing import List
from random import randint
from collections import defaultdict

import numpy as np


def get_random_color_map(
    a_min: int = 0,
    a_max: int = 255,
    n: int = 3,
) -> defaultdict:
    """
    Returns a defaultdict that defaults to a tuple  of n random values between a_min and a_max
    :param a_min: Min array value
    :param a_max: Max array value
    :param n: Number of random values
    :return: Defaultdict with values
    """
    return defaultdict(lambda: tuple(int(randint(a_min, a_max)) for _ in range(n)))


def stack_images(images: List[np.ndarray], border_width: int = 6):
    def hstack(images: List[np.ndarray]):
        assert len(images) % 2 == 0
        h_border = np.zeros((images[0].shape[0], border_width, 3), dtype=np.uint8)
        w_border = np.zeros(
            (border_width, images[0].shape[1] * 2 + border_width, 3), dtype=np.uint8
        )
        stack = []
        for i in range(0, len(images), 2):
            stack.append(np.hstack((images[i], h_border, images[i + 1])))
            stack.append(w_border)

        return stack

    if len(images) == 0:
        return None
    elif len(images) == 1:
        return images[0]
    elif len(images) == 2:
        return hstack(images)[0]
    elif len(images) % 2 == 0:
        return np.vstack(hstack(images)[:-1])
    else:
        pad_w = (border_width // 2) + images[0].shape[1] // 2
        pad = np.zeros_like(images[0])[:, :pad_w, :]

        hstacks = hstack(images[:-1])
        hstacks.append(np.hstack((pad, images[-1], pad)))
        return np.vstack(hstacks)


if __name__ == "__main__":
    import cv2

    image = np.ones((250, 450, 3), dtype=np.uint8) * 200

    for i in range(1, 8):
        stack = stack_images([image for _ in range(i)])
        cv2.imshow("", stack)
        cv2.waitKey()
    cv2.destroyAllWindows()
