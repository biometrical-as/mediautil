from typing import Union, Iterable, Tuple

import cv2
import numpy as np


def draw_polygon(
    image: np.ndarray,
    polygon: Iterable,
    color: Union[Tuple[int, int, int], list] = (255, 255, 255),
) -> np.ndarray:
    """
    Will overlay a set of polygon points on image as a filled polygon.
    :param image: Image to overlay on
    :param keypoints: List of polygon points to use. Must be a list with tuples, arrays or np.ndarrays
    :param color: Color of filled polygon
    """
    polygon = np.array(polygon).astype(int)

    h, w = image.shape[:2]
    scale = (
        np.array([w, h])
        if np.all(polygon[~np.isnan(polygon)] <= 1.0)
        else np.array([1, 1])
    )
    polygon = (polygon * scale).astype(np.int32)

    cv2.fillPoly(image, [polygon], color=color)

    return image
