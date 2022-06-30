from typing import Tuple, Dict, Any

import cv2
import numpy as np


def draw_bbox(
    image: np.ndarray,
    detection: Dict[str, Any],
    color: Tuple[int, int, int] = (255, 255, 255),
    show_label: bool = True,
    show_confidence: bool = True,
) -> np.ndarray:
    """
    Will overlay a bounding box on provided image.
    :param image: Image to overlay on
    :param detection: Detection dict to overlay. must have 'box', 'label' and 'confidence' entries
    :param color: Color to be used
    :param show_label: If True, writes box label in image
    :param show_confidence: If True, write box confidence in image
    :return: Image
    """
    thickness = int(round(max(image.shape) / 350))
    h, w = image.shape[:2]

    scale = (
        np.array([w, h, w, h])
        if np.all(np.array(detection["box"]) <= 1.0)
        else np.array([1, 1, 1, 1])
    )
    xmin, ymin, xmax, ymax = (np.array(detection["box"]) * scale).astype(int)

    # r, g, b = color
    # color = (int(r), int(g), int(b))
    image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness=thickness)

    out_txt = ""
    if show_label:
        out_txt = detection["label"]
    if show_confidence:
        out_txt += "({})".format(round(detection["confidence"], 2))

    if out_txt != "":
        image = cv2.putText(
            image,
            out_txt,
            (xmin, ymin - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            thickness=thickness,
        )
    return image
