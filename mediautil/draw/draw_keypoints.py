from typing import Union, Iterable, Tuple

import cv2
import numpy as np


def draw_keypoints(
    image: np.ndarray,
    keypoints: Iterable,
    limbs: list = None,
    color: Union[Tuple[int, int, int], list] = (255, 255, 255),
    limb_color: Union[Tuple[int, int, int], list] = (255, 255, 255),
    limb_only: bool = True,
) -> np.ndarray:
    """
    Will overlay a set of keypoints on image. If limbs are provided, lines will be drawn to mark them
    Function sees negative values as to be skipped
    :param image: Image to overlay on
    :param keypoints: List of keypoints to use. Must be a list with tuples, arrays or np.ndarrays
    :param limbs: List of limbs to use. Must be a list of index pairs for all limbs. Indexes must match elements in
    keypoints.
    :param color: Color to be used for points. If list, use as lookup for point color
    :param limb_color: Color to be used for limbs. Can be list of equal length to limbs
    :param limb_only: If True, draw only limb connections and no keypoints
    :return: Image
    """
    pose = np.array(keypoints)
    if type(color) is not list:
        color = [color] * pose.shape[0]
    if limbs is not None:
        if type(limb_color) == tuple and len(limb_color) == 3:
            limb_color = [limb_color] * len(limbs)
        else:
            assert len(limb_color) == len(
                limbs
            ), "limbs color must either be rgb values or 1 rgb tuple for each limb"

    thickness = round(max(image.shape) / 350)
    h, w = image.shape[:2]
    scale = (
        np.array([w, h]) if np.all(pose[~np.isnan(pose)] <= 1.0) else np.array([1, 1])
    )
    pose = (pose * scale).astype(int)

    if limbs is not None:
        for (src, dst), limb_c in zip(limbs, limb_color):
            x1, y1, x2, y2 = pose[src][0], pose[src][1], pose[dst][0], pose[dst][1]
            # Dont draw line if one of the points is negative or NaN
            if np.any(pose[(src, dst), :] < 0) or np.any(np.isnan(pose[(src, dst), :])):
                continue
            image = cv2.line(image, (x1, y1), (x2, y2), limb_c, thickness=thickness)

    if not limb_only:
        for i, (x, y) in enumerate(pose):
            if x < 0 or y < 0 or np.any(np.isnan(pose[i, :])):
                continue
            image = cv2.circle(image, (x, y), thickness, color[i], thickness=-1)

    return image
