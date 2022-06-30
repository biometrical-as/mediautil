import numpy as np

from mediautil import Img


def test_numpy_to_Img():
    data = np.random.uniform(0, 255, (1920, 1080, 3)).astype(dtype=np.uint8)
    img = Img(data)

    b = img.bgr[..., 0]
    d_raw = data[..., -1]
    assert np.all(
        b == d_raw
    ), "blue channel in image should be same as blue channel in raw data"
