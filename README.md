# Mediautil
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) 
[![Pytest](https://github.com/biometrical-as/mediautil/actions/workflows/workflow.yaml/badge.svg)](https://github.com/biometrical-as/mediautil/actions/workflows/workflow.yaml)

Media utility repository



## Img
cv2 wrapper class for images.
Can load images from file (with or without lazyloading) or from a numpy-array. 
Class contains rgb, bgr properties to retrieve image, aswell as h, w, hw, wh and shape to get image size, add_watermark method, draw_on_image method, save functionality and a show method. 

# Vid
cv2 wrapper class for videos. 
Loads video from a path and yields frames as np-arrays or as Img objects. 
Class contains properties for wh, hw frame sizes, fps, color-mode as well as a frame-selector and show-method.

## mediautil.video.util
### yield_video_frames
|NB: This function is deprecated. Use Vid object insted
Takes a path to a video file and will yield frame number and video frames in tuples.
Example yields every frame up to frame nr 100 from the video-file 'video_path.mp4'. Each image is 640 by 480 BGR

```python
import cv2
from mediautil.video.util import yield_video_frames
for frame_nr, frame in yield_video_frames(
    'video_path.mp4',
    color_mode='bgr', 
    image_shape=(640, 480), 
    use_pbar=True, 
    frame_window=(0, 100)
): 
    cv2.imshow('video', frame)
    cv2.waitKey(30)
cv2.destroyAllWindows()
```

### get_video_stats
Takes a path to a video file and returns a tuple with`(number of image frames, (image height, image width), video FPS)`

### glob_all_video_paths
Returns a list of all video paths in a folder (recursive or not) with a suffix in 
`['*.mov', '*.MOV', '*.mp4', '*.MP4', '.mkv', '.webm']`


## Class: VideoCreator
A small wrapper for cv2.VideoWriter. Mostly to help remember/simplify how its used. 
Use the object in a with-block or remember to call on .start() and .stop() methods.
example:

```python
from mediautil.video.video_creator import VideoCreator
some_images = [...]
with VideoCreator('output_video.mp4', fps=30, image_shape=(500, 500), color_mode='gray') as vc:
    for image in some_images:
        vc.add_frame(image)
```

## Class: Stabilizer
Takes a Vid object and aplies video stabilization to the video. 
Stabilization prepocessing function is calculated on initalization. 
Images retrieved through __next__ calls has this function applied 

## mediautil.draw.util
### get_random_color_map
Returns a defaultdict that provides unique rgb values for each label. Can be used to generate consistent colors for a set
of labels.

# mediautil.draw
### draw_bounding_box
Function that overlays a detection on an image. Current detection format only supports detections as a dict with the keys
"box", "label" and "confidence".
Example

```python
from mediautil_.image import get_random_color_map, draw_bbox
from matplotlib import pyplot as plt

c = get_random_color_map()
image = ... # Some image
detections = [...] # List of detection-dicts

for det in detections: 
    image = draw_bbox(
        image, 
        det, 
        color=c[det['label']], 
        show_label=True, 
        show_confidence=True
    )
plt.imshow(image)
plt.show()
```

### draw_keypoints
Function that overlays keypoints on an image. One of the optional parameter is "limbs" where the connections between points
can be provided. Both keypoints and limbs bust be iterables that yields two values. For keypoints these values are x and y coordinates
for a landmark, and for limbs these points are two indices for which points in keypoints that are to be connected. 
Example

```python
from mediautil_.image import get_random_color_map, draw_keypoints
from matplotlib import pyplot as plt

c = get_random_color_map()
image = ... # Some image
keypoints = [...] # List of keypoints
connections = [...] # List of which keypoints to be connected

image = draw_keypoints(
    image, 
    keypoints, 
    limbs=connections, 
    color=c['points'], 
    limb_color=c['connection'], 
    limb_only=False
)
plt.imshow(image)
plt.show()
```

### draw_polygon
Function that overlays polygon on an image. Polygon is provided in an iterable of polygon points
Example

```python
from mediautil_.image import get_random_color_map, draw_polygon
from matplotlib import pyplot as plt

c = get_random_color_map()
image = ... # Some image
polygons = [...] # List of polygons

image = draw_polygon(
    image, 
    polygon, 
    color=c['polygon'], 
)
plt.imshow(image)
plt.show()
```
