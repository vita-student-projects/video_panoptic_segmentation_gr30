### Label Map

KITTI-STEP adopts the same 19 classes as defined in
[Cityscapes](https://www.cityscapes-dataset.com/dataset-overview/#class-definitions)
with `pedestrians` and `cars` carefully annotated with track IDs. More
specifically, KITTI-STEP has the following label to index mapping:

Label Name     | Label ID
-------------- | --------
road           | 0
sidewalk       | 1
building       | 2
wall           | 3
fence          | 4
pole           | 5
traffic light  | 6
traffic sign   | 7
vegetation     | 8
terrain        | 9
sky            | 10
person&dagger; | 11
rider          | 12
car&dagger;    | 13
truck          | 14
bus            | 15
train          | 16
motorcycle     | 17
bicycle        | 18
void           | 255

&dagger;: Single instance annotations are available.

The groundtruth panoptic map is encoded as follows in PNG format:

```
R = semantic_id
G = instance_id // 256
B = instance % 256
```

## KittiSTEP Dataset

The KittiSTEP dataset class provides a special argument especially for video training: frames_before.  
The frames_before argument specifies how many frames before the current frame are returned in addition to the indexed frame. If a frame of the frames_before frames is before the beginning of the video, a black frame is returned instead.

The KittiSTEP dataset provides a convenient way to use the \_\_getitem\_\_ method using a tuple (video_id, frame_id) as an index.