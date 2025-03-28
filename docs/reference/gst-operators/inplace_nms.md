![](/docs/images/Ax_Page_Banner_2500x168_01.png)
# inplace_nms

- [inplace\_nms](#inplace_nms)
  - [GST YAML example](#gst-yaml-example)
  - [Description](#description)
  - [Attributes](#attributes)

Non Maximum Suppression

Please find the operator code here:
[inplace_nms](/operators/src/AxInPlaceNms.cpp)

## GST YAML example

```yaml
- instance: axinplace
  lib: libinplace_nms.so
  options: meta_key:yolov5s-relu-coco-onnx;max_boxes:300;nms_threshold:0.45;
    class_agnostic:0;location:CPU
```

## Description
Performs Non Maximum Suppression (NMS) on the detection output stored in the
metadata map with key `meta_key`. Removes boxes from the metadata that it
considers to be duplicates.

## Attributes
*   **meta_key**<br>
    *string*<br>
    Key in the metadata map which stores detection results. The metadata must be
    in the format `AxMetaObjDetection`.

*   **max_boxes**<br>
    *int*<br>
    The maximum number of boxes in the output.

*   **class_agnostic**<br>
    *int*<br>
    If the NMS should be performed for each class separately (0) or for all
    classes together (1).

*   **nms_threshold**<br>
    *float*<br>
    The Intersection over Union (IoU) threshold. A lower number means objects
    are still considered as the same result even though they overlap less.

*   **location**<br>
    *string*<br>
    default `CPU` The subplugin can be run on the CPU or on the GPU using
    OpenCL. Select either `CPU` or `GPU`.

**Mode** `meta`

**Data interface** *unused*
