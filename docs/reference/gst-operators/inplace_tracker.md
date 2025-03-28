![](/docs/images/Ax_Page_Banner_2500x168_01.png)
# inplace_tracker

- [inplace\_tracker](#inplace_tracker)
  - [GST YAML example](#gst-yaml-example)
  - [Description](#description)
  - [Attributes](#attributes)

Tracks detection results.

Please find the operator code here:
[inplace_tracker](/operators/src/AxInPlaceTracker.cpp)

## GST YAML example

```yaml
- instance: axinplace
  lib: libinplace_tracker.so
  options: detection_meta_key:yolo;tracking_meta_key:yolo_tracker;algorithm:sort
```

## Description
Assigns unique IDs to each bounding box that describe the same object over time.

## Attributes
*   **algorithm**<br>
    *string*<br>
    Choose between `scalarmot`, `sort`, `oc-sort` and `bytetrack`.

*   **detection\_meta\_key**<br>
    *string*<br>
    The key in the metadata map that corresponds to the input detections in the
    form of `AxMetaObjDetection`.

*   **tracking\_meta\_key**<br>
    *string*<br>
    The key in the metadata map used to store the result of the tracker in the
    form of `AxMetaTracker`.

*   **streamid\_meta\_key**<br>
    *string*<br>
    In multistream applications, each stream has its own instance of the
    tracker. This parameter describes the key in the metadata map that stores
    the stream IDs.

*   **history\_length**<br>
    *int*<br>
    The maximum number of time steps to be stored in each metadata object.

*   **algo\_params\_json**<br>
    *string*<br>
    A file containing additional parameters for the selected algorithm.

**Mode** none

**Data interface** AxVideoInterface
