![](/docs/images/Ax_Page_Banner_2500x168_01.png)
# inplace_draw

- [inplace\_draw](#inplace_draw)
  - [GST YAML example](#gst-yaml-example)
  - [Description](#description)

Adds visualizations of the inference results to the video.

Please find the operator code here:
[inplace_draw](/operators/src/AxInPlaceDraw.cpp)

## GST YAML example

```yaml
- instance: axinplace
  lib: libinplace_draw.so
  mode: write
```

## Description
This subplugin loops over all results in the metadata map and calls the draw
method. For the available metadata classes, this usually means that OpenCV is
called, which draws labels or bounding boxes onto the video image.

**Mode** `write`

**Data interface** AxVideoInterface
