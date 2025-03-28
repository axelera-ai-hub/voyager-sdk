![](/docs/images/Ax_Page_Banner_2500x168_01.png)
# transform_totensor

- [transform\_totensor](#transform_totensor)
  - [GST YAML example](#gst-yaml-example)
  - [Description](#description)
  - [Attributes](#attributes)

Converts a video frame to a tensor.

Please find the operator code here:
[transform_totensor](/operators/src/AxTransformToTensor.cpp)

## GST YAML example

```yaml
- instance: axtransform
  lib: libtransform_totensor.so
  options: type:int8
```

## Description
If the output tensor is requested in the format `int8`, the subplugin will copy
the uint8 data to remove gaps, as stride and size are defined to be the same in
tensors. The output tensor format will be NHWC. If the output `float32` is
requested, OpenCV DNN's blobFromImage will be called, which will transpose the
image to a tensor format of NCHW and normalize all values to the range [0,1].

## Attributes
*   **type**<br>
    *string*<br>
    The output tensor can either be `int8` or `float32`.

**Input data interface** AxVideoInterface

**Output data interface** AxTensorsInterface
