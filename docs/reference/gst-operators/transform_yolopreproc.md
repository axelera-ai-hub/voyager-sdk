![](/docs/images/Ax_Page_Banner_2500x168_01.png)
# transform_yolopreproc

- [transform\_yolopreproc](#transform_yolopreproc)
  - [GST YAML example](#gst-yaml-example)
  - [Description](#description)
  - [Attributes](#attributes)

Executes the first layer of a YOLO network on the host.

Please find the operator code here:
[transform_yolopreproc](/operators/src/AxTransformYoloPreProc.cpp)

## GST YAML example

```yaml
- instance: axtransform
  lib: libtransform_yolopreproc.so
  options: padding:0,0,0,0,0,0,0,52;fill:114
```

## Description
The first layer of YOLO changes the memory layout of the input such that a patch
of 2x2 pixels is reshaped to one pixel with four times as many channels, e.g. an
image of 640x640x3 will be transformed to 320x320x12. As this operation requires
significant movement in memory, it is most efficiently performed on the host.
This layer is then stripped from the NN that runs on the device. In addition,
this subplugin adds the padding required for the compiled network.

## Attributes
*   **padding**<br>
    *array of int*<br>
    A comma separated list specifying the padding at the beginning, then at the
    end for the first dimension, then the padding at the beginning, then at the
    end of the second dimension, etc for all dimensions. The length of this list
    is thus twice the number of dimensions.

*   **fill**<br>
    *int*<br>
    The value of the padded bytes. Random data if not specified. It is
    recommended to use the quantization zeropoint.

**Input data interface** AxTensorsInterface

**Output data interface** AxTensorsInterface
