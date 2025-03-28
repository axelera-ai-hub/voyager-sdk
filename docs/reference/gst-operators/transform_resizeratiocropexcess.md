![](/docs/images/Ax_Page_Banner_2500x168_01.png)
# transform_resizeratiocropexcess

- [transform\_resizeratiocropexcess](#transform_resizeratiocropexcess)
  - [GST YAML example](#gst-yaml-example)
  - [Description](#description)
  - [Attributes](#attributes)

Changes the resolution of a video frame while keeping the aspect ratio, then
crops the result.

Please find the operator code here:
[transform_resizeratiocropexcess](/operators/src/AxTransformResizeRatioCropExcess.cpp)

## GST YAML example

```yaml
- instance: axtransform
  lib: libtransform_resizeratiocropexcess.so
  options: resize_size:256;final_size_after_crop:224
```

## Description
Upscales or downscales the input video frame using the PillowResize function
with bilinear interpolation and antialiasing. The smallest of input width or
height is resized to the desired resolution, while the aspect ratio is kept.
The longer side of the resulting resized image will be center cropped so that
the output image is square. Optionally, an additional center crop can be
performed.

## Attributes
*   **resize\_size**<br>
    *int*<br>
    Specifies the output width and height to which the images should be resized.
    The resize ratio is determined by the shortest side. The scaled longest side
    will be cropped. The output will be square, so width and height will both be
    equal to resize\_size.

*   **final\_size\_after\_crop**<br>
    *int*<br>
    Optionally, the resized image can be center-cropped to a square image with
    size specified by this parameter. This means that after resizing and
    cropping the longest side, with this parameter an additional crop of both
    width and height is possible.

**Input data interface** AxVideoInterface with AxVideoInterface.info.format =
AxVideoFormat::RGBA

**Output data interface** AxVideoInterface with AxVideoInterface.info.format =
AxVideoFormat::RGBA
