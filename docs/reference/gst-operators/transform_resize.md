![](/docs/images/Ax_Page_Banner_2500x168_01.png)
# transform_resize

- [transform\_resize](#transform_resize)
  - [GST YAML example](#gst-yaml-example)
  - [Description](#description)
  - [Attributes](#attributes)

Changes the resolution of a video frame while optionally keeping the aspect ratio, and
adding padding.

Please find the operator code here:
[transform_resize](/operators/src/AxTransformResize.cpp)

## GST YAML example

```yaml
- instance: axtransform
  lib: libtransform_resize.so
  options: width:640;height:640;padding:114;to_tensor:1;letterbox:1
```

## Description
Upscales or downscales the input video frame using OpenCV's resize function with
bilinear interpolation while optionally keeping the aspect ratio and letterboxing or
pillarboxing.
If letterbox is set the output size is determined from the provided width and height.
If the aspect ratio of the input is greater than the aspect ratio of the output the
scaling ratio is determined by the width otherwise by the height. After scaling to
the new size the image is padded to arrive at the output width and height.
If letterbox is is not set and size is non-zero then the smaller of the input width or
height is scaled to size and the other size scaled to maintain aspect ratio. If size
is not set then the output size is taken from the provided width and height and the
input image scaled to this size.

## Attributes
*   **size**<br>
    *int*<br>
    The size to set the smaller of width or height in the output image.

*   **width**<br>
    *int*<br>
    The width of the resulting image.

*   **height**<br>
    *int*<br>
    The height of the resulting image.

*   **padding**<br>
    *int*<br>
    The color value of the padding. For RGB, all three colors are set to this
    value. The alpha channel is always set to 255.

*   **to\_tensor**<br>
    *int*<br>
    default `0` <br>
    Specify `0` if the output should be a video or `1` if the output should be a
    tensor. The tensor will be NHWC format with 4 channels and 1 byte per
    channel and only works if the video input is RGBA.

*   **letterbox**<br>
    *int*<br>
    default `1` <br>
    Specifies whether the input is letterboxed or pillarboxed or stretched to fit
    the output size

**Input data interface** AxVideoInterface with AxVideoInterface.info.format =
{AxVideoFormat::RGBA, AxVideoFormat::RGB, AxVideoFormat::GRAY8}

**Output data interface** AxVideoInterface or AxTensorsInterface
