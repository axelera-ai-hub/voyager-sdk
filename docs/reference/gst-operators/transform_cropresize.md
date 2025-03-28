![](/docs/images/Ax_Page_Banner_2500x168_01.png)
# transform_cropresize

- [transform\_cropresize](#transform_cropresize)
  - [GST YAML example](#gst-yaml-example)
  - [Description](#description)
  - [Attributes](#attributes)

Crops a video frame based a ROI provided by metadata and resizes it.

Please find the operator code here:
[transform_cropresize](/operators/src/AxTransformCropResize.cpp)

## GST YAML example

```yaml
- instance: axtransform
  lib: libtransform_resizeratiocropexcess.so
  options: meta_key:yolo;width:224;height:224;respect_aspectratio:1
```

## Description
Uses bounding boxes from provided metadata to crop a video frame. Then upscales
or downscales the input video frame using OpenCV's resize function with bilinear
interpolation. The resize either scales both dimensions individually, or the
aspect ratio is kept. In the latter case, the excess will be center cropped.
This element is usually used at later stages in a cascaded pipeline after
`AxDistributor`, which might create a number of subframes.

## Attributes
*   **meta_key**<br>
    *string*<br>
    The key in the metadata map that contains the ROIs in the form of
    `AxMetaBbox`. The box is chosen that corresponds to the respective subframe
    index.

*   **width**<br>
    *int*<br>
    The width of the result.

*   **height**<br>
    *int*<br>
    The height of the result.

*   **respect_aspectratio**<br>
    *int*<br>
    If set to `0`, the ROI is simply resized to the given width and height with
    a scale factor that might differ between width and height. If set to `1`,
    resizing keeps the aspect ratio, and the excess is cropped.

**Input data interface** AxVideoInterface

**Output data interface** AxVideoInterface
