![](/docs/images/Ax_Page_Banner_2500x168_01.png)
# Operators available for use in network YAML files

- [Operators available for use in network YAML files](#operators-available-for-use-in-network-yaml-files)
  - [Input](#input)
  - [Preprocess](#preprocess)
    - [Geometric Transformations](#geometric-transformations)
    - [Pixel Value Transformations](#pixel-value-transformations)
    - [Color Space Transformations](#color-space-transformations)
    - [Tensor Conversion](#tensor-conversion)
  - [Postprocess](#postprocess)

This is a list of non-neural pre- & post-processing operators that may be referred to in the
`pipeline:` sections of network YAML files. The name in brackets is the operator name as used in the
YAML files.

## Input

*   **Input (`input`)**
    *   **Description:** Loads the input image.
    *   **Parameters:**
        *   `color_format`: (String, optional, default: `RGB`) Specifies the color format of the input image. Supported values include: `RGB`, `BGR`, and `Gray`. If not specified, the image is loaded in `RGB` format.

*   **InputFromROI (`input-from-roi`)**
    *   **Description:** Loads a region of interest (ROI) from the input image. This is used in a cascade network where the first model must output a bounding box.
    *   **Parameters:**
        *   `color_format`: (String, optional, default: `RGB`) Specifies the color format of the input image for the ROI. Supported values include: `RGB`, `BGR`, and `Gray`. If not specified, the ROI is loaded assuming the source image is in `RGB` format.
        *   Other parameters related to ROI definition (e.g., coordinates, size) would be listed here.

**Explanation of `color_format` Parameter:**

The `color_format` parameter in the `Input` and `InputFromROI` steps allows you to directly specify the color format of the source image being loaded. By setting this parameter to `RGB`, `BGR`, or `Gray`, the system can load the image in the desired color format from the beginning. The default color format is `RGB`.

**Benefit:**

Specifying the `color_format` at the input stage can potentially eliminate the need for a separate `ConvertColor` preprocessing step later in the pipeline. This can simplify your preprocessing configuration and potentially improve efficiency by avoiding an unnecessary color conversion operation. The system will handle the color format conversion during the image loading process itself.


## Preprocess

### Geometric Transformations
This category includes transformations that modify the spatial arrangement of pixels in an image.

*   **CenterCrop (`centercrop`)**:
    *   **Description:** Crops the input image from the center.
    *   **Equivalence:** Equivalent to torchvision.transforms.CenterCrop().
*   **Letterbox (`letterbox`)**:
    *   **Description:** Resizes the input image while maintaining its aspect ratio and pads the remaining areas with a constant value to fit the target dimensions. This technique is commonly used in YOLO models.
*   **Resize (`resize`)**:
    *   **Description:** Resizes the input image to the specified dimensions.
    *   **Parameters:**
        - width: Target width (integer).
        - height: Target height (integer).
        - size: If specified, the smaller edge of the image is scaled to this size while preserving the aspect ratio. Do not specify both width/height and size.
        - half_pixel_centers: (Boolean, default: False) If True, uses half-pixel centers for resizing (currently supported by the OpenCV backend only).
        - interpolation: (String or InterpolationMode, default: bilinear) Specifies the interpolation algorithm to use. Supported options include: nearest, bilinear, bicubic, and lanczos.
        - Note: The backend may choose a different interpolation mode if the specified one is not supported, and a warning will be logged.
        - Examples:
        - YAML: interpolation: nearest
        - Python: operators.Resize(interpolation=operators.InterpolationMode.nearest) or Resize(interpolation='nearest')

### Pixel Value Transformations
This category includes various normalization techniques that modify the intensity values of the pixels in an image. It's important to understand the color format of your images (e.g., RGB, BGR, Grayscale) to apply these transformations correctly.

*   **ContrastNormalize (`contrast-normalize`)**
    *   **Description:** Performs contrast stretching by linearly mapping the pixel values to the full available range.
    *   **Formula:** `(input - min) / (max - min)`

*   **LinearScaling (`linear-scaling`)**
    *   **Description:** Applies a linear transformation to the pixel values by multiplying with a scale factor and adding an optional bias (shift).
    *   **Typical Use Case:** Transforming pixel values from the range [0, 255] to [-1, 1] (e.g., `input/mean + shift`, commonly used in TensorFlow normalization).

*   **Normalize (`normalize`)**
    *   **Description:** Normalizes the pixel values using a specified mean and standard deviation for each channel.
    *   **Equivalence:** Equivalent to `torchvision.transforms.Normalize(mean, std)`.
    *   **Formula:** `output[channel] = (input[channel] - mean[channel]) / std[channel]`

**Important Note on Color Channels:** When using `Normalize`, ensure you provide the correct `mean` and `std` values for each color channel of your input image. For example, for an RGB image, you would typically provide three values for `mean` and three values for `std`. If you have a multi-channel image but only provide a single `mean` and `std` value, the same values will be applied to all channels.

### Color Space Transformations
This category includes transformations that change the color representation of the image.

*   **ConvertColor (`convert-color`)**
    *   **Description:** Converts the color space of the input image.
    *   **Parameter Format:** Follows OpenCV's cvtColor conventions (e.g., RGB2BGR, YUV2RGB). Developers must know the exact input and output color formats.
    *   **Supported Conversions:**
        - RGB2GRAY
        - GRAY2RGB
        - RGB2BGR
        - BGR2RGB
        - BGR2GRAY
        - GRAY2BGR

### Tensor Conversion

*   **TorchToTensor (`torch-totensor`)**
    *   **Description:** Converts the input image to a PyTorch tensor.
    *   **Functionality:** Permutes the dimensions to NCHW (Number of channels, Height, Width) and, by default, normalizes pixel values to the range [0, 1].
    *   **Parameter:**
        - scale: (Boolean, default: True) If False, the pixel values are not normalized to the range [0, 1].


## Postprocess

*   DecodeSsdMobilenet (decode-ssd-mobilenet)

*   DecodeYolo (decodeyolo)

*   TopK (topk)

*   Multi-Object Tracker (SORT, OC-SORT, ByteTrack)
