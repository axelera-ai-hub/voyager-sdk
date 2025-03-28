![](/docs/images/Ax_Page_Banner_2500x168_01.png)
# transform_dequantize

- [transform\_dequantize](#transform_dequantize)
  - [GST YAML example](#gst-yaml-example)
  - [Description](#description)
  - [Attributes](#attributes)

Dequantizes a tensor from int8 to float32.

Please find the operator code here:
[transform_dequantize](/operators/src/AxTransformDequantize.cpp)

## GST YAML example

```yaml
- instance: axtransform
  lib: libtransform_dequantize.so
  options: dequant_scale:0.13039915263652802;dequant_zeropoint:-70
```

## Description
This is the element immediately after the inference, except if dequantization is
performed in the decoder subplugin. The parameters are available in the file
`manifest.json` after compilation. Dequantization is given by

$$ y = s \cdot ( x - z), $$

where $y$ is the dequantized value in float32, x is the quantized value in int8,
$s$ is the dequantization scale parameter and $z$ is the dequantization
zeropoint parameter.

## Attributes
*   **dequant\_scale**<br>
    *array of int*<br>
    A comma separated list specifying the dequantization scale parameter for
    each tensor in the AxTensorsInterface.

*   **dequant\_zeropoint**<br>
    *array of int*<br>
    A comma separated list specifying the dequantization zeropoint parameter for
    each tensor in the AxTensorsInterface.

*   **transpose**<br>
    *int*<br>
    If the input tensor has to be transposed from NHWC to NCHW (`1`) or if it is
    already in NCHW format (`0`).

**Input data interface** AxTensorsInterface

**Output data interface** AxTensorsInterface
