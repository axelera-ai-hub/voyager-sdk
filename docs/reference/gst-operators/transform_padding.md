![](/docs/images/Ax_Page_Banner_2500x168_01.png)
# transform_padding

- [transform\_padding](#transform_padding)
  - [GST YAML example](#gst-yaml-example)
  - [Description](#description)
  - [Attributes](#attributes)

Adds padding to each dimension of a tensor.

Please find the operator code here:
[transform_padding](/operators/src/AxTransformPadding.cpp)

## GST YAML example

```yaml
- instance: axtransform
  lib: libtransform_padding.so
  options: padding:0,0,0,0,0,8,0,0;fill:114;
    input_shape:1,51840,8,4;output_shape:1,640,1296,4
```

## Description
This is usually the element immediately before the inference. Padding can be
added to the front and back of each dimension. The numbers depend on the
requirements of the compiled neural network and are available in the file
`manifest.json` after compilation. In some cases, padding of the original tensor
is not possible, and the tensor has to be reshaped before padding, and again
after padding. This can be requested using the additional optional keywords for
input and output shape.

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

*   **input\_shape**<br>
    *array of int*<br>
    A comma separated list specifying tensor dimensions. The input tensor can
    optionally be reshaped to these dimensions before applying the padding. Keep
    in mind that reshaping only changes the interpretation of the dimensions of
    the vector, but not the memory layout itself.

*   **output\_shape**<br>
    *array of int*<br>
    The padded tensor can optionally be reshaped before output. See comments
    above.

**Input data interface** AxTensorsInterface

**Output data interface** AxTensorsInterface
