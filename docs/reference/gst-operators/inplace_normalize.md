![](/docs/images/Ax_Page_Banner_2500x168_01.png)
# inplace_normalize

- [inplace\_normalize](#inplace_normalize)
  - [GST YAML example](#gst-yaml-example)
  - [Description](#description)
  - [Attributes](#attributes)

Normalizes a tensor, i.e. performs a constant shift-and-scale on each value of a
tensor. Optionally quantizes the values.

Please find the operator code here:
[inplace_normalize](/operators/src/AxInPlaceNormalize.cpp)

## GST YAML example

```yaml
- instance: axinplace
  lib: libinplace_normalize.so
  mode: write
  options: mean:0.485,0.456,0.406,0.0;std:0.229,0.224,0.225,1.0;simd:avx2;
    quant_scale:0.01863;quant_zeropoint:-14
```

## Description
When training a model with a specific dataset, the values that enter the network
are normalized in order to be in a uniform range for the specific training
dataset. In addition, when dealing with quantized networks, an additional
shift-and-scale operation is usually required to limit the range of the values.
This element performs the following operation

$$ x_{i,c} := \frac 1  {q \cdot s_c} (x_{i,c}-m_c)+ z, $$

where $x_{i,c}$ is a single value from the buffer with channel index $c$, $m_c$
is the mean value corresponding to channel $c$, $s_c$ the standard deviation
corresponding to channel $c$, q is the quantization scale and z the quantization
zeropoint, both of the which are equal for all channels. The operations are
executed in float32 precision and the result is cast to the precision
corresponding to the tensor.<br>
For int8 tensors, the tensor format NHWC is expected, which means the channel
dimension is expected to be the last, i.e. the most contiguous. However, for
float32 tensors, NCHW is expected.

## Attributes
*   **mean**<br>
    *array of float*<br>
    The mean value of the dataset to be used for normalization. It is necessary
    to specify a value for each channel.

*   **std**<br>
    *array of float*<br>
    The standard deviation of the dataset to be used for normalization. It is
    necessary to specify a value for each channel.

*   **quant\_scale**<br>
    *float*<br>
    The quantization scale parameter.

*   **quant\_zeropoint**<br>
    *float*<br>
    The zeropoint quantization parameter.

*   **simd**<br>
    *string*<br>
    For int8 tensors, a huge performance improvement can be achieved by
    using `avx2` or `avx512`. With the help of the SIMDe library, even
    microarchitectures without native AXV instructions (e.g. ARM Neon) might be
    supported.

**Mode** `write`

**Data interface** AxTensorsInterface with AxTensorsInterface.size() = 1
