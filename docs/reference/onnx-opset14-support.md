<!--- SPDX-License-Identifier: Apache-2.0 -->
![](/docs/images/Ax_Page_Banner_2500x168_01.png)
# AIPU support of ONNX opset14 operators

- [AIPU support of ONNX opset14 operators](#aipu-support-of-onnx-opset14-operators)
  - [Onnx Operators](#onnx-operators)
    - [Add](#add)
      - [Parameters](#parameters)
      - [Outputs](#outputs)
      - [Type Constraints](#type-constraints)
      - [Axelera's notes for developers](#axeleras-notes-for-developers)
        - [AIPU Acceleration Constraints](#aipu-acceleration-constraints)
    - [AveragePool](#averagepool)
      - [Parameters](#parameters-1)
      - [Outputs](#outputs-1)
      - [Type Constraints](#type-constraints-1)
      - [Axelera's notes for developers](#axeleras-notes-for-developers-1)
        - [AIPU Acceleration Constraints](#aipu-acceleration-constraints-1)
    - [BatchNormalization](#batchnormalization)
      - [Parameters](#parameters-2)
      - [Outputs (1 - 3)](#outputs-1---3)
      - [Type Constraints](#type-constraints-2)
      - [Axelera's notes for developers](#axeleras-notes-for-developers-2)
        - [AIPU Acceleration Constraints](#aipu-acceleration-constraints-2)
    - [Clip](#clip)
      - [Parameters (1 - 3)](#parameters-1---3)
      - [Outputs](#outputs-2)
      - [Type Constraints](#type-constraints-3)
      - [Axelera's notes for developers](#axeleras-notes-for-developers-3)
        - [AIPU Acceleration Constraints](#aipu-acceleration-constraints-3)
    - [Concat](#concat)
      - [Parameters (1 - ∞)](#parameters-1---)
      - [Outputs](#outputs-3)
      - [Type Constraints](#type-constraints-4)
      - [Axelera's notes for developers](#axeleras-notes-for-developers-4)
        - [AIPU Acceleration Constraints](#aipu-acceleration-constraints-4)
    - [Conv](#conv)
      - [Parameters (2 - 3)](#parameters-2---3)
      - [Outputs](#outputs-4)
      - [Type Constraints](#type-constraints-5)
      - [Axelera's notes for developers](#axeleras-notes-for-developers-5)
        - [AIPU Acceleration Constraints](#aipu-acceleration-constraints-5)
    - [ConvTranspose](#convtranspose)
      - [Parameters (2 - 3)](#parameters-2---3-1)
      - [Outputs](#outputs-5)
      - [Type Constraints](#type-constraints-6)
      - [Axelera's notes for developers](#axeleras-notes-for-developers-6)
        - [AIPU Acceleration Constraints](#aipu-acceleration-constraints-6)
    - [Gemm](#gemm)
      - [Parameters (2 - 3)](#parameters-2---3-2)
      - [Outputs](#outputs-6)
      - [Type Constraints](#type-constraints-7)
      - [Axelera's notes for developers](#axeleras-notes-for-developers-7)
        - [AIPU Acceleration Constraints](#aipu-acceleration-constraints-7)
    - [GlobalAveragePool](#globalaveragepool)
      - [Parameters](#parameters-3)
      - [Outputs](#outputs-7)
      - [Type Constraints](#type-constraints-8)
      - [Axelera's notes for developers](#axeleras-notes-for-developers-8)
        - [AIPU Acceleration Constraints](#aipu-acceleration-constraints-8)
    - [GlobalMaxPool](#globalmaxpool)
      - [Parameters](#parameters-4)
      - [Outputs](#outputs-8)
      - [Type Constraints](#type-constraints-9)
      - [Axelera's notes for developers](#axeleras-notes-for-developers-9)
        - [AIPU Acceleration Constraints](#aipu-acceleration-constraints-9)
    - [HardSigmoid](#hardsigmoid)
      - [Parameters](#parameters-5)
      - [Outputs](#outputs-9)
      - [Type Constraints](#type-constraints-10)
      - [Axelera's notes for developers](#axeleras-notes-for-developers-10)
        - [AIPU Acceleration Constraints](#aipu-acceleration-constraints-10)
    - [LeakyRelu](#leakyrelu)
      - [Parameters](#parameters-6)
      - [Outputs](#outputs-10)
      - [Type Constraints](#type-constraints-11)
      - [Axelera's notes for developers](#axeleras-notes-for-developers-11)
        - [AIPU Acceleration Constraints](#aipu-acceleration-constraints-11)
    - [MaxPool](#maxpool)
      - [Parameters](#parameters-7)
      - [Outputs (1 - 2)](#outputs-1---2)
      - [Type Constraints](#type-constraints-12)
      - [Axelera's notes for developers](#axeleras-notes-for-developers-12)
        - [AIPU Acceleration Constraints](#aipu-acceleration-constraints-12)
    - [Mul](#mul)
      - [Parameters](#parameters-8)
      - [Outputs](#outputs-11)
      - [Type Constraints](#type-constraints-13)
      - [Axelera's notes for developers](#axeleras-notes-for-developers-13)
        - [AIPU Acceleration Constraints](#aipu-acceleration-constraints-13)
    - [PRelu](#prelu)
      - [Parameters](#parameters-9)
      - [Outputs](#outputs-12)
      - [Type Constraints](#type-constraints-14)
      - [Axelera's notes for developers](#axeleras-notes-for-developers-14)
        - [AIPU Acceleration Constraints](#aipu-acceleration-constraints-14)
    - [Pad](#pad)
      - [Parameters (2 - 3)](#parameters-2---3-3)
      - [Outputs](#outputs-13)
      - [Type Constraints](#type-constraints-15)
      - [Axelera's notes for developers](#axeleras-notes-for-developers-15)
        - [AIPU Acceleration Constraints](#aipu-acceleration-constraints-15)
    - [Relu](#relu)
      - [Parameters](#parameters-10)
      - [Outputs](#outputs-14)
      - [Type Constraints](#type-constraints-16)
      - [Axelera's notes for developers](#axeleras-notes-for-developers-16)
        - [AIPU Acceleration Constraints](#aipu-acceleration-constraints-16)
    - [Resize](#resize)
      - [Parameters (1 - 4)](#parameters-1---4)
      - [Outputs](#outputs-15)
      - [Type Constraints](#type-constraints-17)
      - [Axelera's notes for developers](#axeleras-notes-for-developers-17)
        - [AIPU Acceleration Constraints](#aipu-acceleration-constraints-17)
    - [Selu](#selu)
      - [Parameters](#parameters-11)
      - [Outputs](#outputs-16)
      - [Type Constraints](#type-constraints-18)
      - [Axelera's notes for developers](#axeleras-notes-for-developers-18)
        - [AIPU Acceleration Constraints](#aipu-acceleration-constraints-18)
    - [Sigmoid](#sigmoid)
      - [Parameters](#parameters-12)
      - [Outputs](#outputs-17)
      - [Type Constraints](#type-constraints-19)
      - [Axelera's notes for developers](#axeleras-notes-for-developers-19)
        - [AIPU Acceleration Constraints](#aipu-acceleration-constraints-19)
    - [Slice](#slice)
      - [Parameters (3 - 5)](#parameters-3---5)
      - [Outputs](#outputs-18)
      - [Type Constraints](#type-constraints-20)
      - [Axelera's notes for developers](#axeleras-notes-for-developers-20)
        - [AIPU Acceleration Constraints](#aipu-acceleration-constraints-20)
    - [Split](#split)
      - [Parameters (1 - 2)](#parameters-1---2)
      - [Outputs (1 - ∞)](#outputs-1---)
      - [Type Constraints](#type-constraints-21)
      - [Axelera's notes for developers](#axeleras-notes-for-developers-21)
        - [AIPU Acceleration Constraints](#aipu-acceleration-constraints-21)
    - [Sub](#sub)
      - [Parameters](#parameters-13)
      - [Outputs](#outputs-19)
      - [Type Constraints](#type-constraints-22)
      - [Axelera's notes for developers](#axeleras-notes-for-developers-22)
        - [AIPU Acceleration Constraints](#aipu-acceleration-constraints-22)
    - [Tanh](#tanh)
      - [Parameters](#parameters-14)
      - [Outputs](#outputs-20)
      - [Type Constraints](#type-constraints-23)
      - [Axelera's notes for developers](#axeleras-notes-for-developers-23)
        - [AIPU Acceleration Constraints](#aipu-acceleration-constraints-23)
    - [Transpose](#transpose)
      - [Parameters](#parameters-15)
      - [Outputs](#outputs-21)
      - [Type Constraints](#type-constraints-24)
      - [Axelera's notes for developers](#axeleras-notes-for-developers-24)
        - [AIPU Acceleration Constraints](#aipu-acceleration-constraints-24)

## Onnx Operators

|**Operator**|**AIPU Acceleration**|
|-|:-:|
|[Add](#add)|Constrained|
|[AveragePool](#averagepool)|Constrained|
|[BatchNormalization](#batchnormalization)|Supported|
|[Concat](#concat)|Constrained|
|[Conv](#conv)|Constrained|
|[ConvTranspose](#convtranspose)|Constrained|
|[Gemm](#gemm)|Constrained|
|[GlobalAveragePool](#globalaveragepool)|Supported|
|[GlobalMaxPool](#globalmaxpool)|Supported|
|[LeakyRelu](#leakyrelu)|Supported|
|[MaxPool](#maxpool)|Constrained|
|[Mul](#mul)|Constrained|
|[PRelu](#prelu)|Constrained|
|[Pad](#pad)|Constrained|
|[Resize](#resize)|Constrained|
|[Sigmoid](#sigmoid)|Supported|
|[Slice](#slice)|Constrained|
|[Split](#split)|Constrained|
|[Sub](#sub)|Constrained|
|[Tanh](#tanh)|Supported|
|[Transpose](#transpose)|Constrained|


|**Function**|**AIPU Acceleration**|
|-|:-:|
|[Clip](#clip)|Constrained|
|[HardSigmoid](#hardsigmoid)|Constrained|
|[Relu](#relu)|Supported|
|[Selu](#selu)|Constrained|

---
### Add

  Performs element-wise binary addition (with Numpy-style broadcasting support).

  This operator supports **multidirectional (i.e., Numpy-style) broadcasting**.

  (Opset 14 change): Extend supported types to include uint8, int8, uint16, and int16.

#### Parameters

<dl>
<dt><tt>A</tt> : T</dt>
<dd>First operand.</dd>
<dt><tt>B</tt> : T</dt>
<dd>Second operand.</dd>
</dl>

#### Outputs

<dl>
<dt><tt>C</tt> : T</dt>
<dd>Result, has same element type as two inputs</dd>
</dl>

#### Type Constraints

<dl>
<dt><tt>T</tt> : tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(bfloat16)</dt>
<dd>Constrain input and output types to all numeric tensors.</dd>
</dl>

#### Axelera's notes for developers

Given an operand with shape [N, C, H, W], Addition is supported with other operands with shape [N, C, H, W], [1, C, 1, 1], and scalars.

##### AIPU Acceleration Constraints

<dl>
<dt><tt>allow_config</tt> : A.shape == B.shape</dt><dt><tt>allow_config</tt> : len(A.shape)==4 and A.shape[1]==1 and B.shape==(0)</dt><dt><tt>allow_config</tt> : len(B.shape)==4 and B.shape[1]==1 and A.shape==(0)</dt><dt><tt>allow_config</tt> : len(A.shape)==4 and A.shape[1]!=1 and B.shape==(1, A.shape[1], 1, 1)</dt><dt><tt>allow_config</tt> : len(B.shape)==4 and B.shape[1]!=1 and A.shape==(1, B.shape[1], 1, 1)</dt></dl>

---
### AveragePool

  AveragePool consumes an input tensor X and applies average pooling across
   the tensor according to kernel sizes, stride sizes, and pad lengths.
   average pooling consisting of computing the average on all values of a
   subset of the input tensor according to the kernel size and downsampling the
   data into the output tensor Y for further processing. The output spatial shape will be following:
   ```
   output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1)) / strides_spatial_shape[i] + 1)
   ```
   or
   ```
   output_spatial_shape[i] = ceil((input_spatial_shape[i] + pad_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1)) / strides_spatial_shape[i] + 1)
   ```
   if ceil_mode is enabled

   ```
   * pad_shape[i] is sum of pads along axis i
   ```

   `auto_pad` is a DEPRECATED attribute. If you are using them currently, the output spatial shape will be following when ceil_mode is enabled:
   ```
   VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) + 1) / strides_spatial_shape[i])
   SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])
   ```
  or when ceil_mode is disabled:
   ```
   VALID: output_spatial_shape[i] = floor((input_spatial_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) + 1) / strides_spatial_shape[i])
   SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = floor(input_spatial_shape[i] / strides_spatial_shape[i])
   ```

   And pad shape will be following if `SAME_UPPER` or `SAME_LOWER`:
   ```
   pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) - input_spatial_shape[i]
   ```
   The output of each pooling window is divided by the number of elements (exclude pad when attribute count_include_pad is zero).


#### Parameters

<dl>
<dt><tt>X</tt> : T</dt>
<dd>Input data tensor from the previous operator; dimensions for image case are (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data. For non image case, the dimensions are in the form of (N x C x D1 x D2 ... Dn), where N is the batch size. Optionally, if dimension denotation is in effect, the operation expects the input data tensor to arrive with the dimension denotation of [DATA_BATCH, DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...].</dd>
<dt><tt>auto_pad</tt> : string (default is NOTSET)</dt>
<dd>auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID. Where default value is NOTSET, which means explicit padding is used. SAME_UPPER or SAME_LOWER mean pad the input so that `output_shape[i] = ceil(input_shape[i] / strides[i])` for each axis `i`. The padding is split between the two sides equally or almost equally (depending on whether it is even or odd). In case the padding is an odd number, the extra padding is added at the end for SAME_UPPER and at the beginning for SAME_LOWER.</dd>
<dt><tt>ceil_mode</tt> : int (default is 0)</dt>
<dd>Whether to use ceil or floor (default) to compute the output shape.</dd>
<dt><tt>count_include_pad</tt> : int (default is 0)</dt>
<dd>Whether include pad pixels when calculating values for the edges. Default is 0, doesn't count include pad.</dd>
<dt><tt>kernel_shape</tt> : list of ints (required)</dt>
<dd>The size of the kernel along each axis.</dd>
<dt><tt>pads</tt> : list of ints</dt>
<dd>Padding for the beginning and ending along each spatial axis, it can take any value greater than or equal to 0. The value represent the number of pixels added to the beginning and end part of the corresponding axis. `pads` format should be as follow [x1_begin, x2_begin...x1_end, x2_end,...], where xi_begin the number of pixels added at the beginning of axis `i` and xi_end, the number of pixels added at the end of axis `i`. This attribute cannot be used simultaneously with auto_pad attribute. If not present, the padding defaults to 0 along start and end of each spatial axis.</dd>
<dt><tt>strides</tt> : list of ints</dt>
<dd>Stride along each spatial axis. If not present, the stride defaults to 1 along each spatial axis.</dd>
</dl>

#### Outputs

<dl>
<dt><tt>Y</tt> : T</dt>
<dd>Output data tensor from average or max pooling across the input tensor. Dimensions will vary based on various kernel, stride, and pad sizes. Floor value of the dimension is used</dd>
</dl>

#### Type Constraints

<dl>
<dt><tt>T</tt> : tensor(float16), tensor(float), tensor(double)</dt>
<dd>Constrain input and output types to float tensors.</dd>
</dl>

#### Axelera's notes for developers

Only AveragePool operators with explicit padding (i.e., auto_pad = "NOTSET") are currently supported. Moreover, due to torch runtime constraints, symmetric padding along each dimension must be at most half of the kernel size along the same dimension. Lastly, note that count_include_pad is only supported equal to 1. If padding is specified for this operator, count_include_pad !=1 may lead to wrong results.

##### AIPU Acceleration Constraints

<dl>
<dt><tt>rule</tt> : auto_pad=="NOTSET"</dt><dt><tt>rule</tt> : len(pads)==4</dt><dt><tt>rule</tt> : (pads[0]==pads[2] and pads[1]==pads[3] and pads[0]<=0.5*kernel.shape[0] and pads[1]<=0.5*kernel.shape[1]) or (pads[0]!=pads[2]) or (pads[1]!=pads[3])</dt><dt><tt>rule</tt> : (pads!=[0, 0, 0, 0] and count_include_pad==1) or (pads==[0, 0, 0, 0])</dt></dl>

---
### BatchNormalization

  Carries out batch normalization as described in the paper
  https://arxiv.org/abs/1502.03167. Depending on the mode it is being run,
  There are five required inputs 'X', 'scale', 'B', 'input_mean' and
  'input_var'.
  Note that 'input_mean' and 'input_var' are expected to be the estimated
  statistics in inference mode (training_mode=False, default),
  and the running statistics in training mode (training_mode=True).
  There are multiple cases for the number of outputs, which we list below:

  Output case #1: Y, running_mean, running_var (training_mode=True)
  Output case #2: Y (training_mode=False)

  When training_mode=False, extra outputs are invalid.
  The outputs are updated as follows when training_mode=True:
  ```
  running_mean = input_mean * momentum + current_mean * (1 - momentum)
  running_var = input_var * momentum + current_var * (1 - momentum)

  Y = (X - current_mean) / sqrt(current_var + epsilon) * scale + B

  where:

  current_mean = ReduceMean(X, axis=all_except_channel_index)
  current_var =  ReduceVar(X, axis=all_except_channel_index)

  Notice that ReduceVar refers to the population variance, and it equals to
  sum(sqrd(x_i - x_avg)) / N
  where N is the population size (this formula does not use sample size N - 1).

  ```

  When training_mode=False:
  ```
  Y = (X - input_mean) / sqrt(input_var + epsilon) * scale + B
  ```

  For previous (depreciated) non-spatial cases, implementors are suggested
  to flatten the input shape to (N x C * D1 * D2 * ... * Dn) before a BatchNormalization Op.
  This operator has **optional** inputs/outputs. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.

#### Parameters

<dl>
<dt><tt>X</tt> : T</dt>
<dd>Input data tensor from the previous operator; dimensions are in the form of (N x C x D1 x D2 ... Dn), where N is the batch size, C is the number of channels. Statistics are computed for every channel of C over N and D1 to Dn dimensions. For image data, input dimensions become (N x C x H x W). The op also accepts single dimension input of size N in which case C is assumed to be 1</dd>
<dt><tt>scale</tt> : T</dt>
<dd>Scale tensor of shape (C).</dd>
<dt><tt>B</tt> : T</dt>
<dd>Bias tensor of shape (C).</dd>
<dt><tt>input_mean</tt> : U</dt>
<dd>running (training) or estimated (testing) mean tensor of shape (C).</dd>
<dt><tt>input_var</tt> : U</dt>
<dd>running (training) or estimated (testing) variance tensor of shape (C).</dd>
<dt><tt>epsilon</tt> : float (default is 1e-05)</dt>
<dd>The epsilon value to use to avoid division by zero.</dd>
<dt><tt>momentum</tt> : float (default is 0.9)</dt>
<dd>Factor used in computing the running mean and variance.e.g., running_mean = running_mean * momentum + mean * (1 - momentum).</dd>
<dt><tt>training_mode</tt> : int (default is 0)</dt>
<dd>If set to true, it indicates BatchNormalization is being used for training, and outputs 1, 2, 3, and 4 would be populated.</dd>
</dl>

#### Outputs (1 - 3)

<dl>
<dt><tt>Y</tt> : T</dt>
<dd>The output tensor of the same shape as X</dd>
<dt><tt>running_mean</tt> : U</dt>
<dd>The running mean after the BatchNormalization operator.</dd>
<dt><tt>running_var</tt> : U</dt>
<dd>The running variance after the BatchNormalization operator. This op uses the population size (N) for calculating variance, and not the sample size N-1.</dd>
</dl>

#### Type Constraints

<dl>
<dt><tt>T</tt> : tensor(float16), tensor(float), tensor(double), tensor(bfloat16)</dt>
<dd>Constrain input and output types to float tensors.</dd>
<dt><tt>U</tt> : tensor(float16), tensor(float), tensor(double), tensor(bfloat16)</dt>
<dd>Constrain mean and variance types to float tensors. It allows all float type for U.</dd>
</dl>

#### Axelera's notes for developers


##### AIPU Acceleration Constraints

<dl>
<dt><tt>Operator is supported in any configurations.</tt>
</dl>

---
### Clip

  Clip operator limits the given input within an interval. The interval is
  specified by the inputs 'min' and 'max'. They default to
  numeric_limits::lowest() and numeric_limits::max(), respectively.

#### Parameters (1 - 3)

<dl>
<dt><tt>input</tt> : T</dt>
<dd>Input tensor whose elements to be clipped</dd>
<dt><tt>min</tt> : T</dt>
<dd>Minimum value, under which element is replaced by min. It must be a scalar(tensor of empty shape).</dd>
<dt><tt>max</tt> : T</dt>
<dd>Maximum value, above which element is replaced by max. It must be a scalar(tensor of empty shape).</dd>
</dl>

#### Outputs

<dl>
<dt><tt>output</tt> : T</dt>
<dd>Output tensor with clipped input elements</dd>
</dl>

#### Type Constraints

<dl>
<dt><tt>T</tt> : tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(bfloat16)</dt>
<dd>Constrain input and output types to all numeric tensors.</dd>
</dl>

#### Axelera's notes for developers

Only Clip operators implementing ReLU6 (min=0, max=6) and HardTanh (min=-1, max=1) are currently supported.

##### AIPU Acceleration Constraints

<dl>
<dt><tt>allow_config</tt> : min==0 and max==6</dt><dt><tt>allow_config</tt> : min==-1 and max==1</dt></dl>

---
### Concat

  Concatenate a list of tensors into a single tensor. All input tensors must have the same shape, except for the dimension size of the axis to concatenate on.

#### Parameters (1 - &#8734;)

<dl>
<dt><tt>inputs</tt> : T</dt>
<dd>List of tensors for concatenation</dd>
<dt><tt>axis</tt> : int (required)</dt>
<dd>Which axis to concat on. A negative value means counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(inputs)..</dd>
</dl>

#### Outputs

<dl>
<dt><tt>concat_result</tt> : T</dt>
<dd>Concatenated tensor</dd>
</dl>

#### Type Constraints

<dl>
<dt><tt>T</tt> : tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)</dt>
<dd>Constrain output types to any tensor type.</dd>
</dl>

#### Axelera's notes for developers

Only concatenation along the channel dimension for a 4-d feature map is currently supported.

##### AIPU Acceleration Constraints

<dl>
<dt><tt>allow_config</tt> : axis == 1</dt><dt><tt>allow_config</tt> : all([len(x) == 4 for x in inputs.shapes]) and axis == -3</dt></dl>

---
### Conv

  The convolution operator consumes an input tensor and a filter, and
  computes the output.

#### Parameters (2 - 3)

<dl>
<dt><tt>X</tt> : T</dt>
<dd>Input data tensor from previous layer; has size (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and width. Note that this is for the 2D image. Otherwise the size is (N x C x D1 x D2 ... x Dn). Optionally, if dimension denotation is in effect, the operation expects input data tensor to arrive with the dimension denotation of [DATA_BATCH, DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...].</dd>
<dt><tt>W</tt> : T</dt>
<dd>The weight tensor that will be used in the convolutions; has size (M x C/group x kH x kW), where C is the number of channels, and kH and kW are the height and width of the kernel, and M is the number of feature maps. For more than 2 dimensions, the kernel shape will be (M x C/group x k1 x k2 x ... x kn), where (k1 x k2 x ... kn) is the dimension of the kernel. Optionally, if dimension denotation is in effect, the operation expects the weight tensor to arrive with the dimension denotation of [FILTER_OUT_CHANNEL, FILTER_IN_CHANNEL, FILTER_SPATIAL, FILTER_SPATIAL ...]. Assuming zero based indices for the shape array, X.shape[1] == (W.shape[1] * group) == C and W.shape[0] mod G == 0. Or in other words FILTER_IN_CHANNEL multiplied by the number of groups should be equal to DATA_CHANNEL and the number of feature maps M should be a multiple of the number of groups G.</dd>
<dt><tt>B</tt> : T</dt>
<dd>Optional 1D bias to be added to the convolution, has size of M.</dd>
<dt><tt>auto_pad</tt> : string (default is NOTSET)</dt>
<dd>auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID. Where default value is NOTSET, which means explicit padding is used. SAME_UPPER or SAME_LOWER mean pad the input so that `output_shape[i] = ceil(input_shape[i] / strides[i])` for each axis `i`. The padding is split between the two sides equally or almost equally (depending on whether it is even or odd). In case the padding is an odd number, the extra padding is added at the end for SAME_UPPER and at the beginning for SAME_LOWER.</dd>
<dt><tt>dilations</tt> : list of ints</dt>
<dd>dilation value along each spatial axis of the filter. If not present, the dilation defaults is 1 along each spatial axis.</dd>
<dt><tt>group</tt> : int (default is 1)</dt>
<dd>number of groups input channels and output channels are divided into.</dd>
<dt><tt>kernel_shape</tt> : list of ints</dt>
<dd>The shape of the convolution kernel. If not present, should be inferred from input W.</dd>
<dt><tt>pads</tt> : list of ints</dt>
<dd>Padding for the beginning and ending along each spatial axis, it can take any value greater than or equal to 0. The value represent the number of pixels added to the beginning and end part of the corresponding axis. `pads` format should be as follow [x1_begin, x2_begin...x1_end, x2_end,...], where xi_begin the number of pixels added at the beginning of axis `i` and xi_end, the number of pixels added at the end of axis `i`. This attribute cannot be used simultaneously with auto_pad attribute. If not present, the padding defaults to 0 along start and end of each spatial axis.</dd>
<dt><tt>strides</tt> : list of ints</dt>
<dd>Stride along each spatial axis. If not present, the stride defaults is 1 along each spatial axis.</dd>
</dl>

#### Outputs

<dl>
<dt><tt>Y</tt> : T</dt>
<dd>Output data tensor that contains the result of the convolution. The output dimensions are functions of the kernel size, stride size, and pad lengths.</dd>
</dl>

#### Type Constraints

<dl>
<dt><tt>T</tt> : tensor(float16), tensor(float16), tensor(float), tensor(double)</dt>
<dd>Constrain input and output types to float tensors.</dd>
</dl>

#### Axelera's notes for developers

Only Conv operators with explicit padding (i.e., auto_pad = "NOTSET") are currently supported. Grouped convolutions that are not depthwise (i.e., group = nr_channels) are not supported. In the case of depthwise convolutions, only symmetric kernels are supported.

##### AIPU Acceleration Constraints

<dl>
<dt><tt>rule</tt> : auto_pad == "NOTSET"</dt><dt><tt>rule</tt> : group == 1 or (group == X.shape[1] and kernel.shape[0] == kernel.shape[1])</dt></dl>

---
### ConvTranspose

  The convolution transpose operator consumes an input tensor and a filter,
  and computes the output.

  If the pads parameter is provided the shape of the output is calculated via the following equation:

    output_shape[i] = stride[i] * (input_size[i] - 1) + output_padding[i] + ((kernel_shape[i] - 1) * dilations[i] + 1) - pads[start_i] - pads[end_i]

  output_shape can also be explicitly specified in which case pads values are auto generated using these equations:

    total_padding[i] = stride[i] * (input_size[i] - 1) + output_padding[i] + ((kernel_shape[i] - 1) * dilations[i] + 1) - output_shape[i]
    If (auto_pads == SAME_UPPER): pads[start_i] = total_padding[i]/2; pads[end_i] = total_padding[i] - (total_padding[i]/2)
    Else: pads[start_i] = total_padding[i] - (total_padding[i]/2); pads[end_i] = (total_padding[i]/2).



#### Parameters (2 - 3)

<dl>
<dt><tt>X</tt> : T</dt>
<dd>Input data tensor from previous layer; has size (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and width. Note that this is for the 2D image. Otherwise the size is (N x C x D1 x D2 ... x Dn)</dd>
<dt><tt>W</tt> : T</dt>
<dd>The weight tensor that will be used in the convolutions; has size (C x M/group x kH x kW), where C is the number of channels, and kH and kW are the height and width of the kernel, and M is the number of feature maps. For more than 2 dimensions, the weight shape will be (C x M/group x k1 x k2 x ... x kn), where (k1 x k2 x ... x kn) is the dimension of the kernel. The number of channels in the output should be equal to W.shape[1] * group (assuming zero based indices of the shape array)</dd>
<dt><tt>B</tt> : T</dt>
<dd>Optional 1D bias to be added to the convolution, has size of M.</dd>
<dt><tt>auto_pad</tt> : string (default is NOTSET)</dt>
<dd>auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID. Where default value is NOTSET, which means explicit padding is used. SAME_UPPER or SAME_LOWER mean pad the input so that `output_shape[i] = input_shape[i] * strides[i]` for each axis `i`. The padding is split between the two sides equally or almost equally (depending on whether it is even or odd). In case the padding is an odd number, the extra padding is added at the end for SAME_UPPER and at the beginning for SAME_LOWER.</dd>
<dt><tt>dilations</tt> : list of ints</dt>
<dd>dilation value along each spatial axis of the filter. If not present, the dilation defaults to 1 along each spatial axis.</dd>
<dt><tt>group</tt> : int (default is 1)</dt>
<dd>number of groups input channels and output channels are divided into.</dd>
<dt><tt>kernel_shape</tt> : list of ints</dt>
<dd>The shape of the convolution kernel. If not present, should be inferred from input W.</dd>
<dt><tt>output_padding</tt> : list of ints</dt>
<dd>Additional elements added to the side with higher coordinate indices in the output. Each padding value in "output_padding" must be less than the corresponding stride/dilation dimension. By default, this attribute is a zero vector. Note that this attribute doesn't directly affect the computed output values. It only controls the selection of the computed values, so changing this attribute only adds or removes output elements. If "output_shape" is explicitly provided, "output_padding" does not contribute additional size to "output_shape" but participates in the computation of the needed padding amount. This is also called adjs or adjustment in some frameworks.</dd>
<dt><tt>output_shape</tt> : list of ints</dt>
<dd>The shape of the output can be explicitly set which will cause pads values to be auto generated. If output_shape is specified pads values are ignored. See doc for details for equations to generate pads. Note that the output_shape attribute value should not include dimensions for batch size and channels, which are automatically inferred.</dd>
<dt><tt>pads</tt> : list of ints</dt>
<dd>Padding for the beginning and ending along each spatial axis, it can take any value greater than or equal to 0. The value represent the number of pixels added to the beginning and end part of the corresponding axis. `pads` format should be as follow [x1_begin, x2_begin...x1_end, x2_end,...], where xi_begin the number of pixels added at the beginning of axis `i` and xi_end, the number of pixels added at the end of axis `i`. This attribute cannot be used simultaneously with auto_pad attribute. If not present, the padding defaults to 0 along start and end of each spatial axis.</dd>
<dt><tt>strides</tt> : list of ints</dt>
<dd>Stride along each spatial axis. If not present, the stride defaults to 1 along each spatial axis.</dd>
</dl>

#### Outputs

<dl>
<dt><tt>Y</tt> : T</dt>
<dd>Output data tensor that contains the result of the convolution. The output dimensions are functions of the kernel size, stride size, pad lengths and group count. The number of channels in the output should be equal to W.shape[1] * group (assuming zero based indices of the shape array)</dd>
</dl>

#### Type Constraints

<dl>
<dt><tt>T</tt> : tensor(float16), tensor(float), tensor(double)</dt>
<dd>Constrain input and output types to float tensors.</dd>
</dl>

#### Axelera's notes for developers

Only ConvTranspose operators with explicit padding (i.e., auto_pad = "NOTSET") are currently supported.

##### AIPU Acceleration Constraints

<dl>
<dt><tt>rule</tt> : auto_pad == "NOTSET"</dt></dl>

---
### Gemm

  General Matrix multiplication:
  https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3

  * A' = transpose(A) if transA else A
  * B' = transpose(B) if transB else B

  Compute Y = alpha * A' * B' + beta * C, where input tensor A has shape (M, K) or (K, M),
  input tensor B has shape (K, N) or (N, K), input tensor C is broadcastable to shape (M, N),
  and output tensor Y has shape (M, N). A will be transposed before doing the
  computation if attribute transA is non-zero, same for B and transB.
  This operator supports **unidirectional broadcasting** (tensor C should be unidirectional broadcastable to tensor A * B).
  This operator has **optional** inputs/outputs. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.

#### Parameters (2 - 3)

<dl>
<dt><tt>A</tt> : T</dt>
<dd>Input tensor A. The shape of A should be (M, K) if transA is 0, or (K, M) if transA is non-zero.</dd>
<dt><tt>B</tt> : T</dt>
<dd>Input tensor B. The shape of B should be (K, N) if transB is 0, or (N, K) if transB is non-zero.</dd>
<dt><tt>C</tt> : T</dt>
<dd>Optional input tensor C. If not specified, the computation is done as if C is a scalar 0. The shape of C should be unidirectional broadcastable to (M, N).</dd>
<dt><tt>alpha</tt> : float (default is 1.0)</dt>
<dd>Scalar multiplier for the product of input tensors A * B.</dd>
<dt><tt>beta</tt> : float (default is 1.0)</dt>
<dd>Scalar multiplier for input tensor C.</dd>
<dt><tt>transA</tt> : int (default is 0)</dt>
<dd>Whether A should be transposed</dd>
<dt><tt>transB</tt> : int (default is 0)</dt>
<dd>Whether B should be transposed</dd>
</dl>

#### Outputs

<dl>
<dt><tt>Y</tt> : T</dt>
<dd>Output tensor of shape (M, N).</dd>
</dl>

#### Type Constraints

<dl>
<dt><tt>T</tt> : tensor(float16), tensor(float), tensor(double), tensor(uint32), tensor(uint64), tensor(int32), tensor(int64), tensor(bfloat16)</dt>
<dd>Constrain input and output types to float/int tensors.</dd>
</dl>

#### Axelera's notes for developers

Gemm with automatic transposition of the first operand (i.e., transA == 1) is not supported.

##### AIPU Acceleration Constraints

<dl>
<dt><tt>rule</tt> : transA == 0</dt></dl>

---
### GlobalAveragePool

  GlobalAveragePool consumes an input tensor X and applies average pooling across
   the values in the same channel. This is equivalent to AveragePool with kernel size
   equal to the spatial dimension of input tensor.

#### Parameters

<dl>
<dt><tt>X</tt> : T</dt>
<dd>Input data tensor from the previous operator; dimensions for image case are (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data. For non image case, the dimensions are in the form of (N x C x D1 x D2 ... Dn), where N is the batch size.</dd>
</dl>

#### Outputs

<dl>
<dt><tt>Y</tt> : T</dt>
<dd>Output data tensor from pooling across the input tensor. The output tensor has the same rank as the input. The first two dimensions of output shape are the same as the input (N x C), while the other dimensions are all 1.</dd>
</dl>

#### Type Constraints

<dl>
<dt><tt>T</tt> : tensor(float16), tensor(float), tensor(double)</dt>
<dd>Constrain input and output types to float tensors.</dd>
</dl>

#### Axelera's notes for developers


##### AIPU Acceleration Constraints

<dl>
<dt><tt>Operator is supported in any configurations.</tt>
</dl>

---
### GlobalMaxPool

  GlobalMaxPool consumes an input tensor X and applies max pooling across
   the values in the same channel. This is equivalent to MaxPool with kernel size
   equal to the spatial dimension of input tensor.

#### Parameters

<dl>
<dt><tt>X</tt> : T</dt>
<dd>Input data tensor from the previous operator; dimensions for image case are (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data. For non image case, the dimensions are in the form of (N x C x D1 x D2 ... Dn), where N is the batch size.</dd>
</dl>

#### Outputs

<dl>
<dt><tt>Y</tt> : T</dt>
<dd>Output data tensor from pooling across the input tensor. The output tensor has the same rank as the input. The first two dimensions of output shape are the same as the input (N x C), while the other dimensions are all 1.</dd>
</dl>

#### Type Constraints

<dl>
<dt><tt>T</tt> : tensor(float16), tensor(float), tensor(double)</dt>
<dd>Constrain input and output types to float tensors.</dd>
</dl>

#### Axelera's notes for developers


##### AIPU Acceleration Constraints

<dl>
<dt><tt>Operator is supported in any configurations.</tt>
</dl>

---
### HardSigmoid

  HardSigmoid takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where the HardSigmoid function, y = max(0, min(1, alpha * x + beta)),
  is applied to the tensor elementwise.

#### Parameters

<dl>
<dt><tt>X</tt> : T</dt>
<dd>Input tensor</dd>
<dt><tt>alpha</tt> : float (default is 0.2)</dt>
<dd>Value of alpha.</dd>
<dt><tt>beta</tt> : float (default is 0.5)</dt>
<dd>Value of beta.</dd>
</dl>

#### Outputs

<dl>
<dt><tt>Y</tt> : T</dt>
<dd>Output tensor</dd>
</dl>

#### Type Constraints

<dl>
<dt><tt>T</tt> : tensor(float16), tensor(float), tensor(double)</dt>
<dd>Constrain input and output types to float tensors.</dd>
</dl>

#### Axelera's notes for developers

Only HardSigmoid operators with pytorch-like parameters (alpha=0.16666, beta=0.5) are supported.

##### AIPU Acceleration Constraints

<dl>
<dt><tt>rule</tt> : alpha == 0.16666</dt><dt><tt>rule</tt> : beta == 0.5</dt></dl>

---
### LeakyRelu

  LeakyRelu takes input data (Tensor<T>) and an argument alpha, and produces one
  output data (Tensor<T>) where the function `f(x) = alpha * x for x < 0`,
  `f(x) = x for x >= 0`, is applied to the data tensor elementwise.

#### Parameters

<dl>
<dt><tt>X</tt> : T</dt>
<dd>Input tensor</dd>
<dt><tt>alpha</tt> : float (default is 0.01)</dt>
<dd>Coefficient of leakage.</dd>
</dl>

#### Outputs

<dl>
<dt><tt>Y</tt> : T</dt>
<dd>Output tensor</dd>
</dl>

#### Type Constraints

<dl>
<dt><tt>T</tt> : tensor(float16), tensor(float), tensor(double)</dt>
<dd>Constrain input and output types to float tensors.</dd>
</dl>

#### Axelera's notes for developers


##### AIPU Acceleration Constraints

<dl>
<dt><tt>Operator is supported in any configurations.</tt>
</dl>

---
### MaxPool

  MaxPool consumes an input tensor X and applies max pooling across
   the tensor according to kernel sizes, stride sizes, and pad lengths.
   max pooling consisting of computing the max on all values of a
   subset of the input tensor according to the kernel size and downsampling the
   data into the output tensor Y for further processing. The output spatial shape is calculated differently
   depending on whether explicit padding is used, where pads is employed, or auto padding is used, where auto_pad is utilized.
   With explicit padding (https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html?highlight=maxpool#torch.nn.MaxPool2d):
   ```
   output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - dilation[i] * (kernel_shape[i] - 1) - 1) / strides_spatial_shape[i] + 1)
   ```
   or
   ```
   output_spatial_shape[i] = ceil((input_spatial_shape[i] + pad_shape[i] - dilation[i] * (kernel_shape[i] - 1) - 1) / strides_spatial_shape[i] + 1)
   ```
   if ceil_mode is enabled. `pad_shape[i]` is the sum of pads along axis `i`.

   `auto_pad` is a DEPRECATED attribute. If you are using them currently, the output spatial shape will be following when ceil_mode is enabled:
   ```
   VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) + 1) / strides_spatial_shape[i])
   SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])
   ```
   or when ceil_mode is disabled (https://www.tensorflow.org/api_docs/python/tf/keras/layers/AveragePooling2D):
   ```
   VALID: output_spatial_shape[i] = floor((input_spatial_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1)) / strides_spatial_shape[i]) + 1
   SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = floor((input_spatial_shape[i] - 1) / strides_spatial_shape[i]) + 1
   ```
   And pad shape will be following if `SAME_UPPER` or `SAME_LOWER`:
   ```
   pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) - input_spatial_shape[i]
   ```
   The output of each pooling window is maximum number of elements exclude pad.


#### Parameters

<dl>
<dt><tt>X</tt> : T</dt>
<dd>Input data tensor from the previous operator; dimensions for image case are (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data. For non image case, the dimensions are in the form of (N x C x D1 x D2 ... Dn), where N is the batch size. Optionally, if dimension denotation is in effect, the operation expects the input data tensor to arrive with the dimension denotation of [DATA_BATCH, DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...].</dd>
<dt><tt>auto_pad</tt> : string (default is NOTSET)</dt>
<dd>auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID. Where default value is NOTSET, which means explicit padding is used. SAME_UPPER or SAME_LOWER mean pad the input so that `output_shape[i] = ceil(input_shape[i] / strides[i])` for each axis `i`. The padding is split between the two sides equally or almost equally (depending on whether it is even or odd). In case the padding is an odd number, the extra padding is added at the end for SAME_UPPER and at the beginning for SAME_LOWER.</dd>
<dt><tt>ceil_mode</tt> : int (default is 0)</dt>
<dd>Whether to use ceil or floor (default) to compute the output shape.</dd>
<dt><tt>dilations</tt> : list of ints</dt>
<dd>Dilation value along each spatial axis of filter. If not present, the dilation defaults to 1 along each spatial axis.</dd>
<dt><tt>kernel_shape</tt> : list of ints (required)</dt>
<dd>The size of the kernel along each axis.</dd>
<dt><tt>pads</tt> : list of ints</dt>
<dd>Padding for the beginning and ending along each spatial axis, it can take any value greater than or equal to 0. The value represent the number of pixels added to the beginning and end part of the corresponding axis. `pads` format should be as follow [x1_begin, x2_begin...x1_end, x2_end,...], where xi_begin the number of pixels added at the beginning of axis `i` and xi_end, the number of pixels added at the end of axis `i`. This attribute cannot be used simultaneously with auto_pad attribute. If not present, the padding defaults to 0 along start and end of each spatial axis.</dd>
<dt><tt>storage_order</tt> : int (default is 0)</dt>
<dd>The storage order of the tensor. 0 is row major, and 1 is column major. This attribute is used only to convert an n-tuple index value into a single integer value for producing the second output. </dd>
<dt><tt>strides</tt> : list of ints</dt>
<dd>Stride along each spatial axis. If not present, the stride defaults to 1 along each spatial axis.</dd>
</dl>

#### Outputs (1 - 2)

<dl>
<dt><tt>Y</tt> : T</dt>
<dd>Output data tensor from average or max pooling across the input tensor. Dimensions will vary based on various kernel, stride, and pad sizes. Floor value of the dimension is used</dd>
<dt><tt>Indices</tt> : I</dt>
<dd>Indices tensor from max pooling across the input tensor. The dimensions of indices are the same as output tensor. The values in indices of are the indices of the selected values during pooling. The indices are computed as flatten 1-D tensor, and the indices do not consider padding. So the values in indices are in [0, N x C x D1 x ... x Dn).</dd>
</dl>

#### Type Constraints

<dl>
<dt><tt>T</tt> : tensor(float16), tensor(float), tensor(double), tensor(int8), tensor(uint8)</dt>
<dd>Constrain input and output types to float and 8 bit tensors.</dd>
<dt><tt>I</tt> : tensor(int64)</dt>
<dd>Constrain index tensor to int64</dd>
</dl>

#### Axelera's notes for developers

Only MaxPool operators with explicit padding (i.e., auto_pad = "NOTSET") and row major order (i.e. storage_order=0) are currently supported.

##### AIPU Acceleration Constraints

<dl>
<dt><tt>rule</tt> : auto_pad=="NOTSET"</dt><dt><tt>rule</tt> : storage_order==0</dt></dl>

---
### Mul

  Performs element-wise binary multiplication (with Numpy-style broadcasting support).

  This operator supports **multidirectional (i.e., Numpy-style) broadcasting**.

  (Opset 14 change): Extend supported types to include uint8, int8, uint16, and int16.

#### Parameters

<dl>
<dt><tt>A</tt> : T</dt>
<dd>First operand.</dd>
<dt><tt>B</tt> : T</dt>
<dd>Second operand.</dd>
</dl>

#### Outputs

<dl>
<dt><tt>C</tt> : T</dt>
<dd>Result, has same element type as two inputs</dd>
</dl>

#### Type Constraints

<dl>
<dt><tt>T</tt> : tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(bfloat16)</dt>
<dd>Constrain input and output types to all numeric tensors.</dd>
</dl>

#### Axelera's notes for developers

Given an operand with shape [N, C, H, W], Multiplication is supported with other operands with shape [N, C, H, W], [1, C, 1, 1], and scalars.

##### AIPU Acceleration Constraints

<dl>
<dt><tt>allow_config</tt> : A.shape == B.shape</dt><dt><tt>allow_config</tt> : len(A.shape)==4 and A.shape[1]==1 and B.shape==(0)</dt><dt><tt>allow_config</tt> : len(B.shape)==4 and B.shape[1]==1 and A.shape==(0)</dt><dt><tt>allow_config</tt> : len(A.shape)==4 and A.shape[1]!=1 and B.shape==(1, A.shape[1], 1, 1)</dt><dt><tt>allow_config</tt> : len(B.shape)==4 and B.shape[1]!=1 and A.shape==(1, B.shape[1], 1, 1)</dt></dl>

---
### PRelu

  PRelu takes input data (Tensor<T>) and slope tensor as input, and produces one
  output data (Tensor<T>) where the function `f(x) = slope * x for x < 0`,
  `f(x) = x for x >= 0`., is applied to the data tensor elementwise.
  This operator supports **unidirectional broadcasting** (tensor slope should be unidirectional broadcastable to input tensor X).

#### Parameters

<dl>
<dt><tt>X</tt> : T</dt>
<dd>Input tensor</dd>
<dt><tt>slope</tt> : T</dt>
<dd>Slope tensor. The shape of slope can be smaller than first input X; if so, its shape must be unidirectional broadcastable to X</dd>
</dl>

#### Outputs

<dl>
<dt><tt>Y</tt> : T</dt>
<dd>Output tensor (same size as X)</dd>
</dl>

#### Type Constraints

<dl>
<dt><tt>T</tt> : tensor(float16), tensor(float), tensor(double), tensor(uint32), tensor(uint64), tensor(int32), tensor(int64)</dt>
<dd>Constrain input and output types to float/int tensors.</dd>
</dl>

#### Axelera's notes for developers

Due to torch runtime constraints, Prelu is supported with either scalar or per-channel slope parameters.

##### AIPU Acceleration Constraints

<dl>
<dt><tt>allow_config</tt> : slope.size == 1</dt><dt><tt>allow_config</tt> : np.array_equal([x for x in slope.shape if x != 1], [X.shape[1]])</dt></dl>

---
### Pad

  Given a tensor containing the data to be padded (`data`), a tensor containing the number of start and end pad values for axis (`pads`), (optionally) a `mode`, and (optionally) `constant_value`,
  a padded tensor (`output`) is generated.

  The three supported `modes` are (similar to corresponding modes supported by `numpy.pad`):

  1) `constant`(default) - pads with a given constant value as specified by `constant_value` (which defaults to 0, empty string, or False)

  2) `reflect` - pads with the reflection of the vector mirrored on the first and last values of the vector along each axis

  3) `edge` - pads with the edge values of array


  Example 1 (`constant` mode):
    Insert 0 pads to the beginning of the second dimension.

    data =
    [
        [1.0, 1.2],
        [2.3, 3.4],
        [4.5, 5.7],
    ]

    pads = [0, 2, 0, 0]

    mode = 'constant'

    constant_value = 0.0

    output =
    [
        [0.0, 0.0, 1.0, 1.2],
        [0.0, 0.0, 2.3, 3.4],
        [0.0, 0.0, 4.5, 5.7],
    ]


  Example 2 (`reflect` mode):
    data =
    [
        [1.0, 1.2],
        [2.3, 3.4],
        [4.5, 5.7],
    ]

    pads = [0, 2, 0, 0]

    mode = 'reflect'

    output =
    [
        [1.0, 1.2, 1.0, 1.2],
        [2.3, 3.4, 2.3, 3.4],
        [4.5, 5.7, 4.5, 5.7],
    ]


  Example 3 (`edge` mode):
    data =
    [
        [1.0, 1.2],
        [2.3, 3.4],
        [4.5, 5.7],
    ]

    pads = [0, 2, 0, 0]

    mode = 'edge'

    output =
    [
        [1.0, 1.0, 1.0, 1.2],
        [2.3, 2.3, 2.3, 3.4],
        [4.5, 4.5, 4.5, 5.7],
    ]


#### Parameters (2 - 3)

<dl>
<dt><tt>data</tt> : T</dt>
<dd>Input tensor.</dd>
<dt><tt>pads</tt> : tensor(int64)</dt>
<dd>Tensor of integers indicating the number of padding elements to add or remove (if negative) at the beginning and end of each axis. For 2D input tensor, it is the number of pixels. `pads` should be a 1D tensor of shape [2 * input_rank]. `pads` format should be: [x1_begin, x2_begin,...,x1_end, x2_end,...], where xi_begin is the number of pad values added at the beginning of axis `i` and xi_end, the number of pad values added at the end of axis `i`.</dd>
<dt><tt>constant_value</tt> : T</dt>
<dd>(Optional) A scalar value to be used if the mode chosen is `constant` (by default it is 0, empty string or False).</dd>
<dt><tt>mode</tt> : string (default is constant)</dt>
<dd>Supported modes: `constant`(default), `reflect`, `edge`</dd>
</dl>

#### Outputs

<dl>
<dt><tt>output</tt> : T</dt>
<dd>Tensor after padding.</dd>
</dl>

#### Type Constraints

<dl>
<dt><tt>T</tt> : tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)</dt>
<dd>Constrain input and output types to all tensor types.</dd>
</dl>

#### Axelera's notes for developers

Pad is currently supported only for the "constant" mode. Moreover, padding along the batch and channel dimensions is not supported, and should be specified as 0 in the pads parameter.

##### AIPU Acceleration Constraints

<dl>
<dt><tt>rule</tt> : mode not in ["reflect", "edge"]</dt><dt><tt>rule</tt> : pads[:2] == [0, 0]</dt><dt><tt>rule</tt> : pads[2] == pads[3]</dt></dl>

---
### Relu

  Relu takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where the rectified linear function, y = max(0, x), is applied to
  the tensor elementwise.

#### Parameters

<dl>
<dt><tt>X</tt> : T</dt>
<dd>Input tensor</dd>
</dl>

#### Outputs

<dl>
<dt><tt>Y</tt> : T</dt>
<dd>Output tensor</dd>
</dl>

#### Type Constraints

<dl>
<dt><tt>T</tt> : tensor(float), tensor(int32), tensor(int8), tensor(int16), tensor(int64), tensor(float16), tensor(double), tensor(bfloat16)</dt>
<dd>Constrain input and output types to signed numeric tensors.</dd>
</dl>

#### Axelera's notes for developers


##### AIPU Acceleration Constraints

<dl>
<dt><tt>Operator is supported in any configurations.</tt>
</dl>

---
### Resize

  Resize the input tensor. In general, it calculates every value in the output tensor as a weighted average of neighborhood (a.k.a. sampling locations) in the input tensor.
  Each dimension value of the output tensor is:
    output_dimension = floor(input_dimension * (roi_end - roi_start) * scale) if input \"sizes\" is not specified.

#### Parameters (1 - 4)

<dl>
<dt><tt>X</tt> : T1</dt>
<dd>N-D tensor</dd>
<dt><tt>roi</tt> : T2</dt>
<dd>1-D tensor given as [start1, ..., startN, end1, ..., endN], where N is the rank of X. The RoIs' coordinates are normalized in the coordinate system of the input image. It only takes effect when coordinate_transformation_mode is "tf_crop_and_resize"</dd>
<dt><tt>scales</tt> : tensor(float)</dt>
<dd>The scale array along each dimension. It takes value greater than 0. If it's less than 1, it's sampling down, otherwise, it's upsampling. The number of elements of 'scales' should be the same as the rank of input 'X'. One of 'scales' and 'sizes' MUST be specified and it is an error if both are specified. If 'sizes' is needed, the user can use an empty string as the name of 'scales' in this operator's input list.</dd>
<dt><tt>sizes</tt> : tensor(int64)</dt>
<dd>The size of the output tensor. The number of elements of 'sizes' should be the same as the rank of input 'X'. Only one of 'scales' and 'sizes' can be specified.</dd>
<dt><tt>coordinate_transformation_mode</tt> : string (default is half_pixel)</dt>
<dd>
This attribute describes how to transform the coordinate in the resized tensor to the coordinate in the original tensor. <br/>

The coordinate of each dimension is transformed individually. Let's describe a case using axis x as an example.
Denote x_resized as the coordinate of axis x in the resized tensor, x_original as the coordinate of axis x in the original tensor, length_original as the length of the original tensor in axis x, length_resized as the length of the resized tensor in axis x, roi_x = (start_x, end_x) of the axis x in input "roi", scale = length_resized / length_original, <br/>

if coordinate_transformation_mode is "half_pixel", <br/>
x_original = (x_resized + 0.5) / scale - 0.5, <br/>

if coordinate_transformation_mode is "pytorch_half_pixel", <br/>
x_original = length_resized > 1 ? (x_resized + 0.5) / scale - 0.5 : 0, <br/>

if coordinate_transformation_mode is "align_corners", <br/>
x_original = x_resized * (length_original - 1) / (length_resized - 1), <br/>

if coordinate_transformation_mode is "asymmetric", <br/>
x_original = x_resized / scale, <br/>

if coordinate_transformation_mode is "tf_crop_and_resize", <br/>
x_original = length_resized > 1 ? start_x * (length_original - 1) + x_resized * (end_x - start_x) * (length_original - 1) / (length_resized - 1) : 0.5 * (start_x + end_x) * (length_original - 1).</dd>
<dt><tt>cubic_coeff_a</tt> : float (default is -0.75)</dt>
<dd>The coefficient 'a' used in cubic interpolation. Two common choice are -0.5 (in some cases of TensorFlow) and -0.75 (in PyTorch). Check out Equation (4) in https://ieeexplore.ieee.org/document/1163711 for the details. This attribute is valid only if "mode" is "cubic".</dd>
<dt><tt>exclude_outside</tt> : int (default is 0)</dt>
<dd>If set to 1, the weight of sampling locations outside the tensor will be set to 0 and the weight will be renormalized so that their sum is 1.0. The default value is 0.</dd>
<dt><tt>extrapolation_value</tt> : float (default is 0.0)</dt>
<dd>When coordinate_transformation_mode is "tf_crop_and_resize" and x_original is outside the range [0, length_original - 1], this value is used as the corresponding output value. Default is 0.0f.</dd>
<dt><tt>mode</tt> : string (default is nearest)</dt>
<dd>Three interpolation modes: nearest (default), linear and cubic. The "linear" mode includes linear interpolation for 1D tensor and N-linear interpolation for N-D tensor (for example, bilinear interpolation for 2D tensor). The "cubic" mode includes cubic interpolation for 1D tensor and N-cubic interpolation for N-D tensor (for example, bicubic interpolation for 2D tensor).</dd>
<dt><tt>nearest_mode</tt> : string (default is round_prefer_floor)</dt>
<dd>Four modes: round_prefer_floor (default, as known as round half down), round_prefer_ceil (as known as round half up), floor, ceil. Only used by nearest interpolation. It indicates how to get "nearest" pixel in input tensor from x_original, so this attribute is valid only if "mode" is "nearest".</dd>
</dl>

#### Outputs

<dl>
<dt><tt>Y</tt> : T1</dt>
<dd>N-D tensor after resizing</dd>
</dl>

#### Type Constraints

<dl>
<dt><tt>T1</tt> : tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)</dt>
<dd>Constrain input 'X' and output 'Y' to all tensor types.</dd>
<dt><tt>T2</tt> : tensor(float16), tensor(float), tensor(double)</dt>
<dd>Constrain roi type to float or double.</dd>
</dl>

#### Axelera's notes for developers

Resize is currently supported for the nearest and linear modes. The roi parameter is not supported. Linear resizing is only supported for symmetric, integer scaling factors. Nearest is supported for all cases.

##### AIPU Acceleration Constraints

<dl>
<dt><tt>rule</tt> : roi is None</dt><dt><tt>rule</tt> : mode in ["nearest", "linear"]</dt><dt><tt>rule</tt> : (mode == "linear" and Y.shape[-2] % X.shape[-2] == Y.shape[-1] % X.shape[-1]) or mode == "nearest"</dt><dt><tt>rule</tt> : (mode == "linear" and Y.shape[-2] % X.shape[-2] == 0) or mode == "nearest"</dt><dt><tt>rule</tt> : (mode == "linear" and Y.shape[-1] % X.shape[-1] == 0) or mode == "nearest"</dt></dl>

---
### Selu

  Selu takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where the scaled exponential linear unit function,
  `y = gamma * (alpha * e^x - alpha) for x <= 0`, `y = gamma * x for x > 0`,
  is applied to the tensor elementwise.

#### Parameters

<dl>
<dt><tt>X</tt> : T</dt>
<dd>Input tensor</dd>
<dt><tt>alpha</tt> : float (default is 1.67326)</dt>
<dd>Coefficient of SELU default to 1.67326319217681884765625 (i.e., float32 approximation of 1.6732632423543772848170429916717).</dd>
<dt><tt>gamma</tt> : float (default is 1.0507)</dt>
<dd>Coefficient of SELU default to 1.05070102214813232421875 (i.e., float32 approximation of 1.0507009873554804934193349852946).</dd>
</dl>

#### Outputs

<dl>
<dt><tt>Y</tt> : T</dt>
<dd>Output tensor</dd>
</dl>

#### Type Constraints

<dl>
<dt><tt>T</tt> : tensor(float16), tensor(float), tensor(double)</dt>
<dd>Constrain input and output types to float tensors.</dd>
</dl>

#### Axelera's notes for developers

Selu operators are supported if the alpha and gamma parameters are set to the defaults of 1.67326 and 1.0507, respectively.

##### AIPU Acceleration Constraints

<dl>
<dt><tt>rule</tt> : alpha == 1.67326</dt><dt><tt>rule</tt> : gamma == 1.0507</dt></dl>

---
### Sigmoid

  Sigmoid takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where the sigmoid function, y = 1 / (1 + exp(-x)), is applied to the
  tensor elementwise.

#### Parameters

<dl>
<dt><tt>X</tt> : T</dt>
<dd>Input tensor</dd>
</dl>

#### Outputs

<dl>
<dt><tt>Y</tt> : T</dt>
<dd>Output tensor</dd>
</dl>

#### Type Constraints

<dl>
<dt><tt>T</tt> : tensor(float16), tensor(float), tensor(double), tensor(bfloat16)</dt>
<dd>Constrain input and output types to float tensors.</dd>
</dl>

#### Axelera's notes for developers


##### AIPU Acceleration Constraints

<dl>
<dt><tt>Operator is supported in any configurations.</tt>
</dl>

---
### Slice

  Produces a slice of the input tensor along multiple axes. Similar to numpy:
  https://numpy.org/doc/stable/user/basics.indexing.html?highlight=slice#slicing-and-striding

  Slice uses the `starts`, `ends`, `axes` and `steps` inputs to select a sub-tensor
  of its input `data` tensor.

  An effective `starts[i]`, `ends[i]`, and `steps[i]` must be computed for each `i`
  in `[0, ... r-1]` where `r = rank(input)` as follows:

  If `axes` are omitted, they are set to `[0, ..., r-1]`.
  If `steps` are omitted, they are set to `[1, ..., 1]` of length `len(starts)`

  The effective values are initialized as `start[i] = 0`, `ends[i] = dims[i]` where
  `dims` are the dimensions of `input` and `steps[i] = 1`.

  All negative elements of `axes` are made non-negative by adding `r` to them, where
  `r =rank(input)`.

  All negative values in `starts[i]` and `ends[i]` have `dims[axes[i]]` added to them,
  where `dims` are the dimensions of `input`. Then `start[axes[i]]` is the adjusted
  `starts[i]` is clamped into the range `[0, dims[axes[i]]]` for positive stepping
  and `[0, dims[axes[i]]-1]` for negative stepping.

  The clamping for the adjusted `ends[i]` depends on the sign of `steps[i]` and must
  accommodate copying 0 through `dims[axes[i]]` elements, so for positive stepping
  `ends[axes[i]]` is clamped to `[0, dims[axes[i]]]`, while for negative stepping it
  is clamped to `[-1, dims[axes[i]]-1]`.

  Finally, `steps[axes[i]] = steps[i]`.

  For slicing to the end of a dimension with unknown size, it is recommended to pass
  in `INT_MAX` when slicing forward and 'INT_MIN' when slicing backward.

  Example 1:

  ```
  data = [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
  ]
  axes = [0, 1]
  starts = [1, 0]
  ends = [2, 3]
  steps = [1, 2]
  result = [
      [5, 7],
  ]
  ```

  Example 2:

  ```
  data = [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
  ]
  starts = [0, 1]
  ends = [-1, 1000]
  result = [
      [2, 3, 4],
  ]
  ```

#### Parameters (3 - 5)

<dl>
<dt><tt>data</tt> : T</dt>
<dd>Tensor of data to extract slices from.</dd>
<dt><tt>starts</tt> : Tind</dt>
<dd>1-D tensor of starting indices of corresponding axis in `axes`</dd>
<dt><tt>ends</tt> : Tind</dt>
<dd>1-D tensor of ending indices (exclusive) of corresponding axis in `axes`</dd>
<dt><tt>axes</tt> : Tind</dt>
<dd>1-D tensor of axes that `starts` and `ends` apply to. Negative value means counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(data). Behavior is undefined if an axis is repeated.</dd>
<dt><tt>steps</tt> : Tind</dt>
<dd>1-D tensor of slice step of corresponding axis in `axes`. Negative value means slicing backward. 'steps' cannot be 0. Defaults to 1s.</dd>
</dl>

#### Outputs

<dl>
<dt><tt>output</tt> : T</dt>
<dd>Sliced data tensor.</dd>
</dl>

#### Type Constraints

<dl>
<dt><tt>T</tt> : tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)</dt>
<dd>Constrain input and output types to all tensor types.</dd>
<dt><tt>Tind</tt> : tensor(int32), tensor(int64)</dt>
<dd>Constrain indices to integer types</dd>
</dl>

#### Axelera's notes for developers

Slice is supported along one axis only, which must be specified as input to the operator. Stepped slice is currently not supported.

##### AIPU Acceleration Constraints

<dl>
<dt><tt>rule</tt> : steps is None</dt><dt><tt>rule</tt> : axes is not None and len(axes) == 1</dt></dl>

---
### Split

  Split a tensor into a list of tensors, along the specified
  'axis'. Lengths of the parts can be specified using input 'split'.
  Otherwise, the tensor is split to equal sized parts.

#### Parameters (1 - 2)

<dl>
<dt><tt>input</tt> : T</dt>
<dd>The tensor to split</dd>
<dt><tt>split</tt> : tensor(int64)</dt>
<dd>Optional length of each output. Values should be >= 0.Sum of the values must be equal to the dim value at 'axis' specified.</dd>
<dt><tt>axis</tt> : int (default is 0)</dt>
<dd>Which axis to split on. A negative value means counting dimensions from the back. Accepted range is [-rank, rank-1] where r = rank(input).</dd>
</dl>

#### Outputs (1 - &#8734;)

<dl>
<dt><tt>outputs</tt> : T</dt>
<dd>One or more outputs forming list of tensors after splitting</dd>
</dl>

#### Type Constraints

<dl>
<dt><tt>T</tt> : tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)</dt>
<dd>Constrain input and output types to all tensor types.</dd>
</dl>

#### Axelera's notes for developers

Split is supported for axis different than 0. Negative axis values are not supported.

##### AIPU Acceleration Constraints

<dl>
<dt><tt>rule</tt> : axis > 0</dt></dl>

---
### Sub

  Performs element-wise binary subtraction (with Numpy-style broadcasting support).

  This operator supports **multidirectional (i.e., Numpy-style) broadcasting**.

  (Opset 14 change): Extend supported types to include uint8, int8, uint16, and int16.

#### Parameters

<dl>
<dt><tt>A</tt> : T</dt>
<dd>First operand.</dd>
<dt><tt>B</tt> : T</dt>
<dd>Second operand.</dd>
</dl>

#### Outputs

<dl>
<dt><tt>C</tt> : T</dt>
<dd>Result, has same element type as two inputs</dd>
</dl>

#### Type Constraints

<dl>
<dt><tt>T</tt> : tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(bfloat16)</dt>
<dd>Constrain input and output types to all numeric tensors.</dd>
</dl>

#### Axelera's notes for developers

Given an operand with shape [N, C, H, W], Subtraction is supported with other operands with shape [N, C, H, W], [1, C, 1, 1], and scalars.

##### AIPU Acceleration Constraints

<dl>
<dt><tt>allow_config</tt> : A.shape == B.shape</dt><dt><tt>allow_config</tt> : len(A.shape)==4 and A.shape[1]==1 and B.shape==(0)</dt><dt><tt>allow_config</tt> : len(B.shape)==4 and B.shape[1]==1 and A.shape==(0)</dt><dt><tt>allow_config</tt> : len(A.shape)==4 and A.shape[1]!=1 and B.shape==(1, A.shape[1], 1, 1)</dt><dt><tt>allow_config</tt> : len(B.shape)==4 and B.shape[1]!=1 and A.shape==(1, B.shape[1], 1, 1)</dt></dl>

---
### Tanh

  Calculates the hyperbolic tangent of the given input tensor element-wise.

#### Parameters

<dl>
<dt><tt>input</tt> : T</dt>
<dd>Input tensor</dd>
</dl>

#### Outputs

<dl>
<dt><tt>output</tt> : T</dt>
<dd>The hyperbolic tangent values of the input tensor computed element-wise</dd>
</dl>

#### Type Constraints

<dl>
<dt><tt>T</tt> : tensor(bfloat16), tensor(float16), tensor(float), tensor(double)</dt>
<dd>Constrain input and output types to float tensors.</dd>
</dl>

#### Axelera's notes for developers


##### AIPU Acceleration Constraints

<dl>
<dt><tt>Operator is supported in any configurations.</tt>
</dl>

---
### Transpose

  Transpose the input tensor similar to numpy.transpose. For example, when
  perm=(1, 0, 2), given an input tensor of shape (1, 2, 3), the output shape
  will be (2, 1, 3).

#### Parameters

<dl>
<dt><tt>data</tt> : T</dt>
<dd>An input tensor.</dd>
<dt><tt>perm</tt> : list of ints</dt>
<dd>A list of integers. By default, reverse the dimensions, otherwise permute the axes according to the values given.</dd>
</dl>

#### Outputs

<dl>
<dt><tt>transposed</tt> : T</dt>
<dd>Transposed output.</dd>
</dl>

#### Type Constraints

<dl>
<dt><tt>T</tt> : tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(string), tensor(bool), tensor(complex64), tensor(complex128)</dt>
<dd>Constrain input and output types to all tensor types.</dd>
</dl>

#### Axelera's notes for developers

Transpose operations that are not the no-op, trivial case (i.e. perm=[0, 1, 2, 3]), are not supported.

##### AIPU Acceleration Constraints

<dl>
<dt><tt>rule</tt> : perm == [0, 1, 2, 3]</dt></dl>
