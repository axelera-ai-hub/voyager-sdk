![](/docs/images/Ax_Page_Banner_2500x168_01.png)
# decode_ssd2

- [decode\_ssd2](#decode_ssd2)
  - [GST YAML example](#gst-yaml-example)
  - [Description](#description)
  - [Attributes](#attributes)

Decodes SSD output tensors to detection metadata.

Please find the operator code here:
[decode_ssd2](/operators/src/AxDecodeSsd2.cpp)

## GST YAML example

```yaml
- instance: decode_muxer
  lib: libdecode_ssd2.so
  options: meta_key:ssd;confidence_threshold:0.4;classes:90;max_boxes:1000;
    scales:0.9;zero_points:0;transpose:1;class_agnostic:1
```

## Description
In addition to decoding a tensor to metadata, this element also includes the
functionality of the last layer of the SSD network, where the choice is between
sigmoid or softmax functions. In addition, depadding, dequantization of a
quantized tensor and transposition from channel last to channel first are
performed with this element.

## Attributes
*   **meta\_key**<br>
    *string*<br>
    The key in the metadata map where the resulting bounding boxes are to be
    stored.

*   **zero\_points**<br>
    *array of int*<br>
    The dequantization zeropoint parameters per tensor from `manifest.json`.

*   **scales**<br>
    *arroy of float*<br>
    The dequantization scale parameters per tensor from `manifest.json`.

*   **saved\_anchors**<br>
    *string*<br>
    If no value is given here, anchors will be generated, else it will open the
    given file and read anchors from there.

*   **classes**<br>
    *int*<br>
    The number of classes that the network is able to identify.

*   **topk**<br>
    *int*<br>
    The number of boxes to be stored, where boxes with low confidence are
    discarded.

*   **class\_agnostic**<br>
    *int*<br>
    If for one box all classes should be considered (`1`) or only the class with
    the highest confidence (`0`).

*   **confidence\_threshold**<br>
    *float*<br>
    Bounding boxes with a confidence below this threshold will be discarded.

*   **transpose**<br>
    *int*<br>
    If the input tensor has to be transposed from NHWC to NCHW (`1`) or if it is
    already in NCHW format (`0`).

*   **label\_filter**<br>
    *array of int*<br>
    If this property contains any value or list of values, only the classes
    corresponding to those values (class IDs) will be kept.

*   **softmax**<br>
    *int*<br>
    If the final activation layer is using softmax functions (`1`) or sigmoid functions (`0`).

*   **row\_major**<br>
    *int*<br>
    If the boxes inside the inference result are encoded in yxhw format (`1`) or xywh (`0`).

*   **padding**<br>
    *array of int*<br>
    The padding to be removed from beginning and end of each dimension of each tensor.

**Non-tensor data interface** AxVideoInterface
