![](/docs/images/Ax_Page_Banner_2500x168_01.png)
# decode_yolov5

- [decode\_yolov5](#decode_yolov5)
  - [GST YAML example](#gst-yaml-example)
  - [Description](#description)
  - [Attributes](#attributes)

Decodes YOLOv5 output tensors to detection metadata.

Please find the operator code here:
[decode_yolov5](/operators/src/AxDecodeYolo5.cpp)

## GST YAML example

```yaml
- instance: decode_muxer
  lib: libdecode_yolov5.so
  options: meta_key:yolo;
    anchors:1.25,1.625,2.0,3.75,4.125,2.875,1.875,3.8125,3.875,2.8125
      ,3.6875,7.4375,3.625,2.8125,4.875,6.1875,11.65625,10.1875;
    classes:80;confidence_threshold:0.25;
    scales:0.08142165094614029,0.09499982744455338,0.09290479868650436;
    zero_points:70,82,66;topk:100;multiclass:0;sigmoid_in_postprocess:1;
    transpose:1
```

## Description
In addition to decoding a tensor to metadata, this element optionally also
includes the functionality of the last layer of the YOLO network, which is the
sigmoid functions. In addition, the dequantization of a quantized tensor and
transposition from channel last to channel first can be performed with this
element.

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

*   **anchors**<br>
    *array of float*<br>
    The anchors of the model, can be retrieved from the file `model_info.json`.

*   **classes**<br>
    *int*<br>
    The number of classes that the network is able to identify.

*   **topk**<br>
    *int*<br>
    The number of boxes to be stored, where boxes with low confidence are
    discarded.

*   **multiclass**<br>
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

*   **sigmoid\_in\_postprocess**<br>
    *int*<br>
    If the last layer of the YOLO network containing the sigmoid functions
    should be executed in this subplugin (`1`) or not (`0`).

**Non-tensor data interface** AxVideoInterface
