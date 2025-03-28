![](/docs/images/Ax_Page_Banner_2500x168_01.png)
# decode_classification

- [decode\_classification](#decode_classification)
  - [GST YAML example](#gst-yaml-example)
    - [Description](#description)
  - [Attributes](#attributes)

Decodes tensor output of classifiers to metadata.

Please find the operator code here:
[decode_classification](/operators/src/AxDecodeClassification.cpp)

## GST YAML example

```yaml
- instance: decode_muxer
  lib: libdecode_classification.so
  options: meta_key:resnet;classlabels_file:labels.txt;top_k:5;softmax:0
```

### Description
The resulting tensor of classification networks like Resnet or Squeezenet can be
decoded either to `AxMetaClassification` or added to existing
`AxMetaObjDetection`. The latter is useful in a cascaded pipeline where a first
network has already completed object detection and the second network is used to
specify the class of the object.

## Attributes
*   **meta\_key**<br>
    *string*<br>
    The key in the metadata map used to insert the result. Unused if box_meta is set.

*   **box\_meta**<br>
    *string*<br>
    Instead of creating a new entry in the metadata map, an existing entry with
    this key and value of type `AxMetaObjDetection` is updated with the
    classification result. The subframe index is used to find the matching box
    in the detection metadata.

*   **classlabels\_file**<br>
    *string*<br>
    A file containing a list of labels with one label per line. The line number
    corresponds to the class ID.

*   **top\_k**<br>
    *int*<br>
    The number of labels to be stored.

*   **sorted**<br>
    *int*<br>
    If the result should be sorted according to confidence, select `1`, else `0`.

*   **largest**<br>
    *int*<br>
    If the labels with largest confidence should appear first, select `1`, else `0`.

*   **softmax**<br>
    *int*<br>
    If the softmax function should be applied to the non-normalized confidence
    levels, select `1`, else `0`.

**Non-tensor data interface** AxVideoInterface
