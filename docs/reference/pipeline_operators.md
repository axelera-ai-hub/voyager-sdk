![](/docs/images/Ax_Page_Banner_2500x168_01.png)
# Axelera GStreamer Pipeline Operators

- [Axelera GStreamer Pipeline Operators](#axelera-gstreamer-pipeline-operators)
  - [Introduction](#introduction)
  - [GST Pipelines](#gst-pipelines)
      - [Example: YOLOv5s Pipeline](#example-yolov5s-pipeline)
  - [Axelera GStreamer Data Structures](#axelera-gstreamer-data-structures)
    - [AxDataInterface](#axdatainterface)
    - [AxMeta](#axmeta)
      - [Example: Bounding Boxes](#example-bounding-boxes)
    - [AxMetaMap](#axmetamap)
  - [Axelera GST Elements - AxPlugins](#axelera-gst-elements---axplugins)
    - [AxPlugins Options](#axplugins-options)
      - [Subplugin Source Code](#subplugin-source-code)
    - [AxInPlace](#axinplace)
      - [Example: inplace\_draw](#example-inplace_draw)
    - [AxTransform](#axtransform)
      - [Example: Resize](#example-resize)
    - [AxDecodeMuxer](#axdecodemuxer)
    - [AxInferenceNet](#axinferencenet)

## Introduction
The Voyager SDK enables users to easily write performant video processing elements for pre- and
post-processing of video streams, both live and prerecorded. Our pipeline implementation is based
on GStreamer (GST), which provides a robust and low-level interface for defining media pipelines.
Typically, GStreamer pipelines are made up of various stages that are composed of **elements**, with
objects called **pads** defining how media is passed down the pipeline, from each element to the
next. We've also extended GStreamer to support neural networks and their inference results. With the
Voyager SDK it is possible to write GStreamer elements without any knowledge of GStreamer. This is
possible through a set of C++ classes and interfaces that Axelera provides with the Voyager SDK,
which offer advanced functionalities and ease of use beyond what is possible in raw GStreamer.

With the SDK, we include a number of elements built with these tools that cover most of the common
use cases for video processing. Examples include cropping, color normalization, and resizing. We
also include elements are able to decode common neural network output formats, such as YOLO bounding
boxes, and deliver all of these to a user's application. In this document, we provide an overview of
our GStreamer pipeline classes and operators to enable users to understand our pipeline and build
their own elements.


## GST Pipelines


In GStreamer video processing, the output of a GST element is passed as a buffer to the element
(aka plugin) that is immediately connected downstream. In general, data is not stored in elements,
nor is data handed to distant elements further downstream. In cascaded pipelines, where, for example,
a detector is followed by a classifier, this limitation poses a difficulty. Axelera's AxPlugin
GStreamer elements, described in this documentation, alleviate this problem by attaching metadata to
a frame which every downstream element can access even if there is no direct connection.

In a chain of neural networks, the second network does not only require data from the first one but
from the input video as well. Thus, one main video chain should exist from start to end.

In the Voyager SDK, each neural network definition includes the required preprocessing, inference,
postprocessing, and decoding, and can be regarded as one segment of the chain of
cascaded networks. The Voyager SDK provides a GST element that represents this segment in total,
called [AxInferenceNet](#axinferencenet).


#### Example: YOLOv5s Pipeline

The following GStreamer pipeline definition gives an example of a complete video pipeline
incorporating our AxPlugins and the YOLOv5s model. It follows the conventional command line
GStreamer syntax.

```bash
./download_prebuilt.py yolov7-coco
GST_PLUGIN_PATH=`pwd`/operators/lib gst-launch-1.0 \
  filesrc location=media/traffic3_480p.mp4 ! \
  decodebin force-sw-decoders=true caps="video/x-raw(ANY)" expose-all-streams=false ! \
  axinplace lib=libinplace_addstreamid.so mode=meta options="stream_id:0" ! \
  axtransform lib=libtransform_colorconvert.so options=format:rgba ! \
  queue max-size-buffers=4 max-size-time=0 max-size-bytes=0 ! \
  axinferencenet \
    model=build/yolov7-coco/yolov7-coco/1/model.json \
    double_buffer=true \
    dmabuf_inputs=true \
    dmabuf_outputs=true \
    num_children=4 \
    preprocess0_lib=libtransform_resize_cl.so \
    preprocess0_options="width:640;height:640;padding:114;letterbox:1;scale_up:1;to_tensor:1;mean:0.,0.,0.;std:1.,1.,1.;quant_scale:0.003921568859368563;quant_zeropoint:-128.0" \
    preprocess1_lib=libtransform_padding.so \
    preprocess1_options="padding:0,0,1,1,1,15,0,0;fill:0" \
    preprocess1_batch=1 \
    postprocess0_lib=libdecode_yolov5.so \
    postprocess0_options="meta_key:detections;anchors:1.5,2.0,2.375,4.5,5.0,3.5,2.25,4.6875,4.75,3.4375,4.5,9.125,4.4375,3.4375,6.0,7.59375,14.34375,12.53125;classes:80;confidence_threshold:0.25;scales:0.003937006928026676,0.003936995752155781,0.003936977591365576;zero_points:-128,-128,-128;topk:30000;multiclass:0;sigmoid_in_postprocess:0;transpose:1;classlabels_file:ax_datasets/labels/coco.names;model_width:640;model_height:640;scale_up:1;letterbox:1" \
    postprocess0_mode=read \
    postprocess1_lib=libinplace_nms.so \
    postprocess1_options="meta_key:detections;max_boxes:300;nms_threshold:0.45;class_agnostic:0;location:CPU" ! \
  videoconvert ! \
  x264enc ! \
  mp4mux ! \
  filesink location=output_video.mp4
```

In a normal pipeline the filesink would be an appsink and the user would access the inference results
through APIs documented below.

Here, the pipeline consists of

* `filesrc` input source.
* `decodebin` to decode the compressed stream.
* `axinplace` to add a stream_id to the meta data for the buffer. This is used to identify 
* `axinferencenet` which contains
  * resize with letterbox, this fuses also the quantization.
  * padding (the Metis requires that buffers have certain alignments and strides).
  * inference on the Metis.
  * a highly optimised yolo postprocessor that includes dequantisation
  * a configurable NMS
* a sequence of elements to ready the video stream suitable to be written to
* `filesink` element

## Axelera GStreamer Data Structures

This section describes the data structures that underlie our pipeline elements.

### AxDataInterface

In GStreamer, **video buffers** are passed from element to element. The description of how the
memory in those buffers is to be interpreted is called capabilities (caps). Video caps will contain
at least information about the width and height of a video frame and the color format. Voyager SDK
also makes use of **tensor buffers** that store the result of inference networks. For these, Voyager
SDK uses the
[same caps as NNStreamer](https://github.com/nnstreamer/nnstreamer/blob/main/Documentation/data-type-and-flow-control.md#gstreamer-data-types-pad-capabilities).
The user does not have to understand the details of the video and tensor formats, as they are
provided by the following simple interface.

```cpp
using AxDataInterface = std::variant<std::monostate, AxTensorsInterface, AxVideoInterface>;
```

The user can check if the interface provides a video or tensor by using the variant syntax of C++.
When accessing the interface, it is usually set as constant. This means the properties like width
and height cannot be changed (as the buffer allocation is separate from the access to the
interface), but the tensor data accessed by a constant pointer is mutable.

The interface to the video buffer contains the following members:

```cpp
struct AxVideoInterface {
  AxVideoInfo info;
  void *data;
};

struct AxVideoInfo {
  int width = 0;
  int height = 0;
  int stride = 0;
  int offset = 0;
  AxVideoFormat format = AxVideoFormat::UNDEFINED;
};

```

This definition closely follows OpenCV. The Voyager interface can be mapped to an OpenCV matrix in
the following way:

```cpp
auto &input_video = std::get<AxVideoInterface>(input);
cv::Mat input_mat(cv::Size(input_video.info.width, input_video.info.height),
      Ax::opencv_type_u8(input_video.info.format), input_video.data,
      input_video.info.stride);
```
The following tensors interface is given as a vector of tensors, where each tensor interface is
defined by its dimensions and the number of bytes of one element:

```cpp
struct AxTensorInterface {
  std::vector<int> sizes;
  int bytes;
  void *data;
  size_t total () const;
};

using AxTensorsInterface = std::vector<AxTensorInterface>;
```

For a single tensor interface with 1 byte uint8 elements, the mapping to OpenCV is

```cpp
cv::Mat input_mat(input_tensor.sizes, CV_8UC1, input_tensor.data);
```

For other element interpretations, the corresponding OpenCV channel types can be used, e.g. CV_32F1
for 4 byte float32.

### AxMeta

AxMeta is the data structure used to store inference results. Examples of its use would be the
bounding box coordinates and class scores for object detection models, or class for classification
models. From the raw inference tensor results described above, an algorithm decodes it into an
instance of AxMeta. A user may use a custom AxMeta and corresponding decoding algorithm, or one of
the included examples if the inference result format is the same. The name is derived from metadata
in GStreamer, which is all data that is not stored in the video/tensor buffer. The following
paragraph describes the details.

For convenience, AxMeta stores inference results as structured objects. The Voyager SDK allows for
those results to be stored in an arbitrary user-defined C++ class. The only requirement is that it
is derived from the (very simple) following base class `AxMetaBase`.


The `extern_meta` struct enables the Voyager SDK's Python code to read the data.
```cpp
struct extern_meta {
  const char *type{}; // Names the type of metadata
  // e.g. object_meta, classification_meta
  const char *subtype{}; //  Names the subtype of metadata
  // e.g. Object Detection contains BBox, scores, class labels
  int meta_size{}; //  Size of this chunk in bytes
  const char *meta{}; //   Pointer to the raw data.
};

class AxMetaBase
{
  public:
  bool enable_draw = true;
  virtual void draw (const AxVideoInterface &video,
    const std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &meta_map)
  {
    throw std::runtime_error ("Draw function missing");
  }
  virtual size_t get_number_of_subframes () const
  {
    return 1;
  }

  virtual std::vector<extern_meta> get_extern_meta () const = 0;

  virtual ~AxMetaBase ()
  {
  }
};
```

The user can overload the method draw or deactivate it. This only controls drawing with an
AxStreamer subplugin. The method to get the number of subframes is relevant for the GST element
AxDistributor. The struct extern_meta enables the user to export the metadata from GST to the Python
world. Important to remember: the metadata is stored per frame. Subsequent frames have completely
independent metadata.

#### Example: Bounding Boxes

The following code snippets form an example on how to derive from AxMetaBase to create a custom
metadata class for bounding boxes. First, we declare the functions and define the member variables
that store our result in a useful form.

```cpp
struct Box {
  int x1;
  int y1;
  int x2;
  int y2;
};

class AxMetaBoxes : public AxMetaBase
{
  public:
  void draw(const AxVideoInterface &video,
    const std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &meta_map)
    override;

  std::vector<extern_meta> get_extern_meta() const override;

  size_t get_number_of_subframes() const override;

  std::vector<box> boxes;
};
```

The member variable is a vector of bounding boxes, so a subframe could be created of each box:

```cpp
inline size_t AxMetaBoxes::get_number_of_subframes() const override
{
  return boxes.size();
}
```

For exporting the vector of boxes to Python, we define a function that creates a vector of
`extern_meta`, as defined in the first code block of this section.

```cpp
inline std::vector<extern_meta> AxMetaBoxes::get_extern_meta() const override
{
  return { { "boxes", "boxes", int(boxes.size() * sizeof(Box)),
      reinterpret_cast<const char *>(boxes.data()) } };
}
```

Finally, we implement a draw function, making use of OpenCV’s `rectangle` function. As an
alternative, we could also just have disabled the need to override the draw function by setting the
member `enable_draw` to `false`.

```cpp
inline void AxMetaBoxes::draw(const AxVideoInterface &video,
  const std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &meta_map)
  override
{
  cv::Mat cv_mat(cv::Size(video.info.width, video.info.height),
    Ax::opencv_type_u8(video.info.format), video.data, video.info.stride);
  const auto black = cv::Scalar(0, 0, 0);
  for (const auto &box : boxes) {
    cv::rectangle(cv_mat,
      cv::Rect(cv::Point(box.x1, box.y1), cv::Point(box.x2, box.y2)), black);
  }
}
```

### AxMetaMap

It is possible to store any number of metadata objects (derived from AxMetaBase) per frame.
A MetaMap stores references to each of those objects. They can be accessed via a string key.
A reference to this map is passed to all of the AxPlugins. AxDecodeMuxer usually creates a new entry
in this map. The AxInPlace element can be used to create or edit only metadata as well and just
ignore the buffer.

```cpp
using MetaMap = std::unordered_map<std::string, std::unique_ptr<AxMetaBase>>;
```

## Axelera GST Elements - AxPlugins

Axelera provides a set of base plugins that expose a simple and intuitive interface to the user, and
the user can write custom code using those interfaces and compile it into a shared library. There
are three GST elements that enable to the user to access and use the interfaces mentioned above:
AxInPlace, AxTransform and AxDecodeMuxer. Those elements differ by the function they have inside the
pipeline: The simplest element is AxInPlace, which just exposes the buffer to the user so that the
buffer can be edited, e.g. to normalize all elements of a tensor. AxTransform always creates a new
buffer whose properties are specified by the user. Resizing a video image is an example where this
element can be used. AxDecodeMuxer has two inputs - this is usually the inference result stored as a
tensor and the original frame. The latter is passed through the element.

These three elements will be described in greater depth later in this document. First, let's look at
how to use them and specify their parameters.


### AxPlugins Options

The AxPlugins AxInPlace, AxTransform and AxDecodeMuxer plugins each define a GStreamer element that
allows the user to take advantage of Axelera's pipeline features. They must be linked to
user-defined code (called subplugins) to fulfill a useful task. A **plugin** is an individual
GStreamer element, and a **subplugin** is a library loaded by a plugin. Here's an example of our
low-level GStreamer YAML showing a plugin built on AxTransform:

```yaml
  - instance: axtransform
    lib: libtransform_totensor.so
    options: type:int8
```

The `instance` is the name of the underlying AxPlugin, while `lib` specifies the shared library file
containing the subplugin to be loaded. The `options` field allows users to specify parameters for
these subplugins. For this purpose, a map of string keys and string values is created and handed
over to the subplugins. The key-value pairs are specified via `options` and separated via semicolon,
while key and value are separated by colon.

This example shows an AxInPlace plugin which specifies Non-Max Suppression for the output of
SSD-MobileNetV1. Note the use of semicolons to separate key-value pairs under the `options` field:
```yaml
  - instance: axinplace
    lib: libinplace_nms.so
    options: meta_key:SSD-MobileNetV1-COCO;nms_threshold:0.5;class_agnostic:1;max_boxes:200
```

A value can also be a list, in which case the list elements are separated by commas. Here, an
AxTransform plugin specifing a padding operation uses commas to specify the padding dimensions:
```yaml
  - instance: axtransform
    lib: libtransform_padding.so
    options: padding:0,0,0,0,3,13,0,1;fill:-14
    batch: 1
```

#### Subplugin Source Code
When defining a subplugin, the following C symbols are used to configure the options of the
subplugins:

All possible `option` keys must be declared in the `allowed_properties` set. This makes sure that
there will be no typos.
```cpp
extern "C" const std::unordered_set<std::string> &
allowed_properties()
{
  static const std::unordered_set<std::string> allowed_properties{
    "width",
    "height",
  };
  return allowed_properties;
}
```

The object holding the properties must be constructed. This is done by the
`init_and_set_static_properties(...)` function, which also receives the map of key value strings
parsed from the GST yaml. After construction of the properties object, it is placed into a shared
pointer and the static properties are set. Static means that when the properties in an existing
pipeline are changed externally via GST calls, they will still not be updated inside the plugins.
```cpp
extern "C" std::shared_ptr<void>
init_and_set_static_properties(
    const std::unordered_map<std::string, std::string> &input, Ax::Logger &logger)
{
  std::shared_ptr<resize_properties> prop
      = std::make_shared<resize_properties>();
  prop->width = Ax::get_property(input, "width", "resize_properties", prop->width);

  return prop;
}
```

In contrast to the function above, the function `set_dynamic_properties(...)` can be used if the
properties are supposed to be mutable, e.g. when experimenting with thresholds in a livestream
(there is no demo code yet which shows this functionality).
```cpp
extern "C" void
set_dynamic_properties(const std::unordered_map<std::string, std::string> &input,
    nms_properties *prop, Ax::Logger &logger)
{
  prop->nms_threshold = Ax::get_property(
      input, "nms_threshold", "nms_dynamic_properties", prop->nms_threshold);
}
```

The signature of this function receives the same map of up-to-date key value strings, and a pointer
to the properties object constructed before.

All property functions are only loaded optionally. If there is no need for properties, they do not
need to be implemented.

### AxInPlace

AxInPlace can do inplace operations on the buffer with or without the help of metadata, or work in
passthrough mode and operate only on the metadata. Use cases are either in-place arithmetic
operations on the video frame, or operations solely on the metadata. Examples of this functionality
include calling a serialization member function, splitting a metadata entry into two metadata
entries, e.g. split face detector output to masked faces and unmasked faces.
```yaml
# GST yaml
  - instance: axinplace
    lib: libinplace_normalize.so
    mode: write
    options: mean:0.485,0.456,0.406;std:0.229,0.224,0.225
```

To implement a subplugin for AxInPlace, either of the following signatures has to be used:

```cpp
extern "C" void
inplace(
    const AxDataInterface &, const void *, unsigned int, unsigned int,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &, Ax::Logger &);
```

AxInPlace is the only AxPlugin that has the GST property `mode`. Depending how the `mode` property
is set, the first function parameter `AxDataInterface` must be used in different ways. If no mode
is set, the data pointer inside the data interface must not be used, but other values, e.g. width
and height, can be used. If `mode: meta` is set, the data interface must not be used at all. If
`mode: read` is set, the data memory can be read via the data pointer, but it is not writable. If in
doubt, always use `mode: write` to have read and write access to the memory in the buffer.

Subplugin properties are passed to the `inplace(...)` function via the second parameter. The `void`
keyword should be replaced by the type of the property that has been used to create the shared
pointer in `init_and_set_static_properties`.

The following two unsigned int parameters are the subframe index and the number of subframes. This
pipeline concept will be discussed in more detail for the `AxDecodeMuxer`.

The metadata map contains all user-defined data of a frame and has been described previously.

Finally, the logger can be used analogously to `std::cout`.

#### Example: inplace_draw

Consider the rendering subplugin as an example implementation of a subplugin for AxInPlace: A loop
over all metadata map entries calls the draw function of the metadata, which then writes onto the
video frame:

```cpp
extern "C" void
inplace(const AxDataInterface &data, const void *, unsigned int, unsigned int,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &map, Ax::Logger &logger)
{
  if (!std::holds_alternative<AxVideoInterface>(data)) {
    throw std::runtime_error("inplace_draw works on video only");
  }

  for (const auto &ele : map) {
    if (ele.second->enable_draw) {
      ele.second->draw(std::get<AxVideoInterface>(data), map);
    }
  }
}
```

The draw function signature is defined in `AxMetaBase`.

Finally, it should be mentioned that AxInPlace can be used without specifying any lib, in which case
it has the function of telling upstream elements in the GST pipeline to attach a GST internal
metadata to the buffer, containing video info that is not included in the caps.

### AxTransform

AxTransform is a generalization of AxInPlace in the sense that it does not operate on an existing
buffer, instead it receives an input buffer and creates a user-defined output buffer. AxTransform
must be used instead of AxInPlace when the input buffer is not suitable to hold the output of the
element. This can happen for example with a crop or resize operation, where the dimensions of the
tensor are different. Another example is the case where the input of an element is a video, but the
output is another kind of data tensor.

```yaml
# GST yaml
  - instance: axtransform
    lib: libtransform_resize.so
    options: size:640;letterbox:1
    batch: 4
```

The user has to define the following function to create the output buffer:

```cpp
extern "C" AxDataInterface
set_output_interface(const AxDataInterface &, const void *, Ax::Logger &);
```

The return value of the function describes the output buffer, while the first function parameter is
the input buffer. Obviously, the pointer to the data inside if the data interface is null, as no
buffers exist yet in this stage of the pipeline. The second parameter is a pointer to the
user-defined properties of the plugin.

There are times when axtransform is able to pass the input buffer through without modification. This
can be a significant performance boost, avoiding creation of a new buffer and moving the data into that
buffer. For example, if the subplugin is performing a color conversion and the the negotiated input and
output capabilities are the same.
axtransform determines whether it can pass through the input by first testing if the subplugin contains
this function and if so calling it.

```cpp
extern "C" bool
can_passthrough(const AxDataInterface &input, const AxDataInterface &output,
    const void *prop, Ax::Logger &logger)
```

A return value of true  allows the input buffer to be passed on to the next element without
modification. The function uses the input and output descriptions and the subplugin properties to
determine whether any operations need to be applied to the input.
If your subplugin can never passthrough the input then there is no need to implement this function.

If the buffer must be transformed then the element ensures that the memory is ready, the element
calls the function `transform` for each (sub)frame, with the signature given below.

```cpp
extern "C" void
transform(const AxDataInterface &, const AxDataInterface &,
    const void *, unsigned int, unsigned int,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &, Ax::Logger &);
```

The only difference between `transform(...)` and `inplace(...)` is the presence of two data
interfaces for the former, the first data interface for the input buffer, the second one for the
output buffer.

The AxTransform plugin has the GST property `batch`. A number larger than 1 can be specified if the
output buffer as defined by `set_output_interface` is a single (i.e. not a vector) tensor (i.e. not
a video) where the size of the first dimension is 1, i.e. the N in NHWC, e.g. (1,640,640,3). If a
batch of, say, 4 is specified, the output tensor after the AxTransform element would then be of size
(4,640,640,3), because the outputs of 4 calls to `transform`, which each resulted in a (1,640,640,3)
tensor, are pasted together to form a (4,640,640,3) tensor. This batched tensor allows for
performance improvements when running inference on multiple AIPU cores.

#### Example: Resize

We want to implement a subplugin which resizes any video input to a fixed size of 640x480 using
OpenCV’s function `resize`. As the output is fixed, we do not need to take care of subplugin
properties. First we have to define the function that returns the output interface of the subplugin.
We also make sure that the input is indeed a video and not a tensor.

```cpp
extern "C" AxDataInterface
set_output_interface(const AxDataInterface &input, const void *, Ax::Logger &)
{
  if (!std::holds_alternative<AxVideoInterface>(input)) {
    throw std::runtime_error("resize works on video only");
  }

  AxDataInterface output = input;
  auto &output_video = std::get<AxVideoInterface>(output);

  output_video.info.width = 640;
  output_video.info.height = 480;

  return output;
}
```

As a next and final step, we define the `transform` function, where we cast the video interface to
an OpenCV matrix and use this matrix in OpenCV’s `resize` function.

```cpp
#include <opencv2/opencv.hpp>

extern "C" void
transform(const AxDataInterface &input, const AxDataInterface &output,
    const *, unsigned int, unsigned int,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &,
    Ax::Logger &)
{
  auto &input_video = std::get<AxVideoInterface>(input);
  cv::Mat input_mat(cv::Size(input_video.info.width, input_video.info.height),
      Ax::opencv_type_u8(input_video.info.format), input_video.data,
      input_video.info.stride);

  auto &output_video = std::get<AxVideoInterface>(output);
  cv::Mat output_mat(cv::Size(output_video.info.width, output_video.info.height),
      Ax::opencv_type_u8(output_video.info.format), output_video.data,
      output_video.info.stride);

  cv::resize(input_mat, output_mat, output_mat.size());
}
```

### AxDecodeMuxer

This element is similar to the previously described elements, however, it has two input buffers and
one output buffer. Its function is to decode the inference result, i.e. to convert the information
that is inside the output tensor after inference and postprocessing into a structured, user-defined
format with appropriate types that can be interpreted easily.

```cpp
extern "C" void
decode_to_meta(const AxTensorsInterface &,
    const void *, unsigned int, unsigned int,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &,
    const AxDataInterface &, Ax::Logger &);
```

The first function parameter is the interface to a tensor, the input buffer of the second sinkpad.
This is supposed to be the tensor that results from inference after postprocessing. After that
follows as second parameter the pointer to the configurable subplugin properties. The fifth
parameter is the reference to the metadata map. The user is expected to create a new map entry to
store the decoded result. The sixth parameter is the data interface to a read-only loop-through
buffer, i.e. the output buffer corresponds to the unchanged input buffer on the first sinkpad. In
most cases, this parameter can be ignored, however, it can be useful in some cases, for example when
scaling bounding boxes given in relative coordinates to the original frame dimensions.

The third and fourth function parameters are the subframe index and the number of subframes.
`AxDecodeMuxer` has the flexibility to progress asynchronously on both sinkpads. For each frame on
the first sinkpad, there can be none, one or any number of subframes on the second sinkpad. The
`decode_to_meta` function is called repeatedly, as many times as there are subframes. This pipeline
concept is useful if there are, as an example, multiple regions of interest in one video frame and
for each of those, a classification network has to be run. This functionality is encapsulated in the
[AxInferenceNet](#axinferencenet) With the help of the number of subframes and the subframe
index, the user knows how many subframes there are in total and which is the current subframe and
can thus plan the data structures for storing the metadata. There is only one instance of the
metadata map per frame (not per subframe). Do not rely on subframes arriving in order.

### AxInferenceNet

AxInferenceNet is a gstreamer element that acts as the central part of an Axelera based GStreamer
pipeline. If performs efficient and low latency inference, whilst preparing the input by chaining
multiple [AxInPlace](#axinplace), [AxTransform](#axtransform), and one [AxDecodeMuxer](#axdecodemuxer).
It acts as a funnel and accepts multiple input pads, one per input stream.

It take a large array of parameters, which are documented in [C++ AxInferenceNet Reference](/docs/reference/axinferencenet.md).  In general the names match the C++ documentation, but note that for simplicity instead
of using an array type for `preproc` and `postproc`, they are named `preprocessN_subproperty`. For example
instead of `preproc[2].lib` the property name is `preprocess2_lib`. Otherwise the functionality is identical.

An example for Yolov7 is

```yaml
# GST yaml
  - instance: axinferencenet
    name: inference-task0
    model: build/yolov7-coco/yolov7-coco/1/model.json
    devices: metis-0:3:0
    double_buffer: true
    dmabuf_inputs: true
    dmabuf_outputs: true
    num_children: 4
    preprocess0_lib: libtransform_resize_cl.so
    preprocess0_options: width:640;height:640;padding:114;letterbox:1;scale_up:1;to_tensor:1;mean:0.,0.,0.;std:1.,1.,1.;quant_scale:0.003921568859368563;quant_zeropoint:-128.0
    preprocess1_lib: libtransform_padding.so
    preprocess1_options: padding:0,0,1,1,1,15,0,0;fill:0
    preprocess1_batch: 1
    postprocess0_lib: libdecode_yolov5.so
    postprocess0_options: meta_key:detections;anchors:1.5,2.0,2.375,4.5,5.0,3.5,2.25,4.6875,4.75,3.4375,4.5,9.125,4.4375,3.4375,6.0,7.59375,14.34375,12.53125;classes:80;confidence_threshold:0.25;scales:0.003937006928026676,0.003936995752155781,0.003936977591365576;zero_points:-128,-128,-128;topk:30000;multiclass:0;sigmoid_in_postprocess:0;transpose:1;model_width:640;model_height:640;scale_up:1;letterbox:1
    postprocess0_mode: read
    postprocess1_lib: libinplace_nms.so
    postprocess1_options: meta_key:detections;max_boxes:300;nms_threshold:0.45;class_agnostic:0;location:CPU
```

#### Cascaded Inference

A cascaded inference configuration is where two AxInferenceNet elements follow one another and the
second one has a meta property. For example to have the second element perform ROI inference on the
detections from the previous example, add the property :

```yaml
    meta: detections
```

When AxInferenceNet is given frames whose meta data has multiple subframes, it will call the inference
once per subframe.  This is indicated by the AxMetaBase class which has a virtual method which returns
the number of subframes per frame.
```cpp
virtual size_t get_number_of_subframes () const;
```

If this number is 1 (e.g. when the virtual method is not overridden) or no meta property is
specified in the GST yaml, the AxInferenceNet element passes through the input buffer. If the number of
subframes is zero (the parent , the element emits a gap event, which will be propagated through the
pipeline and caught by the first `AxDecodeMuxer` operator. In case the number of subframes is larger
than one.
