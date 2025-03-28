![](/docs/images/Ax_Page_Banner_2500x168_01.png)
# inplace_addstreamid

- [inplace\_addstreamid](#inplace_addstreamid)
  - [GST YAML example](#gst-yaml-example)
  - [Description](#description)
  - [Attributes](#attributes)

Adds a stream ID to the metadata.

Please find the operator code here:
[inplace_addstreamid](/operators/src/AxInPlaceAddStreamId.cpp)

## GST YAML example

```yaml
- instance: axinplace
  lib: libinplace_addstreamid.so
  mode: meta
  options: stream_id:0
```

## Description
If there are multiple input streams, e.g. several video cameras, and they are
serialized into one stream by a funnel element, the ID of each stream has to be
stored before the funnel such that it can be recovered in the application code.

## Attributes
* **meta_key**<br>
  *string*<br>
  default `stream_id`<br>
  The key in the metadata map. The default value should be used in general.

* **stream_id**<br>
  *int*<br>
  The ID of the stream.

**Mode** `meta`

**Data interface** *unused*
