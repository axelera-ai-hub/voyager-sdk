// Copyright Axelera AI, 2025
// collection of utils taken from axstreamer
#pragma once
#include <functional>
#include <gst/gst.h>
#include <memory>
#include <string>
#include "AxLog.hpp"
#include "GstAxDataUtils.hpp"

namespace Ax
{
Severity extract_severity_from_category(GstDebugCategory *category);
void init_logger(Ax::Logger &logger);

std::string get_string(const GValue *value, const std::string &what);

using gstring = std::unique_ptr<char, decltype(&g_free)>;
inline gstring
as_gstring(gchar *s)
{
  return gstring(s, g_free);
}

template <typename T>
using GstHandle = std::unique_ptr<T, std::function<void(T *)>>;

template <typename T>
GstHandle<T>
as_handle(T *p)
{
  return GstHandle<T>(p, ::gst_object_unref);
}

inline GstHandle<GstCaps>
as_handle(GstCaps *p)
{
  return { p, ::gst_caps_unref };
}

inline GstHandle<GstEvent>
as_handle(GstEvent *event)
{
  return { event, ::gst_event_unref };
}

enum {
  AXINFERENCE_PROP_0,
  AXINFERENCE_PROP_MODEL,
  AXINFERENCE_PROP_DOUBLE_BUFFER,
  AXINFERENCE_PROP_DMABUF_INPUTS,
  AXINFERENCE_PROP_DMABUF_OUTPUTS,
  AXINFERENCE_PROP_NUM_CHILDREN,
  AXINFERENCE_PROP_INFERENCE_SKIP_RATE,
  AXINFERENCE_PROP_OPTIONS,
  AXINFERENCE_PROP_META_STRING,
  AXINFERENCE_PROP_DEVICES,
  AXINFERENCE_PROP_STREAM_SELECT,
  AXINFERENCE_PROP_WHICH_CL,
  AXINFERENCE_PROP_ASYNC_MODE,
  AXINFERENCE_PROP_MAX_INFLIGHT,
  AXINFERENCE_PROP_MAX_PENDING,
  AXINFERENCE_PROP_NEXT_AVAILABLE,
};

struct InferenceProperties;

// returns true if the property was handled
bool set_inference_property(InferenceProperties &props, int prop_id, const GValue *value);
bool get_inference_property(const InferenceProperties &props, int prop_id, GValue *value);
void add_string_property(GObjectClass *object_class, int id,
    const std::string &name, const std::string &blurb);
void add_uint_property(GObjectClass *object_klass, int id, const std::string &name,
    const std::string &blurb, uint32_t min, uint32_t max, uint32_t def);
void add_boolean_property(GObjectClass *object_class, int id,
    const std::string &name, const std::string &blurb);
void add_float_property(GObjectClass *object_klass, int id, const std::string &name,
    const std::string &blurb, float min, float max, float def);

// include_async_mode must only be true for elements that supply a real
// completion callback to create_inference(): AsyncPipelinedInference calls it
// unconditionally, so an element that passes an empty callback (e.g.
// GstAxInference) would crash with std::bad_function_call the first time an
// async completion fires if async_mode were reachable on it.
void add_inference_properties(GObjectClass *object_class, bool include_dmabuf_outputs,
    bool include_inference_skip_rate, bool include_async_mode);

// raise runtime_error if the tensors from gst are not compatible with the ax tensors
void ensure_input_tensors_compatible(
    GstTensorsConfig &nn_config, const AxTensorsInterface &ax_tensors);

struct BufferRequirements {
  uint32_t num_buffers;
  bool supports_opencl;
};

BufferRequirements query_downstream_buffers(GstPad *pad);
void gst_query_set_ax_buffer_requirements(
    GstQuery *query, guint num_buffers, gboolean supports_opencl);

} // namespace Ax
