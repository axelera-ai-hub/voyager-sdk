// Copyright Axelera AI, 2024
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
  AXINFERENCE_PROP_NEXT_AVAILABLE,
};

struct InferenceProperties;

// returns true if the property was handled
bool set_inference_property(InferenceProperties &props, int prop_id, const GValue *value);
bool get_inference_property(const InferenceProperties &props, int prop_id, GValue *value);
void add_string_property(GObjectClass *object_class, int id,
    const std::string &name, const std::string &blurb);


void add_inference_properties(GObjectClass *object_class,
    bool include_dmabuf_outputs, bool include_inference_skip_rate);

// raise runtime_error if the tensors from gst are not compatible with the ax tensors
void ensure_input_tensors_compatible(
    GstTensorsConfig &nn_config, const AxTensorsInterface &ax_tensors);

} // namespace Ax
