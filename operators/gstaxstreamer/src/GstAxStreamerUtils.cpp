// Copyright Axelera AI, 2025
#include "GstAxStreamerUtils.hpp"
#include <algorithm>
#include <string>
#include <string_view>
#include <vector>
#include "AxDataInterface.h"
#include "AxInferenceNet.hpp"
#include "AxLog.hpp"
#include "AxStreamerUtils.hpp"

Ax::Severity
Ax::extract_severity_from_category(GstDebugCategory *category)
{
  auto level = gst_debug_category_get_threshold(category);
  switch (level) {
    case GST_LEVEL_NONE:
      return Severity::error;
    case GST_LEVEL_ERROR:
      return Severity::error;
    case GST_LEVEL_WARNING:
      return Severity::warning;
    case GST_LEVEL_FIXME:
      return Severity::fixme;
    case GST_LEVEL_INFO:
      return Severity::info;
    case GST_LEVEL_DEBUG:
      return Severity::debug;
    case GST_LEVEL_LOG:
      return Severity::log;
    case GST_LEVEL_TRACE:
      return Severity::trace;
    default:
      return Severity::error;
  }
}

void
Ax::init_logger(Ax::Logger &logger)
{
  logger.init_log_sink(Ax::Severity::trace,
      [](Ax::SeverityTag severity, Ax::Tag tag, const std::string &message) {
        static std::unordered_map<Ax::Severity, GstDebugLevel> severity_xlate = {
          { Ax::Severity::trace, GST_LEVEL_TRACE },
          { Ax::Severity::log, GST_LEVEL_LOG },
          { Ax::Severity::debug, GST_LEVEL_DEBUG },
          { Ax::Severity::info, GST_LEVEL_INFO },
          { Ax::Severity::fixme, GST_LEVEL_FIXME },
          { Ax::Severity::warning, GST_LEVEL_WARNING },
          { Ax::Severity::error, GST_LEVEL_ERROR },
        };

        auto pos = severity_xlate.find(severity.severity);
        if (pos != severity_xlate.end()) {
          gst_debug_log(static_cast<GstDebugCategory *>(tag.category),
              pos->second, severity.file.c_str(), severity.function.c_str(),
              severity.line, (GObject *) tag.ptr, "%s", message.c_str());
        }
      });
}

std::string
Ax::get_string(const GValue *value, const std::string &what)
{
  if (G_VALUE_HOLDS_STRING(value)) {
    return g_value_get_string(value);
  }
  throw std::runtime_error(what + " must be a string");
}

bool
Ax::set_inference_property(InferenceProperties &props, int prop_id, const GValue *value)
{
  switch (prop_id) {
    case AXINFERENCE_PROP_MODEL:
      props.model = get_string(value, "model");
      break;

    case AXINFERENCE_PROP_DMABUF_INPUTS:
      props.dmabuf_inputs = g_value_get_boolean(value);
      break;

    case AXINFERENCE_PROP_DMABUF_OUTPUTS:
      props.dmabuf_outputs = g_value_get_boolean(value);
      break;

    case AXINFERENCE_PROP_DOUBLE_BUFFER:
      props.double_buffer = g_value_get_boolean(value);
      break;

    case AXINFERENCE_PROP_NUM_CHILDREN:
      props.num_children = g_value_get_int(value);
      break;

    case AXINFERENCE_PROP_INFERENCE_SKIP_RATE:
      {
        const auto s = get_string(value, "skip_rate");
        const auto skip_rate = Ax::parse_skip_rate(s);
        props.skip_count = skip_rate.count;
        props.skip_stride = skip_rate.stride;
        break;
      }

    case AXINFERENCE_PROP_OPTIONS:
      props.options = get_string(value, "options");
      break;

    case AXINFERENCE_PROP_META_STRING:
      props.meta = get_string(value, "meta");
      break;

    case AXINFERENCE_PROP_DEVICES:
      props.devices = get_string(value, "devices");
      break;

    default:
      return false;
  }
  return true;
}

bool
Ax::get_inference_property(const InferenceProperties &props, int prop_id, GValue *value)
{
  switch (prop_id) {
    case AXINFERENCE_PROP_MODEL:
      g_value_set_string(value, props.model.c_str());
      break;

    case AXINFERENCE_PROP_DMABUF_INPUTS:
      g_value_set_boolean(value, props.dmabuf_inputs);
      break;

    case AXINFERENCE_PROP_DMABUF_OUTPUTS:
      g_value_set_boolean(value, props.dmabuf_outputs);
      break;

    case AXINFERENCE_PROP_DOUBLE_BUFFER:
      g_value_set_boolean(value, props.double_buffer);
      break;

    case AXINFERENCE_PROP_NUM_CHILDREN:
      g_value_set_int(value, props.num_children);
      break;

    case AXINFERENCE_PROP_INFERENCE_SKIP_RATE:
      {
        const auto s = std::to_string(props.skip_count) + "/"
                       + std::to_string(props.skip_stride);
        g_value_set_string(value, s.c_str());
        break;
      }

    case AXINFERENCE_PROP_OPTIONS:
      g_value_set_string(value, props.options.c_str());
      break;

    case AXINFERENCE_PROP_META_STRING:
      g_value_set_string(value, props.meta.c_str());
      break;

    case AXINFERENCE_PROP_DEVICES:
      g_value_set_string(value, props.devices.c_str());
      break;

    default:
      return false;
  }
  return true;
}

void
Ax::add_string_property(GObjectClass *object_klass, int id,
    const std::string &name, const std::string &blurb)
{
  g_object_class_install_property(object_klass, id,
      g_param_spec_string(name.c_str(), (name + " string").c_str(),
          blurb.c_str(), "", G_PARAM_READWRITE));
}


void
Ax::add_inference_properties(GObjectClass *object_klass,
    bool include_dmabuf_outputs, bool include_inference_skip_rate)
{
  const InferenceProperties defaults;
  add_string_property(object_klass, AXINFERENCE_PROP_MODEL, "model",
      "String containing lib path to model shared library");
  g_object_class_install_property(object_klass, AXINFERENCE_PROP_DOUBLE_BUFFER,
      g_param_spec_boolean("double_buffer", "whether double buffering is enabled",
          "Whether the model has double buffering enabled.",
          defaults.double_buffer, G_PARAM_READWRITE));
  g_object_class_install_property(object_klass, AXINFERENCE_PROP_DMABUF_INPUTS,
      g_param_spec_boolean("dmabuf_inputs", "use dmabuf_inputs",
          "Use dmabuf for model input", defaults.dmabuf_inputs, G_PARAM_READWRITE));

  if (include_dmabuf_outputs) {
    g_object_class_install_property(object_klass, AXINFERENCE_PROP_DMABUF_OUTPUTS,
        g_param_spec_boolean("dmabuf_outputs", "use dmabuf_outputs",
            "Use dmabuf for model input", defaults.dmabuf_outputs, G_PARAM_READWRITE));
  }

  g_object_class_install_property(object_klass, AXINFERENCE_PROP_NUM_CHILDREN,
      g_param_spec_int("num_children", "num_children int", "Number of child processes",
          0, 4, defaults.num_children, G_PARAM_READWRITE));

  if (include_inference_skip_rate) {
    add_string_property(object_klass, AXINFERENCE_PROP_INFERENCE_SKIP_RATE, "inference_skip_rate",
        "Inference skip rate. For A/B, skip inference in A out of B frames, 0, None or absent == don't skip.\n"
        "e.g for skip_count/skip_stride 4/5 means 4/5 of frames will be skipped.\n"
        "e.g for skip_count/skip_stride 1/5 means 1/5 of frames will be skipped.\n"
        "e.g for skip_count/skip_stride 0/1 means no frames will be skipped");
  }
  add_string_property(object_klass, AXINFERENCE_PROP_OPTIONS, "options",
      "Extra options for inference element");
  add_string_property(object_klass, AXINFERENCE_PROP_DEVICES, "devices", "Devices to connect to");
}


void
Ax::ensure_input_tensors_compatible(
    GstTensorsConfig &given_inputs, const AxTensorsInterface &model_inputs)
{
  // nn tensor config info (nn_config.info.info[N].dimension) is array uint32[8]
  // arranged in opposite order to AxTensorInterface sizes. It also has no
  // explicit length, it is padded with ones.  e.g.
  //  numpy shape   nn_config.info.info[N].dimension
  //  (1024,)       (1024, 1, 1, 1, 1, 1, 1, 1)  \ note these two are
  //  (1, 1024)     (1024, 1, 1, 1, 1, 1, 1, 1)  / indistinguishable
  //  (480, 640, 3) (3, 640, 480, 1, 1, 1, 1, 1)
  if (model_inputs.size() > std::size(given_inputs.info.info)) {
    throw std::runtime_error(
        "Model has more inputs (" + std::to_string(model_inputs.size()) + ") than supported ("
        + std::to_string(std::size(given_inputs.info.info)) + ")");
  }
  if (model_inputs.size() != given_inputs.info.num_tensors) {
    throw std::runtime_error(
        "Input num tensors " + std::to_string(given_inputs.info.num_tensors)
        + " != Model num tensors " + std::to_string(model_inputs.size()));
  }
  for (auto &&[i, model] : Ax::Internal::enumerate(model_inputs)) {
    const auto &nn_info = given_inputs.info.info[i];
    const auto model_shape = Ax::Internal::join(model.sizes, ",");
    if (model.sizes.size() > std::size(nn_info.dimension)) {
      throw std::runtime_error("Model input #" + std::to_string(i) + " ("
                               + model_shape + ") has more dimensions than supported ("
                               + std::to_string(std::size(nn_info.dimension)) + ")");
    }
    const auto ndims = std::max(model.sizes.size(),
        static_cast<size_t>(std::distance(std::begin(nn_info.dimension),
            std::find(std::begin(nn_info.dimension), std::end(nn_info.dimension), 1))));
    const auto rb
        = std::make_reverse_iterator(std::next(nn_info.dimension.begin(), ndims));
    const auto re = std::make_reverse_iterator(nn_info.dimension.begin());
    const auto given_shape = Ax::Internal::join(rb, re, ",");
    if (model_shape != given_shape) {
      throw std::runtime_error("Model input #" + std::to_string(i) + " shape (" + model_shape
                               + ") != given input shape (" + given_shape + ")");
    }
  }
}
