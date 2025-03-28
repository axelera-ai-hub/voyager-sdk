// Copyright Axelera AI, 2025
#include "AxInference.hpp"
#include "AxStreamerUtils.hpp"

#include <fstream>
#include <regex>

#if defined(AXELERA_ENABLE_AXRUNTIME)
#include <axruntime/axruntime.hpp>

using namespace std::string_literals;
using axr::ptr;
using axr::to_ptr;

AxTensorInterface
to_axtensorinfo(const axrTensorInfo &info)
{
  AxTensorInterface tensor;
  tensor.sizes.assign(info.dims, info.dims + info.ndims);
  tensor.bytes = info.bits / 8;
  return tensor;
}

axr::ptr<axrProperties>
create_properties(axrContext *context, bool input_dmabuf, bool output_dmabuf,
    bool double_buffer, int num_sub_devices)
{
  std::string s;
  s += "input_dmabuf=" + std::to_string(int(input_dmabuf)) + "\n";
  s += "output_dmabuf=" + std::to_string(int(output_dmabuf)) + "\n";
  s += "num_sub_devices=" + std::to_string(num_sub_devices) + "\n";
  s += "aipu_cores=" + std::to_string(num_sub_devices) + "\n";
  s += "double_buffer=" + std::to_string(int(double_buffer)) + "\n";
  return to_ptr(axr_create_properties(context, s.c_str()));
}

axr::ptr<axrProperties>
create_conn_properties(axrContext *context)
{
  std::string s;
  s += "device_name=triton-0:1:0";
  return to_ptr(axr_create_properties(context, s.c_str()));
}

void
log(void *arg, axrLogLevel level, const char *msg)
{
  auto &logger = *static_cast<Ax::Logger *>(arg);
  const auto ax_level = static_cast<Ax::Severity>(level);
  const auto tag = Ax::SeverityTag{ ax_level, {}, 0, {} };
  logger(tag) << msg << std::endl;
}

// Conversion from gst debug level to axr log level
static const axrLogLevel gst_levels[] = {
  AXR_LOG_ERROR, // (pad with error just to make array line up with gst values)
  AXR_LOG_ERROR, // 1=ERROR
  AXR_LOG_WARNING, // 2=WARNING
  AXR_LOG_FIXME, // 3=FIXME
  AXR_LOG_INFO, // 4=INFO
  AXR_LOG_DEBUG, // 5=DEBUG
  AXR_LOG_LOG, // 6=LOG
  AXR_LOG_TRACE, // 7=TRACE
};

axrLogLevel
read_gst_debug_level(const std::string &gst_debug)
{
  auto level = AXR_LOG_WARNING;
  for (auto &debug : Ax::Internal::split(gst_debug, ',')) {
    const auto colon = debug.find(':');
    if (colon == std::string::npos) {
      continue;
    }
    // convert wildcard to regex
    const auto expr = std::regex_replace(
        std::string(debug.substr(0, colon)), std::regex("\\*"), ".*");
    if (std::regex_match("axinference", std::regex(expr))) {
      const auto nlevel = std::stoi(std::string(debug.substr(colon + 1)));
      if (nlevel >= 0 && nlevel < std::size(gst_levels)) {
        level = gst_levels[nlevel];
      }
    }
  }
  return level;
}

axr::ptr<axrContext>
create_context(Ax::Logger &logger)
{
  auto ctx = to_ptr(axr_create_context());
  auto level = read_gst_debug_level(getenv("GST_DEBUG") ? getenv("GST_DEBUG") : "");
  axr_set_logger(ctx.get(), level, log, &logger);
  return ctx;
}


static axrDeviceInfo
find_device(axrContext &context, const std::string &name)
{
  axrDeviceInfo *devices = nullptr;
  if (name.empty()) {
    return {};
  }
  const auto device_count = axr_list_devices(&context, &devices);
  if (device_count == 0) {
    throw std::runtime_error("axr_list_devices failed : "s
                             + axr_last_error_string(AXR_OBJECT(&context)));
  }
  std::vector<std::string> found_devices;
  for (size_t devicen = 0; devicen != device_count; ++devicen) {
    if (name == devices[devicen].name) {
      return devices[devicen];
    }
    found_devices.push_back(devices[devicen].name);
  }
  const auto found = Ax::Internal::join(found_devices, ",");
  throw std::runtime_error("Could not find device " + name + ", but did find " + found);
}

class AxRuntimeInference : public Ax::Inference
{
  public:
  AxRuntimeInference(Ax::Logger &logger, const Ax::InferenceProperties &props)
      : logger(logger), context(create_context(logger)), ctx(context.get()),
        model(axr_load_model(ctx, props.model.c_str()))
  {
    auto inputs = axr_num_model_inputs(model);
    for (int n = 0; n != inputs; ++n) {
      input_shapes_.push_back(to_axtensorinfo(axr_get_model_input(model, n)));
    }
    input_args.resize(inputs);
    auto outputs = axr_num_model_outputs(model);
    for (int n = 0; n != outputs; ++n) {
      output_shapes_.push_back(to_axtensorinfo(axr_get_model_output(model, n)));
    }
    output_args.resize(outputs);
    logger(AX_INFO) << "Loaded model " << props.model << " with " << inputs
                    << " inputs and " << outputs << " outputs" << std::endl;

    auto device = find_device(*context, props.devices);
    auto *pdevice = props.devices.empty() ? nullptr : &device;
    // (batch_size is virtual so don't use it)
    const auto num_sub_devices = input_shapes_.front().sizes.front();
    const auto conn_props = create_conn_properties(ctx);
    connection = axr_device_connect(ctx, pdevice, num_sub_devices, conn_props.get());
    if (!connection) {
      throw std::runtime_error(
          "axr_device_connect failed : "s + axr_last_error_string(AXR_OBJECT(ctx)));
    }
    const auto load_props = create_properties(ctx, props.dmabuf_inputs,
        props.dmabuf_outputs, props.double_buffer, num_sub_devices);
    instance = axr_load_model_instance(connection, model, load_props.get());
    if (!instance) {
      throw std::runtime_error("axr_load_model_instance failed : "s
                               + axr_last_error_string(AXR_OBJECT(ctx)));
    }
  }

  int batch_size() const override
  {
    return input_shapes_.front().sizes.front();
  }

  const AxTensorsInterface &input_shapes() const override
  {
    return input_shapes_;
  }

  const AxTensorsInterface &output_shapes() const override
  {
    return output_shapes_;
  }

  void dispatch(const std::vector<std::shared_ptr<void>> &input_ptrs,
      const std::vector<Ax::SharedFD> &input_fds,
      const std::vector<std::shared_ptr<void>> &output_ptrs,
      const std::vector<Ax::SharedFD> &output_fds) override
  {
    const auto use_fds = input_ptrs.empty();
    assert(use_fds ? input_fds.size() == input_shapes().size() :
                     input_ptrs.size() == input_shapes().size());
    assert(use_fds ? input_ptrs.size() == 0 : input_fds.size() == 0);
    if (use_fds) {
      for (auto &&[i, shared_fd] : Ax::Internal::enumerate(input_fds)) {
        input_args[i].fd = shared_fd->fd;
        input_args[i].ptr = nullptr;
        input_args[i].offset = 0;
      }
    } else {
      for (auto &&[i, ptr] : Ax::Internal::enumerate(input_ptrs)) {
        input_args[i].fd = 0;
        input_args[i].ptr = ptr.get();
        input_args[i].offset = 0;
      }
    }
    if (!output_ptrs.empty()) {
      assert(output_ptrs.size() == output_shapes().size());
      for (auto &&[i, ptr] : Ax::Internal::enumerate(output_ptrs)) {
        output_args[i].fd = 0;
        output_args[i].ptr = ptr.get();
        output_args[i].offset = 0;
      }
    } else if (!output_fds.empty()) {
      assert(output_fds.size() == output_shapes().size());
      int i = 0;
      for (auto &&[i, shared_fd] : Ax::Internal::enumerate(output_fds)) {
        output_args[i].fd = shared_fd->fd;
        output_args[i].ptr = nullptr;
        output_args[i].offset = 0;
      }
    }
    auto res = axr_run_model_instance(instance, input_args.data(),
        input_args.size(), output_args.data(), output_args.size());
    if (res != AXR_SUCCESS) {
      throw std::runtime_error(
          "axr_run_model failed with "s + axr_last_error_string(AXR_OBJECT(model)));
    }
  }

  void collect(const std::vector<std::shared_ptr<void>> &output_ptrs) override
  {
  }

  private:
  Ax::Logger &logger;
  axr::ptr<axrContext> context;
  axrContext *ctx;
  axrModel *model;
  axrConnection *connection;
  axrModelInstance *instance;
  std::vector<axrArgument> input_args;
  std::vector<axrArgument> output_args;
  AxTensorsInterface input_shapes_;
  AxTensorsInterface output_shapes_;
};

std::unique_ptr<Ax::Inference>
Ax::create_axruntime_inference(Ax::Logger &logger, const InferenceProperties &props)
{
  return std::make_unique<AxRuntimeInference>(logger, props);
}
#else
std::unique_ptr<Ax::Inference>
Ax::create_axruntime_inference(Ax::Logger &logger, const InferenceProperties &props)
{
  throw std::runtime_error("Axelera AI runtime not installed at compile time");
}
#endif
