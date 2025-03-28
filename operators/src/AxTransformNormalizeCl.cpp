// Copyright Axelera AI, 2023
#include <array>
#include <unordered_map>
#include <unordered_set>
#include "AxDataInterface.h"
#include "AxLog.hpp"
#include "AxMeta.hpp"
#include "AxUtils.hpp"

#include "AxOpenCl.hpp"

const char *kernel_cl = R"##(
__kernel void quantize(const int heightA, const int widthA, const int strideIn, const int strideOut,
__global const uchar4 *in, __global char4 *out, float4 mul, float4 add) {
    const int col = get_global_id(0);
    const int row = get_global_id(1);
    if (row < heightA && col < widthA){
      const int in_idx = row  * (strideIn >> 2) + col;
      const int out_idx = row * (strideOut >> 2) + col;
      out[out_idx] = convert_char4_sat_rte(mad(convert_float4(in[in_idx]), mul, add));
    }
}

)##";

/// This struct represents both video or tensor data
using ax_utils::buffer_details;

using ax_utils::CLProgram;

class CLNormalize
{
  public:
  using buffer = CLProgram::ax_buffer;
  using kernel = CLProgram::ax_kernel;

  CLNormalize(std::string source, Ax::Logger &logger)
      : program(source, logger), quantize{ program.get_kernel("quantize") }
  {
  }

  CLProgram::flush_details run_kernel(
      const kernel &kernel, const buffer_details &out, const buffer &outbuf)
  {
    size_t global_work_size[3] = { 1, 1, 1 };
    global_work_size[0] = out.width;
    global_work_size[1] = out.height;
    error = program.execute_kernel(kernel, 2, global_work_size);
    if (error != CL_SUCCESS) {
      throw std::runtime_error(
          "Unable to execute kernel. Error code: " + std::to_string(error));
    }
    return program.flush_output_buffer_async(outbuf, ax_utils::determine_buffer_size(out));
  }


  std::function<void()> run(const buffer_details &in, const buffer_details &out,
      const std::vector<cl_float> &add, const std::vector<cl_float> &mul)
  {
    auto inpbuf = program.create_buffer(in, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR);
    auto outbuf = program.create_buffer(out, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR);

    program.set_kernel_args(quantize, 0, out.height, out.width, in.stride,
        out.stride, *inpbuf, *outbuf, mul, add);

    auto [error, event, mapped] = run_kernel(quantize, out, outbuf);
    if (error != CL_SUCCESS) {
      throw std::runtime_error(
          "Unable to map output buffer, error = " + std::to_string(error));
    }
    return [this, event, mapped, inpbuf, outbuf]() {
      program.unmap_buffer(event, outbuf, mapped);
    };
  }

  private:
  CLProgram program;
  int error{};
  kernel quantize;
};

struct normalize_properties {
  float quant_scale;
  float quant_zeropoint;
  std::vector<cl_float> add;
  std::vector<cl_float> mul;
  std::unique_ptr<CLNormalize> normalize;
  bool to_tensor{};
};

extern "C" const std::unordered_set<std::string> &
allowed_properties()
{
  static const std::unordered_set<std::string> allowed_properties{
    "mean",
    "std",
    "quant_scale",
    "quant_zeropoint",
    "to_tensor",
  };
  return allowed_properties;
}

extern "C" std::shared_ptr<void>
init_and_set_static_properties(
    const std::unordered_map<std::string, std::string> &input, Ax::Logger &logger)
{
  auto prop = std::make_shared<normalize_properties>();
  prop->to_tensor = Ax::get_property(
      input, "to_tensor", "normalize_dynamic_properties", prop->to_tensor);

  prop->quant_scale = Ax::get_property(
      input, "quant_scale", "normalize_dynamic_properties", prop->quant_scale);
  prop->quant_zeropoint = Ax::get_property(input, "quant_zeropoint",
      "normalize_dynamic_properties", prop->quant_zeropoint);
  auto mean = Ax::get_property(
      input, "mean", "normalize_dynamic_properties", std::vector<cl_float>());
  auto std = Ax::get_property(
      input, "std", "normalize_dynamic_properties", std::vector<cl_float>());
  if (mean.empty() && std.empty()) {
    throw std::runtime_error("mean or std or both must be specified in inplace_normalize");
  }
  if (mean.empty()) {
    mean = std::vector<float>(std.size(), 0.0);
  }
  if (std.empty()) {
    std = std::vector<float>(mean.size(), 1.0);
  }
  if (mean.size() != std.size()) {
    throw std::runtime_error("mean and std must have equal lengths in inplace_normalize");
  }
  const auto max_size = 4;
  auto size = std::max(max_size, static_cast<int>(mean.size()));
  prop->add.resize(size, 0.0F);
  prop->mul.resize(size, 1.0F);
  for (int i = 0; i < mean.size(); ++i) {
    prop->mul[i] = 1.0 / (prop->quant_scale * std[i]);
    prop->add[i] = prop->quant_zeropoint - prop->mul[i] * mean[i];
    constexpr float inv_255 = 1.0 / 255.0;
    prop->mul[i] *= inv_255;
  }
  prop->normalize = std::make_unique<CLNormalize>(kernel_cl, logger);
  return prop;
}

extern "C" void
set_dynamic_properties(const std::unordered_map<std::string, std::string> & /*input*/,
    normalize_properties * /*prop*/, Ax::Logger & /*logger*/)
{
}

extern "C" AxDataInterface
set_output_interface(const AxDataInterface &interface,
    const normalize_properties *prop, Ax::Logger &logger)
{
  if (prop->to_tensor) {
    if (std::holds_alternative<AxVideoInterface>(interface)) {
      auto &info = std::get<AxVideoInterface>(interface).info;
      AxTensorsInterface output = { { { 1, info.height, info.width, 4 }, 1, nullptr } };
      return AxDataInterface(output);
    }
  }
  AxDataInterface output = interface;
  return output;
}


extern "C" std::function<void()>
transform_async(const AxDataInterface &input, const AxDataInterface &output,
    const normalize_properties *prop, unsigned int, unsigned int,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &, Ax::Logger &logger)
{
  //  Here we need to configure the parameters for the OpenCL kernel
  //  and run the kernel
  auto input_details = ax_utils::extract_buffer_details(input);
  if (input_details.size() != 1) {
    throw std::runtime_error("normalize works on single tensor (possibly batched) input only");
  }

  auto output_details = ax_utils::extract_buffer_details(output);
  if (output_details.size() != 1) {
    throw std::runtime_error("normalize works on single tensor (possibly batched) output only");
  }

  return prop->normalize->run(input_details[0], output_details[0], prop->add, prop->mul);
}

extern "C" void
transform(const AxDataInterface &input, const AxDataInterface &output,
    const normalize_properties *prop, unsigned int, unsigned int,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &meta_map,
    Ax::Logger &logger)
{
  logger(AX_WARN) << "Running in synchronous mode, possible performance degradation"
                  << std::endl;
  auto completer = transform_async(input, output, prop, 0, 0, meta_map, logger);
  completer();
}
