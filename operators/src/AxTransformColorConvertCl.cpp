// Copyright Axelera AI, 2024
#include <array>
#include <unordered_map>
#include <unordered_set>
#include "AxDataInterface.h"
#include "AxLog.hpp"
#include "AxMeta.hpp"
#include "AxOpUtils.hpp"
#include "AxOpenCl.hpp"
#include "AxStreamerUtils.hpp"
#include "AxUtils.hpp"

//
// strides hold strides_in[3], strideout
// offsets hold offsets_in[3],
// color_matrix holds the color conversion matrix
//
const char *kernel_sig = R"##(

uchar4 color_convert(uchar4 pixel, float16 matrix) {
    float4 in_pixel = convert_float4(pixel);
    float4 color = mad(in_pixel.x, matrix.s0123, mad(in_pixel.y, matrix.s4567, mad(in_pixel.z, matrix.s89ab, matrix.scdef)));
    color.w = in_pixel.w;
    return convert_uchar4_sat(color);
}

// Utility functions for coordinate transformations
__kernel void informat_to_outformat(int width, int height, int4 strides,
    int4 offsets, int crop_x, int crop_y, float16 color_matrix,
    %s__global %s *out) {
    const int col = get_global_id(0);
    const int row = get_global_id(1);
    if (row >= height || col >= width) {
        return;
    }
)##";


const char *crop_code = R"##(
    corrected.x += crop_x;
    corrected.y += crop_y;
)##";

class CLColorConvert;
struct cc_properties {
  std::string format{ "rgba" };
  std::string flip_method{ "none" };
  mutable std::unique_ptr<CLColorConvert> color_convert;
  mutable int total_time{};
  mutable int num_calls{};
  bool downstream_supports_opencl{};
};

using ax_utils::buffer_details;
using ax_utils::CLProgram;

using ax_utils::opencl_details;

ax_utils::CLProgram::ax_kernel
build_kernel(ax_utils::CLProgram &program, AxVideoFormat in_format,
    AxVideoFormat out_format, int flip_type, int num_planes)
{
  std::string kernel_code = kernel_sig;
  auto input_details = ax_utils::get_input_details(
      in_format, ax_utils::Interpolation::nearest, num_planes);
  auto output_details = ax_utils::get_output_details(in_format, out_format);
  const auto &out_type = output_details.out_type;
  const auto &output_code = output_details.sampler;

  const auto &sampler_code = input_details.sampler;

  auto n = snprintf(nullptr, 0, kernel_code.c_str(),
      input_details.input_params.c_str(), out_type.c_str());
  std::vector<char> buffer(n + 1);
  snprintf(buffer.data(), buffer.size(), kernel_code.c_str(),
      input_details.input_params.c_str(), out_type.c_str());
  auto final_kernel = std::string(buffer.data());
  final_kernel += ax_utils::get_rotation(flip_type);
  final_kernel += crop_code;
  final_kernel += sampler_code;
  final_kernel += output_code;
  final_kernel = ax_utils::get_kernel_utils(flip_type) + final_kernel;
  return program.build_kernel_from_source(final_kernel, "informat_to_outformat");
}

class CLColorConvert
{
  public:
  using buffer = CLProgram::ax_buffer;
  using kernel = CLProgram::ax_kernel;

  CLColorConvert(int flip_type, opencl_details *display, Ax::Logger &logger)
      : program("", display, logger),
        flip_type(flip_type)
  {
  }

  cl_kernel get_converter(ax_utils::CLProgram &program, AxVideoFormat in_format,
      AxVideoFormat out_format, int flip_type, int num_planes)
  {
    auto hash = (static_cast<int>(in_format) << 16)
                + (static_cast<int>(out_format) << 8) + flip_type;
    auto it = std::find_if(std::begin(all_kernels), std::end(all_kernels),
        [hash](auto &x) { return x.hash == hash; });
    if (it != all_kernels.end()) {
      return *it->cl_prog;
    }
    auto k = build_kernel(program, in_format, out_format, flip_type, num_planes);
    return *all_kernels.emplace_back(hash, std::move(k)).cl_prog;
  }

  int run(const buffer_details &in, const buffer_details &out,
      const std::string &format, const cc_properties *prop)
  {
    bool start_flush = prop && prop->downstream_supports_opencl == 0;
    auto num_planes = ax_utils::get_num_planes(in);
    auto converter = get_converter(program, in.format, out.format, flip_type, num_planes);

    auto strides = ax_utils::build_strides(in, out);
    auto offsets = ax_utils::build_offsets(in, out, num_planes);
    auto matrix = ax_utils::get_color_conversion_matrix(in.format, out.format);

    auto in_bufs = program.create_buffers(1, ax_utils::determine_buffer_size(in),
        CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, in.data, in.offsets.size());
    auto outbuf = program.create_buffer(out, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR);

    program.set_kernel_args(converter, 0, out.width, out.height, strides,
        offsets, in.crop_x, in.crop_y, matrix, in_bufs, *outbuf);
    return run_kernel(program, converter, in, out, in_bufs[0], outbuf, start_flush);
  }

  bool can_use_dmabuf() const
  {
    return program.can_use_dmabuf();
  }

  private:
  CLProgram program;
  int flip_type{};
  int error{};
  struct kernels {
    int hash;
    kernel cl_prog{ nullptr };
  };
  std::vector<kernels> all_kernels;
};

std::string_view flips[] = {
  "none",
  "clockwise",
  "rotate-180",
  "counterclockwise",
  "horizontal-flip",
  "vertical-flip",
  "upper-left-diagonal",
  "upper-right-diagonal",
};

int
determine_flip_type(std::string_view flip)
{
  auto it = std::find(std::begin(flips), std::end(flips), flip);
  return it != std::end(flips) ? std::distance(std::begin(flips), it) : -1;
}

extern "C" const std::unordered_set<std::string> &
allowed_properties()
{
  static const std::unordered_set<std::string> allowed_properties{
    "format",
    "flip_method",
  };
  return allowed_properties;
}

extern "C" std::shared_ptr<void>
init_and_set_static_properties_with_context(
    const std::unordered_map<std::string, std::string> &input, void *context, Ax::Logger &logger)
{
  auto prop = std::make_shared<cc_properties>();
  prop->format = Ax::get_property(input, "format", "ColorConvertProperties", prop->format);
  prop->flip_method = Ax::get_property(
      input, "flip_method", "ColorConvertProperties", prop->flip_method);
  auto flip_type = determine_flip_type(prop->flip_method);
  if (flip_type == -1) {
    logger(AX_ERROR) << "Invalid flip_method type: " << prop->flip_method
                     << " defaulting to none" << std::endl;
    flip_type = 0;
  }
  prop->color_convert = std::make_unique<CLColorConvert>(
      flip_type, static_cast<ax_utils::opencl_details *>(context), logger);
  return prop;
}

extern "C" void
set_dynamic_properties(const std::unordered_map<std::string, std::string> &input,
    cc_properties *prop, Ax::Logger & /*logger*/)
{
  prop->downstream_supports_opencl = Ax::get_property(input, "downstream_supports_opencl",
      "ColorConvertProperties", prop->downstream_supports_opencl);
}

bool
is_a_rotate(std::string_view flip_type)
{
  static std::string_view rotates[] = {
    "clockwise",
    "counterclockwise",
    "upper-left-diagonal",
    "upper-right-diagonal",
  };
  return std::find(std::begin(rotates), std::end(rotates), flip_type) != std::end(rotates);
}

struct {
  std::string color;
  AxVideoFormat format;
} valid_formats[] = {
  { "rgba", AxVideoFormat::RGBA },
  { "bgra", AxVideoFormat::BGRA },
  { "rgb", AxVideoFormat::RGB },
  { "bgr", AxVideoFormat::BGR },
  { "gray", AxVideoFormat::GRAY8 },
};

extern "C" AxDataInterface
set_output_interface(const AxDataInterface &interface,
    const cc_properties *prop, Ax::Logger &logger)
{
  AxDataInterface output{};
  if (std::holds_alternative<AxVideoInterface>(interface)) {
    auto in_info = std::get<AxVideoInterface>(interface);
    auto out_info = in_info;
    auto stride_factor = (prop->format == "gray") ? 1 : 4;
    out_info.info.stride = out_info.info.width * stride_factor;
    if (is_a_rotate(prop->flip_method)) {
      std::swap(out_info.info.width, out_info.info.height);
      out_info.info.stride = out_info.info.width * stride_factor;
      out_info.strides = { size_t(out_info.info.stride) };
    };
    auto fmt_found = std::find_if(std::begin(valid_formats), std::end(valid_formats),
        [fmt = prop->format](auto f) { return f.color == fmt; });
    if (fmt_found == std::end(valid_formats)) {
      logger(AX_ERROR)
          << "Invalid output format given in color conversion: " << prop->format
          << std::endl;
      throw std::runtime_error(
          "Invalid output format given in color conversion: " + prop->format);
    }
    logger(AX_INFO) << "Setting output format to " << prop->format << std::endl;
    out_info.info.format = fmt_found->format;
    output = out_info;
  }
  return output;
}

/// @brief  Check if the plugin has any work to do
/// @param input
/// @param output
/// @param logger
/// @return true if the plugin can pass through the input to output
extern "C" bool
can_passthrough(const AxDataInterface &input, const AxDataInterface &output,
    const cc_properties *prop, Ax::Logger &logger)
{
  if (!std::holds_alternative<AxVideoInterface>(input)) {
    throw std::runtime_error("color_convert works on video input only");
  }

  if (!std::holds_alternative<AxVideoInterface>(output)) {
    throw std::runtime_error("color_convert works on video input only");
  }
  auto input_details = ax_utils::extract_buffer_details(input);
  if (input_details.size() != 1) {
    throw std::runtime_error("color_convert works on single video (possibly batched) input only");
  }

  auto output_details = ax_utils::extract_buffer_details(output);
  if (output_details.size() != 1) {
    throw std::runtime_error("color_convert works on single video (possibly batched) output only");
  }
  // When output is GRAY and input is NV12, NV16, or I420, we can pass through,
  // as the yuv image already has the gray image as luminance (Y) component in the beginning of the buffer
  bool gray_out_bypass = (input_details[0].format == AxVideoFormat::I420
                             || input_details[0].format == AxVideoFormat::NV12
                             || input_details[0].format == AxVideoFormat::NV16)
                         && (output_details[0].format == AxVideoFormat::GRAY8
                             && input_details[0].width == output_details[0].width
                             && input_details[0].height == output_details[0].height);

  auto flip_type = determine_flip_type(prop->flip_method);
  if (flip_type == -1) {
    flip_type = 0;
  }

  return (flip_type == 0 && input_details[0].format == output_details[0].format
             && input_details[0].width == output_details[0].width
             && input_details[0].height == output_details[0].height)
         || gray_out_bypass;
}


extern "C" void
transform(const AxDataInterface &input, const AxDataInterface &output,
    const cc_properties *prop, unsigned int, unsigned int,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &, Ax::Logger &logger)
{
  //  These must be video interfaces as we have already checked in can_passthrough
  auto in_info = std::get<AxVideoInterface>(input);
  auto out_info = std::get<AxVideoInterface>(output);
  //  Validate input and output formats

  auto input_details = ax_utils::extract_buffer_details(input);
  if (input_details.size() != 1) {
    throw std::runtime_error("color_convert works on single tensor (possibly batched) input only");
  }

  auto output_details = ax_utils::extract_buffer_details(output);
  if (output_details.size() != 1) {
    throw std::runtime_error(
        "color_convert works on single tensor (possibly batched) output only");
  }
  if (std::holds_alternative<void *>(input_details[0].data)) {
    const int pagesize = 4096;
    auto ptr = std::get<void *>(input_details[0].data);
    if ((reinterpret_cast<uintptr_t>(ptr) & (pagesize - 1)) != 0) {
      logger(AX_DEBUG) << "Input buffer is not page aligned" << std::endl;
    }
  }
  prop->color_convert->run(input_details[0], output_details[0], prop->format, prop);
}

extern "C" bool
query_supports(Ax::PluginFeature feature, const cc_properties *prop, Ax::Logger &logger)
{
  if (feature == Ax::PluginFeature::opencl_buffers) {
    return true;
  }
  if (feature == Ax::PluginFeature::dmabuf_buffers) {
    return prop->color_convert->can_use_dmabuf();
  }
  if (feature == Ax::PluginFeature::crop_meta) {
    return true;
  }
  return Ax::PluginFeatureDefaults(feature);
}
