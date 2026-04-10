// Copyright Axelera AI, 2026
#include <array>
#include <unordered_map>
#include <unordered_set>
#include "AxDataInterface.h"
#include "AxLog.hpp"
#include "AxOpUtils.hpp"
#include "AxOpenCl.hpp"
#include "AxPlugin.hpp"
#include "AxRoiCropCommon.h"
#include "AxStreamerUtils.hpp"
#include "AxUtils.hpp"

class CLRoiCrop;
struct roicrop_cl_properties : RoiCropParams {
  std::string out_format{};
  std::unique_ptr<CLRoiCrop> roicrop;
};

constexpr std::array valid_output_formats = {
  AxVideoFormat::RGB,
  AxVideoFormat::BGR,
  AxVideoFormat::RGBA,
  AxVideoFormat::BGRA,
  AxVideoFormat::GRAY8,
};

static AxVideoFormat
add_alpha(AxVideoFormat format)
{
  if (format == AxVideoFormat::BGR) {
    return AxVideoFormat::BGRA;
  } else if (format == AxVideoFormat::RGB) {
    return AxVideoFormat::RGBA;
  }
  return format;
}

// Returns a kernel snippet that checks whether top_left is out of bounds,
// writes black to the output pixel, and returns early.  This is inserted
// between the kernel preamble and the sampler so the sampler itself stays
// a clean, unconditional pixel read.
static std::string
oob_fill_snippet(AxVideoFormat out_format)
{
  switch (out_format) {
    case AxVideoFormat::RGBA:
    case AxVideoFormat::BGRA:
      return R"##(
    if (top_left.x < 0 || top_left.x >= width || top_left.y < 0 || top_left.y >= height) {
        __global uchar4 *pout = advance_uchar4_ptr(out, row * strides.w);
        pout[col] = (uchar4)(0, 0, 0, 255);
        return;
    }
)##";
    case AxVideoFormat::RGB:
    case AxVideoFormat::BGR:
      return R"##(
    if (top_left.x < 0 || top_left.x >= width || top_left.y < 0 || top_left.y >= height) {
        __global uchar *pout = advance_uchar_ptr(out, row * strides.w);
        vstore3((uchar3)(0, 0, 0), col, pout);
        return;
    }
)##";
    case AxVideoFormat::GRAY8:
      return R"##(
    if (top_left.x < 0 || top_left.x >= width || top_left.y < 0 || top_left.y >= height) {
        __global uchar *pout = advance_uchar_ptr(out, row * strides.w);
        pout[col] = 0;
        return;
    }
)##";
    default:
      return {};
  }
}

// Kernel template: %s placeholders filled with input and output element types
// (e.g. "uchar4" for RGBA, "uchar" for RGB/GRAY8).
// The kernel body is completed by appending the format-specific sampler snippet
// and output snippet from get_input_details / get_output_details, followed by
// the closing brace that comes from the output snippet.
const char *roicrop_kernel_template = R"##(
uchar4 color_convert(uchar4 pixel, float16 matrix) {
    float4 in_pixel = convert_float4(pixel);
    float4 color = mad(in_pixel.x, matrix.s0123,
                   mad(in_pixel.y, matrix.s4567,
                   mad(in_pixel.z, matrix.s89ab, matrix.scdef)));
    color.w = in_pixel.w;
    return convert_uchar4_sat(color);
}

__kernel void roicrop_cl(__global const %s *p_in, __global %s *out, int4 image_dims,
                         int crop_x, int crop_y,
                         int4 strides, int4 offsets, float16 color_matrix) {
    const int col = get_global_id(0);
    const int row = get_global_id(1);
    if (row >= image_dims.w || col >= image_dims.z) {
        return;
    }
    int width = image_dims.x;
    int height = image_dims.y;
    int2 top_left = (int2)(col + crop_x, row + crop_y);
)##";

using ax_utils::buffer_details;
using ax_utils::CLProgram;
using ax_utils::opencl_details;

class CLRoiCrop
{
  using buffer = CLProgram::ax_buffer;
  using kernel = CLProgram::ax_kernel;

  public:
  CLRoiCrop(opencl_details *ocl, Ax::Logger &logger)
      : program("", ocl, logger)
  {
  }

  kernel build_kernel(AxVideoFormat in_format, AxVideoFormat out_format)
  {
    std::string kernel_code = roicrop_kernel_template;

    auto [_, in_type, sampler_code]
        = ax_utils::get_input_details(in_format, ax_utils::Interpolation::nearest);
    auto [__, out_type, output_code] = ax_utils::get_output_details(in_format, out_format);

    auto n = snprintf(nullptr, 0, kernel_code.c_str(), in_type.c_str(), out_type.c_str());
    std::vector<char> buf(n + 1);
    snprintf(buf.data(), buf.size(), kernel_code.c_str(), in_type.c_str(),
        out_type.c_str());
    auto final_kernel = std::string(buf.data());
    final_kernel += oob_fill_snippet(out_format);
    final_kernel += sampler_code;
    final_kernel += output_code;
    final_kernel = ax_utils::get_kernel_utils(0) + final_kernel;

    return program.build_kernel_from_source(final_kernel, "roicrop_cl");
  }

  cl_kernel get_converter(AxVideoFormat in_format, AxVideoFormat out_format)
  {
    auto hash = (static_cast<int>(in_format) << 16)
                + (static_cast<int>(out_format) << 8) + 0;
    auto it = std::find_if(std::begin(all_kernels), std::end(all_kernels),
        [hash](auto &x) { return x.hash == hash; });
    if (it != all_kernels.end()) {
      return *it->cl_prog;
    }
    auto k = build_kernel(in_format, out_format);
    return *all_kernels.emplace_back(hash, std::move(k)).cl_prog;
  }

  int run(const buffer_details &in, const buffer_details &out, int crop_x,
      int crop_y, bool downstream_supports_opencl)
  {
    auto converter = get_converter(in.format, out.format);

    bool start_flush = !downstream_supports_opencl;
    auto outbuf = program.create_buffer(out, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR);
    auto inbuf = program.create_buffer(in, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR);

    auto image_dims = (cl_int4){ in.width, in.height, out.width, out.height };
    auto strides = ax_utils::build_strides(in, out);
    auto offsets = ax_utils::build_offsets(in, out);
    auto matrix = ax_utils::get_color_conversion_matrix(in.format, out.format);

    program.set_kernel_args(converter, 0, *inbuf, *outbuf, image_dims,
        (cl_int) crop_x, (cl_int) crop_y, strides, offsets, matrix);
    return run_kernel(program, converter, in, out, inbuf, outbuf, start_flush);
  }

  bool can_use_dmabuf() const
  {
    return program.can_use_dmabuf();
  }

  private:
  CLProgram program;
  struct kernels {
    int hash;
    kernel cl_prog{ nullptr };
  };
  std::vector<kernels> all_kernels;
  AxVideoFormat last_in_format{ AxVideoFormat::UNDEFINED };
  AxVideoFormat last_out_format{ AxVideoFormat::UNDEFINED };
};


extern "C" const std::unordered_set<std::string> &
allowed_properties()
{
  static const std::unordered_set<std::string> props = [] {
    auto result = roicrop_allowed_properties();
    result.insert("format");
    return result;
  }();
  return props;
}

extern "C" std::shared_ptr<void>
init_and_set_static_properties_with_context(
    const std::unordered_map<std::string, std::string> &input, void *context, Ax::Logger &logger)
{
  auto prop = std::make_shared<roicrop_cl_properties>();
  parse_roicrop_params(input, *prop, "roicrop_cl", logger);
  prop->out_format = Ax::get_property(input, "format", "roicrop_cl", prop->out_format);
  prop->roicrop
      = std::make_unique<CLRoiCrop>(static_cast<opencl_details *>(context), logger);
  return prop;
}

extern "C" void
set_dynamic_properties(const std::unordered_map<std::string, std::string> &input,
    roicrop_cl_properties *prop, Ax::Logger & /*logger*/)
{
  prop->downstream_supports_opencl = Ax::get_property(input,
      "downstream_supports_opencl", "roicrop_cl", prop->downstream_supports_opencl);
}

extern "C" AxDataInterface
set_output_interface_from_meta(const AxDataInterface &interface,
    const roicrop_cl_properties *prop, unsigned int subframe_index, unsigned int number_of_subframes,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &meta_map,
    Ax::Logger &logger)
{
  auto output = roicrop_set_output_interface(interface, *prop, subframe_index,
      number_of_subframes, meta_map, "roicrop_cl", logger);
  auto &out_info = std::get<AxVideoInterface>(output).info;
  if (!out_info.cropped) {
    // The kernel will run (ROI is out of bounds); apply format conversion.
    auto in_format = std::get<AxVideoInterface>(interface).info.format;
    auto effective_format = prop->out_format.empty() ?
                                add_alpha(in_format) :
                                AxVideoFormatFromString(prop->out_format);
    Ax::validate_output_format(
        effective_format, prop->out_format, "roicrop_cl", valid_output_formats);
    out_info.format = effective_format;
  }
  // else: crop is fully in bounds; transform is skipped via crop_meta, so
  // the output format stays the same as the input.
  return output;
}

extern "C" void
transform(const AxDataInterface &input, const AxDataInterface &output,
    const roicrop_cl_properties *prop, unsigned int subframe_index, unsigned int subframe_number,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &map, Ax::Logger &logger)
{
  auto input_details = ax_utils::extract_buffer_details(input);
  if (input_details.size() != 1) {
    throw std::runtime_error("roicrop_cl works on single video input only");
  }
  auto output_details = ax_utils::extract_buffer_details(output);
  if (output_details.size() != 1) {
    throw std::runtime_error("roicrop_cl works on single video output only");
  }

  constexpr std::array valid_input_formats = {
    AxVideoFormat::RGB,
    AxVideoFormat::BGR,
    AxVideoFormat::RGBA,
    AxVideoFormat::BGRA,
    AxVideoFormat::NV12,
    AxVideoFormat::NV16,
    AxVideoFormat::I420,
    AxVideoFormat::YUY2,
    AxVideoFormat::GRAY8,
  };
  if (std::none_of(valid_input_formats.begin(), valid_input_formats.end(),
          [&](auto fmt) { return fmt == input_details[0].format; })) {
    throw std::runtime_error("roicrop_cl does not support input format: "
                             + AxVideoFormatToString(input_details[0].format));
  }

  auto [x1, y1, x2, y2] = roicrop_get_roi_with_margin(
      *prop, subframe_index, subframe_number, map, "roicrop_cl", logger);

  prop->roicrop->run(input_details[0], output_details[0], x1, y1,
      prop->downstream_supports_opencl);
}

extern "C" bool
query_supports(Ax::PluginFeature feature, const roicrop_cl_properties *prop, Ax::Logger &logger)
{
  if (feature == Ax::PluginFeature::opencl_buffers) {
    return true;
  }
  if (feature == Ax::PluginFeature::crop_meta) {
    return true;
  }
  if (feature == Ax::PluginFeature::dmabuf_buffers) {
    return prop->roicrop->can_use_dmabuf();
  }
  return Ax::PluginFeatureDefaults(feature);
}
