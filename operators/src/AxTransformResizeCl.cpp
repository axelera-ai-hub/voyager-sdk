// Copyright Axelera AI, 2024
#include <array>
#include <unordered_map>
#include <unordered_set>
#include "AxDataInterface.h"
#include "AxLog.hpp"
#include "AxMeta.hpp"
#include "AxOpUtils.hpp"
#include "AxOpenCl.hpp"
#include "AxPlugin.hpp"
#include "AxStreamerUtils.hpp"
#include "AxUtils.hpp"

class CLResize;
struct resize_properties {
  int width{};
  int height{};
  int size{};
  bool letterbox{};
  bool scale_up{ true };
  int fill{ 114 };
  bool downstream_supports_opencl{};

  AxVideoFormat format{ AxVideoFormat::UNDEFINED };

  bool to_tensor{};
  float quant_scale{ 1.0F / 255.0F };
  float quant_zeropoint{};
  bool normalization_active{};
  std::unique_ptr<CLResize> resize;
};

const char *resize_kernel = R"##(

__kernel void resize_kernel_cl(%s__global %s *out, int4 image_dims, int crop_x, int crop_y,
                            int4 strides, int4 offsets, float xscale, float yscale, int scaled_width,
                            int scaled_height, uchar fill, float16 color_matrix) {

    const int col = get_global_id(0);
    const int row = get_global_id(1);
    if (row >= image_dims.w || col >= image_dims.z) {
      return;
    }

    int xoffset = (image_dims.z - scaled_width) / 2;
    int yoffset = (image_dims.w - scaled_height) / 2;

    float2 corrected = ((float2)(0.5F + col - xoffset, 0.5F + row - yoffset) * (float2)(xscale, yscale));
    image_description img = {image_dims, strides, offsets, (int4)(xoffset, yoffset, scaled_width, scaled_height), (int4)(crop_x, crop_y, 0, 0)};

)##";


using ax_utils::buffer_details;
using ax_utils::CLProgram;
using ax_utils::opencl_details;

AxVideoFormat
add_alpha(AxVideoFormat format)
{
  if (format == AxVideoFormat::BGR) {
    return AxVideoFormat::BGRA;
  } else if (format == AxVideoFormat::RGB) {
    return AxVideoFormat::RGBA;
  }
  return format;
}

ax_utils::CLProgram::ax_kernel
build_kernel(ax_utils::CLProgram &program, AxVideoFormat in_format,
    AxVideoFormat out_format, int flip_type, const resize_properties &prop, int num_planes)
{
  std::string kernel_code = resize_kernel;

  auto input_details = ax_utils::get_input_details(
      in_format, ax_utils::Interpolation::bilinear, num_planes);
  auto output_details = prop.normalization_active ?
                            ax_utils::get_output_norm_details(in_format, out_format) :
                            ax_utils::get_output_details(in_format, out_format);
  const auto &out_type = output_details.out_type;
  const auto &output_code = output_details.sampler;

  const auto &sampler_code = input_details.sampler;

  auto n = snprintf(nullptr, 0, kernel_code.c_str(),
      input_details.input_params.c_str(), out_type.c_str());
  std::vector<char> buffer(n + 1);
  snprintf(buffer.data(), buffer.size(), kernel_code.c_str(),
      input_details.input_params.c_str(), out_type.c_str());
  auto final_kernel = std::string(buffer.data());

  final_kernel += sampler_code;
  final_kernel += output_code;
  final_kernel = ax_utils::get_kernel_utils(flip_type, program.has_fp16()) + final_kernel;

  return program.build_kernel_from_source(final_kernel, "resize_kernel_cl");
}

class CLResize
{
  using buffer = CLProgram::ax_buffer;
  using kernel = CLProgram::ax_kernel;

  public:
  CLResize(opencl_details *display, Ax::Logger &logger)
      : program("", display, logger)
  {
  }

  void set_normalization(std::vector<cl_float> mul, std::vector<cl_float> add)
  {
    mul_ = std::move(mul);
    add_ = std::move(add);
  }

  cl_kernel get_converter(ax_utils::CLProgram &program, AxVideoFormat in_format,
      AxVideoFormat out_format, int flip_type, const resize_properties &prop, int num_planes)
  {
    auto hash = (static_cast<int>(in_format) << 16)
                + (static_cast<int>(out_format) << 8) + flip_type;
    auto it = std::find_if(std::begin(all_kernels), std::end(all_kernels),
        [hash](auto &x) { return x.hash == hash; });
    if (it != all_kernels.end()) {
      return *it->cl_prog;
    }
    auto k = build_kernel(program, in_format, out_format, flip_type, prop, num_planes);
    return *all_kernels.emplace_back(hash, std::move(k)).cl_prog;
  }

  ax_utils::CLProgram::ax_buffer create_buffer(const buffer_details &info, cl_mem_flags flags)
  {
    return program.create_buffer(info, flags);
  }

  int run(const buffer_details &in, const buffer_details &out, const resize_properties &prop)
  {
    auto num_planes = ax_utils::get_num_planes(in);
    auto converter = get_converter(program, in.format, out.format, 0, prop, num_planes);
    bool start_flush = prop.downstream_supports_opencl == 0;
    cl_float xscale = (float) in.width / out.width;
    cl_float yscale = (float) in.height / out.height;
    cl_int scaled_width = out.width;
    cl_int scaled_height = out.height;
    if (prop.letterbox) {
      bool scale_to_height = static_cast<double>(prop.width) / prop.height
                             > static_cast<double>(in.width) / in.height;

      auto scale_factor = scale_to_height ?
                              static_cast<double>(prop.height) / in.height :
                              static_cast<double>(prop.width) / in.width;

      auto height = std::lround(in.height * scale_factor);
      auto width = std::lround(in.width * scale_factor);

      xscale = 1.0F / scale_factor;
      yscale = 1.0F / scale_factor;
      scaled_width = width;
      scaled_height = height;
    }

    if (in.width < out.width && in.height < out.height && !prop.scale_up) {
      xscale = 1.0F;
      yscale = 1.0F;
      scaled_width = in.width;
      scaled_height = in.height;
    }
    cl_uchar fill = prop.fill;
    auto image_dims = (cl_int4){ in.width, in.height, out.width, out.height };
    auto strides = ax_utils::build_strides(in, out);
    auto offsets = ax_utils::build_offsets(in, out, num_planes);
    auto outbuf = program.create_buffer(out, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR);
    auto in_bufs = program.create_buffers(1, ax_utils::determine_buffer_size(in),
        CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, in.data, in.offsets.size());
    auto matrix_f32 = prop.normalization_active ?
                          ax_utils::get_color_conversion_matrix_with_norm(
                              in.format, out.format, mul_, add_) :
                          ax_utils::get_color_conversion_matrix(in.format, out.format);

    program.set_kernel_args(converter, 0, in_bufs, *outbuf, image_dims,
        in.crop_x, in.crop_y, strides, offsets, xscale, yscale, scaled_width,
        scaled_height, fill, matrix_f32);
    return run_kernel(program, converter, in, out, in_bufs[0], outbuf, start_flush);
  }

  bool can_use_dmabuf() const
  {
    return program.can_use_dmabuf();
  }

  private:
  CLProgram program;
  int error{};
  std::vector<cl_float> mul_{ 1.0F, 1.0F, 1.0F, 1.0F };
  std::vector<cl_float> add_{ 0.0F, 0.0F, 0.0F, 0.0F };
  struct kernels {
    int hash;
    kernel cl_prog{ nullptr };
  };
  std::vector<kernels> all_kernels;
  struct last_buffer_details {
    buffer mem{ nullptr };
    buffer_details in{};
  } last_buffer;
};


extern "C" const std::unordered_set<std::string> &
allowed_properties()
{
  static const std::unordered_set<std::string> allowed_properties{
    "width",
    "height",
    "size",
    "letterbox",
    "padding",
    "format",
    "scale_up",
    //  For normalisation
    "mean",
    "std",
    "quant_scale",
    "quant_zeropoint",
    "to_tensor",

  };
  return allowed_properties;
}

extern "C" std::shared_ptr<void>
init_and_set_static_properties_with_context(
    const std::unordered_map<std::string, std::string> &input, void *context, Ax::Logger &logger)
{
  auto prop = std::make_shared<resize_properties>();
  prop->resize = std::make_unique<CLResize>(
      static_cast<ax_utils::opencl_details *>(context), logger);
  prop->size = Ax::get_property(input, "size", "resize_cl_static_properties", prop->size);
  prop->width = Ax::get_property(input, "width", "resize_cl_static_properties", prop->width);
  prop->height
      = Ax::get_property(input, "height", "resize_cl_static_properties", prop->height);
  prop->letterbox = Ax::get_property(
      input, "letterbox", "resize_cl_static_properties", prop->letterbox);

  prop->scale_up = Ax::get_property(
      input, "scale_up", "resize_cl_static_properties", prop->scale_up);
  auto format = Ax::get_property(
      input, "format", "resize_cl_static_properties", std::string{});
  if (format == "rgba") {
    prop->format = AxVideoFormat::RGBA;
  } else if (format == "bgra") {
    prop->format = AxVideoFormat::BGRA;
  } else if (format == "") {
    prop->format = AxVideoFormat::UNDEFINED;
  } else {
    throw std::runtime_error(
        "Resize with color convert only outputs RGBA or BGRA, given: " + format);
  }
  prop->fill = Ax::get_property(input, "padding", "resize_cl_static_properties", prop->fill);
  if (prop->letterbox) {
    if (prop->width == 0) {
      prop->width = prop->height;
    } else if (prop->height == 0) {
      prop->height = prop->width;
    }
  }
  if (prop->size > 0 && (prop->width > 0 || prop->height > 0)) {
    throw std::runtime_error("You must provide only one of width/height or size");
  }
  if (prop->size == 0 && (prop->width <= 0 || prop->height <= 0)) {
    throw std::runtime_error("Invalid width or height");
  }

  prop->to_tensor = Ax::get_property(
      input, "to_tensor", "resize_cl_static_properties", prop->to_tensor);

  prop->quant_scale = Ax::get_property(
      input, "quant_scale", "resize_cl_static_properties", prop->quant_scale);
  prop->quant_zeropoint = Ax::get_property(input, "quant_zeropoint",
      "resize_cl_static_properties", prop->quant_zeropoint);
  auto mean = Ax::get_property(
      input, "mean", "resize_cl_static_properties", std::vector<cl_float>());
  auto std = Ax::get_property(
      input, "std", "resize_cl_static_properties", std::vector<cl_float>());
  if (mean.empty()) {
    mean = std::vector<float>(std.size(), 0.0);
  }
  if (std.empty()) {
    std = std::vector<float>(mean.size(), 1.0);
  }
  if (mean.size() != std.size()) {
    throw std::runtime_error("mean and std must have equal lengths in resize_cl");
  }
  if (mean.empty() && std.empty()) {
    prop->normalization_active = false;
  } else {
    prop->normalization_active = true;
    const auto max_size = 4;
    const auto resize_size = std::min(max_size, static_cast<int>(mean.size()));
    mean.resize(resize_size, 0.0F);
    std.resize(resize_size, 1.0F);
    std::vector<cl_float> mul(max_size, 1.0F);
    std::vector<cl_float> add(max_size, 0.0F);
    for (int i = 0; i < (int) mean.size(); ++i) {
      mul[i] = 1.0 / (255.0 * prop->quant_scale * std[i]);
      add[i] = 255 * prop->quant_zeropoint * std[i] * prop->quant_scale - (255 * mean[i]);
      add[i] *= mul[i];
    }
    prop->resize->set_normalization(std::move(mul), std::move(add));
  }
  return prop;
}

extern "C" void
set_dynamic_properties(const std::unordered_map<std::string, std::string> &input,
    resize_properties *prop, Ax::Logger & /*logger*/)
{
  prop->downstream_supports_opencl = Ax::get_property(input, "downstream_supports_opencl",
      "resize_cl_static_properties", prop->downstream_supports_opencl);
}

std::pair<int, int>
determine_width_height(const AxVideoInterface &in_info, int size)
{
  auto width = in_info.info.width;
  auto height = in_info.info.height;
  auto height_is_shortest = height < width;
  auto scale = height_is_shortest ? static_cast<double>(size) / height :
                                    static_cast<double>(size) / width;
  return { static_cast<int>(std::round(width * scale)),
    static_cast<int>(std::round(height * scale)) };
}

extern "C" AxDataInterface
set_output_interface(const AxDataInterface &interface,
    const resize_properties *prop, Ax::Logger &logger)
{
  AxDataInterface output{};
  if (std::holds_alternative<AxVideoInterface>(interface)) {
    auto in_info = std::get<AxVideoInterface>(interface);
    auto out_info = in_info;
    auto [width, height] = prop->size ? determine_width_height(in_info, prop->size) :
                                        std::make_pair(prop->width, prop->height);
    out_info.info.width = width;
    out_info.info.height = height;
    out_info.info.actual_height = height;
    out_info.info.format = prop->format == AxVideoFormat::UNDEFINED ?
                               add_alpha(in_info.info.format) :
                               prop->format;
    output = out_info;
  }
  if (prop->to_tensor) {
    auto &info = std::get<AxVideoInterface>(output).info;
    const auto channels = info.format == AxVideoFormat::GRAY8 ? 1 : 4;
    AxTensorsInterface output
        = { { { 1, info.height, info.width, channels }, 1, nullptr } };
    return output;
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
    const resize_properties *prop, Ax::Logger &logger)
{
  if (!std::holds_alternative<AxVideoInterface>(input)) {
    logger(AX_WARN) << "Resize works on video input only" << std::endl;
    return false;
  }

  auto input_details = ax_utils::extract_buffer_details(input);
  auto output_details = ax_utils::extract_buffer_details(output);

  return (input_details[0].width == output_details[0].width
          && input_details[0].height == output_details[0].height
          && (input_details[0].format == output_details[0].format
              || output_details[0].format == AxVideoFormat::UNDEFINED)
          && !prop->normalization_active);
}

extern "C" void
transform(const AxDataInterface &input, const AxDataInterface &output,
    const resize_properties *prop, unsigned int, unsigned int,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &, Ax::Logger &logger)
{

  auto input_details = ax_utils::extract_buffer_details(input);
  if (input_details.size() != 1) {
    throw std::runtime_error("resize works on single video input only");
  }

  auto output_details = ax_utils::extract_buffer_details(output);
  if (output_details.size() != 1) {
    throw std::runtime_error("resize works on single video output only");
  }
  auto valid_formats = std::array{
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
  if (std::none_of(valid_formats.begin(), valid_formats.end(), [input_details](auto format) {
        return format == input_details[0].format;
      })) {
    throw std::runtime_error("Resize does not work with the input format: "
                             + AxVideoFormatToString(input_details[0].format));
  }
  if (output_details[0].format == AxVideoFormat::UNDEFINED) {
    output_details[0].format = prop->format == AxVideoFormat::UNDEFINED ?
                                   add_alpha(input_details[0].format) :
                                   prop->format;
  }
  if (output_details[0].format != AxVideoFormat::RGBA
      && output_details[0].format != AxVideoFormat::BGRA
      && output_details[0].format != AxVideoFormat::GRAY8) {
    throw std::runtime_error("Resize does not work with the output format: "
                             + AxVideoFormatToString(output_details[0].format));
  }
  prop->resize->run(input_details[0], output_details[0], *prop);
  return;
}

extern "C" bool
query_supports(Ax::PluginFeature feature,
    const resize_properties *resize_properties, Ax::Logger &logger)
{
  if (feature == Ax::PluginFeature::opencl_buffers) {
    return true;
  }
  if (feature == Ax::PluginFeature::crop_meta) {
    return true;
  }
  if (feature == Ax::PluginFeature::dmabuf_buffers) {
    return resize_properties->resize->can_use_dmabuf();
  }
  return Ax::PluginFeatureDefaults(feature);
}
