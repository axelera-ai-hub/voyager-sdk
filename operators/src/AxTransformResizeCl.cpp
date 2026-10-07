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

enum class Interpolation { nearest = 0, bilinear = 1, pillow_bilinear = 2 };

class CLResize;
struct resize_properties {
  int width{};
  int height{};
  int size{};
  bool letterbox{};
  bool scale_up{ true };
  int fill{ 114 };
  //  Extra border (top, bottom, left, right in pixels) added around the
  //  resized/letterboxed content to satisfy the AIPU's input-shape contract.
  //  Empty means no extra padding. Filled with aipu_fill, independently of
  //  the letterbox fill above.
  std::vector<int> aipu_padding{};
  int aipu_fill{};
  Interpolation interpolation{ Interpolation::bilinear };
  bool downstream_supports_opencl{};

  AxVideoFormat format{ AxVideoFormat::UNDEFINED };

  bool to_tensor{};
  float quant_scale{ 1.0F / 255.0F };
  float quant_zeropoint{};
  bool normalization_active{};
  std::unique_ptr<CLResize> resize;
};

const char *resize_kernel_preamble = R"##(

__kernel void resize_kernel_cl(%s__global %s *out, int4 image_dims,
                            int4 strides, int4 offsets, int4 roi, int crop_x, int crop_y,
                            uchar fill, float16 color_matrix, float16 transform_matrix,
                            int4 aipu_pad, uchar aipu_fill) {

    const int col = get_global_id(0);
    const int row = get_global_id(1);
    if (row >= image_dims.w || col >= image_dims.z) {
      return;
    }

)##";

// Letterbox padding and AIPU input-shape padding must both be colour/
// normalisation-converted exactly like a real sampled pixel (see the
// corresponding gray/rgba_output[_norm]_cl snippets below) so that, e.g. with
// a quantization zero-point baked into color_matrix (to_tensor +
// mean/std/quant_scale/quant_zeropoint), the padded border ends up at the
// same quantized "neutral gray" value the model saw at training time,
// instead of the raw uint8 `border_fill` byte. `border_fill` resolves to
// either the letterbox `fill` or the AIPU `aipu_fill`, whichever border the
// pixel falls in -- see the aipu_pad-vs-roi nesting where these are used.
//
// The %s placeholders take the chroma channels of the pre-conversion pixel:
// for YUV-family inputs this must be the neutral chroma "128, 128" (matching
// what every input sampler returns for out-of-bounds/letterbox pixels), not
// `border_fill`, otherwise color_matrix (which expects YUV-space input) tints
// the border instead of producing a neutral gray.
const char *fill_write_gray_plain = R"##(
        advance_uchar_ptr(out, row * strides.w)[col] =
            convert_uchar_sat(color_convert_float((uchar4)(border_fill, %s, 255), color_matrix).x);
)##";

const char *fill_write_gray_norm = R"##(
        advance_uchar_ptr(out, row * strides.w)[col] = convert_uchar(convert_char_sat_rte(
            color_convert_float((uchar4)(border_fill, %s, 255), color_matrix).x));
)##";

const char *fill_write_rgba_plain = R"##(
        advance_uchar4_ptr(out, row * strides.w)[col] =
            convert_uchar4_sat(color_convert_float((uchar4)(border_fill, %s, 255), color_matrix));
)##";

const char *fill_write_rgba_norm = R"##(
        advance_uchar4_ptr(out, row * strides.w)[col] = convert_uchar4(convert_char4_sat_rte(
            color_convert_float((uchar4)(border_fill, %s, 255), color_matrix)));
)##";

bool
is_yuv_format(AxVideoFormat format)
{
  switch (format) {
    case AxVideoFormat::NV12:
    case AxVideoFormat::NV16:
    case AxVideoFormat::I420:
    case AxVideoFormat::Y42B:
    case AxVideoFormat::Y444:
    case AxVideoFormat::YUY2:
      return true;
    default:
      return false;
  }
}

const char *resize_kernel_coord = R"##(
    const float col_c = col + 0.5f, row_c = row + 0.5f;
    // transform_matrix's third row (s8,s9,sa) is always {0,0,1} for every matrix
    // this kernel receives (see fold_crop_into_transform, which only ever writes
    // the first two rows on top of the hand-built scale/translate matrix in
    // CLResize::run), so the perspective divide is always by a constant 1.0 --
    // skip computing and dividing by it per pixel.
    float2 corrected = (float2)(
        transform_matrix.s0 * col_c + transform_matrix.s1 * row_c + transform_matrix.s2,
        transform_matrix.s4 * col_c + transform_matrix.s5 * row_c + transform_matrix.s6
    );
    float xscale = transform_matrix.s0, yscale = transform_matrix.s5;
    image_description img = { .image_dims = image_dims, .strides = strides,
        .offsets = offsets, .crop = (int2)(crop_x, crop_y) };

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
  auto interpolation = prop.interpolation == Interpolation::pillow_bilinear ?
                           ax_utils::Interpolation::pillow_bilinear :
                       prop.interpolation == Interpolation::nearest ?
                           ax_utils::Interpolation::nearest :
                           ax_utils::Interpolation::bilinear;
  auto input_details = ax_utils::get_input_details(in_format, interpolation, num_planes);
  auto output_details = prop.normalization_active ?
                            ax_utils::get_output_norm_details(in_format, out_format) :
                            ax_utils::get_output_details(in_format, out_format);
  const auto &out_type = output_details.out_type;
  const auto &output_code = output_details.sampler;
  const auto &sampler_code = input_details.sampler;

  auto n = snprintf(nullptr, 0, resize_kernel_preamble,
      input_details.input_params.c_str(), out_type.c_str());
  std::vector<char> buffer(n + 1);
  snprintf(buffer.data(), buffer.size(), resize_kernel_preamble,
      input_details.input_params.c_str(), out_type.c_str());
  auto final_kernel = std::string(buffer.data());

  const char *fill_write_fmt
      = out_format == AxVideoFormat::GRAY8 ?
            (prop.normalization_active ? fill_write_gray_norm : fill_write_gray_plain) :
            (prop.normalization_active ? fill_write_rgba_norm : fill_write_rgba_plain);
  const char *chroma = is_yuv_format(in_format) ? "128, 128" : "border_fill, border_fill";
  auto fn = snprintf(nullptr, 0, fill_write_fmt, chroma);
  std::vector<char> fill_write_buf(fn + 1);
  snprintf(fill_write_buf.data(), fill_write_buf.size(), fill_write_fmt, chroma);
  //  A pixel outside roi is either in the letterbox border (still inside the
  //  aipu_pad-free inner canvas) or in the AIPU's outer padding border; roi is
  //  always contained within the inner canvas, so this nesting is exhaustive.
  //  Checked (and, on a miss, written) before any of the sampling/transform
  //  work below, since border pixels never need it. With no letterboxing, roi
  //  covers the whole inner canvas so the outer branch is never taken; with no
  //  aipu_padding, aipu_pad is all zero so the inner branch is never taken -
  //  this code is always safe to include unconditionally.
  final_kernel += R"##(
    if (col < roi.x || row < roi.y || col >= roi.x + roi.z || row >= roi.y + roi.w) {
        bool in_aipu_pad = col < aipu_pad.s2 || row < aipu_pad.s0
            || col >= image_dims.z - aipu_pad.s3 || row >= image_dims.w - aipu_pad.s1;
        uchar border_fill = in_aipu_pad ? aipu_fill : fill;
)##";
  final_kernel += fill_write_buf.data();
  final_kernel += R"##(
        return;
    }
)##";

  final_kernel += resize_kernel_coord;
  final_kernel += sampler_code;
  final_kernel += output_code;
  bool use_fp16 = (interpolation == ax_utils::Interpolation::bilinear)
                  && program.has_fp16();
  final_kernel = ax_utils::get_kernel_utils(flip_type, use_fp16) + final_kernel;

  return program.build_kernel_from_source(final_kernel, "resize_kernel_cl");
}

class CLResize
{
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
    auto hash = (static_cast<int>(in_format) << 16) + (static_cast<int>(out_format) << 8)
                + flip_type + (static_cast<int>(prop.interpolation) << 24);
    return kernel_cache.get_or_build(hash, [&] {
      return build_kernel(program, in_format, out_format, flip_type, prop, num_planes);
    });
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
    //  aipu_padding, when set, is an extra border (independent of letterboxing)
    //  added around the resized/letterboxed content to satisfy the AIPU's
    //  input-shape contract. out.width/out.height already include it (see
    //  set_output_interface), so the content itself must be scaled/centred
    //  against the inner (un-padded) size, not the full padded canvas.
    cl_int pad_top = 0, pad_bottom = 0, pad_left = 0, pad_right = 0;
    if (prop.aipu_padding.size() == 4) {
      pad_top = prop.aipu_padding[0];
      pad_bottom = prop.aipu_padding[1];
      pad_left = prop.aipu_padding[2];
      pad_right = prop.aipu_padding[3];
    }
    cl_int inner_width = out.width - pad_left - pad_right;
    cl_int inner_height = out.height - pad_top - pad_bottom;
    cl_float xscale = (float) in.width / inner_width;
    cl_float yscale = (float) in.height / inner_height;
    cl_int scaled_width = inner_width;
    cl_int scaled_height = inner_height;
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

    if (in.width < inner_width && in.height < inner_height && !prop.scale_up) {
      xscale = 1.0F;
      yscale = 1.0F;
      scaled_width = in.width;
      scaled_height = in.height;
    }
    cl_int xoffset = (inner_width - scaled_width) / 2 + pad_left;
    cl_int yoffset = (inner_height - scaled_height) / 2 + pad_top;
    cl_int4 roi = { xoffset, yoffset, scaled_width, scaled_height };
    // clang-format off
    cl_float16 transform = {
        xscale, 0.0f, -xscale * xoffset, 0.0f,
        0.0f, yscale, -yscale * yoffset, 0.0f,
        0.0f, 0.0f,                1.0f, 0.0f,
        0.0f, 0.0f,                0.0f, 0.0f
    };
    // clang-format on
    transform = ax_utils::fold_crop_into_transform(transform, in.crop_x, in.crop_y);
    cl_uchar fill = prop.fill;
    cl_int4 aipu_pad = { pad_top, pad_bottom, pad_left, pad_right };
    cl_uchar aipu_fill = prop.aipu_fill;
    auto image_dims = ax_utils::image_dims_for_crop(in, out);
    auto outbuf = program.create_buffer(out, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR);
    auto ri = prop.normalization_active ?
                  ax_utils::prepare_run_inputs(program, in, out, num_planes, mul_, add_) :
                  ax_utils::prepare_run_inputs(program, in, out, num_planes);

    program.set_kernel_args(converter, 0, ri.in_bufs, *outbuf, image_dims,
        ri.strides, ri.offsets, roi, (cl_int) in.crop_x, (cl_int) in.crop_y,
        fill, ri.color_matrix, transform, aipu_pad, aipu_fill);
    return run_kernel(program, converter, in, out, ri.in_bufs[0], outbuf, start_flush);
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
  ax_utils::KernelCache kernel_cache;
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
    "interpolation",
    //  For normalisation
    "mean",
    "std",
    "quant_scale",
    "quant_zeropoint",
    "to_tensor",
    //  For fusing the AIPU's extra input-shape padding into this kernel
    "aipu_padding",
    "aipu_fill",

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
  prop->aipu_padding = Ax::get_property(
      input, "aipu_padding", "resize_cl_static_properties", prop->aipu_padding);
  if (!prop->aipu_padding.empty() && prop->aipu_padding.size() != 4) {
    throw std::runtime_error("aipu_padding must have exactly 4 values: top,bottom,left,right");
  }
  prop->aipu_fill = Ax::get_property(
      input, "aipu_fill", "resize_cl_static_properties", prop->aipu_fill);
  prop->interpolation = static_cast<Interpolation>(Ax::get_property(input, "interpolation",
      "resize_cl_static_properties", static_cast<int>(prop->interpolation)));
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
    if (prop->aipu_padding.size() == 4) {
      width += prop->aipu_padding[2] + prop->aipu_padding[3]; // left + right
      height += prop->aipu_padding[0] + prop->aipu_padding[1]; // top + bottom
    }
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
  ax_utils::require_opencl_input_format(input_details[0].format, "Resize");
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
