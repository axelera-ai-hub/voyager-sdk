// Copyright Axelera AI, 2024
#include <array>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include "AxDataInterface.h"
#include "AxLog.hpp"
#include "AxMeta.hpp"
#include "AxOpUtils.hpp"
#include "AxOpenCl.hpp"
#include "AxUtils.hpp"


class CLBarrelCorrect;
struct barrelcorrect_properties {
  int width{};
  int height{};
  int size{};
  std::vector<cl_float> camera_props;
  std::vector<cl_float> distort_coefs;
  bool normalised{ true };
  std::string out_format{};
  bool downstream_supports_opencl{ false };
  std::unique_ptr<CLBarrelCorrect> barrelcorrect;
};

const char *const barrel_correct = R"##(
uchar4 color_convert(uchar4 pixel, float16 matrix) {
    float4 in_pixel = convert_float4(pixel);
    float4 color = mad(in_pixel.x, matrix.s0123, mad(in_pixel.y, matrix.s4567, mad(in_pixel.z, matrix.s89ab, matrix.scdef)));
    color.w = in_pixel.w;
    return convert_uchar4_sat(color);
}

float2 barrel_distortion_correction(
    float x, float y, const float2 focal, const float2 centre,
    float4 new_camera_props, __constant const float *coeffs)
{
    const float k1 = coeffs[0];
    const float k2 = coeffs[1];
    const float p1 = coeffs[2];
    const float p2 = coeffs[3];
    const float k3 = coeffs[4];
    const float fx_new = new_camera_props.x;
    const float fy_new = new_camera_props.y;
    const float cx_new = new_camera_props.z;
    const float cy_new = new_camera_props.w;
    // Convert pixel coordinates to normalized coordinates (x_n, y_n)
    const float2 new_xy = (float2)(x - cx_new, y - cy_new) / (float2)(fx_new, fy_new);
    const float x_n = new_xy.x;
    const float y_n = new_xy.y;
    const float r2 = x_n * x_n + y_n * y_n;
    const float r4 = r2 * r2;
    const float r6 = r4 * r2;

    // Radial and tangential distortion
    const float radial = 1.0f + k1 * r2 + k2 * r4 + k3 * r6;
    const float2 tangential = (float2)(2.0f * p1 * x_n * y_n + p2 * (r2 + 2.0f * x_n * x_n),
                                            p1 * (r2 + 2.0f * y_n * y_n) + 2.0f * p2 * x_n * y_n);

    const float2 distorted = mad(radial, new_xy, tangential);
    // Map back to pixel coordinates in the original image
    return  mad(focal, distorted, centre);
}

__kernel void barrel_correct(__global const %s *in, __global %s *out, int4 image_dims,
                        int4 strides, int4 offsets, const float x_scale, const float y_scale, const float4 camera_props,
                        float4 new_camera_props, __constant const float *coeffs, float16 color_matrix) {

    const int col = get_global_id(0);
    const int row = get_global_id(1);
    if (row >= image_dims.w || col >= image_dims.z) {
      return;
    }
    float x = (col + 0.5F) * x_scale;
    float y = (row + 0.5F) * y_scale;
    float2 focal = camera_props.xy;
    float2 centre = camera_props.zw;
    float2 corrected = barrel_distortion_correction(x, y, focal, centre, new_camera_props, coeffs);
    image_description img = {image_dims, strides, offsets, (int4)(0, 0, image_dims.z, image_dims.w), (int4)(0,0,0,0)};
    uchar fill = 0;
)##";


using ax_utils::buffer_details;
using ax_utils::CLProgram;
using ax_utils::opencl_details;

ax_utils::CLProgram::ax_kernel
build_kernel(ax_utils::CLProgram &program, AxVideoFormat in_format, AxVideoFormat out_format)
{
  std::string kernel_code = barrel_correct;

  auto [unused1, in_type, sampler_code] = ax_utils::get_input_details(in_format);
  auto [unused2, out_type, output_code]
      = ax_utils::get_output_details(in_format, out_format);

  auto n = snprintf(nullptr, 0, kernel_code.c_str(), in_type.c_str(), out_type.c_str());
  std::vector<char> buffer(n + 1);
  snprintf(buffer.data(), buffer.size(), kernel_code.c_str(), in_type.c_str(),
      out_type.c_str());
  auto final_kernel = std::string(buffer.data());
  final_kernel += sampler_code;
  final_kernel += output_code;
  final_kernel = ax_utils::get_kernel_utils() + final_kernel;
  return program.build_kernel_from_source(final_kernel, "barrel_correct");
}

cv::Mat
determine_optimal_matrix(const std::vector<cl_float> &distort_coefs,
    std::span<float> camera_props, const buffer_details &in)
{
  //  Create the input matrix
  auto input_matrix = cv::Mat({ 3, 3 }, {
                                            camera_props[0],
                                            0.0F,
                                            camera_props[2],
                                            0.0F,
                                            camera_props[1],
                                            camera_props[3],
                                            0.0F,
                                            0.0F,
                                            1.0F,
                                        });
  // [TODO]
  // return cv::getOptimalNewCameraMatrix(input_matrix, distort_coefs,
  //     cv::Size(in.width, in.height), 0, cv::Size(in.width, in.height));
  return input_matrix;
}

class CLBarrelCorrect
{
  using buffer = CLProgram::ax_buffer;
  using kernel = CLProgram::ax_kernel;

  public:
  CLBarrelCorrect(opencl_details *context, Ax::Logger &logger)
      : program("", context, logger)
  {
  }

  int run(const buffer_details &in, const buffer_details &out,
      const barrelcorrect_properties &prop)
  {
    if (!converter) {
      converter = build_kernel(program, in.format, out.format);
    }
    bool start_flush = !prop.downstream_supports_opencl;
    auto outbuf = program.create_buffer(out, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR);
    auto width = prop.normalised ? static_cast<float>(in.width) : 1.0F;
    auto height = prop.normalised ? static_cast<float>(in.height) : 1.0F;
    auto original_camera_props = std::array<cl_float, 4>{
      prop.camera_props[0] * width,
      prop.camera_props[1] * height,
      prop.camera_props[2] * width,
      prop.camera_props[3] * height,
    };

    auto x_scale = static_cast<float>(in.width) / static_cast<float>(out.width);
    auto y_scale = static_cast<float>(in.height) / static_cast<float>(out.height);

    if (!distort_coeffs) {
      distort_coeffs = program.create_buffer(1,
          prop.distort_coefs.size() * sizeof(prop.distort_coefs[0]),
          CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
          const_cast<float *>(prop.distort_coefs.data()), 1);
    }

    if (camera_props.empty()) {
      auto new_matrix
          = determine_optimal_matrix(prop.distort_coefs, original_camera_props, in);
      camera_props = {
        new_matrix.at<float>(0, 0),
        new_matrix.at<float>(1, 1),
        new_matrix.at<float>(0, 2),
        new_matrix.at<float>(1, 2),
      };
    }

    std::array<cl_int, 4> image_dims = { in.width, in.height, out.width, out.height };
    auto strides = ax_utils::build_strides(in, out);
    auto offsets = ax_utils::build_offsets(in, out);
    auto matrix = ax_utils::get_color_conversion_matrix(in.format, out.format);
    auto inbuf_y = program.create_buffer(in, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR);

    program.set_kernel_args(*converter, 0, *inbuf_y, *outbuf, image_dims,
        strides, offsets, x_scale, y_scale, original_camera_props, camera_props,
        *distort_coeffs, matrix);
    return run_kernel(program, *converter, in, out, inbuf_y, outbuf, start_flush);
  }

  private:
  CLProgram program;
  int error{};
  kernel converter{ nullptr };
  std::vector<cl_float> camera_props{};
  CLProgram::ax_buffer distort_coeffs{ nullptr };
};


extern "C" const std::unordered_set<std::string> &
allowed_properties()
{
  static const std::unordered_set<std::string> allowed_properties{
    "camera_props",
    "distort_coefs",
    "out_format",
    "format",
    "normalized_properties",
    "width",
    "height",
    "size",
  };
  return allowed_properties;
}

extern "C" std::shared_ptr<void>
init_and_set_static_properties_with_context(
    const std::unordered_map<std::string, std::string> &input, void *context, Ax::Logger &logger)
{
  auto prop = std::make_shared<barrelcorrect_properties>();

  prop->camera_props = Ax::get_property(input, "camera_props",
      "barrelcorrect_static_properties", prop->camera_props);
  prop->distort_coefs = Ax::get_property(input, "distort_coefs",
      "barrelcorrect_static_properties", prop->distort_coefs);
  prop->out_format = Ax::get_property(
      input, "out_format", "barrelcorrect_dynamic_properties", prop->out_format);
  prop->out_format = Ax::get_property(
      input, "format", "barrelcorrect_dynamic_properties", prop->out_format);

  prop->normalised = Ax::get_property(input, "normalized_properties",
      "barrelcorrect_static_properties", prop->normalised);
  prop->size = Ax::get_property(input, "size", "barrelcorrect_static_properties", prop->size);
  prop->width = Ax::get_property(
      input, "width", "barrelcorrect_static_properties", prop->width);
  prop->height = Ax::get_property(
      input, "height", "barrelcorrect_static_properties", prop->height);

  constexpr auto camera_props_size = 4;
  if (prop->camera_props.size() != camera_props_size) {
    throw std::runtime_error("camera_props must have 4 values");
  }
  constexpr auto distort_coefs_size = 5;
  if (prop->distort_coefs.size() != distort_coefs_size) {
    throw std::runtime_error("distort_coefs must have 5 values");
  }
  prop->barrelcorrect = std::make_unique<CLBarrelCorrect>(
      static_cast<opencl_details *>(context), logger);
  if (prop->size > 0 && (prop->width > 0 || prop->height > 0)) {
    throw std::runtime_error("You must provide only one of width/height or size");
  }

  return prop;
}

extern "C" void
set_dynamic_properties(const std::unordered_map<std::string, std::string> &input,
    barrelcorrect_properties *prop, Ax::Logger & /*logger*/)
{
  prop->downstream_supports_opencl = Ax::get_property(input, "downstream_supports_opencl",
      "barrelcorrect_dynamic_properties", prop->downstream_supports_opencl);
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

constexpr std::array valid_formats = {
  AxVideoFormat::RGB,
  AxVideoFormat::BGR,
  AxVideoFormat::RGBA,
  AxVideoFormat::BGRA,
  AxVideoFormat::GRAY8,
};

const char *const name = "Barrel Correction";

extern "C" AxDataInterface
set_output_interface(const AxDataInterface &interface,
    const barrelcorrect_properties *prop, Ax::Logger &logger)
{
  AxDataInterface output{};
  if (std::holds_alternative<AxVideoInterface>(interface)) {
    auto in_info = std::get<AxVideoInterface>(interface);
    auto out_info = in_info;
    auto width = in_info.info.width;
    int height = in_info.info.height;
    if (prop->size != 0) {
      std::tie(width, height) = determine_width_height(in_info, prop->size);
    } else if (prop->width != 0 && prop->height != 0) {
      width = prop->width;
      height = prop->height;
    }
    out_info.info.width = width;
    out_info.info.height = height;
    out_info.info.actual_height = height;

    auto format = prop->out_format.empty() ? out_info.info.format :
                                             AxVideoFormatFromString(prop->out_format);
    out_info.info.format = format;
    Ax::validate_output_format(out_info.info.format, prop->out_format, name, valid_formats);
    output = out_info;
  }
  return output;
}

extern "C" void
transform(const AxDataInterface &input, const AxDataInterface &output,
    const barrelcorrect_properties *prop, unsigned int /*subframe_idx*/,
    unsigned int /*total_subframes*/,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> & /*meta*/,
    Ax::Logger &logger)
{
  auto in_info = std::get<AxVideoInterface>(input);
  auto out_info = std::get<AxVideoInterface>(output);

  //  Validate input and output formats

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
    throw std::runtime_error("Barrel Correction does not work with the input format: "
                             + AxVideoFormatToString(input_details[0].format));
  }
  Ax::validate_output_format(output_details[0].format, prop->out_format, name, valid_formats);
  prop->barrelcorrect->run(input_details[0], output_details[0], *prop);
}

extern "C" int
query_supports(Ax::PluginFeature feature, const barrelcorrect_properties *prop,
    Ax::Logger &logger)
{
  if (feature == Ax::PluginFeature::opencl_buffers) {
    return 1;
  }
  return Ax::PluginFeatureDefaults(feature);
}
