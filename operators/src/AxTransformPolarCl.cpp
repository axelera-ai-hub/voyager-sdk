// Copyright Axelera AI, 2025
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


class CLPolarTransform;
struct polar_properties {
  int width{};
  int height{};
  int size{};
  float center_x{ 0.5f };
  float center_y{ 0.5f };
  float start_angle{ M_PI / 2.0f };
  bool rotate180{ true };
  float max_radius{};
  bool inverse{ false };
  bool linear_polar{ true };
  std::string format{};
  bool downstream_supports_opencl{ false };
  std::unique_ptr<CLPolarTransform> polar_transform;
};

const char *polar_transform_kernel = R"##(

#undef M_PI
#define M_PI 3.14159265358979323846f

float2 polar_to_cartesian(float start_angle, float rho, float theta, float center_x, float center_y, float max_radius, int linear_polar)
{
    if (linear_polar) {
        // Linear polar mapping
        rho = rho * max_radius;
    } else {
        // Semi-log polar mapping
        rho = exp(rho * log(max_radius + 1.0f)) - 1.0f;
    }

    float x = center_x + rho * cos(theta-start_angle);
    float y = center_y + rho * sin(theta-start_angle);

    return (float2)(x, y);
}

float2 cartesian_to_polar(float start_angle, float x, float y, float center_x, float center_y, float max_radius, int linear_polar)
{
    float dx = x - center_x;
    float dy = y - center_y;

    float rho = sqrt(dx * dx + dy * dy);
    float theta = atan2(dy, dx);
    theta -= start_angle;

    // Normalize theta to [0, 2*PI]
    if (theta < 0.0f) theta += 2.0f * (float)M_PI;

    if (linear_polar) {
        // Linear polar mapping
        rho = rho / max_radius;
    } else {
        // Semi-log polar mapping
        rho = log(rho + 1.0f) / log(max_radius + 1.0f);
    }

    return (float2)(rho, theta);
}


float2 transform_coordinates(int row, int col, int in_width, int in_height, int out_width, int out_height,
                           float center_x, float center_y, float max_radius, int inverse, int linear_polar,
                           float start_angle, int rotate180) {
    float2 src_coords;
    if (!inverse) {
        // Polar to Cartesian (unwrap polar image back to cartesian)
        float rho = (float)row / (float)(out_height - 1);
        float theta = (float)col / (float)(out_width - 1) * 2.0f * (float)M_PI;
        if (rotate180) {
          rho = 1.0f - rho;
          theta += (float)M_PI;
          if (theta >= 2.0f * (float)M_PI) theta -= 2.0f * (float)M_PI;
        }
        src_coords = polar_to_cartesian(start_angle, rho, theta, center_x * in_width, center_y * in_height, max_radius, linear_polar);
    } else {
        // Cartesian to Polar (wrap cartesian image to polar)
        float x = (float)col;
        float y = (float)row;
        float2 polar = cartesian_to_polar(start_angle, x, y, center_x * out_width, center_y * out_height, max_radius,  linear_polar);

        // Map polar coordinates to output space
        float rho_norm = polar.x;
        float theta_norm = polar.y / (2.0f * (float)M_PI);

        src_coords.x = theta_norm * (in_width - 1);
        src_coords.y = rho_norm * (in_height - 1);

        if (rotate180) {
            // Rotate input coordinates by 180 degrees
            src_coords.x = in_width - 1 - src_coords.x;
            src_coords.y = in_height - 1 - src_coords.y;
        }
    }
    return src_coords;
}

__kernel void polar_transform(%s__global %s *out, int4 image_dims,
                        int4 strides, int4 offsets, float center_x, float center_y, float max_radius, int inverse, int linear_polar, float start_angle, int rotate180,
                        float16 color_matrix) {

    const int col = get_global_id(0);
    const int row = get_global_id(1);
    if (row >= image_dims.w || col >= image_dims.z) {
      return;
    }
    float2 corrected = transform_coordinates(row, col, image_dims.x, image_dims.y, image_dims.z, image_dims.w,
                           center_x, center_y, max_radius, inverse, linear_polar,
                           start_angle, rotate180);
    image_description img = {image_dims, strides, offsets, (int4)(0, 0, image_dims.z, image_dims.w), (int4)(0,0,0,0)};
    uchar fill = 0;
)##";

using ax_utils::buffer_details;
using ax_utils::CLProgram;
using ax_utils::opencl_details;
class CLPolarTransform
{
  using buffer = CLProgram::ax_buffer;
  using kernel = CLProgram::ax_kernel;

  public:
  CLPolarTransform(opencl_details *context, Ax::Logger &logger)
      : program("", context, logger)
  {
  }

  float calculate_max_radius(const buffer_details &in, float center_x, float center_y)
  {
    float cx = center_x * in.width;
    float cy = center_y * in.height;

    // Calculate distance to corners
    float d1 = sqrt(cx * cx + cy * cy);
    float d2 = sqrt((in.width - cx) * (in.width - cx) + cy * cy);
    float d3 = sqrt(cx * cx + (in.height - cy) * (in.height - cy));
    float d4 = sqrt(
        (in.width - cx) * (in.width - cx) + (in.height - cy) * (in.height - cy));

    return fmax(fmax(d1, d2), fmax(d3, d4));
  }

  ax_utils::CLProgram::ax_kernel build_kernel(ax_utils::CLProgram &program,
      AxVideoFormat in_format, AxVideoFormat out_format, int flip_type, int num_planes)
  {
    std::string kernel_code = polar_transform_kernel;

    auto input_details = ax_utils::get_input_details(
        in_format, ax_utils::Interpolation::bilinear, num_planes);
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

    final_kernel += sampler_code;
    final_kernel += output_code;
    final_kernel = ax_utils::get_kernel_utils(flip_type, program.has_fp16()) + final_kernel;

    return program.build_kernel_from_source(final_kernel, "polar_transform");
  }


  int run(const buffer_details &in, const buffer_details &out, const polar_properties &prop)
  {
    auto num_planes = ax_utils::get_num_planes(in);
    if (!converter) {
      converter = build_kernel(program, in.format, out.format, 0, num_planes);
    }
    bool start_flush = prop.downstream_supports_opencl == 0;
    auto outbuf = program.create_buffer(out, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR);

    cl_float max_radius = prop.max_radius;
    if (max_radius == 0.0f) {
      max_radius = calculate_max_radius(in, prop.center_x, prop.center_y);
    }

    std::array<cl_int, 4> image_dims = { in.width, in.height, out.width, out.height };
    auto strides = ax_utils::build_strides(in, out);
    auto offsets = ax_utils::build_offsets(in, out, num_planes);
    auto in_bufs = program.create_buffers(1, ax_utils::determine_buffer_size(in),
        CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, in.data, in.offsets.size());
    auto matrix = ax_utils::get_color_conversion_matrix(in.format, out.format);

    program.set_kernel_args(*converter, 0, in_bufs, *outbuf, image_dims,
        strides, offsets, static_cast<cl_float>(prop.center_x),
        static_cast<cl_float>(prop.center_y), max_radius,
        static_cast<cl_int>(prop.inverse), static_cast<cl_int>(prop.linear_polar),
        static_cast<cl_float>(prop.start_angle),
        static_cast<cl_int>(prop.rotate180), matrix);
    return run_kernel(program, *converter, in, out, in_bufs[0], outbuf, start_flush);
  }

  private:
  CLProgram program;
  int error{};
  kernel converter{ nullptr };
};


extern "C" const std::unordered_set<std::string> &
allowed_properties()
{
  static const std::unordered_set<std::string> allowed_properties{ "center_x",
    "center_y", "max_radius", "inverse", "linear_polar", "format", "width",
    "height", "size", "start_angle", "rotate180" };
  return allowed_properties;
}

extern "C" std::shared_ptr<void>
init_and_set_static_properties_with_context(
    const std::unordered_map<std::string, std::string> &input, void *context, Ax::Logger &logger)
{
  auto prop = std::make_shared<polar_properties>();

  prop->center_x
      = Ax::get_property(input, "center_x", "polar_static_properties", prop->center_x);
  prop->center_y
      = Ax::get_property(input, "center_y", "polar_static_properties", prop->center_y);
  prop->max_radius = Ax::get_property(
      input, "max_radius", "polar_static_properties", prop->max_radius);
  prop->inverse
      = Ax::get_property(input, "inverse", "polar_static_properties", prop->inverse);
  prop->linear_polar = Ax::get_property(
      input, "linear_polar", "polar_static_properties", prop->linear_polar);
  prop->format = Ax::get_property(input, "format", "polar_dynamic_properties", prop->format);
  prop->size = Ax::get_property(input, "size", "polar_static_properties", prop->size);
  prop->width = Ax::get_property(input, "width", "polar_static_properties", prop->width);
  prop->height = Ax::get_property(input, "height", "polar_static_properties", prop->height);
  prop->start_angle = Ax::get_property(
      input, "start_angle", "polar_static_properties", prop->start_angle);
  prop->rotate180 = Ax::get_property(
      input, "rotate180", "polar_static_properties", prop->rotate180);

  prop->polar_transform = std::make_unique<CLPolarTransform>(
      static_cast<opencl_details *>(context), logger);
  if (prop->size > 0 && (prop->width > 0 || prop->height > 0)) {
    throw std::runtime_error("You must provide only one of width/height or size");
  }

  return prop;
}

extern "C" void
set_dynamic_properties(const std::unordered_map<std::string, std::string> &input,
    polar_properties *prop, Ax::Logger & /*logger*/)
{
  prop->downstream_supports_opencl = Ax::get_property(input, "downstream_supports_opencl",
      "polar_dynamic_properties", prop->downstream_supports_opencl);
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

std::array valid_formats = {
  AxVideoFormat::RGB,
  AxVideoFormat::BGR,
  AxVideoFormat::RGBA,
  AxVideoFormat::BGRA,
  AxVideoFormat::GRAY8,
};

const char *name = "Polar Transform";

extern "C" AxDataInterface
set_output_interface(const AxDataInterface &interface,
    const polar_properties *prop, Ax::Logger &logger)
{
  AxDataInterface output{};
  if (std::holds_alternative<AxVideoInterface>(interface)) {
    auto in_info = std::get<AxVideoInterface>(interface);
    auto out_info = in_info;
    auto width = in_info.info.width;
    int height = in_info.info.height;
    if (prop->size) {
      std::tie(width, height) = determine_width_height(in_info, prop->size);
    } else if (prop->width != 0 && prop->height != 0) {
      width = prop->width;
      height = prop->height;
    }
    out_info.info.width = width;
    out_info.info.height = height;
    out_info.info.actual_height = height;

    auto format = prop->format.empty() ? out_info.info.format :
                                         AxVideoFormatFromString(prop->format);
    out_info.info.format = format;
    Ax::validate_output_format(out_info.info.format, prop->format, name, valid_formats);
    out_info.info.stride = width * AxVideoFormatNumChannels(format);
    out_info.strides.assign(1, size_t(out_info.info.stride));
    out_info.offsets.assign(1, size_t{ 0 });
    output = out_info;
  }
  return output;
}

extern "C" void
transform(const AxDataInterface &input, const AxDataInterface &output,
    const polar_properties *prop, unsigned int, unsigned int,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &, Ax::Logger &logger)
{
  auto in_info = std::get<AxVideoInterface>(input);
  auto out_info = std::get<AxVideoInterface>(output);

  auto input_details = ax_utils::extract_buffer_details(input);
  if (input_details.size() != 1) {
    throw std::runtime_error("polar transform works on single video input only");
  }

  auto output_details = ax_utils::extract_buffer_details(output);
  if (output_details.size() != 1) {
    throw std::runtime_error("polar transform works on single video output only");
  }
  auto valid_formats = std::array{
    AxVideoFormat::RGB,
    AxVideoFormat::BGR,
    AxVideoFormat::RGBA,
    AxVideoFormat::BGRA,
    AxVideoFormat::NV12,
    AxVideoFormat::NV16,
    AxVideoFormat::I420,
    AxVideoFormat::Y42B,
    AxVideoFormat::YUY2,
    AxVideoFormat::GRAY8,
  };
  if (std::none_of(valid_formats.begin(), valid_formats.end(), [input_details](auto format) {
        return format == input_details[0].format;
      })) {
    throw std::runtime_error("Polar Transform does not work with the input format: "
                             + AxVideoFormatToString(input_details[0].format));
  }
  Ax::validate_output_format(output_details[0].format, prop->format, name, valid_formats);
  prop->polar_transform->run(input_details[0], output_details[0], *prop);
}

extern "C" int
query_supports(Ax::PluginFeature feature, const polar_properties *prop, Ax::Logger &logger)
{
  if (feature == Ax::PluginFeature::opencl_buffers) {
    return 1;
  }
  return 0;
}
