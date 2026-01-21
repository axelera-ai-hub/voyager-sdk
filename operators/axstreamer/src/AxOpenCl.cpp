// Copyright Axelera AI, 2026
#include "AxOpenCl.hpp"

#include <iostream>
#include <string_view>
#include <vector>
#include "AxStreamerUtils.hpp"

namespace ax_utils
{
struct local_size {
  size_t width;
  size_t height;
};

struct opencl_error {
  cl_int code;
  std::string_view message;
};

constexpr std::array<opencl_error, 59> opencl_error_tab = { {
    { 0, "Success" },
    { -1, "Device not found" },
    { -2, "Device not available" },
    { -3, "Compiler not available" },
    { -4, "Memory object allocation failure" },
    { -5, "Out of resources" },
    { -6, "Out of host memory" },
    { -7, "Profiling information not available" },
    { -8, "Memory copy overlap" },
    { -9, "Image format mismatch" },
    { -10, "Image format not supported" },
    { -11, "Build program failure" },
    { -12, "Map failure" },
    { -13, "Misaligned sub buffer offset" },
    { -14, "Exec status error for events in wait list" },
    { -15, "Compile program failure" },
    { -16, "Linker not available" },
    { -17, "Link program failure" },
    { -18, "Device partition failed" },
    { -19, "Kernel arg info not available" },
    { -30, "Invalid value" },
    { -31, "Invalid device type" },
    { -32, "Invalid platform" },
    { -33, "Invalid device" },
    { -34, "Invalid context" },
    { -35, "Invalid queue properties" },
    { -36, "Invalid command queue" },
    { -37, "Invalid host pointer" },
    { -38, "Invalid memory object" },
    { -39, "Invalid image format descriptor" },
    { -40, "Invalid image size" },
    { -41, "Invalid sampler" },
    { -42, "Invalid binary" },
    { -43, "Invalid build options" },
    { -44, "Invalid program" },
    { -45, "Invalid program executable" },
    { -46, "Invalid kernel name" },
    { -47, "Invalid kernel definition" },
    { -48, "Invalid kernel" },
    { -49, "Invalid arg index" },
    { -50, "Invalid arg value" },
    { -51, "Invalid arg size" },
    { -52, "Invalid kernel args" },
    { -53, "Invalid work dimension" },
    { -54, "Invalid work group size" },
    { -55, "Invalid work item size" },
    { -56, "Invalid global offset" },
    { -57, "Invalid event wait list" },
    { -58, "Invalid event" },
    { -59, "Invalid operation" },
    { -60, "Invalid GL object" },
    { -61, "Invalid buffer size" },
    { -62, "Invalid mip level" },
    { -63, "Invalid global work size" },
    { -64, "Invalid property" },
    { -65, "Invalid image descriptor" },
    { -66, "Invalid compiler options" },
    { -67, "Invalid linker options" },
    { -68, "Invalid device partition count" },
} };

std::string
cl_error_to_string(cl_int code)
{
  auto e = std::find_if(opencl_error_tab.begin(), opencl_error_tab.end(),
      [code](const opencl_error &err) { return err.code == code; });
  if (e != opencl_error_tab.end()) {
    return std::string{ e->message };
  }
  return "Unknown OpenCL error";
}

local_size
determine_local_work_size(size_t max_work_size)
{
  auto width = max_work_size;
  auto height = max_work_size;
  while (width * height > max_work_size) {
    if (width > height)
      width /= 2;
    else
      height /= 2;
  }
  return { width, height };
}

output_format
get_output_format(AxVideoFormat format, bool ignore_alpha)
{
  switch (format) {
    case AxVideoFormat::RGBA:
      return ignore_alpha ? RGB_OUTPUT : RGBA_OUTPUT;
    case AxVideoFormat::BGRA:
      return ignore_alpha ? BGR_OUTPUT : BGRA_OUTPUT;
    case AxVideoFormat::RGB:
      return RGB_OUTPUT;
    case AxVideoFormat::BGR:
      return BGR_OUTPUT;
    case AxVideoFormat::GRAY8:
      return GRAY_OUTPUT;
    default:
      throw std::runtime_error(
          "Unsupported output format: " + AxVideoFormatToString(format));
  }
}

std::mutex CLProgram::cl_mutex;

// Test platform functionality
bool
platform_is_functional(cl_platform_id platform)
{
  cl_uint num_devices{};
  cl_device_id test_device{};
  cl_int test_error
      = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &test_device, &num_devices);
  if (test_error != CL_SUCCESS || num_devices == 0) {
    return false;
  }

  // Try to create a context to verify the platform works
  cl_context test_context
      = clCreateContext(nullptr, 1, &test_device, nullptr, nullptr, &test_error);
  if (test_error != CL_SUCCESS) {
    return false;
  }
  clReleaseContext(test_context);
  return true;
}

std::vector<cl_platform_id>
get_platform_ids(Ax::Logger &logger)
{
  cl_uint numPlatforms;
  auto error = clGetPlatformIDs(0, nullptr, &numPlatforms);
  if (error != CL_SUCCESS) {
    logger.throw_error("OpenCL not functional: Failed to get platform count! Error: "
                       + cl_error_to_string(error));
  }

  std::vector<cl_platform_id> platforms(numPlatforms);
  error = clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
  if (error != CL_SUCCESS) {
    logger.throw_error("OpenCL not functional: Failed to get platform IDs! Error: "
                       + cl_error_to_string(error));
  }
  return platforms;
}

std::vector<std::string>
get_order_preference(std::string_view preference, Ax::Logger &logger)
{
  using namespace std::string_literals;
  if (preference == "intel") {
    return { "Intel"s };
  } else if (preference == "arm") {
    return { "ARM"s, "rusticl"s };
  } else if (preference == "cpu") {
    return { "Portable Computing Language"s };
  } else if (preference == "gpu") {
    return { "NVIDIA"s, "Intel"s, "ARM"s };
  } else if (preference == "nvidia") {
    return { "NVIDIA"s };
  } else if (preference != "auto" && !preference.empty()) { // AUTO or any other value
    logger(AX_WARN) << "Unknown OpenCL preference: " << preference
                    << ", using auto" << std::endl;
  }
  return std::vector<std::string>{
    "Intel",
    "ARM",
    "rusticl",
    "NVIDIA",
    "Portable Computing Language",
  };
}

struct platform_info {
  cl_platform_id id;
  std::string name;
};

// Find a preferred platform based on the given preference and available platforms
// Returns an empty platform_info if no suitable platform is found
// If preferred is empty, it will return the first functional platform
platform_info
find_preferred_platform(std::string_view preferred, const std::vector<cl_platform_id> &platforms)
{
  for (cl_platform_id platform : platforms) {
    char platform_name[256];
    clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(platform_name),
        platform_name, nullptr);
    std::string platform_str(platform_name);

    if ((preferred.empty() || platform_str.find(preferred) != std::string::npos)
        && platform_is_functional(platform)) {
      return { platform, platform_str };
    }
  }
  return {};
}

opencl_details
build_cl_details(Ax::Logger &logger, const char *which_cl, void *display)
{
  opencl_details details{};
  details.version = AX_ALLOCATION_CONTEXT_VERSION;

  try {

    // Get all available platforms
    std::vector<cl_platform_id> platforms = get_platform_ids(logger);
    if (platforms.empty()) {
      logger.throw_error("OpenCL not functional: No platforms found!");
    }

    cl_platform_id selected_platform{};

    // Get OpenCL preference from environment variable
    // AX_OPENCL_PREFERENCE can be: "INTEL", "CPU", "GPU", "AUTO" (default)
    std::string preference = which_cl ? which_cl : "auto";
    // Define preference order for each setting
    auto preference_order = get_order_preference(preference, logger);

    // Try platforms in preference order
    for (auto preferred_platform : preference_order) {
      auto preferred = find_preferred_platform(preferred_platform, platforms);
      if (preferred.id) {
        selected_platform = preferred.id;
        logger(AX_INFO) << "Selected OpenCL platform: " << preferred.name
                        << " (preference: " << preference << ")" << std::endl;
        break; // Found a functional platform
      }
    }

    // If no preferred platform works, handle based on preference type
    if (!selected_platform) {
      if (preference == "auto") {
        // For AUTO mode, try any functional platform in remaining order
        logger(AX_DEBUG) << "No preferred platform functional in AUTO mode, trying remaining platforms"
                         << std::endl;
        auto preferred = find_preferred_platform("", platforms);
        if (preferred.id) {
          selected_platform = preferred.id;
          logger(AX_INFO) << "Selected OpenCL platform: " << preferred.name
                          << " (preference: " << preference << ")" << std::endl;

        } else {
          // For explicit preferences (INTEL, GPU, CPU), fail with clear error
          std::vector<std::string> available_platforms{};
          for (cl_platform_id platform : platforms) {
            char platform_name[256];
            clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(platform_name),
                platform_name, nullptr);
            available_platforms.push_back(std::string(platform_name));
          }

          logger.throw_error("Requested OpenCL platform '" + preference
                             + "' is not available or functional. " + "Available platforms: ["
                             + Ax::Internal::join(available_platforms, ", ") + "]. "
                             + "Use '--cl-platform auto' for automatic selection.");
        }
      }
    }

    if (!selected_platform) {
      logger.throw_error("No functional OpenCL platform of type '" + preference
                         + "' found. Available platform may be installed but not working correctly.");
    }

    details.extensions = init_extensions(selected_platform, display);

    cl_uint num_devices;
    auto error = get_device_id(
        selected_platform, &details.device_id, &num_devices, details.extensions);
    if (error != CL_SUCCESS) {
      logger.throw_error("OpenCL not functional: Failed to get device! Error: "
                         + cl_error_to_string(error));
    }

    details.context
        = create_context(selected_platform, details.device_id, details.extensions);
    if (error != CL_SUCCESS || !details.context) {
      logger.throw_error("OpenCL not functional: Failed to create OpenCL context, error: "
                         + cl_error_to_string(error));
    }
    details.commands = clCreateCommandQueue(details.context, details.device_id, 0, &error);
    if (error != CL_SUCCESS) {
      clReleaseContext(details.context);
      logger.throw_error("OpenCL not functional: Failed to create OpenCL command queue, error: "
                         + cl_error_to_string(error));
    }
    cl_bool unified{};
    clGetDeviceInfo(details.device_id, CL_DEVICE_HOST_UNIFIED_MEMORY,
        sizeof(unified), &unified, NULL);
    details.extensions.unified_memory = unified;

  } catch (std::exception &e) {
    logger(AX_ERROR) << "OpenCL initialization failed: " << e.what() << std::endl;
    details.device_id = nullptr;
    details.context = nullptr;
    details.commands = nullptr;
    details.exception = std::current_exception();
  }
  return details;
}


AxAllocationContextHandle
clone_context(AxAllocationContext *context)
{
  return context ? AxAllocationContextHandle(new AxAllocationContext{ *context }) :
                   AxAllocationContextHandle();
}

opencl_details
copy_context_and_retain(opencl_details *context)
{
  opencl_details details(*context);

  // Retain the context and command queue to avoid premature release
  if (details.context)
    clRetainContext(details.context);
  if (details.commands)
    clRetainCommandQueue(details.commands);
  return details;
}

CLProgram::CLProgram(const std::string &source, opencl_details *context, Ax::Logger &log)
    : logger(log), cl_details(context ? copy_context_and_retain(context) :
                                        build_cl_details(logger, nullptr, nullptr))
{
  if (cl_details.version != AX_ALLOCATION_CONTEXT_VERSION) {
    throw std::runtime_error(
        "Incompatible AxAllocationContext version, expected "
        + std::to_string(AX_ALLOCATION_CONTEXT_VERSION) + ", got "
        + std::to_string(cl_details.version)
        + "\nThis is probably due to an incompatible axstreamer and gstaxstreamer version.");
  }
  cl_int error = CL_SUCCESS;
  if (!cl_details.context) {
    std::rethrow_exception(cl_details.exception);
  }
  const char *sources[] = { source.c_str() };
  {
    std::lock_guard<std::mutex> lock(cl_mutex);
    //  We need to lock here as clCreateProgramWithSource and
    //  clBuildProgram are not thread safe on some platforms
    //  (e.g. Intel)
    //  See https://community.intel.com/t5/Intel-Graphics-Technology/Thread-safety-of-clCreateProgramWithSource/m-p/1247554
    program = clCreateProgramWithSource(cl_details.context, 1, sources, NULL, &error);
    error = error == CL_SUCCESS ? clBuildProgram(program, 0, NULL, NULL, NULL, NULL) : error;
  }
  if (error != CL_SUCCESS) {
    size_t param_value_size_ret;
    clGetProgramBuildInfo(program, cl_details.device_id, CL_PROGRAM_BUILD_LOG,
        0, NULL, &param_value_size_ret);
    std::vector<char> build_log(param_value_size_ret + 1);
    clGetProgramBuildInfo(program, cl_details.device_id, CL_PROGRAM_BUILD_LOG,
        param_value_size_ret, build_log.data(), NULL);
    std::cerr << "Build log:\n" << build_log.data() << std::endl;
    throw std::runtime_error("Failed to create OpenCL program");
  }
  clGetDeviceInfo(cl_details.device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE,
      sizeof max_work_group_size, &max_work_group_size, nullptr);
}


CLProgram::ax_kernel
CLProgram::get_kernel(const std::string &kernel_name) const
{
  int error = CL_SUCCESS;
  auto kernel = clCreateKernel(program, kernel_name.c_str(), &error);
  if (error != CL_SUCCESS) {
    throw std::runtime_error("Failed to create OpenCL kernel " + kernel_name
                             + ", error: " + cl_error_to_string(error));
  }
  return ax_kernel{ kernel };
}

std::vector<CLProgram::ax_buffer>
CLProgram::create_buffers(int elem_size, int num_elems, int flags,
    const buffer_initializer &ptr, int num_planes) const
{
  cl_int error = CL_SUCCESS;
  if (auto *p = std::get_if<opencl_buffer *>(&ptr)) {
    auto *p1 = *p;
    if (p1->event) {
      //  If we get here then upstream has begun mapping the buffer, so we
      //  need to wait for that to complete and then unmap
      clWaitForEvents(1, &*p1->event);
      p1->event.reset();
      clEnqueueUnmapMemObject(cl_details.commands, p1->buffer, p1->mapped, 0, NULL, NULL);
    }
  }
  auto buffers = create_optimal_buffer(cl_details.context, cl_details.extensions,
      elem_size, num_elems, flags, ptr, num_planes, error);
  auto wrapped_buffers = std::vector<ax_buffer>{};
  std::transform(buffers.begin(), buffers.end(), std::back_inserter(wrapped_buffers),
      [this](cl_mem buffer) { return ax_buffer{ buffer }; });
  return wrapped_buffers;
}

CLProgram::ax_buffer
CLProgram::create_buffer(int elem_size, int num_elems, int flags,
    const buffer_initializer &ptr, int num_planes) const
{
  auto buffers = create_buffers(elem_size, num_elems, flags, ptr, num_planes);
  auto buffer = buffers.empty() ? ax_buffer{ nullptr } : std::move(buffers[0]);
  return buffer;
}

CLProgram::ax_buffer
CLProgram::create_buffer(const buffer_details &details, int flags)
{
  return create_buffer(1, ax_utils::determine_buffer_size(details), flags,
      details.data, details.offsets.size());
}

CLProgram::flush_details
CLProgram::flush_output_buffer_async(const ax_buffer &out, int size)
{
  int ret = CL_SUCCESS;
  auto event = ax_event{ nullptr };
  auto mapped = clEnqueueMapBuffer(cl_details.commands, *out, CL_FALSE,
      CL_MAP_READ, 0, size, 0, NULL, &*event, &ret);
  return { ret, std::move(event), mapped };
}

int
CLProgram::unmap_buffer(ax_event event, const ax_buffer &out, void *mapped)
{
  auto num_events = event ? 1 : 0;
  cl_event *events = event ? &*event : nullptr;
  auto ret = clEnqueueUnmapMemObject(
      cl_details.commands, *out, mapped, num_events, events, NULL);
  if (ret != CL_SUCCESS) {
    throw std::runtime_error(
        "Failed to unmap output buffer, error: " + cl_error_to_string(ret));
  }
  return ret;
}

int
CLProgram::flush_output_buffer(const ax_buffer &out, int size)
{
  auto [result, event, mapped] = flush_output_buffer_async(out, size);
  if (result != CL_SUCCESS) {
    throw std::runtime_error(
        "Failed to map output buffer, error: " + cl_error_to_string(result));
  }
  return unmap_buffer(std::move(event), out, mapped);
}

CLProgram::flush_details
CLProgram::start_flush_output_buffer(const ax_buffer &out, int size)
{
  auto details = flush_output_buffer_async(out, size);
  if (details.result != CL_SUCCESS) {
    throw std::runtime_error("Failed to map output buffer, error: "
                             + cl_error_to_string(details.result));
  }
  return details;
}

int
CLProgram::acquireva(std::span<cl_mem> input_buffers)
{
  return acquire_va(cl_details.commands, cl_details.extensions, input_buffers);
}

int
CLProgram::releaseva(std::span<cl_mem> input_buffers)
{
  return release_va(cl_details.commands, cl_details.extensions, input_buffers);
}


int
CLProgram::execute_kernel(cl_kernel kernel, int num_dims, size_t global_work_size[3])
{
  auto local_size = determine_local_work_size(max_work_group_size);
  size_t local[3] = { local_size.width, local_size.height, 1 };
  size_t global[3] = { 0 };
  global[0] = (global_work_size[0] + local[0] - 1) & ~(local[0] - 1);
  global[1] = (global_work_size[1] + local[1] - 1) & ~(local[1] - 1);
  global[2] = global_work_size[2];
  auto *local_ptr = RPi_Hack ? nullptr : local;
  auto result = clEnqueueNDRangeKernel(cl_details.commands, kernel, num_dims,
      NULL, global, local_ptr, 0, NULL, NULL);
  if (result != CL_SUCCESS) {
    RPi_Hack = true;
    result = clEnqueueNDRangeKernel(cl_details.commands, kernel, num_dims, NULL,
        global, nullptr, 0, NULL, NULL);
    if (result != CL_SUCCESS) {
      throw std::runtime_error(
          "Failed to execute kernel, error: " + cl_error_to_string(result));
    }
  }
  return CL_SUCCESS;
}

CLProgram::~CLProgram()
{
  if (program)
    clReleaseProgram(program);
  if (cl_details.commands)
    clReleaseCommandQueue(cl_details.commands);
  if (cl_details.context)
    clReleaseContext(cl_details.context);
}

std::string
get_kernel_utils(int rotate_type)
{

  std::string utils = R"##(

#define advance_uchar_ptr(ptr, offset) ((__global uchar *)ptr + offset)
#define advance_uchar2_ptr(ptr, offset) ((__global uchar2 *)((__global uchar *)ptr + offset))
#define advance_uchar3_ptr(ptr, offset) ((__global uchar3 *)((__global uchar *)ptr + offset))
#define advance_uchar4_ptr(ptr, offset) ((__global uchar4 *)((__global uchar *)ptr + offset))

typedef enum output_format {
      RGBA_OUTPUT = 0,
      BGRA_OUTPUT = 1,
      RGB_OUTPUT = 3,
      BGR_OUTPUT = 4,
      GRAY_OUTPUT = 5
} output_format;

uchar RGB_to_GRAY(uchar3 rgb) {
    // Convert RGB to grayscale using the formula: Y = 0.299*R + 0.587*G + 0.114*B
    return (uchar)((rgb.x * 77 + rgb.y * 150 + rgb.z * 29) >> 8);
}


uchar3 YUV_to_RGB(uchar Y, uchar U, uchar V) {
    int C = (Y - 16) * 19071; // 1.164f * 16384
    int D = U - 128;
    int E = V - 128;

    // Integer approximation of YUV to RGB conversion
    int4 rgb;
    rgb.x = (C + (26149 * E)) >> 14;            // 1.596 * 16384
    rgb.y = (C - (6406 * D + 13320 * E)) >> 14; // 0.391F * 16384, 0.813 * 16384
    rgb.z = (C + (33063 * D)) >> 14;            // 2.018 * 16384
    return convert_uchar4_sat(rgb).xyz;
}

//  Convert YUV values to RGBA
uchar4 convert_YUV2RGBA(float3 yuv) {

    float Y = yuv.x - 16.0f;
    float U = yuv.y - 128.0f;
    float V = yuv.z - 128.0f;

    Y *= 1.164f;
    float4 rgba = (float4)(Y + 1.596F * V, Y - 0.391F * U - 0.813F * V, Y + 2.018F * U, 255.0f);
    return convert_uchar4_sat(rgba);
}

float3 bilinear(float3 p00, float3 p01, float3 p10, float3 p11, float xfrac, float yfrac) {
    float3 i1 = mix(p00, p01, xfrac);
    float3 i2 = mix(p10, p11, xfrac);
    return mix(i1, i2, yfrac);
}

typedef struct nv12_image {
    int width;
    int height;
    int ystride;
    int uvstride;
    int crop_x;
    int crop_y;
} nv12_image;

uchar4 nv12_sampler(__global const uchar *y_image, __global uchar2 *uv, float fx, float fy, const nv12_image *img) {
  float xpixel_left = fx - 0.5f;
  float ypixel_top = fy - 0.5f;

  int x1 = xpixel_left;
  int y1 = ypixel_top;
  float xfrac = xpixel_left - x1;
  float yfrac = ypixel_top - y1;

  x1 = max(x1, 0);
  y1 = max(y1, 0);
  int x2 = min(x1 + 1,img->width - 1);
  int y2 = min(y1 + 1,img->height - 1);
  x1 += img->crop_x;
  y1 += img->crop_y;
  x2 += img->crop_x;
  y2 += img->crop_y;

  float y00 = convert_float(y_image[y1 * img->ystride + x1]);
  float y01 = convert_float(y_image[y1 * img->ystride + x2]);
  float y10 = convert_float(y_image[y2 * img->ystride + x1]);
  float y11 = convert_float(y_image[y2 * img->ystride + x2]);

  int ux1 = x1 / 2;
  int uy1 = y1 / 2;
  int ux2 = x2 / 2;
  int uy2 = y2 / 2;

  bool need_right = ux1 == ux2;
  bool need_bottom = uy1 == uy2;
#define NV12_READ(x, y, p, stride) convert_float2(p[y * stride + x])

  float2 uv00 = NV12_READ(ux1, uy1, uv, img->uvstride);
  float2 uv01 = need_right ? NV12_READ(ux2, uy1, uv, img->uvstride) : uv00;
  float2 uv10 = need_bottom ? NV12_READ(ux1, uy2, uv, img->uvstride) : uv00;
  float2 uv11 = need_right ? (need_bottom ? NV12_READ(ux2, uy2, uv, img->uvstride) : uv01) : uv10;

  float3 yuv0 = (float3)(y00, uv00);
  float3 yuv1 = (float3)(y01, uv01);
  float3 yuv2 = (float3)(y10, uv10);
  float3 yuv3 = (float3)(y11, uv11);

  float3 yuv = bilinear(yuv0, yuv1, yuv2, yuv3, xfrac, yfrac);
  return convert_YUV2RGBA(yuv);
}

typedef struct i420_image {
    int width;
    int height;
    int ystride;
    int ustride;
    int vstride;
    int crop_x;
    int crop_y;
} i420_image;

uchar4 i420_sampler(__global const uchar *y_image, __global const uchar *u, __global const uchar *v, float fx, float fy, const i420_image *img) {
  float xpixel_left = fx - 0.5f;
  float ypixel_top = fy - 0.5f;

  int x1 = xpixel_left;
  int y1 = ypixel_top;
  float xfrac = xpixel_left - x1;
  float yfrac = ypixel_top - y1;

  x1 = max(x1, 0);
  y1 = max(y1, 0);
  int x2 = min(x1 + 1,img->width - 1);
  int y2 = min(y1 + 1,img->height - 1);
  x1 += img->crop_x;
  y1 += img->crop_y;
  x2 += img->crop_x;
  y2 += img->crop_y;

  float y00 = convert_float(y_image[y1 * img->ystride + x1]);
  float y01 = convert_float(y_image[y1 * img->ystride + x2]);
  float y10 = convert_float(y_image[y2 * img->ystride + x1]);
  float y11 = convert_float(y_image[y2 * img->ystride + x2]);

  int ux1 = x1 / 2;
  int uy1 = y1 / 2;
  int ux2 = x2 / 2;
  int uy2 = y2 / 2;

  bool need_right = ux1 == ux2;
  bool need_bottom = uy1 == uy2;

#define I420_READ(x, y, pu, pv, ustride, vstride) convert_float2((uchar2)(pu[y * ustride + x], pv[y * vstride + x]))

  float2 uv00 = I420_READ(ux1, uy1, u, v, img->ustride, img->vstride);
  float2 uv01 = need_right ? I420_READ(ux2, uy1, u, v, img->ustride, img->vstride) : uv00;
  float2 uv10 = need_bottom ? I420_READ(ux1, uy2, u, v, img->ustride, img->vstride) : uv00;
  float2 uv11 = need_right ? (need_bottom ? I420_READ(ux2, uy2, u, v, img->ustride, img->vstride) : uv01) : uv10;

  float3 yuv0 = (float3)(y00, uv00);
  float3 yuv1 = (float3)(y01, uv01);
  float3 yuv2 = (float3)(y10, uv10);
  float3 yuv3 = (float3)(y11, uv11);

  float3 yuv = bilinear(yuv0, yuv1, yuv2, yuv3, xfrac, yfrac);
  return convert_YUV2RGBA(yuv);
}


typedef struct yuyv_image {
    int width;
    int height;
    int stride;
    int crop_x;
    int crop_y;
} yuyv_image;

uchar4  yuyv_sampler(__global uchar4 *in, float fx, float fy, const yuyv_image *img) {
  float xpixel_left = fx - 0.5f;
  float ypixel_top = fy - 0.5f;

  int x1 = xpixel_left;
  int y1 = ypixel_top;
  float xfrac = xpixel_left - x1;
  float yfrac = ypixel_top - y1;

  int x2 = min(x1 + 1, img->width - 1);
  int y2 = min(y1 + 1, img->height - 1);

  x1 += img->crop_x;
  y1 += img->crop_y;
  x2 += img->crop_x;
  y2 += img->crop_y;

  // Each YUYV pixel pair is stored as Y1 U Y2 V
  int idx1 = y1 * img->stride + (x1 >> 1);
  int idx2 = y1 * img->stride + (x2 >> 1);
  int idx3 = y2 * img->stride + (x1 >> 1);
  int idx4 = y2 * img->stride + (x2 >> 1);

  float4 p00 = convert_float4(in[idx1]);
  float4 p01 = convert_float4(in[idx2]);
  float4 p10 = convert_float4(in[idx3]);
  float4 p11 = convert_float4(in[idx4]);

  // Select correct Y and UV values based on even/odd position
  float3 in00 = (x1 & 1) ? (float3)(p00.z, p00.y, p00.w) : (float3)(p00.x, p00.y, p00.w);
  float3 in01 = (x2 & 1) ? (float3)(p01.z, p01.y, p01.w) : (float3)(p01.x, p01.y, p01.w);
  float3 in10 = (x1 & 1) ? (float3)(p10.z, p10.y, p10.w) : (float3)(p10.x, p10.y, p10.w);
  float3 in11 = (x2 & 1) ? (float3)(p11.z, p11.y, p11.w) : (float3)(p11.x, p11.y, p11.w);

  float3 yuv = bilinear(in00, in01, in10, in11, xfrac, yfrac);
  return convert_YUV2RGBA(yuv);
}

typedef struct rgb_image {
    int width;
    int height;
    int stride;
    int crop_x;
    int crop_y;
} rgb_image;

typedef struct gray8_image {
    int width;
    int height;
    int stride;
    int crop_x;
    int crop_y;
} gray8_image;

uchar gray8_sampler_bl(__global const uchar *image, float fx, float fy, const gray8_image *img) {
    //  Here we add in the offsets to the pixel from the crop meta
    float xpixel_left = clamp(fx - 0.5f, 0.0f, (float)(img->width - 1));
    float ypixel_top = clamp(fy - 0.5f, 0.0f, (float)(img->height - 1));

    int x1 = (int)floor(xpixel_left);
    int y1 = (int)floor(ypixel_top);
    float xfrac = xpixel_left - x1;
    float yfrac = ypixel_top - y1;

    int x2 = min(x1 + 1, img->width - 1);
    int y2 = min(y1 + 1, img->height - 1);

    x1 += img->crop_x;
    y1 += img->crop_y;
    x2 += img->crop_x;
    y2 += img->crop_y;

    float p00 = (float)image[y1 * img->stride + x1];
    float p01 = (float)image[y1 * img->stride + x2];
    float p10 = (float)image[y2 * img->stride + x1];
    float p11 = (float)image[y2 * img->stride + x2];

    //  Performs bilinear interpolation with higher precision
    float i1 = mix(p00, p01, xfrac);
    float i2 = mix(p10, p11, xfrac);
    float value = mix(i1, i2, yfrac);

    return convert_uchar_sat(value);
}
uchar4 rgba_sampler_bl(__global const uchar4 *image, float fx, float fy, const rgb_image *img ) {
    //  Here we add in the offsets to the pixel from the crop meta
    float xpixel_left = fx - 0.5f;
    float ypixel_top = fy - 0.5f;

    int x1 = xpixel_left;
    int y1 = ypixel_top;
    float xfrac = xpixel_left - x1;
    float yfrac = ypixel_top - y1;

    x1 = max(x1, 0);
    y1 = max(y1, 0);
    int x2 = min(x1 + 1,img->width - 1);
    int y2 = min(y1 + 1,img->height - 1);

    x1 += img->crop_x;
    y1 += img->crop_y;
    x2 += img->crop_x;
    y2 += img->crop_y;
    float4 p00 = convert_float4(image[y1 * img->stride + x1]);
    float4 p01 = convert_float4(image[y1 * img->stride + x2]);
    float4 p10 = convert_float4(image[y2 * img->stride + x1]);
    float4 p11 = convert_float4(image[y2 * img->stride + x2]);

    //  Performs bilinear interpolation
    //  frac is the fraction of the pixel that is color2
    //  color = color1 + (color2 - color1) * frac

    float4 i1 = mix(p00, p01, xfrac);
    float4 i2 = mix(p10, p11, xfrac);
    uchar4 result = convert_uchar4_sat(mix(i1, i2, yfrac));
    return result;
}

uchar4 rgb_sampler_bl(__global const uchar *image, float fx, float fy, const rgb_image *img ) {
    //  Here we add in the offsets to the pixel from the crop meta
    float xpixel_left = fx - 0.5f;
    float ypixel_top = fy - 0.5f;

    int x1 = xpixel_left;
    int y1 = ypixel_top;
    float xfrac = xpixel_left - x1;
    float yfrac = ypixel_top - y1;

    x1 = max(x1, 0);
    y1 = max(y1, 0);
    int x2 = min(x1 + 1,img->width - 1);
    int y2 = min(y1 + 1,img->height - 1);

    x1 += img->crop_x;
    y1 += img->crop_y;
    x2 += img->crop_x;
    y2 += img->crop_y;
    __global const uchar * p_in = advance_uchar_ptr(image, y1 * img->stride);
    float3 p00 = convert_float3(vload3(x1, p_in));
    float3 p01 = convert_float3(vload3(x2, p_in));
    p_in = advance_uchar_ptr(image, y2 * img->stride);
    float3 p10 = convert_float3(vload3(x1, p_in));
    float3 p11 = convert_float3(vload3(x2, p_in));

    //  Performs bilinear interpolation
    //  frac is the fraction of the pixel that is color2
    //  color = color1 + (color2 - color1) * frac

    float3 i1 = mix(p00, p01, xfrac);
    float3 i2 = mix(p10, p11, xfrac);
    uchar4 result = convert_uchar4_sat((float4)(mix(i1, i2, yfrac), 255.0f));
    return result;
}

//  upper-right-diagonal
inline int2 get_input_coords_urd(int row, int col, int width, int height) {
  int new_x = (height - row) - 1;
  int new_y = (width - col) - 1;
  return (int2)(new_x, new_y);
}

//  upper-left-diagonal
inline int2 get_input_coords_uld(int row, int col, int width, int height) {
  int new_x = row;
  int new_y = col;
  return (int2)(new_x, new_y);
}


//  Rotate 90 clockwise
inline int2 get_input_coords_clockwise(int row, int col, int width, int height) {
  int new_x = row;
  int new_y = (width - col) - 1;
  return (int2)(new_x, new_y);
}

//  Rotate 90 counter clockwise
inline int2 get_input_coords_counter_clockwise(int row, int col, int width, int height) {
  int new_x = (height - row) - 1;
  int new_y = col;
  return (int2)(new_x, new_y);
}

//  Rotate 180
inline int2 get_input_coords_rotate180(int row, int col, int width, int height) {
  int new_x = (width - col) - 1;
  int new_y = (height - row) - 1;
  return (int2)(new_x, new_y);
}

//  Vertical flip
inline int2 get_input_coords_vertical(int row, int col, int width, int height) {
  int new_x = col;
  int new_y = (height - row) - 1;
  return (int2)(new_x, new_y);
}

//  Horizontal flip
inline int2 get_input_coords_horizontal(int row, int col, int width, int height) {
  int new_x = (width - col) - 1;
  int new_y = row;
  return (int2)(new_x, new_y);
}

//  Do nothing
inline int2 get_input_coords_none(int row, int col, int width, int height) {
  return (int2)(col, row);
}


)##";

  std::array xlate_names = {
    "get_input_coords_none"s,
    "get_input_coords_clockwise"s,
    "get_input_coords_rotate180"s,
    "get_input_coords_counter_clockwise"s,
    "get_input_coords_horizontal"s,
    "get_input_coords_vertical"s,
    "get_input_coords_uld"s,
    "get_input_coords_urd"s,
  };

  if (0 <= rotate_type && rotate_type < xlate_names.size()) {
    utils += "#define get_input_coords " + xlate_names[rotate_type];
  } else {
    utils += "#define get_input_coords " + xlate_names[0];
  }
  return utils;
}

} // namespace ax_utils
