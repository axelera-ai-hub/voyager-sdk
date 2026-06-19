// Copyright Axelera AI, 2024
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
  } else if (preference == "amd") {
    return { "AMD"s };
  } else if (preference != "auto" && !preference.empty()) { // AUTO or any other value
    logger(AX_WARN) << "Unknown OpenCL preference: " << preference
                    << ", using auto" << std::endl;
  }
  return std::vector<std::string>{
    "Intel",
    "ARM",
    "AMD",
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
build_cl_details(Ax::Logger &logger, const char *cl_choice, void *display)
{
  opencl_details details{};
  details.version = AX_ALLOCATION_CONTEXT_VERSION;

  std::string which_cl = cl_choice ? cl_choice : "";
  std::transform(which_cl.begin(), which_cl.end(), which_cl.begin(),
      [](unsigned char c) { return std::tolower(c); });

  try {

    // Get all available platforms
    std::vector<cl_platform_id> platforms = get_platform_ids(logger);
    if (platforms.empty()) {
      logger.throw_error("OpenCL not functional: No platforms found!");
    }

    cl_platform_id selected_platform{};

    // Get OpenCL preference from environment variable
    // AX_OPENCL_PREFERENCE can be: "INTEL", "CPU", "GPU", "AUTO" (default)
    std::string preference = which_cl.empty() ? "auto" : which_cl;
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
    auto error = get_device_id(selected_platform, &details.device_id,
        &num_devices, details.extensions, which_cl);
    if (error != CL_SUCCESS) {
      logger.throw_error("OpenCL not functional: Failed to get device! Error: "
                         + cl_error_to_string(error));
    }
    details.context
        = create_context(selected_platform, details.device_id, details.extensions);
    cl_command_queue_properties cq_props{};
    clGetDeviceInfo(details.device_id, CL_DEVICE_QUEUE_PROPERTIES,
        sizeof(cq_props), &cq_props, NULL);
    cl_queue_properties ooo_enable = (cq_props & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) ?
                                         CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE :
                                         0;
    cl_queue_properties props[] = { CL_QUEUE_PROPERTIES, ooo_enable, 0 };
    details.commands = clCreateCommandQueueWithProperties(details.context,
        details.device_id, ooo_enable ? props : nullptr, &error);
    if (error != CL_SUCCESS) {
      clReleaseContext(details.context);
      logger.throw_error("OpenCL not functional: Failed to create OpenCL command queue, error: "
                         + cl_error_to_string(error));
    }
    details.map_commands = clCreateCommandQueueWithProperties(
        details.context, details.device_id, nullptr, &error);
    if (error != CL_SUCCESS) {
      clReleaseContext(details.context);
      logger.throw_error("OpenCL not functional: Failed to create OpenCL command queue, error: "
                         + cl_error_to_string(error));
    }
    cl_bool unified{};
    clGetDeviceInfo(details.device_id, CL_DEVICE_HOST_UNIFIED_MEMORY,
        sizeof(unified), &unified, NULL);
    details.extensions.unified_memory = unified;

    cl_device_fp_config half_fp_config{};
    clGetDeviceInfo(details.device_id, CL_DEVICE_HALF_FP_CONFIG,
        sizeof(half_fp_config), &half_fp_config, nullptr);
    // Only enable fp16 for native hardware support; exclude software-only emulation.
    details.extensions.has_fp16
        = (half_fp_config != 0) && (half_fp_config != CL_FP_SOFT_FLOAT);

  } catch (std::exception &e) {
    logger(AX_ERROR) << "OpenCL initialization failed: " << e.what() << std::endl;
    details.device_id = nullptr;
    details.context = nullptr;
    details.commands = nullptr;
    details.map_commands = nullptr;
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
  if (details.map_commands)
    clRetainCommandQueue(details.map_commands);
  return details;
}

CLProgram::CLProgram(const std::string &source, opencl_details *context, Ax::Logger &log)
    : logger(log),
      cl_details(context ? copy_context_and_retain(context) :
                           build_cl_details(logger, nullptr, nullptr))
{
  if (cl_details.version != AX_ALLOCATION_CONTEXT_VERSION) {
    throw std::runtime_error(
        "Incompatible AxAllocationContext version, expected "
        + std::to_string(AX_ALLOCATION_CONTEXT_VERSION) + ", got "
        + std::to_string(cl_details.version)
        + "\nThis is probably due to an incompatible axstreamer and gstaxstreamer version.");
  }
  if (!cl_details.context) {
    std::rethrow_exception(cl_details.exception);
  }
  if (!source.empty()) {
    build_kernel_from_source(source, "");
  }
  clGetDeviceInfo(cl_details.device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE,
      sizeof max_work_group_size, &max_work_group_size, nullptr);
}

CLProgram::ax_kernel
CLProgram::build_kernel_from_source(const std::string &source, const std::string &kernel_name)
{
  const char *sources[] = { source.c_str() };
  cl_int error = CL_SUCCESS;
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
  return kernel_name.empty() ? ax_kernel{ nullptr } : get_kernel(kernel_name);
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
  if (auto *p = std::get_if<opencl_buffer *>(&ptr); p && *p) {
    auto *p1 = *p;
    if (p1->event && p1->mapped) {
      //  If we get here then upstream has begun mappinng the buffer, so we
      //  need to wait for that to complete and then unamp
      clEnqueueUnmapMemObject(
          cl_details.commands, p1->buffer, p1->mapped, 1, &*p1->event, NULL);
      p1->event.reset();
      p1->mapped = nullptr;
    }
  } else if (auto *p = std::get_if<opencl_planes *>(&ptr); p && *p) {
    //  For multi-plane input, flush any pending map on the Y plane
    auto *y = (*p)->planes.empty() ? nullptr : (*p)->planes[0];
    if (y && y->event && y->mapped) {
      clEnqueueUnmapMemObject(cl_details.commands, y->buffer, y->mapped, 1, &*y->event, NULL);
      y->event.reset();
      y->mapped = nullptr;
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
CLProgram::flush_output_buffer_async(const ax_buffer &out, int size, ax_event ev)
{
  int ret = CL_SUCCESS;
  auto event = *ev;
  auto num_events = event ? 1 : 0;
  auto new_event = ax_event{ nullptr };
  auto mapped = clEnqueueMapBuffer(cl_details.commands, *out, CL_FALSE,
      CL_MAP_READ, 0, size, num_events, &event, &*new_event, &ret);
  return { ret, std::move(new_event), mapped };
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

CLProgram::flush_details
CLProgram::start_flush_output_buffer(const ax_buffer &out, int size, ax_event ev)
{
  auto details = flush_output_buffer_async(out, size, std::move(ev));
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

static CLProgram::ax_event
enqueue_dmabuf_khr_op(cl_command_queue commands, clEnqueueAcquireExternalMemObjectsKHR_fn fn,
    std::span<cl_mem> buffers, CLProgram::ax_event wait_event)
{
  if (!fn || buffers.empty())
    return wait_event;
  auto result = CLProgram::ax_event{ nullptr };
  auto num_waits = cl_uint(wait_event ? 1 : 0);
  auto *wait = wait_event ? &*wait_event : nullptr;
  if (auto err = fn(commands, buffers.size(), buffers.data(), num_waits, wait, &*result);
      err != CL_SUCCESS) {
    throw std::runtime_error("clEnqueueExternalMemObjectsKHR failed: "
                             + ax_utils::cl_error_to_string(err));
  }
  return result;
}

CLProgram::ax_event
CLProgram::acquire_dmabuf_khr(std::span<cl_mem> buffers, ax_event wait_event)
{
  return enqueue_dmabuf_khr_op(cl_details.commands,
      cl_details.extensions.clEnqueueAcquireExternalMemObjectsKHR, buffers,
      std::move(wait_event));
}

CLProgram::ax_event
CLProgram::release_dmabuf_khr(std::span<cl_mem> buffers, ax_event wait_event)
{
  return enqueue_dmabuf_khr_op(cl_details.commands,
      cl_details.extensions.clEnqueueReleaseExternalMemObjectsKHR, buffers,
      std::move(wait_event));
}


CLProgram::ax_event
CLProgram::execute_kernel(cl_kernel kernel, int num_dims,
    size_t global_work_size[3], ax_event wait_event)
{
  auto local_size = determine_local_work_size(max_work_group_size);
  size_t local[3] = { local_size.width, local_size.height, 1 };
  size_t global[3] = { 0 };
  global[0] = (global_work_size[0] + local[0] - 1) & ~(local[0] - 1);
  global[1] = (global_work_size[1] + local[1] - 1) & ~(local[1] - 1);
  global[2] = global_work_size[2];
  auto *local_ptr = RPi_Hack ? nullptr : local;
  auto event = ax_event{ nullptr };
  auto num_wait_events = wait_event ? 1 : 0;
  cl_event *wait_events = wait_event ? &*wait_event : nullptr;
  auto result = clEnqueueNDRangeKernel(cl_details.commands, kernel, num_dims,
      NULL, global, local_ptr, num_wait_events, wait_events, &*event);
  if (result != CL_SUCCESS) {
    RPi_Hack = true;
    result = clEnqueueNDRangeKernel(cl_details.commands, kernel, num_dims, NULL,
        global, nullptr, num_wait_events, wait_events, &*event);
    if (result != CL_SUCCESS) {
      throw std::runtime_error(
          "Failed to execute kernel, error: " + cl_error_to_string(result));
    }
  }
  return event;
}

CLProgram::~CLProgram()
{
  if (program)
    clReleaseProgram(program);
  if (cl_details.commands)
    clReleaseCommandQueue(cl_details.commands);
  if (cl_details.map_commands)
    clReleaseCommandQueue(cl_details.map_commands);
  if (cl_details.context)
    clReleaseContext(cl_details.context);
}

std::string
get_kernel_utils(int rotate_type, bool use_fp16)
{
  const char *precision_preamble_fp32 = R"##(
typedef float  REAL;
typedef float2 REAL2;
typedef float4 REAL4;
typedef float16 REAL16;
#define CONVERT_REAL4(x)  convert_float4(x)
#define CONVERT_REAL2(x)  convert_float2(x)
#define CONVERT_REAL(x)   convert_float(x)
)##";

  const char *precision_preamble_fp16 = R"##(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
typedef half   REAL;
typedef half2  REAL2;
typedef half4  REAL4;
typedef half16 REAL16;
#define CONVERT_REAL4(x)  convert_half4(x)
#define CONVERT_REAL2(x)  convert_half2(x)
#define CONVERT_REAL(x)   convert_half(x)
)##";

  std::string utils = use_fp16 ? precision_preamble_fp16 : precision_preamble_fp32;
  utils += R"##(

#define advance_uchar_ptr(ptr, offset) ((__global uchar *)ptr + offset)
#define advance_uchar2_ptr(ptr, offset) ((__global uchar2 *)((__global uchar *)ptr + offset))
#define advance_uchar3_ptr(ptr, offset) ((__global uchar3 *)((__global uchar *)ptr + offset))
#define advance_uchar4_ptr(ptr, offset) ((__global uchar4 *)((__global uchar *)ptr + offset))

REAL4 bilinear(REAL4 p00, REAL4 p01, REAL4 p10, REAL4 p11, float xfrac, float yfrac) {
    REAL4 i1 = mix(p00, p01, (REAL)xfrac);
    REAL4 i2 = mix(p10, p11, (REAL)xfrac);
    return mix(i1, i2, (REAL)yfrac);
}

typedef struct image_description {
    int4 image_dims;
    int4 strides;
    int4 offsets;
    int4 letterbox;
    int4 crop;
} image_description;


#define NV12_READ(x, y, p, stride) CONVERT_REAL2(p[y * stride + x])

uchar4 nv12_sampler_two_plane(__global const uchar *y_image, __global const uchar *uv_image, int out_x, int out_y, float fx, float fy, const image_description *img, uchar fill) {
  if (fx < 0 || fx >= img->image_dims.x || fy < 0 || fy >= img->image_dims.y) {
    return (uchar4)(16, 128, 128, 255);
  }

  float xpixel_left = fx - 0.5f;
  float ypixel_top = fy - 0.5f;

  int x1 = xpixel_left;
  int y1 = ypixel_top;

  int adj_x = out_x - img->letterbox.x;
  int adj_y = out_y - img->letterbox.y;
  if (adj_x < 0 || adj_y < 0 || adj_x >= img->letterbox.z || adj_y >= img->letterbox.w) {
    return (uchar4)(fill, 128, 128, 255);
  }

  float xfrac = xpixel_left - x1;
  float yfrac = ypixel_top - y1;

  x1 = max(x1, 0);
  y1 = max(y1, 0);
  int x2 = min(x1 + 1,img->image_dims.x - 1);
  int y2 = min(y1 + 1,img->image_dims.y - 1);
  x1 += img->crop.x;
  y1 += img->crop.y;
  x2 += img->crop.x;
  y2 += img->crop.y;

  int ystride = img->strides.x;
  REAL y00 = CONVERT_REAL(y_image[y1 * ystride + x1]);
  REAL y01 = CONVERT_REAL(y_image[y1 * ystride + x2]);
  REAL y10 = CONVERT_REAL(y_image[y2 * ystride + x1]);
  REAL y11 = CONVERT_REAL(y_image[y2 * ystride + x2]);

  int ux1 = x1 / 2;
  int uy1 = y1 / 2;
  int ux2 = x2 / 2;
  int uy2 = y2 / 2;

  bool need_right = ux1 != ux2;
  bool need_bottom = uy1 != uy2;

  __global uchar2 *in_uv2 = (__global uchar2 *)uv_image;
  int uvstride = img->strides.y / 2;
  REAL2 uv00 = NV12_READ(ux1, uy1, in_uv2, uvstride);
  REAL2 uv01 = need_right ? NV12_READ(ux2, uy1, in_uv2, uvstride) : uv00;
  REAL2 uv10 = need_bottom ? NV12_READ(ux1, uy2, in_uv2, uvstride) : uv00;
  REAL2 uv11 = need_right ? (need_bottom ? NV12_READ(ux2, uy2, in_uv2, uvstride) : uv01) : uv10;

  REAL4 yuv0 = (REAL4)(y00, uv00, 255);
  REAL4 yuv1 = (REAL4)(y01, uv01, 255);
  REAL4 yuv2 = (REAL4)(y10, uv10, 255);
  REAL4 yuv3 = (REAL4)(y11, uv11, 255);

  REAL4 yuv = bilinear(yuv0, yuv1, yuv2, yuv3, xfrac, yfrac);
  return convert_uchar4_sat(yuv);
}

uchar4 nv12_sampler(__global const uchar *y_image, int out_x, int out_y, float fx, float fy, const image_description *img, uchar fill) {
  return nv12_sampler_two_plane(y_image, y_image + img->offsets.y, out_x, out_y, fx, fy, img, fill);
}


uchar4 nv16_sampler_two_plane(__global const uchar *y_image, __global const uchar *uv_image, int out_x, int out_y, float fx, float fy, const image_description *img, uchar fill) {
  if (fx < 0 || fx >= img->image_dims.x || fy < 0 || fy >= img->image_dims.y) {
    return (uchar4)(16, 128, 128, 255);
  }
  float xpixel_left = fx - 0.5f;
  float ypixel_top = fy - 0.5f;

  int x1 = xpixel_left;
  int y1 = ypixel_top;
  int adj_x = out_x - img->letterbox.x;
  int adj_y = out_y - img->letterbox.y;
  if (adj_x < 0 || adj_y < 0 || adj_x >= img->letterbox.z || adj_y >= img->letterbox.w) {
    return (uchar4)(fill, 128, 128, 255);
  }

  float xfrac = xpixel_left - x1;
  float yfrac = ypixel_top - y1;

  x1 = max(x1, 0);
  y1 = max(y1, 0);
  int x2 = min(x1 + 1,img->image_dims.x - 1);
  int y2 = min(y1 + 1,img->image_dims.y - 1);
  x1 += img->crop.x;
  y1 += img->crop.y;
  x2 += img->crop.x;
  y2 += img->crop.y;

  int ystride = img->strides.x;
  REAL y00 = CONVERT_REAL(y_image[y1 * ystride + x1]);
  REAL y01 = CONVERT_REAL(y_image[y1 * ystride + x2]);
  REAL y10 = CONVERT_REAL(y_image[y2 * ystride + x1]);
  REAL y11 = CONVERT_REAL(y_image[y2 * ystride + x2]);

  int ux1 = x1 / 2;
  int uy1 = y1;
  int ux2 = x2 / 2;
  int uy2 = y2;

  bool need_right = ux1 != ux2;
  bool need_bottom = uy1 != uy2;

  __global uchar2 *in_uv2 = (__global uchar2 *)uv_image;
  int uvstride = img->strides.y / 2;
  REAL2 uv00 = NV12_READ(ux1, uy1, in_uv2, uvstride);
  REAL2 uv01 = need_right ? NV12_READ(ux2, uy1, in_uv2, uvstride) : uv00;
  REAL2 uv10 = need_bottom ? NV12_READ(ux1, uy2, in_uv2, uvstride) : uv00;
  REAL2 uv11 = need_right ? (need_bottom ? NV12_READ(ux2, uy2, in_uv2, uvstride) : uv01) : uv10;

  REAL4 yuv0 = (REAL4)(y00, uv00, 255);
  REAL4 yuv1 = (REAL4)(y01, uv01, 255);
  REAL4 yuv2 = (REAL4)(y10, uv10, 255);
  REAL4 yuv3 = (REAL4)(y11, uv11, 255);

  REAL4 yuv = bilinear(yuv0, yuv1, yuv2, yuv3, xfrac, yfrac);
  return convert_uchar4_sat(yuv);
}

uchar4 nv16_sampler(__global const uchar *y_image, int out_x, int out_y, float fx, float fy, const image_description *img, uchar fill) {
  return nv16_sampler_two_plane(y_image, y_image + img->offsets.y, out_x, out_y, fx, fy, img, fill);
}


#define I420_READ(x, y, pu, pv, ustride, vstride) CONVERT_REAL2((uchar2)(pu[y * ustride + x], pv[y * vstride + x]))

// I420 three-plane: Y in y_image, U in u_image, V in v_image (all offsets = 0).
uchar4 i420_sampler_three_plane(__global const uchar *y_image, __global const uchar *u_image, __global const uchar *v_image, int out_x, int out_y, float fx, float fy, const image_description *img, uchar fill) {
  if (fx < 0 || fx >= img->image_dims.x || fy < 0 || fy >= img->image_dims.y) {
    return (uchar4)(16, 128, 128, 255);
  }
  float xpixel_left = fx - 0.5f;
  float ypixel_top = fy - 0.5f;

  int x1 = xpixel_left;
  int y1 = ypixel_top;
  int adj_x = out_x - img->letterbox.x;
  int adj_y = out_y - img->letterbox.y;
  if (adj_x < 0 || adj_y < 0 || adj_x >= img->letterbox.z || adj_y >= img->letterbox.w) {
    return (uchar4)(fill, 128, 128, 255);
  }
  float xfrac = xpixel_left - x1;
  float yfrac = ypixel_top - y1;

  x1 = max(x1, 0);
  y1 = max(y1, 0);
  int x2 = min(x1 + 1, img->image_dims.x - 1);
  int y2 = min(y1 + 1, img->image_dims.y - 1);
  x1 += img->crop.x;
  y1 += img->crop.y;
  x2 += img->crop.x;
  y2 += img->crop.y;

  int ystride = img->strides.x;
  float y00 = convert_float(y_image[y1 * ystride + x1]);
  float y01 = convert_float(y_image[y1 * ystride + x2]);
  float y10 = convert_float(y_image[y2 * ystride + x1]);
  float y11 = convert_float(y_image[y2 * ystride + x2]);

  int ux1 = x1 / 2;
  int uy1 = y1 / 2;
  int ux2 = x2 / 2;
  int uy2 = y2 / 2;

  bool need_right = ux1 != ux2;
  bool need_bottom = uy1 != uy2;

  REAL2 uv00 = I420_READ(ux1, uy1, u_image, v_image, img->strides.y, img->strides.z);
  REAL2 uv01 = need_right ? I420_READ(ux2, uy1, u_image, v_image, img->strides.y, img->strides.z) : uv00;
  REAL2 uv10 = need_bottom ? I420_READ(ux1, uy2, u_image, v_image, img->strides.y, img->strides.z) : uv00;
  REAL2 uv11 = need_right ? (need_bottom ? I420_READ(ux2, uy2, u_image, v_image, img->strides.y, img->strides.z) : uv01) : uv10;

  REAL4 yuv0 = (REAL4)(y00, uv00, 255);
  REAL4 yuv1 = (REAL4)(y01, uv01, 255);
  REAL4 yuv2 = (REAL4)(y10, uv10, 255);
  REAL4 yuv3 = (REAL4)(y11, uv11, 255);

  REAL4 yuv = bilinear(yuv0, yuv1, yuv2, yuv3, xfrac, yfrac);
  return convert_uchar4_sat(yuv);
}

// I420 two-plane: Y in y_image, U+V back-to-back in uv_image.
// U starts at offset 0; V starts at offsets.z within uv_image.
uchar4 i420_sampler_two_plane(__global const uchar *y_image, __global const uchar *uv_image, int out_x, int out_y, float fx, float fy, const image_description *img, uchar fill) {
  return i420_sampler_three_plane(y_image, uv_image, uv_image + img->offsets.z, out_x, out_y, fx, fy, img, fill);
}

uchar4 i420_sampler(__global const uchar *y_image, int out_x, int out_y, float fx, float fy, const image_description *img, uchar fill) {
  return i420_sampler_three_plane(y_image, y_image + img->offsets.y, y_image + img->offsets.z, out_x, out_y, fx, fy, img, fill);
}


uchar4 yuyv_sampler(__global const uchar *in, int out_x, int out_y, float fx, float fy, const image_description *img, uchar fill) {
  __global const uchar4 *in4 = (__global const uchar4 *)in;
  if (fx < 0 || fx >= img->image_dims.x || fy < 0 || fy >= img->image_dims.y) {
    return (uchar4)(16, 128, 128, 255);
  }
  float xpixel_left = fx - 0.5f;
  float ypixel_top = fy - 0.5f;

  int x1 = xpixel_left;
  int y1 = ypixel_top;
  int adj_x = out_x - img->letterbox.x;
  int adj_y = out_y - img->letterbox.y;
  if (adj_x < 0 || adj_y < 0 || adj_x >= img->letterbox.z || adj_y >= img->letterbox.w) {
    return (uchar4)(fill, 128, 128, 255);
  }
  float xfrac = xpixel_left - x1;
  float yfrac = ypixel_top - y1;

  int x2 = min(x1 + 1, img->image_dims.x - 1);
  int y2 = min(y1 + 1, img->image_dims.y - 1);

  x1 += img->crop.x;
  y1 += img->crop.y;
  x2 += img->crop.x;
  y2 += img->crop.y;
  int stride = img->strides.x / sizeof(uchar4);
  // Each YUYV pixel pair is stored as Y1 U Y2 V
  int idx1 = y1 * stride + (x1 >> 1);
  int idx2 = y1 * stride + (x2 >> 1);
  int idx3 = y2 * stride + (x1 >> 1);
  int idx4 = y2 * stride + (x2 >> 1);

  REAL4 p00 = CONVERT_REAL4(in4[idx1]);
  REAL4 p01 = CONVERT_REAL4(in4[idx2]);
  REAL4 p10 = CONVERT_REAL4(in4[idx3]);
  REAL4 p11 = CONVERT_REAL4(in4[idx4]);

  // Select correct Y and UV values based on even/odd position
  REAL4 in00 = (x1 & 1) ? (REAL4)(p00.z, p00.y, p00.w, 255) : (REAL4)(p00.x, p00.y, p00.w, 255);
  REAL4 in01 = (x2 & 1) ? (REAL4)(p01.z, p01.y, p01.w, 255) : (REAL4)(p01.x, p01.y, p01.w, 255);
  REAL4 in10 = (x1 & 1) ? (REAL4)(p10.z, p10.y, p10.w, 255) : (REAL4)(p10.x, p10.y, p10.w, 255);
  REAL4 in11 = (x2 & 1) ? (REAL4)(p11.z, p11.y, p11.w, 255) : (REAL4)(p11.x, p11.y, p11.w, 255);

  REAL4 yuv = bilinear(in00, in01, in10, in11, xfrac, yfrac);
  return convert_uchar4_sat(yuv);
}

uchar gray8_sampler_bl(__global const uchar *image, int out_x, int out_y, float fx, float fy, const image_description *img, uchar fill) {
    //  Here we add in the offsets to the pixel from the crop meta
    if (fx < 0 || fx >= img->image_dims.x || fy < 0 || fy >= img->image_dims.y) {
      return fill;
    }
    float xpixel_left = clamp(fx - 0.5f, 0.0f, (float)(img->image_dims.x - 1));
    float ypixel_top = clamp(fy - 0.5f, 0.0f, (float)(img->image_dims.y - 1));

    int x1 = (int)floor(xpixel_left);
    int y1 = (int)floor(ypixel_top);

    int adj_x = out_x - img->letterbox.x;
    int adj_y = out_y - img->letterbox.y;
    if (adj_x < 0 || adj_y < 0 || adj_x >= img->letterbox.z || adj_y >= img->letterbox.w) {
      return fill;
    }

    float xfrac = xpixel_left - x1;
    float yfrac = ypixel_top - y1;

    int x2 = min(x1 + 1, img->image_dims.x - 1);
    int y2 = min(y1 + 1, img->image_dims.y - 1);

    x1 += img->crop.x;
    y1 += img->crop.y;
    x2 += img->crop.x;
    y2 += img->crop.y;

    REAL p00 = (REAL)image[y1 * img->strides.x + x1];
    REAL p01 = (REAL)image[y1 * img->strides.x + x2];
    REAL p10 = (REAL)image[y2 * img->strides.x + x1];
    REAL p11 = (REAL)image[y2 * img->strides.x + x2];

    //  Performs bilinear interpolation with higher precision
    REAL i1 = mix(p00, p01, (REAL)xfrac);
    REAL i2 = mix(p10, p11, (REAL)xfrac);
    REAL value = mix(i1, i2, (REAL)yfrac);

    return convert_uchar_sat(value);
}

uchar4 rgba_sampler_bl(__global const uchar *image, int out_x, int out_y, float fx, float fy, const image_description *img, uchar fill) {
    __global const uchar4 *image4 = (__global const uchar4 *)image;
    //  Here we add in the offsets to the pixel from the crop meta
    if (fx < 0 || fx >= img->image_dims.x || fy < 0 || fy >= img->image_dims.y) {
      return (uchar4)(fill, fill, fill, 255);
    }
    float xpixel_left = fx - 0.5f;
    float ypixel_top = fy - 0.5f;

    int x1 = xpixel_left;
    int y1 = ypixel_top;
    int adj_x = out_x - img->letterbox.x;
    int adj_y = out_y - img->letterbox.y;
    if (adj_x < 0 || adj_y < 0 || adj_x >= img->letterbox.z || adj_y >= img->letterbox.w) {
      return (uchar4)(fill, fill, fill, 255);
    }

    float xfrac = xpixel_left - x1;
    float yfrac = ypixel_top - y1;

    x1 = max(x1, 0);
    y1 = max(y1, 0);
    int x2 = min(x1 + 1,img->image_dims.x - 1);
    int y2 = min(y1 + 1,img->image_dims.y - 1);

    x1 += img->crop.x;
    y1 += img->crop.y;
    x2 += img->crop.x;
    y2 += img->crop.y;
    int stride = img->strides.x / sizeof(uchar4);
    REAL4 p00 = CONVERT_REAL4(image4[y1 * stride + x1]);
    REAL4 p01 = CONVERT_REAL4(image4[y1 * stride + x2]);
    REAL4 p10 = CONVERT_REAL4(image4[y2 * stride + x1]);
    REAL4 p11 = CONVERT_REAL4(image4[y2 * stride + x2]);

    //  Performs bilinear interpolation
    //  frac is the fraction of the pixel that is color2
    //  color = color1 + (color2 - color1) * frac

    REAL4 i1 = mix(p00, p01, (REAL)xfrac);
    REAL4 i2 = mix(p10, p11, (REAL)xfrac);
    uchar4 result = convert_uchar4_sat(mix(i1, i2, (REAL)yfrac));
    return result;
}

uchar4 rgb_sampler_bl(__global const uchar *image, int out_x, int out_y, float fx, float fy, const image_description *img, uchar fill) {
    //  Here we add in the offsets to the pixel from the crop meta
    if (fx < 0 || fx >= img->image_dims.x || fy < 0 || fy >= img->image_dims.y) {
      return (uchar4)(0, 0, 0, 255);
    }
    float xpixel_left = fx - 0.5f;
    float ypixel_top = fy - 0.5f;

    int x1 = xpixel_left;
    int y1 = ypixel_top;
    int adj_x = out_x - img->letterbox.x;
    int adj_y = out_y - img->letterbox.y;
    if (adj_x < 0 || adj_y < 0 || adj_x >= img->letterbox.z || adj_y >= img->letterbox.w) {
      return (uchar4)(fill, fill, fill, 255);
    }

    float xfrac = xpixel_left - x1;
    float yfrac = ypixel_top - y1;

    x1 = max(x1, 0);
    y1 = max(y1, 0);
    int x2 = min(x1 + 1,img->image_dims.x - 1);
    int y2 = min(y1 + 1,img->image_dims.y - 1);

    x1 += img->crop.x;
    y1 += img->crop.y;
    x2 += img->crop.x;
    y2 += img->crop.y;
    int stride = img->strides.x;
    __global const uchar * p_in = advance_uchar_ptr(image, y1 * stride);
    REAL4 p00 = CONVERT_REAL4((uchar4)(vload3(x1, p_in), 255));
    REAL4 p01 = CONVERT_REAL4((uchar4)(vload3(x2, p_in), 255));
    p_in = advance_uchar_ptr(image, y2 * stride);
    REAL4 p10 = CONVERT_REAL4((uchar4)(vload3(x1, p_in), 255));
    REAL4 p11 = CONVERT_REAL4((uchar4)(vload3(x2, p_in), 255));

    //  Performs bilinear interpolation
    //  frac is the fraction of the pixel that is color2
    //  color = color1 + (color2 - color1) * frac

    REAL4 i1 = mix(p00, p01, (REAL)xfrac);
    REAL4 i2 = mix(p10, p11, (REAL)xfrac);
    uchar4 result = convert_uchar4_sat(mix(i1, i2, (REAL)yfrac));
    return result;
}

float4 color_convert_float(uchar4 pixel, float16 matrix) {
    float4 in_pixel = convert_float4(pixel);
    float4 color = mad(in_pixel.x, matrix.s0123, mad(in_pixel.y, matrix.s4567, mad(in_pixel.z, matrix.s89ab, matrix.scdef)));
    color.w = in_pixel.w;
    return color;
}

)##";
  return utils;
}

std::string
get_rotation(int rotate_type)
{
  const char *urd = R"##(
  int new_x = (height - row) - 1;
  int new_y = (width - col) - 1;
  int2 corrected = (int2)(new_x, new_y);
)##";

  const char *uld = R"##(
  int new_x = row;
  int new_y = col;
  int2 corrected = (int2)(new_x, new_y);
)##";

  const char *clockwise = R"##(
  int new_x = row;
  int new_y = (width - col) - 1;
  int2 corrected = (int2)(new_x, new_y);
)##";

  const char *counter_clockwise = R"##(
  int new_x = (height - row) - 1;
  int new_y = col;
  int2 corrected = (int2)(new_x, new_y);
)##";

  const char *rotate180 = R"##(
  int new_x = (width - col) - 1;
  int new_y = (height - row) - 1;
  int2 corrected = (int2)(new_x, new_y);
)##";

  const char *vertical = R"##(
  int new_x = col;
  int new_y = (height - row) - 1;
  int2 corrected = (int2)(new_x, new_y);
)##";

  const char *horizontal = R"##(
  int new_x = (width - col) - 1;
  int new_y = row;
  int2 corrected = (int2)(new_x, new_y);
)##";

  const char *none = R"##(
  int2 corrected = (int2)(col, row);
)##";

  std::array flips = {
    none,
    clockwise,
    rotate180,
    counter_clockwise,
    horizontal,
    vertical,
    uld,
    urd,
  };
  return 0 <= rotate_type && rotate_type < flips.size() ? flips[rotate_type] : none;
}

CLProgram::flush_details
exec_kernel(CLProgram &program, cl_kernel k, const buffer_details &in,
    const buffer_details &out, CLProgram::ax_buffer &inbuf,
    CLProgram::ax_buffer &outbuf, bool start_flush)
{
  auto event = CLProgram::ax_event{ nullptr };
  if (auto *p = std::get_if<opencl_buffer *>(&in.data); p && *p) {
    auto *p1 = *p;
    if (p1->event && !p1->mapped) {
      //  This event synchronizes the kernel with the upstream kernel
      event = std::move(p1->event);
    }
  } else if (auto *p = std::get_if<opencl_planes *>(&in.data); p && *p) {
    //  For multi-plane input, synchronize on the Y plane's upstream event
    auto *y = (*p)->planes.empty() ? nullptr : (*p)->planes[0];
    if (y && y->event && !y->mapped) {
      event = std::move(y->event);
    }
  }

  //  KHR-imported DMA-bufs require explicit acquire/release around kernel use;
  //  ARM-imported ones handle synchronisation internally.
  cl_mem khr_bufs[2] = {};
  size_t khr_count = 0;
  if (program.uses_khr_dmabuf_import()) {
    if (ax_utils::is_dmabuf(in))
      khr_bufs[khr_count++] = *inbuf;
    if (ax_utils::is_dmabuf(out))
      khr_bufs[khr_count++] = *outbuf;
  }

  event = program.acquire_dmabuf_khr({ khr_bufs, khr_count }, std::move(event));

  size_t global_work_size[3] = { 1, 1, 1 };
  global_work_size[0] = out.width;
  global_work_size[1] = out.height;
  auto ev = program.execute_kernel(k, 2, global_work_size, std::move(event));

  ev = program.release_dmabuf_khr({ khr_bufs, khr_count }, std::move(ev));

  //  If the output is a dmabuf the downstream takes the fd directly — no
  //  mapping to host memory is needed or appropriate.
  if (start_flush && !ax_utils::is_dmabuf(out)) {
    return program.start_flush_output_buffer(
        outbuf, out.stride * out.height, std::move(ev));
  }
  return { CL_SUCCESS, std::move(ev), nullptr };
}

int
run_kernel(CLProgram &program, cl_kernel k, const buffer_details &in,
    const buffer_details &out, CLProgram::ax_buffer &inbuf,
    CLProgram::ax_buffer &outbuf, bool start_flush)
{
  auto details = exec_kernel(program, k, in, out, inbuf, outbuf, start_flush);
  if (details.event) {
    // The downstream does not support OpenCL buffers so the buffer has begun
    //  mapping to system memory. The event will be signalled when complete.
    //  Store this away so that when the buffer is mapped we just wait on the
    //  event.
    if (auto *p = std::get_if<opencl_buffer *>(&out.data)) {
      (*p)->event = std::move(details.event);
      (*p)->mapped = details.mapped;
    } else if (ax_utils::is_dmabuf(out)) {
      clWaitForEvents(1, &*details.event);
      details.event.reset();
    } else {
      // Non-dmabuf, non-opencl_buffer output: wait for the host-mapped buffer.
      clWaitForEvents(1, &*details.event);
      if (details.mapped) {
        program.unmap_buffer(CLProgram::ax_event{ {} }, outbuf, details.mapped);
      }
      details.event.reset();
    }
  }
  return 0;
}


bool
is_rgb(AxVideoFormat format)
{
  return format == AxVideoFormat::RGB || format == AxVideoFormat::RGBA;
}

bool
is_rgb_or_gray(AxVideoFormat format)
{
  return is_rgb(format) || format == AxVideoFormat::GRAY8;
}

bool
is_bgr(AxVideoFormat format)
{
  return format == AxVideoFormat::BGR || format == AxVideoFormat::BGRA;
}

bool
output_needs_swizzle(AxVideoFormat in_format, AxVideoFormat out_format)
{
  return is_bgr(in_format) != is_bgr(out_format);
}


std::array<float, 16> yuv_to_rgb_matrix = {
  // clang-format off
  1.16406F,  1.16406F, 1.16406F,   0.0F,
  0.00000F, -0.39100F, 2.01800F,   0.0F,
  1.59600F, -0.81300F, 0.00000F,   0.0F,
  -222.91F,   135.48F, -276.92F, 255.0F
  // clang-format on
};

std::array<float, 16> yuv_to_gray_matrix = {
  // clang-format off
  1.16406F, 1.16406F, 1.16406F,   0.0F,
  0.00052F, 0.00052F, 0.00052F,   0.0F,
  0.00003F, 0.00003F, 0.00003F,   0.0F,
  -18.703F, -18.703F, -18.703F, 255.0F
  // clang-format on
};

std::array<float, 16> rgb_to_gray_matrix = {
  // clang-format off
  0.299F, 0.299F, 0.299F,   0.0F,
  0.587F, 0.587F, 0.587F,   0.0F,
  0.114F, 0.114F, 0.114F,   0.0F,
    0.0F,   0.0F,   0.0F, 255.0F,
  // clang-format on
};

std::array<float, 16> bgr_to_gray_matrix = {
  // clang-format off
  0.114F, 0.114F, 0.114F,   0.0F,
  0.587F, 0.587F, 0.587F,   0.0F,
  0.299F, 0.299F, 0.299F,   0.0F,
    0.0F,   0.0F,   0.0F, 255.0F,
  // clang-format on
};

std::array<float, 16> identity_matrix = {
  // clang-format off
  1.0F, 0.0F, 0.0F, 0.0F,
  0.0F, 1.0F, 0.0F, 0.0F,
  0.0F, 0.0F, 1.0F, 0.0F,
  0.0F, 0.0F, 0.0F, 1.0F,
  // clang-format on
};

struct color_matrix_key {
  AxVideoFormat in_format;
  AxVideoFormat out_format;
  std::array<float, 16> *matrix;
};
color_matrix_key converters[] = {
  { AxVideoFormat::NV12, AxVideoFormat::RGB, &yuv_to_rgb_matrix },
  { AxVideoFormat::NV12, AxVideoFormat::BGR, &yuv_to_rgb_matrix },
  { AxVideoFormat::NV12, AxVideoFormat::GRAY8, &yuv_to_gray_matrix },
  { AxVideoFormat::NV16, AxVideoFormat::RGB, &yuv_to_rgb_matrix },
  { AxVideoFormat::NV16, AxVideoFormat::BGR, &yuv_to_rgb_matrix },
  { AxVideoFormat::NV16, AxVideoFormat::GRAY8, &yuv_to_gray_matrix },
  { AxVideoFormat::I420, AxVideoFormat::RGB, &yuv_to_rgb_matrix },
  { AxVideoFormat::I420, AxVideoFormat::BGR, &yuv_to_rgb_matrix },
  { AxVideoFormat::I420, AxVideoFormat::GRAY8, &yuv_to_gray_matrix },
  { AxVideoFormat::YUY2, AxVideoFormat::RGB, &yuv_to_rgb_matrix },
  { AxVideoFormat::YUY2, AxVideoFormat::BGR, &yuv_to_rgb_matrix },
  { AxVideoFormat::YUY2, AxVideoFormat::GRAY8, &yuv_to_gray_matrix },
  { AxVideoFormat::RGB, AxVideoFormat::GRAY8, &rgb_to_gray_matrix },
  { AxVideoFormat::BGR, AxVideoFormat::GRAY8, &bgr_to_gray_matrix },
  { AxVideoFormat::RGB, AxVideoFormat::RGB, &identity_matrix },
  { AxVideoFormat::RGB, AxVideoFormat::BGR, &identity_matrix },
  { AxVideoFormat::BGR, AxVideoFormat::BGR, &identity_matrix },
  { AxVideoFormat::BGR, AxVideoFormat::RGB, &identity_matrix },
  { AxVideoFormat::GRAY8, AxVideoFormat::GRAY8, &identity_matrix },
};

AxVideoFormat
remove_alpha_channel(AxVideoFormat format)
{
  switch (format) {
    case AxVideoFormat::RGBA:
      return AxVideoFormat::RGB;
    case AxVideoFormat::BGRA:
      return AxVideoFormat::BGR;
    default:
      return format;
  }
}

std::array<float, 16>
get_color_conversion_matrix(AxVideoFormat in_format, AxVideoFormat out_format)
{
  auto in = remove_alpha_channel(in_format);
  auto out = remove_alpha_channel(out_format);
  auto *p = std::find_if(std::begin(converters), std::end(converters),
      [in, out](const color_matrix_key &key) {
        return key.in_format == in && key.out_format == out;
      });
  if (p != std::end(converters)) {
    auto result = *(p->matrix);
    if (output_needs_swizzle(in_format, out_format)) {
      // Swizzle R and B channels
      std::swap(result[0], result[2]);
      std::swap(result[4], result[6]);
      std::swap(result[8], result[10]);
      std::swap(result[12], result[14]);
    }
    return result;
  }
  throw std::runtime_error("Unsupported color conversion from " + AxVideoFormatToString(in_format)
                           + " to " + AxVideoFormatToString(out_format));
}

std::array<float, 16>
get_color_conversion_matrix_with_norm(AxVideoFormat in_format, AxVideoFormat out_format,
    const std::vector<cl_float> &mul, const std::vector<cl_float> &add)
{
  auto M = get_color_conversion_matrix(in_format, out_format);
  // Fuse per-channel affine norm (out_i = color_i * mul_i + add_i) into the
  // matrix. Rows 0-2 are input channel weights; row 3 is the bias vector.
  for (int i = 0; i < 4; ++i) {
    float m = mul[i];
    float a = add[i];
    M[0 * 4 + i] *= m;
    M[1 * 4 + i] *= m;
    M[2 * 4 + i] *= m;
    M[3 * 4 + i] = M[3 * 4 + i] * m + a;
  }
  return M;
}

std::array<cl_int, 4>
build_strides(const buffer_details &in, const buffer_details &out)
{
  if (in.format == AxVideoFormat::NV12 || in.format == AxVideoFormat::NV16) {
    return {
      static_cast<cl_int>(in.strides[0]),
      static_cast<cl_int>(in.strides[1]),
      0,
      static_cast<cl_int>(out.strides[0]),
    };
  }
  if (in.format == AxVideoFormat::I420) {
    return {
      static_cast<cl_int>(in.strides[0]),
      static_cast<cl_int>(in.strides[1]),
      static_cast<cl_int>(in.strides[2]),
      static_cast<cl_int>(out.strides[0]),
    };
  }
  return {
    static_cast<cl_int>(in.strides[0]),
    0,
    0,
    static_cast<cl_int>(out.strides[0]),
  };
}

std::array<cl_int, 4>
build_offsets(const buffer_details &in, const buffer_details &out, int num_planes)
{
  if (in.format == AxVideoFormat::NV12 || in.format == AxVideoFormat::NV16) {
    if (num_planes == 2) {
      //  Separate UV buffer: starts at byte 0.
      return { 0, 0, 0, 0 };
    }
    //  Single buffer: UV at in.offsets[1].
    return { 0, static_cast<cl_int>(in.offsets[1]), 0, 0 };
  }
  if (in.format == AxVideoFormat::I420) {
    if (num_planes == 2) {
      //  UV combined buffer: U at offset 0, V at (offsets[2] - offsets[1]).
      return { 0, 0,
        static_cast<cl_int>(in.offsets[2]) - static_cast<cl_int>(in.offsets[1]), 0 };
    }
    if (num_planes == 3) {
      //  Each plane is a separate buffer starting at byte 0.
      return { 0, 0, 0, 0 };
    }
    return { 0, static_cast<cl_int>(in.offsets[1]),
      static_cast<cl_int>(in.offsets[2]), 0 };
  }
  return { 0, 0, 0, 0 };
}

const char *rgb_sampler = R"##(
    uchar4 pixel = rgb_sampler_bl(in, col, row, corrected.x, corrected.y, &img, fill);
)##";

const char *rgba_sampler = R"##(
    uchar4 pixel = rgba_sampler_bl(in, col, row, corrected.x, corrected.y, &img, fill);
)##";

const char *nv12_sampler = R"##(
    uchar4 pixel = nv12_sampler(in, col, row, corrected.x, corrected.y, &img, fill);
)##";

const char *nv12_sampler_two_plane = R"##(
    uchar4 pixel = nv12_sampler_two_plane(in, in_uv, col, row, corrected.x, corrected.y, &img, fill);
)##";

const char *nv16_sampler = R"##(
    uchar4 pixel = nv16_sampler(in, col, row, corrected.x, corrected.y, &img, fill);
)##";

const char *nv16_sampler_two_plane = R"##(
    uchar4 pixel = nv16_sampler_two_plane(in, in_uv, col, row, corrected.x, corrected.y, &img, fill);
)##";

const char *i420_sampler = R"##(
    uchar4 pixel = i420_sampler(in, col, row, corrected.x, corrected.y, &img, fill);
)##";

const char *i420_sampler_two_plane = R"##(
    uchar4 pixel = i420_sampler_two_plane(in, in_uv, col, row, corrected.x, corrected.y, &img, fill);
)##";

const char *i420_sampler_three_plane = R"##(
    uchar4 pixel = i420_sampler_three_plane(in, in_u, in_v, col, row, corrected.x, corrected.y, &img, fill);
)##";

const char *yuyv_sampler = R"##(
    uchar4 pixel = yuyv_sampler(in, col, row, corrected.x, corrected.y, &img, fill);
)##";

const char *gray8_sampler = R"##(
    uchar pixel = gray8_sampler_bl(in, col, row, corrected.x, corrected.y, &img, fill);
)##";

const char *nv12_nn_sampler = R"##(
    uchar y = in[corrected.y * strides.x + corrected.x];
    __global const uchar *inuv = advance_uchar_ptr(in, offsets.y);
    int uv_idx = corrected.y / 2 * strides.y + (corrected.x & ~1);
    uchar4 pixel = (uchar4)(y, inuv[uv_idx], inuv[uv_idx + 1], 255);
)##";

const char *nv16_nn_sampler = R"##(
    uchar y = in[corrected.y * strides.x + corrected.x];
    __global const uchar *inuv = advance_uchar_ptr(in, offsets.y);
    int uv_idx = corrected.y * strides.y + (corrected.x & ~1);
    uchar4 pixel = (uchar4)(y, inuv[uv_idx], inuv[uv_idx + 1], 255);
)##";

const char *i420_nn_sampler = R"##(
    uchar y = in[corrected.y * strides.x + corrected.x];
    __global const uchar *u = advance_uchar_ptr(in, offsets.y);
    __global const uchar *v = advance_uchar_ptr(in, offsets.z);
    uchar4 pixel = (uchar4)(y, u[corrected.y / 2 * strides.y + corrected.x / 2], v[corrected.y / 2 * strides.z + corrected.x / 2], 255);
)##";

const char *yuyv_nn_sampler = R"##(
    __global uchar4 *in4 = advance_uchar4_ptr(in, corrected.y * strides.x);
    uchar4 i = in4[corrected.x / 2];
    uchar y = corrected.x % 2 == 0 ? i.x : i.z;
    uchar4 pixel = (uchar4)(y, i.y, i.w, 255);
)##";

const char *rgba_nn_sampler = R"##(
    __global const uchar4 *row_ptr = advance_uchar4_ptr(in, corrected.y * strides.x);
    uchar4 pixel = row_ptr[corrected.x];
)##";

const char *rgb_nn_sampler = R"##(
    __global const uchar *p = advance_uchar_ptr(in, corrected.y * strides.x);
    uchar4 pixel = (uchar4)(vload3(corrected.x, p), 255);
)##";

const char *gray8_nn_sampler = R"##(
    uchar pixel = in[corrected.y * strides.x + corrected.x];
)##";

const char *nv12_nn_sampler_two_plane = R"##(
    uchar y = in[corrected.y * strides.x + corrected.x];
    int uv_idx = corrected.y / 2 * strides.y + (corrected.x & ~1);
    uchar4 pixel = (uchar4)(y, in_uv[uv_idx], in_uv[uv_idx + 1], 255);
)##";

const char *nv16_nn_sampler_two_plane = R"##(
    uchar y = in[corrected.y * strides.x + corrected.x];
    int uv_idx = corrected.y * strides.y + (corrected.x & ~1);
    uchar4 pixel = (uchar4)(y, in_uv[uv_idx], in_uv[uv_idx + 1], 255);
)##";

const char *i420_nn_sampler_two_plane = R"##(
    uchar y = in[corrected.y * strides.x + corrected.x];
    __global const uchar *inu = in_uv;
    __global const uchar *inv = advance_uchar_ptr(in_uv, offsets.z);
    uchar4 pixel = (uchar4)(y, inu[corrected.y / 2 * strides.y + corrected.x / 2], inv[corrected.y / 2 * strides.z + corrected.x / 2], 255);
)##";

const char *i420_nn_sampler_three_plane = R"##(
    uchar y = in[corrected.y * strides.x + corrected.x];
    uchar4 pixel = (uchar4)(y, in_u[corrected.y / 2 * strides.y + corrected.x / 2], in_v[corrected.y / 2 * strides.z + corrected.x / 2], 255);
)##";

const char *rgb_output_cl = R"##(
    int strideOut = strides.w;
    __global uchar* prgb = advance_uchar_ptr(out, row * strideOut);
    vstore3(convert_uchar3_sat(color_convert_float(pixel, color_matrix).xyz), col, prgb);
}
)##";

const char *rgba_output_cl = R"##(
    int strideOut = strides.w;
    __global uchar4* prgb = advance_uchar4_ptr(out, row * strideOut);
    prgb[col] = convert_uchar4_sat(color_convert_float(pixel, color_matrix));
}
)##";

const char *gray_output_cl = R"##(
    int strideOut = strides.w;
    __global uchar* p_gray = advance_uchar_ptr(out, row * strideOut);
    p_gray[col] = convert_uchar_sat(color_convert_float(pixel, color_matrix).x);
}
)##";

const char *gray_in_output_cl = R"##(
    int strideOut = strides.w;
    __global uchar* p_gray = advance_uchar_ptr(out, row * strideOut);
    p_gray[col] = pixel;
}
)##";

const char *rgb_output_norm_cl = R"##(
    int strideOut = strides.w;
    __global uchar* prgb = advance_uchar_ptr(out, row * strideOut);
    float4 new_pixel = color_convert_float(pixel, color_matrix);
    char4 pix = convert_char4_sat_rte(new_pixel);
    vstore3(convert_uchar3(pix.xyz), col, prgb);
}
)##";

const char *rgba_output_norm_cl = R"##(
    int strideOut = strides.w;
    __global uchar4* prgb = advance_uchar4_ptr(out, row * strideOut);
    float4 new_pixel = color_convert_float(pixel, color_matrix);
    char4 pix = convert_char4_sat_rte(new_pixel);
    prgb[col] = convert_uchar4(pix);
}
)##";

const char *gray_output_norm_cl = R"##(
    int strideOut = strides.w;
    __global uchar* p_gray = advance_uchar_ptr(out, row * strideOut);
    float4 new_pixel = color_convert_float(pixel, color_matrix);
    p_gray[col] = convert_uchar(convert_char_sat_rte(new_pixel.x));
}
)##";

const char *gray_in_output_norm_cl = R"##(
    int strideOut = strides.w;
    __global uchar* p_gray = advance_uchar_ptr(out, row * strideOut);
    float fgray = convert_float(pixel);
    fgray = fgray * color_matrix.s0 + color_matrix.sc;
    p_gray[col] = convert_uchar(convert_char_sat_rte(fgray));
}
)##";


std::vector<kernel_arg_details> input_details_tab = {
  { AxVideoFormat::NV12, "uchar", { nv12_sampler, nv12_sampler_two_plane } },
  { AxVideoFormat::NV16, "uchar", { nv16_sampler, nv16_sampler_two_plane } },
  { AxVideoFormat::I420, "uchar",
      { i420_sampler, i420_sampler_two_plane, i420_sampler_three_plane } },
  { AxVideoFormat::YUY2, "uchar", { yuyv_sampler } },
  { AxVideoFormat::RGBA, "uchar", { rgba_sampler } },
  { AxVideoFormat::BGRA, "uchar", { rgba_sampler } },
  { AxVideoFormat::RGB, "uchar", { rgb_sampler } },
  { AxVideoFormat::BGR, "uchar", { rgb_sampler } },
  { AxVideoFormat::GRAY8, "uchar", { gray8_sampler } },
};

std::vector<kernel_arg_details> nn_input_details_tab = {
  { AxVideoFormat::NV12, "uchar", { nv12_nn_sampler, nv12_nn_sampler_two_plane } },
  { AxVideoFormat::NV16, "uchar", { nv16_nn_sampler, nv16_nn_sampler_two_plane } },
  { AxVideoFormat::I420, "uchar",
      { i420_nn_sampler, i420_nn_sampler_two_plane, i420_nn_sampler_three_plane } },
  { AxVideoFormat::YUY2, "uchar4", { yuyv_nn_sampler } },
  { AxVideoFormat::RGBA, "uchar4", { rgba_nn_sampler } },
  { AxVideoFormat::BGRA, "uchar4", { rgba_nn_sampler } },
  { AxVideoFormat::RGB, "uchar", { rgb_nn_sampler } },
  { AxVideoFormat::BGR, "uchar", { rgb_nn_sampler } },
  { AxVideoFormat::GRAY8, "uchar", { gray8_nn_sampler } },
};

kernel_args
get_input_details(AxVideoFormat format, Interpolation interp, int num_planes)
{
  auto &tab = interp == Interpolation::nearest ? nn_input_details_tab : input_details_tab;
  for (const auto &details : tab) {
    if (details.in_format == format) {
      auto idx = num_planes - 1;
      const auto &sampler = (idx > 0 && !details.samplers[idx].empty()) ?
                                details.samplers[idx] :
                                details.samplers[0];
      return { details.out_type, sampler,
        std::string("__global const uchar *in, ") + uv_kernel_params(num_planes) };
    }
  }
  throw std::runtime_error("Unsupported input format");
}

std::vector<kernel_arg_details> out_details_tab = {
  { AxVideoFormat::RGBA, "uchar4", { rgba_output_cl } },
  { AxVideoFormat::BGRA, "uchar4", { rgba_output_cl } },
  { AxVideoFormat::RGB, "uchar", { rgb_output_cl } },
  { AxVideoFormat::BGR, "uchar", { rgb_output_cl } },
  { AxVideoFormat::GRAY8, "uchar", { gray_output_cl } },
};

std::vector<kernel_arg_details> out_details_norm_tab = {
  { AxVideoFormat::RGBA, "uchar4", { rgba_output_norm_cl } },
  { AxVideoFormat::BGRA, "uchar4", { rgba_output_norm_cl } },
  { AxVideoFormat::RGB, "uchar", { rgb_output_norm_cl } },
  { AxVideoFormat::BGR, "uchar", { rgb_output_norm_cl } },
  { AxVideoFormat::GRAY8, "uchar", { gray_output_norm_cl } },
};


kernel_args
get_output_details(AxVideoFormat in_format, AxVideoFormat out_format)
{
  if (in_format == out_format && in_format == AxVideoFormat::GRAY8) {
    return { "uchar", gray_in_output_cl };
  }
  for (const auto &details : out_details_tab) {
    if (details.in_format == out_format) {
      return { details.out_type, details.samplers[0] };
    }
  }
  throw std::runtime_error("Unsupported output format");
}

kernel_args
get_output_norm_details(AxVideoFormat in_format, AxVideoFormat out_format)
{
  if (in_format == out_format && in_format == AxVideoFormat::GRAY8) {
    return { "uchar", gray_in_output_norm_cl };
  }
  for (const auto &details : out_details_norm_tab) {
    if (details.in_format == out_format) {
      return { details.out_type, details.samplers[0] };
    }
  }
  throw std::runtime_error("Unsupported output format");
}


} // namespace ax_utils
