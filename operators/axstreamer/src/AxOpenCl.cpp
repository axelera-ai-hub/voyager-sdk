// Copyright Axelera AI, 2024
#include "AxOpenCl.hpp"

#include <iostream>
#include <string_view>
#include <vector>
#include "AxStreamerUtils.hpp"

namespace ax_utils
{

std::string_view build_options
    = "-cl-strict-aliasing -cl-mad-enable -cl-no-signed-zeros -cl-denorms-are-zero -cl-fast-relaxed-math -cl-finite-math-only";

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
    error = error == CL_SUCCESS ? clBuildProgram(program, 0, NULL,
                                      ax_utils::build_options.data(), NULL, NULL) :
                                  error;
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
    REAL4 i1 = floor(mix(p00, p01, (REAL)xfrac) + (REAL)0.5f);
    REAL4 i2 = floor(mix(p10, p11, (REAL)xfrac) + (REAL)0.5f);
    return floor(mix(i1, i2, (REAL)yfrac) + (REAL)0.5f);
}

REAL bilinear1(REAL p00, REAL p01, REAL p10, REAL p11, float xfrac, float yfrac) {
    REAL i1 = floor(mix(p00, p01, (REAL)xfrac) + (REAL)0.5f);
    REAL i2 = floor(mix(p10, p11, (REAL)xfrac) + (REAL)0.5f);
    return floor(mix(i1, i2, (REAL)yfrac) + (REAL)0.5f);
}

REAL2 bilinear2(REAL2 p00, REAL2 p01, REAL2 p10, REAL2 p11, float xfrac, float yfrac) {
    REAL2 i1 = floor(mix(p00, p01, (REAL)xfrac) + (REAL2)(0.5f));
    REAL2 i2 = floor(mix(p10, p11, (REAL)xfrac) + (REAL2)(0.5f));
    return floor(mix(i1, i2, (REAL)yfrac) + (REAL2)(0.5f));
}

typedef struct image_description {
    int4 image_dims;
    int4 strides;
    int4 offsets;
    int4 letterbox;
    int4 crop;
} image_description;


#define NV12_READ(x, y, p, stride) CONVERT_REAL2(p[y * stride + x])

uchar4 nv_semiplanar_sampler_two_plane(__global const uchar *y_image, __global const uchar *uv_image, int out_x, int out_y, float fx, float fy, const image_description *img, uchar fill, int uv_div_y) {
  if (fx < 0 || fx >= img->image_dims.x || fy < 0 || fy >= img->image_dims.y) {
    return (uchar4)(16, 128, 128, 255);
  }
  float xpixel_left = fx - 0.5f;
  float ypixel_top = fy - 0.5f;

  int adj_x = out_x - img->letterbox.x;
  int adj_y = out_y - img->letterbox.y;
  if (adj_x < 0 || adj_y < 0 || adj_x >= img->letterbox.z || adj_y >= img->letterbox.w) {
    return (uchar4)(fill, 128, 128, 255);
  }

  int xi = (int)floor(xpixel_left);
  int yi = (int)floor(ypixel_top);
  float xfrac = xpixel_left - xi;
  float yfrac = ypixel_top - yi;

  int x1 = max(xi,     0) + img->crop.x;
  int x2 = min(xi + 1, img->image_dims.x - 1) + img->crop.x;
  int y1 = max(yi,     0) + img->crop.y;
  int y2 = min(yi + 1, img->image_dims.y - 1) + img->crop.y;

  int ystride = img->strides.x;
  REAL y00 = CONVERT_REAL(y_image[y1 * ystride + x1]);
  REAL y01 = CONVERT_REAL(y_image[y1 * ystride + x2]);
  REAL y10 = CONVERT_REAL(y_image[y2 * ystride + x1]);
  REAL y11 = CONVERT_REAL(y_image[y2 * ystride + x2]);
  REAL y_val = bilinear1(y00, y01, y10, y11, xfrac, yfrac);

  float uv_xsrc = xpixel_left / 2;
  float uv_ysrc = ypixel_top / uv_div_y;
  int uxi = (int)floor(uv_xsrc);
  int uyi = (int)floor(uv_ysrc);
  float uv_xfrac = uv_xsrc - uxi;
  float uv_yfrac = uv_ysrc - uyi;
  int uv_crop_x = img->crop.x / 2;
  int uv_crop_y = img->crop.y / uv_div_y;
  int ux1 = max(uxi, 0) + uv_crop_x;
  int uy1 = max(uyi, 0) + uv_crop_y;
  int ux2 = min(uxi + 1, img->image_dims.x / 2 - 1) + uv_crop_x;
  int uy2 = min(uyi + 1, img->image_dims.y / uv_div_y - 1) + uv_crop_y;

  __global uchar2 *in_uv2 = (__global uchar2 *)uv_image;
  int uvstride = img->strides.y / 2;
  REAL2 uv00 = NV12_READ(ux1, uy1, in_uv2, uvstride);
  REAL2 uv01 = NV12_READ(ux2, uy1, in_uv2, uvstride);
  REAL2 uv10 = NV12_READ(ux1, uy2, in_uv2, uvstride);
  REAL2 uv11 = NV12_READ(ux2, uy2, in_uv2, uvstride);
  REAL2 uv_val = bilinear2(uv00, uv01, uv10, uv11, uv_xfrac, uv_yfrac);

  return convert_uchar4_sat((float4)((float)y_val, (float)uv_val.x, (float)uv_val.y, 255.0f));
}

uchar4 nv12_sampler_two_plane(__global const uchar *y_image, __global const uchar *uv_image, int out_x, int out_y, float fx, float fy, const image_description *img, uchar fill) {
  return nv_semiplanar_sampler_two_plane(y_image, uv_image, out_x, out_y, fx, fy, img, fill, 2);
}

uchar4 nv12_sampler(__global const uchar *y_image, int out_x, int out_y, float fx, float fy, const image_description *img, uchar fill) {
  return nv12_sampler_two_plane(y_image, y_image + img->offsets.y, out_x, out_y, fx, fy, img, fill);
}

uchar4 nv16_sampler_two_plane(__global const uchar *y_image, __global const uchar *uv_image, int out_x, int out_y, float fx, float fy, const image_description *img, uchar fill) {
  return nv_semiplanar_sampler_two_plane(y_image, uv_image, out_x, out_y, fx, fy, img, fill, 1);
}

uchar4 nv16_sampler(__global const uchar *y_image, int out_x, int out_y, float fx, float fy, const image_description *img, uchar fill) {
  return nv16_sampler_two_plane(y_image, y_image + img->offsets.y, out_x, out_y, fx, fy, img, fill);
}


#define I420_READ(x, y, pu, pv, ustride, vstride) CONVERT_REAL2((uchar2)(pu[y * ustride + x], pv[y * vstride + x]))

uchar4 yuv_planar_bilinear_three_plane(__global const uchar *y_image, __global const uchar *u_image, __global const uchar *v_image, int out_x, int out_y, float fx, float fy, const image_description *img, uchar fill, int uv_div_x, int uv_div_y) {
  if (fx < 0 || fx >= img->image_dims.x || fy < 0 || fy >= img->image_dims.y) {
    return (uchar4)(16, 128, 128, 255);
  }
  float xpixel_left = fx - 0.5f;
  float ypixel_top = fy - 0.5f;

  int adj_x = out_x - img->letterbox.x;
  int adj_y = out_y - img->letterbox.y;
  if (adj_x < 0 || adj_y < 0 || adj_x >= img->letterbox.z || adj_y >= img->letterbox.w) {
    return (uchar4)(fill, 128, 128, 255);
  }
  int xi = (int)floor(xpixel_left);
  int yi = (int)floor(ypixel_top);
  float xfrac = xpixel_left - xi;
  float yfrac = ypixel_top - yi;

  int x1 = max(xi,     0);
  int x2 = min(xi + 1, img->image_dims.x - 1);
  int y1 = max(yi,     0);
  int y2 = min(yi + 1, img->image_dims.y - 1);
  x1 += img->crop.x;
  y1 += img->crop.y;
  x2 += img->crop.x;
  y2 += img->crop.y;

  int ystride = img->strides.x;
  REAL y00 = CONVERT_REAL(y_image[y1 * ystride + x1]);
  REAL y01 = CONVERT_REAL(y_image[y1 * ystride + x2]);
  REAL y10 = CONVERT_REAL(y_image[y2 * ystride + x1]);
  REAL y11 = CONVERT_REAL(y_image[y2 * ystride + x2]);
  REAL y_val = bilinear1(y00, y01, y10, y11, xfrac, yfrac);

  float uv_xsrc = xpixel_left / uv_div_x;
  float uv_ysrc = ypixel_top / uv_div_y;
  int uxi = (int)floor(uv_xsrc);
  int uyi = (int)floor(uv_ysrc);
  float uv_xfrac = uv_xsrc - uxi;
  float uv_yfrac = uv_ysrc - uyi;
  int uv_crop_x = img->crop.x / uv_div_x;
  int uv_crop_y = img->crop.y / uv_div_y;
  int ux1 = max(uxi, 0) + uv_crop_x;
  int uy1 = max(uyi, 0) + uv_crop_y;
  int ux2 = min(uxi + 1, img->image_dims.x / uv_div_x - 1) + uv_crop_x;
  int uy2 = min(uyi + 1, img->image_dims.y / uv_div_y - 1) + uv_crop_y;

  REAL2 uv00 = I420_READ(ux1, uy1, u_image, v_image, img->strides.y, img->strides.z);
  REAL2 uv01 = I420_READ(ux2, uy1, u_image, v_image, img->strides.y, img->strides.z);
  REAL2 uv10 = I420_READ(ux1, uy2, u_image, v_image, img->strides.y, img->strides.z);
  REAL2 uv11 = I420_READ(ux2, uy2, u_image, v_image, img->strides.y, img->strides.z);
  REAL2 uv_val = bilinear2(uv00, uv01, uv10, uv11, uv_xfrac, uv_yfrac);

  return convert_uchar4_sat((float4)((float)y_val, (float)uv_val.x, (float)uv_val.y, 255.0f));
}

// I420 three-plane: Y in y_image, U in u_image, V in v_image (all offsets = 0).
uchar4 i420_sampler_three_plane(__global const uchar *y_image, __global const uchar *u_image, __global const uchar *v_image, int out_x, int out_y, float fx, float fy, const image_description *img, uchar fill) {
  return yuv_planar_bilinear_three_plane(y_image, u_image, v_image, out_x, out_y, fx, fy, img, fill, 2, 2);
}

// I420 two-plane: Y in y_image, U+V back-to-back in uv_image.
// U starts at offset 0; V starts at offsets.z within uv_image.
uchar4 i420_sampler_two_plane(__global const uchar *y_image, __global const uchar *uv_image, int out_x, int out_y, float fx, float fy, const image_description *img, uchar fill) {
  return i420_sampler_three_plane(y_image, uv_image, uv_image + img->offsets.z, out_x, out_y, fx, fy, img, fill);
}

uchar4 i420_sampler(__global const uchar *y_image, int out_x, int out_y, float fx, float fy, const image_description *img, uchar fill) {
  return i420_sampler_three_plane(y_image, y_image + img->offsets.y, y_image + img->offsets.z, out_x, out_y, fx, fy, img, fill);
}

uchar4 y444_sampler_three_plane(__global const uchar *y_image, __global const uchar *u_image, __global const uchar *v_image, int out_x, int out_y, float fx, float fy, const image_description *img, uchar fill) {
  return yuv_planar_bilinear_three_plane(y_image, u_image, v_image, out_x, out_y, fx, fy, img, fill, 1, 1);
}

uchar4 y444_sampler_two_plane(__global const uchar *y_image, __global const uchar *uv_image, int out_x, int out_y, float fx, float fy, const image_description *img, uchar fill) {
  return y444_sampler_three_plane(y_image, uv_image, uv_image + img->offsets.z, out_x, out_y, fx, fy, img, fill);
}

uchar4 y444_sampler(__global const uchar *y_image, int out_x, int out_y, float fx, float fy, const image_description *img, uchar fill) {
  return y444_sampler_three_plane(y_image, y_image + img->offsets.y, y_image + img->offsets.z, out_x, out_y, fx, fy, img, fill);
}

uchar4 y42b_sampler_three_plane(__global const uchar *y_image, __global const uchar *u_image, __global const uchar *v_image, int out_x, int out_y, float fx, float fy, const image_description *img, uchar fill) {
  return yuv_planar_bilinear_three_plane(y_image, u_image, v_image, out_x, out_y, fx, fy, img, fill, 2, 1);
}

uchar4 y42b_sampler_two_plane(__global const uchar *y_image, __global const uchar *uv_image, int out_x, int out_y, float fx, float fy, const image_description *img, uchar fill) {
  return y42b_sampler_three_plane(y_image, uv_image, uv_image + img->offsets.z, out_x, out_y, fx, fy, img, fill);
}

uchar4 y42b_sampler(__global const uchar *y_image, int out_x, int out_y, float fx, float fy, const image_description *img, uchar fill) {
  return y42b_sampler_three_plane(y_image, y_image + img->offsets.y, y_image + img->offsets.z, out_x, out_y, fx, fy, img, fill);
}

uchar4 yuyv_sampler(__global const uchar *in, int out_x, int out_y, float fx, float fy, const image_description *img, uchar fill) {
  __global const uchar4 *in4 = (__global const uchar4 *)in;
  if (fx < 0 || fx >= img->image_dims.x || fy < 0 || fy >= img->image_dims.y) {
    return (uchar4)(16, 128, 128, 255);
  }
  float xpixel_left = fx - 0.5f;
  float ypixel_top = fy - 0.5f;

  int adj_x = out_x - img->letterbox.x;
  int adj_y = out_y - img->letterbox.y;
  if (adj_x < 0 || adj_y < 0 || adj_x >= img->letterbox.z || adj_y >= img->letterbox.w) {
    return (uchar4)(fill, 128, 128, 255);
  }
  int xi = (int)floor(xpixel_left);
  int yi = (int)floor(ypixel_top);
  float xfrac = xpixel_left - xi;
  float yfrac = ypixel_top - yi;

  int x1 = max(xi,     0);
  int x2 = min(xi + 1, img->image_dims.x - 1);
  int y1 = max(yi,     0);
  int y2 = min(yi + 1, img->image_dims.y - 1);

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

    REAL i1 = floor(mix(p00, p01, (REAL)xfrac) + (REAL)0.5f);
    REAL i2 = floor(mix(p10, p11, (REAL)xfrac) + (REAL)0.5f);
    REAL value = floor(mix(i1, i2, (REAL)yfrac) + (REAL)0.5f);

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

    int adj_x = out_x - img->letterbox.x;
    int adj_y = out_y - img->letterbox.y;
    if (adj_x < 0 || adj_y < 0 || adj_x >= img->letterbox.z || adj_y >= img->letterbox.w) {
      return (uchar4)(fill, fill, fill, 255);
    }

    int xi = (int)floor(xpixel_left);
    int yi = (int)floor(ypixel_top);
    float xfrac = xpixel_left - xi;
    float yfrac = ypixel_top - yi;

    int x1 = max(xi,     0);
    int x2 = min(xi + 1, img->image_dims.x - 1);
    int y1 = max(yi,     0);
    int y2 = min(yi + 1, img->image_dims.y - 1);

    x1 += img->crop.x;
    y1 += img->crop.y;
    x2 += img->crop.x;
    y2 += img->crop.y;
    int stride = img->strides.x / sizeof(uchar4);
    REAL4 p00 = CONVERT_REAL4(image4[y1 * stride + x1]);
    REAL4 p01 = CONVERT_REAL4(image4[y1 * stride + x2]);
    REAL4 p10 = CONVERT_REAL4(image4[y2 * stride + x1]);
    REAL4 p11 = CONVERT_REAL4(image4[y2 * stride + x2]);

    REAL4 i1 = floor(mix(p00, p01, (REAL)xfrac) + (REAL)0.5f);
    REAL4 i2 = floor(mix(p10, p11, (REAL)xfrac) + (REAL)0.5f);
    uchar4 result = convert_uchar4_sat(floor(mix(i1, i2, (REAL)yfrac) + (REAL)0.5f));
    return result;
}

uchar4 rgb_sampler_bl(__global const uchar *image, int out_x, int out_y, float fx, float fy, const image_description *img, uchar fill) {
    //  Here we add in the offsets to the pixel from the crop meta
    if (fx < 0 || fx >= img->image_dims.x || fy < 0 || fy >= img->image_dims.y) {
      return (uchar4)(fill, fill, fill, 255);
    }
    float xpixel_left = fx - 0.5f;
    float ypixel_top = fy - 0.5f;

    int adj_x = out_x - img->letterbox.x;
    int adj_y = out_y - img->letterbox.y;
    if (adj_x < 0 || adj_y < 0 || adj_x >= img->letterbox.z || adj_y >= img->letterbox.w) {
      return (uchar4)(fill, fill, fill, 255);
    }

    int xi = (int)floor(xpixel_left);
    int yi = (int)floor(ypixel_top);
    float xfrac = xpixel_left - xi;
    float yfrac = ypixel_top - yi;

    int x1 = max(xi,     0);
    int x2 = min(xi + 1, img->image_dims.x - 1);
    int y1 = max(yi,     0);
    int y2 = min(yi + 1, img->image_dims.y - 1);

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

    REAL4 i1 = floor(mix(p00, p01, (REAL)xfrac) + (REAL)0.5f);
    REAL4 i2 = floor(mix(p10, p11, (REAL)xfrac) + (REAL)0.5f);
    uchar4 result = convert_uchar4_sat(floor(mix(i1, i2, (REAL)yfrac) + (REAL)0.5f));
    return result;
}

float4 color_convert_float(uchar4 pixel, float16 matrix) {
    float4 in_pixel = convert_float4(pixel);
    float4 color = mad(in_pixel.x, matrix.s0123, mad(in_pixel.y, matrix.s4567, mad(in_pixel.z, matrix.s89ab, matrix.scdef)));
    color.w = in_pixel.w;
    return color;
}

float pillow_bilinear_weight(float x) {
    return max(0.0f, 1.0f - fabs(x));
}

uchar4 rgba_sampler_pb(__global const uchar *image, int out_x, int out_y, float fx, float fy, const image_description *img, uchar fill, float filter_scale_x, float filter_scale_y) {
    if (fx < 0 || fx >= img->image_dims.x || fy < 0 || fy >= img->image_dims.y) {
        return (uchar4)(fill, fill, fill, 255);
    }
    int adj_x = out_x - img->letterbox.x;
    int adj_y = out_y - img->letterbox.y;
    if (adj_x < 0 || adj_y < 0 || adj_x >= img->letterbox.z || adj_y >= img->letterbox.w) {
        return (uchar4)(fill, fill, fill, 255);
    }
    float src_x = fx - 0.5f;
    float src_y = fy - 0.5f;
    int cx = (int)floor(src_x);
    int cy = (int)floor(src_y);
    float fs_x = max(1.0f, filter_scale_x);
    float fs_y = max(1.0f, filter_scale_y);
    int half_x = (int)ceil(fs_x);
    int half_y = (int)ceil(fs_y);
    int stride = img->strides.x / 4;
    __global const uchar4 *image4 = (__global const uchar4 *)image;
    float4 sum = (float4)(0.0f);
    float wsum = 0.0f;
    for (int j = -(half_y - 1); j <= half_y; j++) {
        float wy = pillow_bilinear_weight((src_y - (float)(cy + j)) / fs_y);
        int sy = clamp(cy + j, 0, img->image_dims.y - 1) + img->crop.y;
        for (int i = -(half_x - 1); i <= half_x; i++) {
            float w = wy * pillow_bilinear_weight((src_x - (float)(cx + i)) / fs_x);
            int sx = clamp(cx + i, 0, img->image_dims.x - 1) + img->crop.x;
            sum += w * convert_float4(image4[sy * stride + sx]);
            wsum += w;
        }
    }
    if (wsum != 0.0f) sum /= wsum;
    return convert_uchar4_sat(sum);
}

uchar4 rgb_sampler_pb(__global const uchar *image, int out_x, int out_y, float fx, float fy, const image_description *img, uchar fill, float filter_scale_x, float filter_scale_y) {
    if (fx < 0 || fx >= img->image_dims.x || fy < 0 || fy >= img->image_dims.y) {
        return (uchar4)(fill, fill, fill, 255);
    }
    int adj_x = out_x - img->letterbox.x;
    int adj_y = out_y - img->letterbox.y;
    if (adj_x < 0 || adj_y < 0 || adj_x >= img->letterbox.z || adj_y >= img->letterbox.w) {
        return (uchar4)(fill, fill, fill, 255);
    }
    float src_x = fx - 0.5f;
    float src_y = fy - 0.5f;
    int cx = (int)floor(src_x);
    int cy = (int)floor(src_y);
    float fs_x = max(1.0f, filter_scale_x);
    float fs_y = max(1.0f, filter_scale_y);
    int half_x = (int)ceil(fs_x);
    int half_y = (int)ceil(fs_y);
    int stride = img->strides.x;
    float4 sum = (float4)(0.0f);
    float wsum = 0.0f;
    for (int j = -(half_y - 1); j <= half_y; j++) {
        float wy = pillow_bilinear_weight((src_y - (float)(cy + j)) / fs_y);
        int sy = clamp(cy + j, 0, img->image_dims.y - 1) + img->crop.y;
        __global const uchar *p_in = advance_uchar_ptr(image, sy * stride);
        for (int i = -(half_x - 1); i <= half_x; i++) {
            float w = wy * pillow_bilinear_weight((src_x - (float)(cx + i)) / fs_x);
            int sx = clamp(cx + i, 0, img->image_dims.x - 1) + img->crop.x;
            float4 px = convert_float4((uchar4)(vload3(sx, p_in), 255));
            sum += w * px;
            wsum += w;
        }
    }
    if (wsum != 0.0f) sum /= wsum;
    return convert_uchar4_sat(sum);
}

uchar gray8_sampler_pb(__global const uchar *image, int out_x, int out_y, float fx, float fy, const image_description *img, uchar fill, float filter_scale_x, float filter_scale_y) {
    if (fx < 0 || fx >= img->image_dims.x || fy < 0 || fy >= img->image_dims.y) {
        return fill;
    }
    int adj_x = out_x - img->letterbox.x;
    int adj_y = out_y - img->letterbox.y;
    if (adj_x < 0 || adj_y < 0 || adj_x >= img->letterbox.z || adj_y >= img->letterbox.w) {
        return fill;
    }
    float src_x = fx - 0.5f;
    float src_y = fy - 0.5f;
    int cx = (int)floor(src_x);
    int cy = (int)floor(src_y);
    float fs_x = max(1.0f, filter_scale_x);
    float fs_y = max(1.0f, filter_scale_y);
    int half_x = (int)ceil(fs_x);
    int half_y = (int)ceil(fs_y);
    float sum = 0.0f, wsum = 0.0f;
    for (int j = -(half_y - 1); j <= half_y; j++) {
        float wy = pillow_bilinear_weight((src_y - (float)(cy + j)) / fs_y);
        int sy = clamp(cy + j, 0, img->image_dims.y - 1) + img->crop.y;
        for (int i = -(half_x - 1); i <= half_x; i++) {
            float w = wy * pillow_bilinear_weight((src_x - (float)(cx + i)) / fs_x);
            int sx = clamp(cx + i, 0, img->image_dims.x - 1) + img->crop.x;
            sum += w * convert_float(image[sy * img->strides.x + sx]);
            wsum += w;
        }
    }
    if (wsum != 0.0f) sum /= wsum;
    return convert_uchar_sat(sum);
}

uchar4 nv_semiplanar_sampler_pb_two_plane(__global const uchar *y_image, __global const uchar *uv_image, int out_x, int out_y, float fx, float fy, const image_description *img, uchar fill, float filter_scale_x, float filter_scale_y, int uv_div_y) {
  if (fx < 0 || fx >= img->image_dims.x || fy < 0 || fy >= img->image_dims.y) {
    return (uchar4)(16, 128, 128, 255);
  }
  int adj_x = out_x - img->letterbox.x;
  int adj_y = out_y - img->letterbox.y;
  if (adj_x < 0 || adj_y < 0 || adj_x >= img->letterbox.z || adj_y >= img->letterbox.w) {
    return (uchar4)(fill, 128, 128, 255);
  }
  float src_x = fx - 0.5f;
  float src_y = fy - 0.5f;
  int cx = (int)floor(src_x);
  int cy = (int)floor(src_y);
  float fs_x = max(1.0f, filter_scale_x);
  float fs_y = max(1.0f, filter_scale_y);
  int half_x = (int)ceil(fs_x);
  int half_y = (int)ceil(fs_y);
  int ystride = img->strides.x;
  float y_sum = 0.0f, y_wsum = 0.0f;
  for (int j = -(half_y - 1); j <= half_y; j++) {
    float wy = pillow_bilinear_weight((src_y - (float)(cy + j)) / fs_y);
    int sy = clamp(cy + j, 0, img->image_dims.y - 1) + img->crop.y;
    for (int i = -(half_x - 1); i <= half_x; i++) {
      float w = wy * pillow_bilinear_weight((src_x - (float)(cx + i)) / fs_x);
      int sx = clamp(cx + i, 0, img->image_dims.x - 1) + img->crop.x;
      y_sum += w * convert_float(y_image[sy * ystride + sx]);
      y_wsum += w;
    }
  }
  float y_val = (y_wsum != 0.0f) ? (y_sum / y_wsum) : 16.0f;
  float xfrac = src_x - cx;
  float yfrac = src_y - cy;
  int x1 = max(cx, 0) + img->crop.x;
  int y1 = max(cy, 0) + img->crop.y;
  int x2 = min(cx + 1, img->image_dims.x - 1) + img->crop.x;
  int y2 = min(cy + 1, img->image_dims.y - 1) + img->crop.y;
  int ux1 = x1 / 2, uy1 = y1 / uv_div_y;
  int ux2 = x2 / 2, uy2 = y2 / uv_div_y;
  __global uchar2 *in_uv2 = (__global uchar2 *)uv_image;
  int uvstride = img->strides.y / 2;
  REAL2 uv00 = NV12_READ(ux1, uy1, in_uv2, uvstride);
  REAL2 uv01 = NV12_READ(ux2, uy1, in_uv2, uvstride);
  REAL2 uv10 = NV12_READ(ux1, uy2, in_uv2, uvstride);
  REAL2 uv11 = NV12_READ(ux2, uy2, in_uv2, uvstride);
  REAL2 uv_val = mix(mix(uv00, uv01, xfrac), mix(uv10, uv11, xfrac), yfrac);
  return convert_uchar4_sat((float4)(y_val, (float)uv_val.x, (float)uv_val.y, 255.0f));
}

uchar4 nv12_sampler_pb_two_plane(__global const uchar *y_image, __global const uchar *uv_image, int out_x, int out_y, float fx, float fy, const image_description *img, uchar fill, float filter_scale_x, float filter_scale_y) {
  return nv_semiplanar_sampler_pb_two_plane(y_image, uv_image, out_x, out_y, fx, fy, img, fill, filter_scale_x, filter_scale_y, 2);
}

uchar4 nv12_sampler_pb(__global const uchar *y_image, int out_x, int out_y, float fx, float fy, const image_description *img, uchar fill, float filter_scale_x, float filter_scale_y) {
  return nv12_sampler_pb_two_plane(y_image, y_image + img->offsets.y, out_x, out_y, fx, fy, img, fill, filter_scale_x, filter_scale_y);
}

uchar4 nv16_sampler_pb_two_plane(__global const uchar *y_image, __global const uchar *uv_image, int out_x, int out_y, float fx, float fy, const image_description *img, uchar fill, float filter_scale_x, float filter_scale_y) {
  return nv_semiplanar_sampler_pb_two_plane(y_image, uv_image, out_x, out_y, fx, fy, img, fill, filter_scale_x, filter_scale_y, 1);
}

uchar4 nv16_sampler_pb(__global const uchar *y_image, int out_x, int out_y, float fx, float fy, const image_description *img, uchar fill, float filter_scale_x, float filter_scale_y) {
  return nv16_sampler_pb_two_plane(y_image, y_image + img->offsets.y, out_x, out_y, fx, fy, img, fill, filter_scale_x, filter_scale_y);
}

uchar4 yuv_planar_sampler_pb_three_plane(__global const uchar *y_image, __global const uchar *u_image, __global const uchar *v_image, int out_x, int out_y, float fx, float fy, const image_description *img, uchar fill, float filter_scale_x, float filter_scale_y, int uv_div_x, int uv_div_y) {
  if (fx < 0 || fx >= img->image_dims.x || fy < 0 || fy >= img->image_dims.y) {
    return (uchar4)(16, 128, 128, 255);
  }
  int adj_x = out_x - img->letterbox.x;
  int adj_y = out_y - img->letterbox.y;
  if (adj_x < 0 || adj_y < 0 || adj_x >= img->letterbox.z || adj_y >= img->letterbox.w) {
    return (uchar4)(fill, 128, 128, 255);
  }
  float src_x = fx - 0.5f;
  float src_y = fy - 0.5f;
  int cx = (int)floor(src_x);
  int cy = (int)floor(src_y);
  float fs_x = max(1.0f, filter_scale_x);
  float fs_y = max(1.0f, filter_scale_y);
  int half_x = (int)ceil(fs_x);
  int half_y = (int)ceil(fs_y);
  int ystride = img->strides.x;
  float y_sum = 0.0f, y_wsum = 0.0f;
  for (int j = -(half_y - 1); j <= half_y; j++) {
    float wy = pillow_bilinear_weight((src_y - (float)(cy + j)) / fs_y);
    int sy = clamp(cy + j, 0, img->image_dims.y - 1) + img->crop.y;
    for (int i = -(half_x - 1); i <= half_x; i++) {
      float w = wy * pillow_bilinear_weight((src_x - (float)(cx + i)) / fs_x);
      int sx = clamp(cx + i, 0, img->image_dims.x - 1) + img->crop.x;
      y_sum += w * convert_float(y_image[sy * ystride + sx]);
      y_wsum += w;
    }
  }
  float y_val = (y_wsum != 0.0f) ? (y_sum / y_wsum) : 16.0f;
  float uv_xsrc = src_x / uv_div_x;
  float uv_ysrc = src_y / uv_div_y;
  int uxi = (int)floor(uv_xsrc);
  int uyi = (int)floor(uv_ysrc);
  float uv_xfrac = uv_xsrc - uxi;
  float uv_yfrac = uv_ysrc - uyi;
  int uv_crop_x = img->crop.x / uv_div_x;
  int uv_crop_y = img->crop.y / uv_div_y;
  int ux1 = max(uxi, 0) + uv_crop_x;
  int uy1 = max(uyi, 0) + uv_crop_y;
  int ux2 = min(uxi + 1, img->image_dims.x / uv_div_x - 1) + uv_crop_x;
  int uy2 = min(uyi + 1, img->image_dims.y / uv_div_y - 1) + uv_crop_y;
  REAL2 uv00 = I420_READ(ux1, uy1, u_image, v_image, img->strides.y, img->strides.z);
  REAL2 uv01 = I420_READ(ux2, uy1, u_image, v_image, img->strides.y, img->strides.z);
  REAL2 uv10 = I420_READ(ux1, uy2, u_image, v_image, img->strides.y, img->strides.z);
  REAL2 uv11 = I420_READ(ux2, uy2, u_image, v_image, img->strides.y, img->strides.z);
  REAL2 uv_val = mix(mix(uv00, uv01, uv_xfrac), mix(uv10, uv11, uv_xfrac), uv_yfrac);
  return convert_uchar4_sat((float4)(y_val, (float)uv_val.x, (float)uv_val.y, 255.0f));
}

uchar4 i420_sampler_pb_three_plane(__global const uchar *y_image, __global const uchar *u_image, __global const uchar *v_image, int out_x, int out_y, float fx, float fy, const image_description *img, uchar fill, float filter_scale_x, float filter_scale_y) {
  return yuv_planar_sampler_pb_three_plane(y_image, u_image, v_image, out_x, out_y, fx, fy, img, fill, filter_scale_x, filter_scale_y, 2, 2);
}

uchar4 i420_sampler_pb_two_plane(__global const uchar *y_image, __global const uchar *uv_image, int out_x, int out_y, float fx, float fy, const image_description *img, uchar fill, float filter_scale_x, float filter_scale_y) {
  return i420_sampler_pb_three_plane(y_image, uv_image, uv_image + img->offsets.z, out_x, out_y, fx, fy, img, fill, filter_scale_x, filter_scale_y);
}

uchar4 i420_sampler_pb(__global const uchar *y_image, int out_x, int out_y, float fx, float fy, const image_description *img, uchar fill, float filter_scale_x, float filter_scale_y) {
  return i420_sampler_pb_three_plane(y_image, y_image + img->offsets.y, y_image + img->offsets.z, out_x, out_y, fx, fy, img, fill, filter_scale_x, filter_scale_y);
}

uchar4 yuyv_sampler_pb(__global const uchar *in, int out_x, int out_y, float fx, float fy, const image_description *img, uchar fill, float filter_scale_x, float filter_scale_y) {
  if (fx < 0 || fx >= img->image_dims.x || fy < 0 || fy >= img->image_dims.y) {
    return (uchar4)(16, 128, 128, 255);
  }
  int adj_x = out_x - img->letterbox.x;
  int adj_y = out_y - img->letterbox.y;
  if (adj_x < 0 || adj_y < 0 || adj_x >= img->letterbox.z || adj_y >= img->letterbox.w) {
    return (uchar4)(fill, 128, 128, 255);
  }
  float src_x = fx - 0.5f;
  float src_y = fy - 0.5f;
  int cx = (int)floor(src_x);
  int cy = (int)floor(src_y);
  float fs_x = max(1.0f, filter_scale_x);
  float fs_y = max(1.0f, filter_scale_y);
  int half_x = (int)ceil(fs_x);
  int half_y = (int)ceil(fs_y);
  __global const uchar4 *in4 = (__global const uchar4 *)in;
  int stride = img->strides.x / 4;
  float4 sum = (float4)(0.0f);
  float wsum = 0.0f;
  for (int j = -(half_y - 1); j <= half_y; j++) {
    float wy = pillow_bilinear_weight((src_y - (float)(cy + j)) / fs_y);
    int sy = clamp(cy + j, 0, img->image_dims.y - 1) + img->crop.y;
    for (int i = -(half_x - 1); i <= half_x; i++) {
      float w = wy * pillow_bilinear_weight((src_x - (float)(cx + i)) / fs_x);
      int sx = clamp(cx + i, 0, img->image_dims.x - 1) + img->crop.x;
      float4 p = convert_float4(in4[sy * stride + (sx >> 1)]);
      float4 px = (sx & 1) ? (float4)(p.z, p.y, p.w, 255) : (float4)(p.x, p.y, p.w, 255);
      sum += w * px;
      wsum += w;
    }
  }
  if (wsum != 0.0f) sum /= wsum;
  return convert_uchar4_sat(sum);
}

uchar4 y444_sampler_pb_three_plane(__global const uchar *y_image, __global const uchar *u_image, __global const uchar *v_image, int out_x, int out_y, float fx, float fy, const image_description *img, uchar fill, float filter_scale_x, float filter_scale_y) {
  return yuv_planar_sampler_pb_three_plane(y_image, u_image, v_image, out_x, out_y, fx, fy, img, fill, filter_scale_x, filter_scale_y, 1, 1);
}

uchar4 y444_sampler_pb_two_plane(__global const uchar *y_image, __global const uchar *uv_image, int out_x, int out_y, float fx, float fy, const image_description *img, uchar fill, float filter_scale_x, float filter_scale_y) {
  return y444_sampler_pb_three_plane(y_image, uv_image, uv_image + img->offsets.z, out_x, out_y, fx, fy, img, fill, filter_scale_x, filter_scale_y);
}

uchar4 y444_sampler_pb(__global const uchar *y_image, int out_x, int out_y, float fx, float fy, const image_description *img, uchar fill, float filter_scale_x, float filter_scale_y) {
  return y444_sampler_pb_three_plane(y_image, y_image + img->offsets.y, y_image + img->offsets.z, out_x, out_y, fx, fy, img, fill, filter_scale_x, filter_scale_y);
}

uchar4 y42b_sampler_pb_three_plane(__global const uchar *y_image, __global const uchar *u_image, __global const uchar *v_image, int out_x, int out_y, float fx, float fy, const image_description *img, uchar fill, float filter_scale_x, float filter_scale_y) {
  return yuv_planar_sampler_pb_three_plane(y_image, u_image, v_image, out_x, out_y, fx, fy, img, fill, filter_scale_x, filter_scale_y, 2, 1);
}

uchar4 y42b_sampler_pb_two_plane(__global const uchar *y_image, __global const uchar *uv_image, int out_x, int out_y, float fx, float fy, const image_description *img, uchar fill, float filter_scale_x, float filter_scale_y) {
  return y42b_sampler_pb_three_plane(y_image, uv_image, uv_image + img->offsets.z, out_x, out_y, fx, fy, img, fill, filter_scale_x, filter_scale_y);
}

uchar4 y42b_sampler_pb(__global const uchar *y_image, int out_x, int out_y, float fx, float fy, const image_description *img, uchar fill, float filter_scale_x, float filter_scale_y) {
  return y42b_sampler_pb_three_plane(y_image, y_image + img->offsets.y, y_image + img->offsets.z, out_x, out_y, fx, fy, img, fill, filter_scale_x, filter_scale_y);
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

// GStreamer Numeric Enum Layouts
typedef enum {
  GST_RANGE_UNKNOWN = 0,
  GST_RANGE_LIMITED = 1,
  GST_RANGE_FULL = 2
} GstRange;
typedef enum {
  GST_MATRIX_UNKNOWN = 0,
  GST_MATRIX_RGB = 1,
  GST_MATRIX_FCC = 2,
  GST_MATRIX_BT601 = 3,
  GST_MATRIX_BT709 = 4,
  GST_MATRIX_SMPTE240M = 5,
  GST_MATRIX_BT2020 = 6
} GstMatrix;

struct luma_weights {
  float Kr;
  float Kb;
};

luma_weights
get_luma_weights(int matrix_idx)
{
  switch (matrix_idx) {
    case GST_MATRIX_FCC:
    case GST_MATRIX_BT601:
      return { 0.2990f, 0.1140f };
    case GST_MATRIX_BT709:
      return { 0.2126f, 0.0722f };
    case GST_MATRIX_SMPTE240M:
      return { 0.2120f, 0.0870f };
    case GST_MATRIX_BT2020:
      return { 0.2627f, 0.0593f };
    default:
      return { 0.2990f, 0.1140f };
  }
}


cl_float16
get_yuv_to_rgb_matrix(int range_idx, int matrix_idx)
{
  float y_scale, cb_cr_scale, y_offset, c_offset;
  if (range_idx == GST_RANGE_FULL) {
    y_scale = 1.0f;
    cb_cr_scale = 1.0f;
    y_offset = 0.0f;
    c_offset = (matrix_idx == GST_MATRIX_RGB) ? 0.0f : 128.0f;
  } else {
    y_scale = 255.0f / (235.0f - 16.0f);
    cb_cr_scale = 255.0f / (240.0f - 16.0f);
    y_offset = 16.0f;
    c_offset = 128.0f;
  }

  if (matrix_idx == GST_MATRIX_RGB) {
    // clang-format off
    return { { y_scale, 0.0f, 0.0f, 0.0f,
             0.0f, y_scale, 0.0f, 0.0f,
             0.0f, 0.0f, y_scale, 0.0f,
             -y_scale * y_offset, -y_scale * y_offset, -y_scale * y_offset, 1.0f } };
    // clang-format on
  }

  auto [Kr, Kb] = get_luma_weights(matrix_idx);
  float Kg = 1.0f - Kr - Kb;
  float r_cr = 2.0f * (1.0f - Kr);
  float b_cb = 2.0f * (1.0f - Kb);
  float g_cb = (2.0f * Kb * (1.0f - Kb)) / Kg;
  float g_cr = (2.0f * Kr * (1.0f - Kr)) / Kg;

  //            R               G                    B               offset
  cl_float16 m{};
  m.s[0] = y_scale;
  m.s[1] = y_scale;
  m.s[2] = y_scale;
  m.s[5] = -cb_cr_scale * g_cb;
  m.s[6] = cb_cr_scale * b_cb;
  m.s[8] = cb_cr_scale * r_cr;
  m.s[9] = -cb_cr_scale * g_cr;
  m.s[12] = (-y_scale * y_offset) - (m.s[8] * c_offset);
  m.s[13] = (-y_scale * y_offset) - (m.s[5] * c_offset) - (m.s[9] * c_offset);
  m.s[14] = (-y_scale * y_offset) - (m.s[6] * c_offset);
  m.s[15] = 1.0f;
  return m;
}

cl_float16
get_yuv_to_gray_matrix(int range_idx, int matrix_idx)
{
  cl_float16 m{};
  if (matrix_idx == GST_MATRIX_RGB) {
    constexpr float Kr = 0.2990f, Kg = 0.5870f, Kb = 0.1140f;
    // clang-format off
    m.s[0] = m.s[1] = m.s[2] = Kr;
    m.s[4] = m.s[5] = m.s[6] = Kg;
    m.s[8] = m.s[9] = m.s[10] = Kb;
    m.s[15] = 1.0f;
    // clang-format on
    return m;
  }
  float y_scale = (range_idx == GST_RANGE_FULL) ? 1.0f : 255.0f / (235.0f - 16.0f);
  float y_offset = (range_idx == GST_RANGE_FULL) ? 0.0f : 16.0f;
  float bias = -y_scale * y_offset;
  // clang-format off
  m.s[0] = m.s[5] = m.s[10] = y_scale;
  m.s[12] = m.s[13] = m.s[14] = bias;
  m.s[15] = 1.0f;
  // clang-format on
  return m;
}

cl_float16
get_color_conversion_matrix(AxVideoFormat in_format, AxVideoFormat out_format)
{
  int range_idx = GST_RANGE_UNKNOWN;
  int color_matrix_idx = GST_MATRIX_UNKNOWN;
  // RGB-type inputs are always full-range; colour matrix/range from the stream are irrelevant.
  if (is_rgb(in_format) || is_bgr(in_format) || in_format == AxVideoFormat::GRAY8) {
    range_idx = GST_RANGE_FULL;
    color_matrix_idx = GST_MATRIX_RGB;
  } else {
    // YUV inputs: treat unknown (0) as the default limited/BT.601 behaviour.
    if (range_idx == GST_RANGE_UNKNOWN)
      range_idx = GST_RANGE_LIMITED;
    if (color_matrix_idx == GST_MATRIX_UNKNOWN)
      color_matrix_idx = GST_MATRIX_BT601;
  }
  if (in_format == AxVideoFormat::GRAY8 && out_format == AxVideoFormat::GRAY8) {
    return get_yuv_to_rgb_matrix(GST_RANGE_FULL, GST_MATRIX_RGB);
  }
  if (out_format == AxVideoFormat::GRAY8) {
    auto result = get_yuv_to_gray_matrix(range_idx, color_matrix_idx);
    if (is_bgr(in_format)) {
      std::swap(result.s[0], result.s[8]);
      std::swap(result.s[1], result.s[9]);
      std::swap(result.s[2], result.s[10]);
      std::swap(result.s[3], result.s[11]);
    }
    return result;
  }
  auto result = get_yuv_to_rgb_matrix(range_idx, color_matrix_idx);
  if (output_needs_swizzle(in_format, out_format)) {
    std::swap(result.s[0], result.s[2]);
    std::swap(result.s[4], result.s[6]);
    std::swap(result.s[8], result.s[10]);
    std::swap(result.s[12], result.s[14]);
  }
  return result;
}

cl_float16
get_color_conversion_matrix_with_norm(AxVideoFormat in_format, AxVideoFormat out_format,
    const std::vector<cl_float> &mul, const std::vector<cl_float> &add)
{
  auto M = get_color_conversion_matrix(in_format, out_format);
  // Fuse per-channel affine norm (out_i = color_i * mul_i + add_i) into the
  // matrix. Rows 0-2 are input channel weights; row 3 is the bias vector.
  for (int i = 0; i < 4; ++i) {
    float m = mul[i];
    float a = add[i];
    M.s[0 * 4 + i] *= m;
    M.s[1 * 4 + i] *= m;
    M.s[2 * 4 + i] *= m;
    M.s[3 * 4 + i] = M.s[3 * 4 + i] * m + a;
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
  if (in.format == AxVideoFormat::I420 || in.format == AxVideoFormat::Y42B
      || in.format == AxVideoFormat::Y444) {
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
  if (in.format == AxVideoFormat::I420 || in.format == AxVideoFormat::Y42B
      || in.format == AxVideoFormat::Y444) {
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

const char *y444_sampler = R"##(
    uchar4 pixel = y444_sampler(in, col, row, corrected.x, corrected.y, &img, fill);
)##";

const char *y444_sampler_two_plane = R"##(
    uchar4 pixel = y444_sampler_two_plane(in, in_uv, col, row, corrected.x, corrected.y, &img, fill);
)##";

const char *y444_sampler_three_plane = R"##(
    uchar4 pixel = y444_sampler_three_plane(in, in_u, in_v, col, row, corrected.x, corrected.y, &img, fill);
)##";

const char *y42b_sampler = R"##(
    uchar4 pixel = y42b_sampler(in, col, row, corrected.x, corrected.y, &img, fill);
)##";

const char *y42b_sampler_two_plane = R"##(
    uchar4 pixel = y42b_sampler_two_plane(in, in_uv, col, row, corrected.x, corrected.y, &img, fill);
)##";

const char *y42b_sampler_three_plane = R"##(
    uchar4 pixel = y42b_sampler_three_plane(in, in_u, in_v, col, row, corrected.x, corrected.y, &img, fill);
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

const char *y444_nn_sampler = R"##(
    uchar y = in[corrected.y * strides.x + corrected.x];
    __global const uchar *u = advance_uchar_ptr(in, offsets.y);
    __global const uchar *v = advance_uchar_ptr(in, offsets.z);
    uchar4 pixel = (uchar4)(y, u[corrected.y * strides.y + corrected.x], v[corrected.y * strides.z + corrected.x], 255);
)##";

const char *y444_nn_sampler_two_plane = R"##(
    uchar y = in[corrected.y * strides.x + corrected.x];
    __global const uchar *inu = in_uv;
    __global const uchar *inv = advance_uchar_ptr(in_uv, offsets.z);
    uchar4 pixel = (uchar4)(y, inu[corrected.y * strides.y + corrected.x], inv[corrected.y * strides.z + corrected.x], 255);
)##";

const char *y444_nn_sampler_three_plane = R"##(
    uchar y = in[corrected.y * strides.x + corrected.x];
    uchar4 pixel = (uchar4)(y, in_u[corrected.y * strides.y + corrected.x], in_v[corrected.y * strides.z + corrected.x], 255);
)##";

const char *nv12_pb_sampler = R"##(
    uchar4 pixel = nv12_sampler_pb(in, col, row, corrected.x, corrected.y, &img, fill, xscale, yscale);
)##";

const char *nv12_pb_sampler_two_plane = R"##(
    uchar4 pixel = nv12_sampler_pb_two_plane(in, in_uv, col, row, corrected.x, corrected.y, &img, fill, xscale, yscale);
)##";

const char *nv16_pb_sampler = R"##(
    uchar4 pixel = nv16_sampler_pb(in, col, row, corrected.x, corrected.y, &img, fill, xscale, yscale);
)##";

const char *nv16_pb_sampler_two_plane = R"##(
    uchar4 pixel = nv16_sampler_pb_two_plane(in, in_uv, col, row, corrected.x, corrected.y, &img, fill, xscale, yscale);
)##";

const char *i420_pb_sampler = R"##(
    uchar4 pixel = i420_sampler_pb(in, col, row, corrected.x, corrected.y, &img, fill, xscale, yscale);
)##";

const char *i420_pb_sampler_two_plane = R"##(
    uchar4 pixel = i420_sampler_pb_two_plane(in, in_uv, col, row, corrected.x, corrected.y, &img, fill, xscale, yscale);
)##";

const char *i420_pb_sampler_three_plane = R"##(
    uchar4 pixel = i420_sampler_pb_three_plane(in, in_u, in_v, col, row, corrected.x, corrected.y, &img, fill, xscale, yscale);
)##";

const char *yuyv_pb_sampler = R"##(
    uchar4 pixel = yuyv_sampler_pb(in, col, row, corrected.x, corrected.y, &img, fill, xscale, yscale);
)##";

const char *gray8_pb_sampler = R"##(
    uchar pixel = gray8_sampler_pb(in, col, row, corrected.x, corrected.y, &img, fill, xscale, yscale);
)##";

const char *rgb_pb_sampler = R"##(
    uchar4 pixel = rgb_sampler_pb(in, col, row, corrected.x, corrected.y, &img, fill, xscale, yscale);
)##";

const char *rgba_pb_sampler = R"##(
    uchar4 pixel = rgba_sampler_pb(in, col, row, corrected.x, corrected.y, &img, fill, xscale, yscale);
)##";

const char *y444_pb_sampler = R"##(
    uchar4 pixel = y444_sampler_pb(in, col, row, corrected.x, corrected.y, &img, fill, xscale, yscale);
)##";

const char *y444_pb_sampler_two_plane = R"##(
    uchar4 pixel = y444_sampler_pb_two_plane(in, in_uv, col, row, corrected.x, corrected.y, &img, fill, xscale, yscale);
)##";

const char *y444_pb_sampler_three_plane = R"##(
    uchar4 pixel = y444_sampler_pb_three_plane(in, in_u, in_v, col, row, corrected.x, corrected.y, &img, fill, xscale, yscale);
)##";

const char *y42b_pb_sampler = R"##(
    uchar4 pixel = y42b_sampler_pb(in, col, row, corrected.x, corrected.y, &img, fill, xscale, yscale);
)##";

const char *y42b_pb_sampler_two_plane = R"##(
    uchar4 pixel = y42b_sampler_pb_two_plane(in, in_uv, col, row, corrected.x, corrected.y, &img, fill, xscale, yscale);
)##";

const char *y42b_pb_sampler_three_plane = R"##(
    uchar4 pixel = y42b_sampler_pb_three_plane(in, in_u, in_v, col, row, corrected.x, corrected.y, &img, fill, xscale, yscale);
)##";

// Y42B (planar 4:2:2): chroma full vertical resolution, half horizontal (corrected.x / 2).
const char *y42b_nn_sampler = R"##(
    uchar y = in[corrected.y * strides.x + corrected.x];
    __global const uchar *u = advance_uchar_ptr(in, offsets.y);
    __global const uchar *v = advance_uchar_ptr(in, offsets.z);
    uchar4 pixel = (uchar4)(y, u[corrected.y * strides.y + corrected.x / 2], v[corrected.y * strides.z + corrected.x / 2], 255);
)##";

const char *y42b_nn_sampler_two_plane = R"##(
    uchar y = in[corrected.y * strides.x + corrected.x];
    __global const uchar *inu = in_uv;
    __global const uchar *inv = advance_uchar_ptr(in_uv, offsets.z);
    uchar4 pixel = (uchar4)(y, inu[corrected.y * strides.y + corrected.x / 2], inv[corrected.y * strides.z + corrected.x / 2], 255);
)##";

const char *y42b_nn_sampler_three_plane = R"##(
    uchar y = in[corrected.y * strides.x + corrected.x];
    uchar4 pixel = (uchar4)(y, in_u[corrected.y * strides.y + corrected.x / 2], in_v[corrected.y * strides.z + corrected.x / 2], 255);
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
  { AxVideoFormat::Y42B, "uchar",
      { y42b_sampler, y42b_sampler_two_plane, y42b_sampler_three_plane } },
  { AxVideoFormat::Y444, "uchar",
      { y444_sampler, y444_sampler_two_plane, y444_sampler_three_plane } },
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
  { AxVideoFormat::Y42B, "uchar",
      { y42b_nn_sampler, y42b_nn_sampler_two_plane, y42b_nn_sampler_three_plane } },
  { AxVideoFormat::Y444, "uchar",
      { y444_nn_sampler, y444_nn_sampler_two_plane, y444_nn_sampler_three_plane } },
  { AxVideoFormat::YUY2, "uchar4", { yuyv_nn_sampler } },
  { AxVideoFormat::RGBA, "uchar4", { rgba_nn_sampler } },
  { AxVideoFormat::BGRA, "uchar4", { rgba_nn_sampler } },
  { AxVideoFormat::RGB, "uchar", { rgb_nn_sampler } },
  { AxVideoFormat::BGR, "uchar", { rgb_nn_sampler } },
  { AxVideoFormat::GRAY8, "uchar", { gray8_nn_sampler } },
};

std::vector<kernel_arg_details> pillow_bilinear_input_details_tab = {
  { AxVideoFormat::NV12, "uchar", { nv12_pb_sampler, nv12_pb_sampler_two_plane } },
  { AxVideoFormat::NV16, "uchar", { nv16_pb_sampler, nv16_pb_sampler_two_plane } },
  { AxVideoFormat::I420, "uchar",
      { i420_pb_sampler, i420_pb_sampler_two_plane, i420_pb_sampler_three_plane } },
  { AxVideoFormat::Y444, "uchar",
      { y444_pb_sampler, y444_pb_sampler_two_plane, y444_pb_sampler_three_plane } },
  { AxVideoFormat::Y42B, "uchar",
      { y42b_pb_sampler, y42b_pb_sampler_two_plane, y42b_pb_sampler_three_plane } },
  { AxVideoFormat::YUY2, "uchar", { yuyv_pb_sampler } },
  { AxVideoFormat::RGBA, "uchar", { rgba_pb_sampler } },
  { AxVideoFormat::BGRA, "uchar", { rgba_pb_sampler } },
  { AxVideoFormat::RGB, "uchar", { rgb_pb_sampler } },
  { AxVideoFormat::BGR, "uchar", { rgb_pb_sampler } },
  { AxVideoFormat::GRAY8, "uchar", { gray8_pb_sampler } },
};

kernel_args
get_input_details(AxVideoFormat format, Interpolation interp, int num_planes)
{
  auto &tab = interp == Interpolation::nearest ? nn_input_details_tab :
              interp == Interpolation::pillow_bilinear ? pillow_bilinear_input_details_tab :
                                                         input_details_tab;
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
