// Copyright Axelera AI, 2025
#include "AxOpenCl.hpp"

#include <iostream>
#include <vector>


struct local_size {
  size_t width;
  size_t height;
};
local_size
determine_local_work_size(size_t width, size_t height, size_t max_work_size)
{
  while (width * height > max_work_size) {
    if (width > height)
      width /= 2;
    else
      height /= 2;
  }
  return { width, height };
}

namespace ax_utils
{
std::mutex CLProgram::cl_mutex;

CLProgram::CLProgram(const std::string &source, Ax::Logger &log) : logger(log)
{
  std::lock_guard<std::mutex> lock(cl_mutex);

  cl_platform_id platformId;
  cl_uint numPlatforms;

  auto error = clGetPlatformIDs(1, &platformId, &numPlatforms);
  if (error != CL_SUCCESS) {
    throw std::runtime_error("Failed to get platform ID! Error = " + std::to_string(error));
  }
  extensions = init_extensions(platformId);

  cl_uint num_devices = 1;
  error = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_GPU, 1, &device_id, &num_devices);

  if (error != CL_SUCCESS) {
    throw std::runtime_error("Unable to find appropriate OpenCL device.");
  }
  context = create_context(platformId, device_id, extensions);
  if (error != CL_SUCCESS) {
    logger(AX_ERROR) << "Failed to create OpenCL context, error = " << error << std::endl;
    throw std::runtime_error(
        "Failed to create OpenCL context, error = " + std::to_string(error));
  }
  commands = clCreateCommandQueue(context, device_id, 0, &error);
  if (error != CL_SUCCESS) {
    logger(AX_ERROR) << "Failed to create OpenCL command queue, error = " << error
                     << std::endl;
    throw std::runtime_error(
        "Failed to create OpenCL command queue, error = " + std::to_string(error));
  }

  const char *sources[] = { source.c_str() };
  program = error == CL_SUCCESS ?
                clCreateProgramWithSource(context, 1, sources, NULL, &error) :
                cl_program{};
  error = error == CL_SUCCESS ? clBuildProgram(program, 0, NULL, NULL, NULL, NULL) : error;
  if (error != CL_SUCCESS) {
    size_t param_value_size_ret;
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL,
        &param_value_size_ret);
    std::vector<char> build_log(param_value_size_ret + 1);
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG,
        param_value_size_ret, build_log.data(), NULL);
    std::cerr << "Build log:\n" << build_log.data() << std::endl;
    throw std::runtime_error("Failed to create OpenCL program");
  }
  clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE,
      sizeof max_work_group_size, &max_work_group_size, NULL);
}

CLProgram::ax_kernel
CLProgram::get_kernel(const std::string &kernel_name) const
{
  int error = CL_SUCCESS;
  auto kernel = clCreateKernel(program, kernel_name.c_str(), &error);
  if (error != CL_SUCCESS) {
    throw std::runtime_error("Failed to create kernel " + kernel_name
                             + ", error = " + std::to_string(error));
  }
  return ax_kernel{ kernel };
}

CLProgram::ax_buffer
CLProgram::create_buffer(int elem_size, int num_elems, int flags,
    const std::variant<void *, int> &ptr) const
{
  cl_int error = CL_SUCCESS;
  return ax_buffer{ create_optimal_buffer(
      context, extensions, elem_size, num_elems, flags, ptr, error) };
}

CLProgram::ax_buffer
CLProgram::create_buffer(const buffer_details &details, int flags)
{
  return create_buffer(
      1, ax_utils::determine_buffer_size(details), flags, details.data);
}

int
CLProgram::write_buffer(const ax_buffer &buffer, int elem_size, int num_elems, const void *data)
{
  return clEnqueueWriteBuffer(
      commands, *buffer, CL_TRUE, 0, elem_size * num_elems, data, 0, NULL, NULL);
}

int
CLProgram::read_buffer(const ax_buffer &buffer, int elem_size, int num_elems, void *data)
{
  return clEnqueueReadBuffer(
      commands, *buffer, CL_TRUE, 0, elem_size * num_elems, data, 0, NULL, NULL);
}

CLProgram::flush_details
CLProgram::flush_output_buffer_async(const ax_buffer &out, int size)
{
  int ret = CL_SUCCESS;
  auto event = cl_event{};
  auto mapped = clEnqueueMapBuffer(
      commands, *out, CL_FALSE, CL_MAP_READ, 0, size, 0, NULL, &event, &ret);
  return { ret, event, mapped };
}

int
CLProgram::unmap_buffer(const ax_buffer &out, void *mapped)
{
  auto ret = clEnqueueUnmapMemObject(commands, *out, mapped, 0, NULL, NULL);
  if (ret != CL_SUCCESS) {
    throw std::runtime_error(
        "Failed to unmap output buffer, error = " + std::to_string(ret));
  }
  return ret;
}

int
CLProgram::unmap_buffer(cl_event event, const ax_buffer &out, void *mapped)
{
  auto ret = clWaitForEvents(1, &event);
  if (ret != CL_SUCCESS) {
    throw std::runtime_error("Failed to wait for event, error = " + std::to_string(ret));
  }
  clReleaseEvent(event);
  return unmap_buffer(out, mapped);
}

int
CLProgram::flush_output_buffer(const ax_buffer &out, int size)
{
  auto [result, event, mapped] = flush_output_buffer_async(out, size);
  if (result != CL_SUCCESS) {
    throw std::runtime_error(
        "Failed to map output buffer, error = " + std::to_string(result));
    return result;
  }

  int ret = clWaitForEvents(1, &event);
  return unmap_buffer(out, mapped);
}

int
CLProgram::execute_kernel(const ax_kernel &kernel, int num_dims, size_t global_work_size[3])
{
  //  TODO: determine local work size
  size_t local[3] = { 16, 16, 1 };
  size_t global[3] = { 0 };
  global[0] = (global_work_size[0] + local[0] - 1) & ~(local[0] - 1);
  global[1] = (global_work_size[1] + local[1] - 1) & ~(local[1] - 1);
  global[2] = global_work_size[2];
  return clEnqueueNDRangeKernel(
      commands, *kernel, num_dims, NULL, global, local, 0, NULL, NULL);
}

CLProgram::~CLProgram()
{
  if (program)
    clReleaseProgram(program);
  if (commands)
    clReleaseCommandQueue(commands);
  if (context)
    clReleaseContext(context);
}

const char *
get_kernel_utils()
{

  return R"##(

//  Convert YUV values to RGBA
uchar4 convert_YUV2RGBA(float3 yuv) {

    float Y = yuv.x - 16.0f;
    float U = yuv.y - 128.0f;
    float V = yuv.z - 128.0f;

    Y *= 1.164f;
    float4 rgba = (float4)(Y + 1.596F * V, Y - 0.391F * U - 0.813F * V, Y + 2.018F * U, 255.0f);
    return convert_uchar4_sat_rte(rgba);
}

float3 bilinear(float3 p00, float3 p01, float3 p10, float3 p11, float xfrac, float yfrac) {
    float3 i1 = mad((p01 - p00), xfrac, p00);
    float3 i2 = mad((p11 - p10), xfrac, p10);
    return mad((i2 - i1), yfrac, i1);
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

  x1 = max(x1, 0);
  y1 = max(y1, 0);
  int x2 = min(x1 + 1,img->width - 1);
  int y2 = min(y1 + 1,img->height - 1);
  x1 += img->crop_x;
  y1 += img->crop_y;
  x2 += img->crop_x;
  y2 += img->crop_y;

  float4 top = convert_float4(in[y1 * img->stride + x1 / 2]);
  float4 bottom = convert_float4(in[y2 * img->stride + x2 / 2]);

  float3 in00 = x1 % 2 == 0 ? top.xyw : top.zyw;
  float3 in01 = x1 % 2 == 0 ? convert_float4(in[y1 * img->stride + x1 / 2 + 1]).xyw : top.zyw;
  float3 in10 = x1 % 2 == 0 ? bottom.xyw : bottom.zyw;
  float3 in11 = x1 % 2 == 0 ? convert_float4(in[y2 * img->stride + x1 / 2 + 1]).xyw : bottom.zyw;

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

uchar4 rgb_sampler_bl(__global const uchar4 *image, float fx, float fy, const rgb_image *img ) {
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

    float4 i1 = mad((p01 - p00), xfrac, p00);
    float4 i2 = mad((p11 - p10), xfrac, p10);
    uchar4 result = convert_uchar4_sat_rte(mad((i2 - i1), yfrac, i1));
    return result;
}
)##";
}

} // namespace ax_utils
