// Copyright Axelera AI, 2024
#pragma once

#include "AxDataInterface.h"
#include "AxLog.hpp"
#include "AxUtils.hpp"

#include <gmodule.h>
#include <gst/gst.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>


//  Indicates the type of tensor data.
//  Currently, we support only INT8 and FLOAT32.
enum class tensor_type {
  INT8,
  UINT8,
  INT16,
  UINT16,
  INT32,
  UINT32,
  FLOAT64,
  FLOAT32,
  INT64,
  UINT64,
  FLOAT16,

  LAST,
};

enum class tensor_format { STATIC, FLEXIBLE, LAST };

constexpr auto AX_TENSOR_RANK_LIMIT = 8;
using tensor_dim = std::array<uint32_t, AX_TENSOR_RANK_LIMIT>;

struct GstTensorInfo {
  tensor_type type; // Tensor type
  tensor_dim dimension; // Tensor dimensions (we support up to 8 dimensions)
};

constexpr auto AX_TENSOR_SIZE_LIMIT = 32;
struct GstTensorsInfo {
  unsigned int num_tensors{};
  std::array<GstTensorInfo, AX_TENSOR_SIZE_LIMIT> info; /**< The list of tensor info */
  tensor_format format{};
};

struct GstTensorsConfig {
  int rate_n{}; // Frame rate numerator
  int rate_d{}; // Frame rate denominator
  GstTensorsInfo info{}; // Actual tensor info
};

#define TENSOR_MIME_TYPE "other/tensor"
#define TENSORS_MIME_TYPE "other/tensors"
#define GST_TENSOR_RATE_RANGE "(fraction) [ 0, max ]"


#define GST_TENSOR_CAP_DEFAULT \
  TENSOR_MIME_TYPE ", "        \
                   "framerate = " GST_TENSOR_RATE_RANGE

#define GST_TENSORS_CAP_DEFAULT                                                  \
  TENSORS_MIME_TYPE ", "                                                         \
                    "format = (string) static, num_tensors = (int) [ 1, 16 ],  " \
                    "framerate = (fraction) [ 0, max ]"

#define GST_TENSORS_CAP_MAKE(fmt)                 \
  TENSORS_MIME_TYPE ", "                          \
                    "format = (string) " fmt ", " \
                    "framerate = " GST_TENSOR_RATE_RANGE

#define GST_TENSORS_FLEX_CAP_DEFAULT GST_TENSORS_CAP_MAKE("flexible")


bool structure_is_tensor_stream(const GstStructure *structure);

bool gst_tensors_config_from_structure(GstTensorsConfig &config, const GstStructure *structure);

int tensor_type_size(tensor_type type);

bool validate_tensors_config(const GstTensorsConfig &config);

guint get_buffer_n_tensor(GstBuffer *buffer);

GstCaps *get_possible_pad_caps_from_config(GstPad *pad, const GstTensorsConfig &config);

void update_tensor_dimensions(GstCaps *caps, GstCaps *peer_caps);

#define GST_AX_VIDEO_FORMATS(GST_AX_VIDEO_FORMAT_REGISTER) \
  GST_AX_VIDEO_FORMAT_REGISTER(RGB)                        \
  GST_AX_VIDEO_FORMAT_REGISTER(RGBA)                       \
  GST_AX_VIDEO_FORMAT_REGISTER(RGBx)                       \
  GST_AX_VIDEO_FORMAT_REGISTER(BGR)                        \
  GST_AX_VIDEO_FORMAT_REGISTER(BGRA)                       \
  GST_AX_VIDEO_FORMAT_REGISTER(BGRx)                       \
  GST_AX_VIDEO_FORMAT_REGISTER(GRAY8)                      \
  GST_AX_VIDEO_FORMAT_REGISTER(NV12)                       \
  GST_AX_VIDEO_FORMAT_REGISTER(NV16)                       \
  GST_AX_VIDEO_FORMAT_REGISTER(I420)                       \
  GST_AX_VIDEO_FORMAT_REGISTER(YUY2)

#define AX_GST_GENERATE_VIDEO_CAPS(x) GST_VIDEO_CAPS_MAKE(#x) ";"
#define AX_GST_VIDEO_FORMATS_CAPS \
  GST_AX_VIDEO_FORMATS(AX_GST_GENERATE_VIDEO_CAPS)

AxDataInterface interface_from_caps_and_meta(GstCaps *caps, GstBuffer *buffer);
size_t size_from_interface(const AxDataInterface &interface);
void copy_or_fixate_framerate(GstCaps *from, GstCaps *to);
void assign_data_ptrs_to_interface(
    const std::vector<GstMapInfo> &info, AxDataInterface &interface);
void assign_vaapi_ptrs_to_interface(
    const std::vector<GstMapInfo> &info, AxDataInterface &interface);
void assign_opencl_ptrs_to_interface(AxDataInterface &input, GstBuffer *buffer);

void assign_fds_to_interface(AxDataInterface &input, GstBuffer *buffer);
void add_video_meta_from_interface_to_buffer(
    GstBuffer *buffer, const AxDataInterface &interface);
GstCaps *caps_from_interface(const AxDataInterface &interface);

std::vector<GstMapInfo> get_mem_map(GstBuffer *buffer, GstMapFlags flags, GObject *self);
void unmap_mem(std::vector<GstMapInfo> &mapInfoVec);

extern "C" GstAllocator *gst_tensor_dmabuf_allocator_get(const char *device);

extern "C" GstAllocator *gst_aligned_allocator_get(void);

extern "C" GstAllocator *gst_opencl_allocator_get(const char *cl_platform, Ax::Logger *logger);

extern "C" gboolean gst_is_opencl_memory(GstMemory *mem);

extern "C" opencl_buffer *gst_opencl_mem_get_opencl_buffer(GstMemory *mem);

extern "C" AxAllocationContext *gst_opencl_allocator_get_context(
    const char *cl_platform, Ax::Logger *logger);

extern "C" void gst_opencl_mem_reset(GstMemory *mem);

extern "C" void gst_opencl_memory_add_dependency(GstMemory *mem, GstBuffer *dependency);

// Custom query type for querying buffer pool requirements
#define GST_QUERY_AX_BUFFER_REQUIREMENTS ((GstQueryType) (GST_QUERY_CUSTOM))
#define GST_QUERY_SUPPORTS_OPENCL ((GstQueryType) (GST_QUERY_CUSTOM + 1))
