// Copyright Axelera AI, 2026
/**
 * Custom UDMABUF Allocator
 */
#include <fcntl.h>
#include <glib.h>
#include <gst/allocators/allocators.h>
#include <gst/gst.h>
#include <stdio.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include "GstAxDataUtils.hpp"
#include "GstAxStreamerUtils.hpp"
#if HAVE_OPENCL
#include "AxOpenCl.hpp"


//  Design.
//
//  This allocator is designed to be our default allocator for buffers.
//  It will allocate buffers in system memory that can be used in the same
//  way as a buffer from the default allocator would be used.
//
//  It provides buffers that are suitably aligned for direct mapping into GPU
//  space on integrated GPUs. If the subplugin acknowledges that it can
//  use OpenCL buffers, then the buffer will not be mapped before being passed
//  to the subplugin. If the buffer already contains cl_mem object it will be
//  used directly, otherwise a new cl_mem object will be created from the
//  buffer data.
//  Only when the buffer is mapped will the cl_mem object be mapped back into
//  cpu space and the cl_mem and related objects released.
//
//  This allows us to only do the potential CPU->GPU->CPU transfers when we
//  actually need to avoiding traffic backwards and forwards over the PCIe bus


GST_DEBUG_CATEGORY(openclallocator_debug);
#define GST_CAT_DEFAULT openclallocator_debug

#define gst_opencl_allocator_parent_class parent_class

extern "C" {
GST_EXPORT void gst_opencl_allocator_register();
GST_EXPORT GstAllocator *gst_opencl_allocator_get(const char *which_cl, Ax::Logger *logger);
GST_EXPORT gboolean gst_is_opencl_memory(GstMemory *mem);
GST_EXPORT GType gst_opencl_allocator_get_type(void);
}
#define GST_OPENCL_ALLOCATOR "GstOpenCLAllocator"
#define GST_TYPE_OPENCL_ALLOCATOR (gst_opencl_allocator_get_type())
#define GST_OPENCL_ALLOCATOR_CAST(obj) ((GstOpenCLAllocator *) (obj))
#define GST_OPENCL_ALLOCATOR_GET_CLASS(obj) \
  (G_TYPE_INSTANCE_GET_CLASS((obj), GST_TYPE_OPENCL_ALLOCATOR, GstOpenCLAllocatorClass))

typedef struct {
  GstMemory mem;
  opencl_buffer buffer{}; // OpenCL buffer
} GstOpenCLMemory;

#define GST_OPENCL_MEMORY_NAME "GstOpenCLMemory"
#define GST_OPENCL_MEMORY_CAST(mem) ((GstOpenCLMemory *) (mem))
enum { PROP_0, PROP_DEVICE, N_PROPERTIES };
static GParamSpec *obj_properties[N_PROPERTIES] = {
  NULL,
};

/*
 * @brief struct for type GstTensorOpenCLAllocator
 */
typedef struct {
  GstAllocator parent;
  ax_utils::opencl_details cl_details; // OpenCL details
} GstOpenCLAllocator;

/**
 * @brief struct for class GstTensorDmaBufAllocatorClass
 */
typedef struct {
  GstAllocatorClass parent_class;
} GstOpenCLAllocatorClass;

G_DEFINE_TYPE(GstOpenCLAllocator, gst_opencl_allocator, GST_TYPE_ALLOCATOR);

/**
 * @brief DMABUF allocation function
 */
static GstMemory *
gst_opencl_alloc(GstAllocator *allocator, gsize size, GstAllocationParams *params)
{
  auto *self = GST_OPENCL_ALLOCATOR_CAST(allocator);
  auto *mem = new GstOpenCLMemory{};
  auto page_size = 4096;
  auto adjusted_size = (size + page_size - 1) & ~(page_size - 1);
  auto *data = std::aligned_alloc(page_size, adjusted_size);
  mem->buffer.data = std::span<uint8_t>(reinterpret_cast<uint8_t *>(data), adjusted_size);
  gst_memory_init(GST_MEMORY_CAST(mem), (GstMemoryFlags) 0, allocator, NULL,
      adjusted_size, 0, 0, size);
  return GST_MEMORY_CAST(mem);
}

static void
release_buffer_dependencies(GstOpenCLMemory *ocl_mem)
{
  //  Release the GstMemory objects that the OpenCL buffer depends on
  for (auto mem : ocl_mem->buffer.gst_memories) {
    gst_buffer_unref((GstBuffer *) mem);
  }
  ocl_mem->buffer.gst_memories.clear();
  if (ocl_mem->buffer.event) {
    ocl_mem->buffer.event.reset();
  }
}

static cl_command_queue
get_command_queue()
{
  GstOpenCLAllocator *opencl_allocator = GST_OPENCL_ALLOCATOR_CAST(
      gst_opencl_allocator_get("auto", nullptr)); // Get the OpenCL allocator
  if (opencl_allocator) {
    auto commands = opencl_allocator->cl_details.commands; // Get the command queue
    gst_object_unref(opencl_allocator); // Don't forget to unref when done
    return commands;
  }
  return nullptr;
}

static gpointer
gst_opencl_map(GstMemory *mem, gsize maxsize, GstMapFlags flags)
{
  auto *ocl_mem = GST_OPENCL_MEMORY_CAST(mem);
  if (!gst_memory_is_type(mem, GST_OPENCL_MEMORY_NAME)) {
    GST_ERROR_OBJECT(ocl_mem, "Unable to map non OpenCL memory");
    return nullptr;
  }
  //  If we have a buffer, then we need to map it into CPU space
  //  Once we have done that we can release the buffer and any dependencies
  if (ocl_mem->buffer.buffer) {
    cl_command_queue commands = get_command_queue();
    if (!commands) {
      GST_ERROR_OBJECT(ocl_mem, "No OpenCL command queue available for mapping");
      return nullptr;
    }
    int error = CL_SUCCESS;
    if (ocl_mem->buffer.event) {
      clWaitForEvents(1, &*ocl_mem->buffer.event);
      ocl_mem->buffer.event.reset();
    } else {
      auto cl_flags = (flags & GST_MAP_WRITE) != 0 ? CL_MAP_WRITE_INVALIDATE_REGION : CL_MAP_READ;
      ocl_mem->buffer.mapped = clEnqueueMapBuffer(commands, ocl_mem->buffer.buffer,
          CL_TRUE, cl_flags, 0, maxsize, 0, nullptr, nullptr, &error);
    }
    release_buffer_dependencies(ocl_mem);
    if (error != CL_SUCCESS) {
      GST_ERROR_OBJECT(ocl_mem, "Failed to map the buffer to CPU space: %d", error);
      return nullptr;
    }
    return ocl_mem->buffer.mapped;
  }
  return ocl_mem->buffer.data.data(); // Return the data pointer
}

static void
gst_opencl_unmap(GstMemory *mem)
{
  auto *ocl_mem = GST_OPENCL_MEMORY_CAST(mem);
  if (!gst_memory_is_type(mem, GST_OPENCL_MEMORY_NAME)) {
    GST_ERROR_OBJECT(ocl_mem, "Unable to map non OpenCL memory");
  }
  if (ocl_mem->buffer.buffer) {
    cl_command_queue commands = get_command_queue();
    if (!commands) {
      GST_ERROR_OBJECT(ocl_mem, "No OpenCL command queue available for mapping");
      return;
    }
    clEnqueueUnmapMemObject(
        commands, ocl_mem->buffer.buffer, ocl_mem->buffer.mapped, 0, NULL, NULL);
  }
}

static void
gst_opencl_free(GstAllocator *allocator, GstMemory *mem)
{
  GstOpenCLAllocator *self = GST_OPENCL_ALLOCATOR_CAST(allocator);

  if (!gst_memory_is_type(mem, GST_OPENCL_MEMORY_NAME)) {
    GST_ERROR_OBJECT(self, "Unable to free non opencl memory");
    return;
  }
  GstOpenCLMemory *ocl_mem = GST_OPENCL_MEMORY_CAST(mem);
  if (ocl_mem->buffer.buffer) {
    cl_int error = clReleaseMemObject(ocl_mem->buffer.buffer);
    if (error != CL_SUCCESS) {
      GST_ERROR_OBJECT(self, "Failed to release OpenCL buffer: %d", error);
    }
  }
  release_buffer_dependencies(ocl_mem);
  std::free(ocl_mem->buffer.data.data());
  delete ocl_mem;
}

static void
gst_opencl_allocator_finalize(GObject *obj)
{
  GstOpenCLAllocator *self = GST_OPENCL_ALLOCATOR_CAST(obj);
  //  Close device platofrm etc.
  if (self->cl_details.context) {
    clReleaseContext(self->cl_details.context);
    self->cl_details.context = nullptr;
  }
  if (self->cl_details.commands) {
    clReleaseCommandQueue(self->cl_details.commands);
    self->cl_details.commands = nullptr;
  }

  G_OBJECT_CLASS(parent_class)->finalize(obj);
  GST_LOG_OBJECT(self, "OpenCL Allocator finalize");
}

/**
 * @brief class initialization for GstOpenCLAllocatorClass
 */
static void
gst_opencl_allocator_class_init(GstOpenCLAllocatorClass *klass)
{
  GstAllocatorClass *alloc = (GstAllocatorClass *) klass;
  GObjectClass *gobject_class = G_OBJECT_CLASS(klass);

  alloc->alloc = GST_DEBUG_FUNCPTR(gst_opencl_alloc);
  alloc->free = GST_DEBUG_FUNCPTR(gst_opencl_free);
  GST_DEBUG_CATEGORY_INIT(openclallocator_debug, "openclallocator", 0, "GstOPENCLAllocator debug");
}

/**
 * @brief initialization for GstOpenCLAllocatorClass
 */
static void
gst_opencl_allocator_init(GstOpenCLAllocator *self)
{
  GstAllocator *alloc = GST_ALLOCATOR_CAST(self);
  GObjectClass *object_class = G_OBJECT_CLASS(GST_OPENCL_ALLOCATOR_GET_CLASS(self));
  GstAllocator *sysmem_alloc = gst_aligned_allocator_get();
  alloc->mem_type = GST_OPENCL_MEMORY_NAME;
  alloc->mem_map = gst_opencl_map;
  alloc->mem_unmap = gst_opencl_unmap;

  alloc->mem_copy = sysmem_alloc->mem_copy;
  alloc->mem_share = sysmem_alloc->mem_share;
  alloc->mem_is_span = sysmem_alloc->mem_is_span;

  gst_object_unref(sysmem_alloc);
  object_class->finalize = gst_opencl_allocator_finalize;
}

void
gst_opencl_allocator_register()
{
  auto alloc = static_cast<GstAllocator *>(g_object_new(GST_TYPE_OPENCL_ALLOCATOR, NULL));
  gst_allocator_register(GST_OPENCL_MEMORY_NAME, alloc);
}

extern "C" GstAllocator *
gst_opencl_allocator_get(const char *cl_platform, Ax::Logger *logger)
{
  auto alloc = gst_allocator_find(GST_OPENCL_MEMORY_NAME);
  if (!alloc) {
    auto details = ax_utils::build_cl_details(*logger, cl_platform, nullptr);
    gst_opencl_allocator_register();
    auto cl_alloc = GST_OPENCL_ALLOCATOR_CAST(gst_allocator_find(GST_OPENCL_MEMORY_NAME));
    cl_alloc->cl_details = details;
    return GST_ALLOCATOR_CAST(cl_alloc);
  }
  return alloc;
}

extern "C" gboolean
gst_is_opencl_memory(GstMemory *mem)
{
  return gst_memory_is_type(mem, GST_OPENCL_MEMORY_NAME);
}

extern "C" AxAllocationContext *
gst_opencl_allocator_get_context(const char *cl_platform, Ax::Logger *logger)
{
  auto alloc = Ax::as_handle(
      GST_OPENCL_ALLOCATOR_CAST(gst_opencl_allocator_get(cl_platform, logger)));
  return new AxAllocationContext(alloc->cl_details);
}

opencl_buffer *
gst_opencl_mem_get_opencl_buffer(GstMemory *mem)
{
  if (!gst_is_opencl_memory(mem)) {
    GST_ERROR("Memory is not OpenCL memory");
    return nullptr;
  }
  auto *ocl_mem = GST_OPENCL_MEMORY_CAST(mem);
  return &ocl_mem->buffer;
}


void
gst_opencl_mem_reset(GstMemory *mem)
{
  if (!gst_is_opencl_memory(mem)) {
    GST_ERROR("Memory is not OpenCL memory");
    return;
  }
  auto *ocl_mem = GST_OPENCL_MEMORY_CAST(mem);
  release_buffer_dependencies(ocl_mem);
}

void
gst_opencl_memory_add_dependency(GstMemory *mem, GstBuffer *dependency)
{
  if (!gst_is_opencl_memory(mem)) {
    return;
  }
  auto *ocl_mem = GST_OPENCL_MEMORY_CAST(mem);
  ocl_mem->buffer.gst_memories.push_back(dependency);
  gst_buffer_ref(dependency); // Increase reference count for the dependency
}

#else

extern "C" GstAllocator *
gst_opencl_allocator_get(const char *cl_platform, Ax::Logger *logger)
{
  //  If we have no OpenCL fall back to the aligned allocator
  return gst_aligned_allocator_get();
}

extern "C" gboolean
gst_is_opencl_memory(GstMemory *mem)
{
  return false;
}

opencl_buffer *
gst_opencl_mem_get_opencl_buffer(GstMemory *mem)
{
  return nullptr;
}

extern "C" AxAllocationContext *
gst_opencl_allocator_get_context(const char *cl_platform, Ax::Logger *logger)
{
  return {};
}

void
gst_opencl_mem_reset(GstMemory * /*mem*/)
{
}

void
gst_opencl_memory_add_dependency(GstMemory * /*mem*/, GstBuffer * /*dependency*/)
{
}
#endif
