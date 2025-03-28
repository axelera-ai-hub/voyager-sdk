/**
 * Custom UDMABUF Allocator
 */
#include <glib.h>
#include <gst/allocators/allocators.h>
#include <gst/gst.h>
#include <stdio.h>
#include <sys/types.h>

GST_DEBUG_CATEGORY(alignedallocator_debug);
#define GST_CAT_DEFAULT alignedallocator_debug

#define gst_aligned_allocator_parent_class parent_class

extern "C" {
GST_EXPORT void gst_aligned_allocator_register(void);
GST_EXPORT GstAllocator *gst_aligned_allocator_get(void);
GST_EXPORT gboolean gst_is_aligned_memory(GstMemory *mem);
GST_EXPORT GType gst_aligned_allocator_get_type(void);
}

#define GST_ALIGNED_ALLOCATOR "GstAlignedAllocator"
#define GST_TYPE_ALIGNED_ALLOCATOR (gst_aligned_allocator_get_type())
#define GST_ALIGNED_ALLOCATOR_CAST(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_ALIGNED_ALLOCATOR, GstAlignedAllocator))
#define GST_ALIGNED_ALLOCATOR_GET_CLASS(obj) \
  (G_TYPE_INSTANCE_GET_CLASS((obj), GST_TYPE_ALIGNED_ALLOCATOR, GstAlignedAllocatorClass))

typedef struct {
  GstMemory mem;
} GstAlignedMemory;

#define GST_ALIGNED_MEMORY_NAME "GstAlignedMemory"
#define GST_ALIGNED_MEMORY_CAST(mem) ((GstAlignedMemory *) (mem))
enum { PROP_0, PROP_DEVICE, N_PROPERTIES };
static GParamSpec *obj_properties[N_PROPERTIES] = {
  NULL,
};

/*
 * @brief struct for type GstAlignedAllocator
 */
typedef struct {
  GstAllocator parent;
  GstAllocator *sysmem_allocator;
  GstMemory *(*default_alloc)(
      GstAllocator *allocator, gsize size, GstAllocationParams *params);
  void (*default_free)(GstAllocator *allocator, GstMemory *mem);
} GstAlignedAllocator;

/**
 * @brief struct for class GstAlignedAllocatorClass
 */
typedef struct {
  GstFdAllocatorClass parent_class;
} GstAlignedAllocatorClass;

G_DEFINE_TYPE(GstAlignedAllocator, gst_aligned_allocator, GST_TYPE_FD_ALLOCATOR);

/**
 * @brief DMABUF allocation function
 */
static GstMemory *
gst_aligned_alloc(GstAllocator *allocator, gsize size, GstAllocationParams *params)
{
  GstAlignedAllocator *aligned_allocator = GST_ALIGNED_ALLOCATOR_CAST(allocator);
  gsize maxsize = size + params->prefix + params->padding;
  GstAllocationParams aligned_params = *params;
  const int pagesize = 4096;
  aligned_params.align |= (pagesize - 1);
  maxsize = (maxsize + aligned_params.align) & ~aligned_params.align;
  return gst_allocator_alloc(aligned_allocator->sysmem_allocator, maxsize, &aligned_params);
}

static void
gst_aligned_free(GstAllocator *allocator, GstMemory *mem)
{
  GstAlignedAllocator *aligned_allocator = (GstAlignedAllocator *) allocator;
  gst_allocator_free(aligned_allocator->sysmem_allocator, mem);
}

static void
gst_aligned_allocator_finalize(GObject *obj)
{
  GstAlignedAllocator *self = GST_ALIGNED_ALLOCATOR_CAST(obj);
  g_object_unref(self->sysmem_allocator);
  G_OBJECT_CLASS(parent_class)->finalize(obj);
  GST_LOG_OBJECT(self, "Aligned Allocator finalize");
}

/**
 * @brief class initization for GstAlignedAllocatorClass
 */
static void
gst_aligned_allocator_class_init(GstAlignedAllocatorClass *klass)
{
  GstAllocatorClass *alloc = (GstAllocatorClass *) klass;
  GObjectClass *gobject_class = G_OBJECT_CLASS(klass);

  alloc->alloc = GST_DEBUG_FUNCPTR(gst_aligned_alloc);
  alloc->free = GST_DEBUG_FUNCPTR(gst_aligned_free);

  GST_DEBUG_CATEGORY_INIT(alignedallocator_debug, "alignedallocator", 0,
      "GstAlignedAllocator debug");
}

/**
 * @brief initialzation for GstAlignedAllocator
 */
static void
gst_aligned_allocator_init(GstAlignedAllocator *self)
{
  GstAllocator *alloc = GST_ALLOCATOR_CAST(self);
  GObjectClass *object_class = G_OBJECT_CLASS(GST_ALIGNED_ALLOCATOR_GET_CLASS(self));

  GstAllocator *sysmem_alloc = gst_allocator_find(GST_ALLOCATOR_SYSMEM);

  alloc->mem_type = GST_ALIGNED_MEMORY_NAME;

  self->sysmem_allocator = sysmem_alloc;
  alloc->mem_map = sysmem_alloc->mem_map;
  alloc->mem_unmap = sysmem_alloc->mem_unmap;

  alloc->mem_copy = sysmem_alloc->mem_copy;
  alloc->mem_share = sysmem_alloc->mem_share;
  alloc->mem_is_span = sysmem_alloc->mem_is_span;

  object_class->finalize = gst_aligned_allocator_finalize;
}

extern "C" void
gst_aligned_allocator_register(void)
{
  auto alloc = static_cast<GstAllocator *>(g_object_new(GST_TYPE_ALIGNED_ALLOCATOR, NULL));
  gst_allocator_register(GST_ALIGNED_ALLOCATOR, alloc);
}

extern "C" GstAllocator *
gst_aligned_allocator_get(void)
{
  auto alloc = gst_allocator_find(GST_ALIGNED_ALLOCATOR);
  if (!alloc) {
    gst_aligned_allocator_register();
    alloc = gst_allocator_find(GST_ALIGNED_ALLOCATOR);
  }

  return alloc;
}

extern "C" gboolean
gst_is_aligned_memory(GstMemory *mem)
{
  return gst_memory_is_type(mem, GST_ALIGNED_MEMORY_NAME);
}
