/**
 * Custom UDMABUF Allocator
 */
#include <fcntl.h>
#include <glib.h>
#include <gst/allocators/allocators.h>
#include <gst/allocators/gstfdmemory.h>
#include <gst/gst.h>
#include <linux/udmabuf.h>
#include <stdio.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

GST_DEBUG_CATEGORY(dmabufallocator_debug);
#define GST_CAT_DEFAULT dmabufallocator_debug

#define gst_tensor_dmabuf_allocator_parent_class parent_class

extern "C" {
GST_EXPORT void gst_tensor_dmabuf_allocator_register(const char *device);
GST_EXPORT GstAllocator *gst_tensor_dmabuf_allocator_get(const char *device);
GST_EXPORT gboolean gst_is_tensor_dmabuf_memory(GstMemory *mem);
GST_EXPORT GType gst_tensor_dmabuf_allocator_get_type(void);
}
#define GST_TENSOR_DMABUF_ALLOCATOR "GstTensorDmaBufAllocator"
#define GST_TYPE_TENSOR_DMABUF_ALLOCATOR \
  (gst_tensor_dmabuf_allocator_get_type())
#define GST_TENSOR_DMABUF_ALLOCATOR_CAST(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_TENSOR_DMABUF_ALLOCATOR, GstTensorDmaBufAllocator))
#define GST_TENSOR_DMABUF_ALLOCATOR_GET_CLASS(obj) \
  (G_TYPE_INSTANCE_GET_CLASS((obj), GST_TYPE_TENSOR_DMABUF_ALLOCATOR, GstTensorDmaBufAllocatorClass))

typedef struct {
  GstMemory mem;
  GstFdMemoryFlags flags;
  gint fd;
  gpointer data;
  gint mmapping_flags;
  gint mmap_count;
  GMutex lock;
} GstDMABUFMemory;

#define GST_TENSOR_DMABUF_MEMORY_NAME "GstDMABUFMemory"
#define GST_TENSOR_DMABUF_MEMORY_CAST(mem) ((GstDMABUFMemory *) (mem))
enum { PROP_0, PROP_DEVICE, N_PROPERTIES };
static GParamSpec *obj_properties[N_PROPERTIES] = {
  NULL,
};

/*
 * @brief struct for type GstTensorDmaBufAllocator
 */
typedef struct {
  GstFdAllocator parent;
  int uDmaBufDeviceFD;
  gchar *uDmaBufDeviceName;
  int id;
} GstTensorDmaBufAllocator;

/**
 * @brief struct for class GstTensorDmaBufAllocatorClass
 */
typedef struct {
  GstFdAllocatorClass parent_class;
} GstTensorDmaBufAllocatorClass;

G_DEFINE_TYPE(GstTensorDmaBufAllocator, gst_tensor_dmabuf_allocator, GST_TYPE_FD_ALLOCATOR);

#define DMABUF_HEAP_SYSTEM "/dev/dma_heap/system"
#define DMABUF_HEAP_RESERVED "/dev/dma_heap/reserved"
struct dma_heap_allocation_data {
  __u64 len;
  __u32 fd;
  __u32 fd_flags;
  __u64 heap_flags;
};

#define DMA_HEAP_IOC_MAGIC 'H'

/**
 * DOC: DMA_HEAP_IOCTL_ALLOC - allocate memory from pool
 *
 * Takes a dma_heap_allocation_data struct and returns it with the fd field
 * populated with the dmabuf handle of the allocation.
 */
#define DMA_HEAP_IOCTL_ALLOC \
  _IOWR(DMA_HEAP_IOC_MAGIC, 0x0, struct dma_heap_allocation_data)

/**
 * @brief DMABUF allocation function
 */
static GstMemory *
gst_tensor_dmabuf_alloc(GstAllocator *allocator, gsize size, GstAllocationParams *params)
{
  char name[16];
  int memfd, ret, pages, dmabuffd;
  GstMemory *mem;
  gsize totalsize;
  struct udmabuf_create create;
  GstTensorDmaBufAllocator *self = GST_TENSOR_DMABUF_ALLOCATOR_CAST(allocator);
  GST_OBJECT_LOCK(self);
  if (self->uDmaBufDeviceFD <= 0) {
    GST_ERROR_OBJECT(self, "UDMABUF not opened");
    return NULL;
  }

  if (!strncmp(self->uDmaBufDeviceName, DMABUF_HEAP_SYSTEM, sizeof(DMABUF_HEAP_SYSTEM))
      || !strncmp(self->uDmaBufDeviceName, DMABUF_HEAP_RESERVED, sizeof(DMABUF_HEAP_RESERVED))) {
    GST_DEBUG_OBJECT(self, "DMABUF use %s", self->uDmaBufDeviceName);
    pages = ((size - 1) / getpagesize()) + 1;
    totalsize = pages * getpagesize();

    struct dma_heap_allocation_data data = {
      .len = totalsize,
      .fd = 0,
      .fd_flags = O_RDWR | O_CLOEXEC,
      .heap_flags = 0,
    };

    ret = ioctl(self->uDmaBufDeviceFD, DMA_HEAP_IOCTL_ALLOC, &data);
    if (ret < 0) {
      GST_ERROR_OBJECT(self, "Fail to alloc dmabuf heap system\n");
      return NULL;
    }
    dmabuffd = (int) data.fd;
  } else {
    GST_ERROR_OBJECT(self, "Unsupported dmabuf device: %s", self->uDmaBufDeviceName);
    return NULL;
  }

  /* Get a dmabuf gstmemory with the fd */
  mem = gst_fd_allocator_alloc(
      allocator, dmabuffd, size, static_cast<GstFdMemoryFlags>(0));

  if (G_UNLIKELY(!mem)) {
    GST_ERROR_OBJECT(self, "GstDmaBufMemory allocation failed");
    close(dmabuffd);
    return NULL;
  }

  GST_DEBUG_OBJECT(self, "DMABUF created with fd: %d, at address: %p", dmabuffd, mem);
  GST_OBJECT_UNLOCK(self);
  return mem;
}

static gpointer
gst_tensor_dmabuf_map(GstMemory *mem, gsize maxsize, GstMapFlags flags)
{

  int fd = gst_fd_memory_get_fd(mem);
  GstTensorDmaBufAllocator *self = GST_TENSOR_DMABUF_ALLOCATOR_CAST(mem->allocator);
  GstDMABUFMemory *dmabuf = GST_TENSOR_DMABUF_MEMORY_CAST(mem);

  gpointer ptr = mmap(NULL, maxsize, flags, GST_MEMORY_FLAGS(mem), fd, 0);
  if (ptr == NULL || ptr == MAP_FAILED) {
    GST_ERROR_OBJECT(self, "Unable to map dmabuf: %d error: %s", fd, strerror(errno));
    return NULL;
  }
  GST_DEBUG_OBJECT(self, "mmap: %p", ptr);
  dmabuf->data = ptr;

  return ptr;
}

static void
gst_tensor_dmbuf_unmap(GstMemory *mem)
{
  GstTensorDmaBufAllocator *self = GST_TENSOR_DMABUF_ALLOCATOR_CAST(mem->allocator);
  GstDMABUFMemory *dmabuf = GST_TENSOR_DMABUF_MEMORY_CAST(mem);
  gpointer ptr = dmabuf->data;
  if (ptr == NULL) {
    GST_ERROR_OBJECT(self, "Invalid memory: %p", ptr);
    return;
  }
  int ret = munmap((void *) ptr, mem->maxsize);
  if (ret != 0) {
    GST_ERROR_OBJECT(self, "Unable to unmap, return: %d, error: %d", ret, errno);
    return;
  }
  GST_DEBUG_OBJECT(self, "unmmap: %p", ptr);
  dmabuf->data = NULL;
}

static void
gst_tensor_dmabuf_free(GstAllocator *allocator, GstMemory *mem)
{
  int fd = -1;
  GstTensorDmaBufAllocator *self = GST_TENSOR_DMABUF_ALLOCATOR_CAST(allocator);

  if (!gst_memory_is_type(mem, GST_TENSOR_DMABUF_MEMORY_NAME)) {
    GST_ERROR_OBJECT(self, "Unable to free non dmabuf memory");
    return;
  }

  fd = gst_fd_memory_get_fd(mem);
  GST_DEBUG_OBJECT(self, "Free DMABUF with fd: %d", fd);
  if (fd <= 0) {
    GST_ERROR_OBJECT(self, "Invalid fd");
    return;
  }

  GST_ALLOCATOR_CLASS(parent_class)->free(allocator, mem);
}

static void
gst_tensor_dmabuf_allocator_finalize(GObject *obj)
{
  GstTensorDmaBufAllocator *self = GST_TENSOR_DMABUF_ALLOCATOR_CAST(obj);
  close(self->uDmaBufDeviceFD);
  self->uDmaBufDeviceFD = -1;
  G_OBJECT_CLASS(parent_class)->finalize(obj);
  GST_LOG_OBJECT(self, "DMABUF Allocator finalize");
}

static void
gst_tensor_dmabuf_allocator_set_property(
    GObject *object, guint property_id, const GValue *value, GParamSpec *pspec)
{
  GstTensorDmaBufAllocator *self = GST_TENSOR_DMABUF_ALLOCATOR_CAST(object);
  switch (property_id) {
    case PROP_DEVICE:
      self->uDmaBufDeviceName = g_value_dup_string(value);
      break;
    default:
      /* We don't have any other property... */
      G_OBJECT_WARN_INVALID_PROPERTY_ID(object, property_id, pspec);
      break;
  }
}

static void
gst_tensor_dmabuf_allocator_get_property(
    GObject *object, guint property_id, GValue *value, GParamSpec *pspec)
{
  GstTensorDmaBufAllocator *self = GST_TENSOR_DMABUF_ALLOCATOR_CAST(object);
  switch (property_id) {
    case PROP_DEVICE:
      g_value_set_string(value, self->uDmaBufDeviceName);
      break;

    default:
      /* We don't have any other property... */
      G_OBJECT_WARN_INVALID_PROPERTY_ID(object, property_id, pspec);
      break;
  }
}
/**
 * @brief class initization for GstTensorDmaBufAllocatorClass
 */
static void
gst_tensor_dmabuf_allocator_class_init(GstTensorDmaBufAllocatorClass *klass)
{
  GstAllocatorClass *alloc = (GstAllocatorClass *) klass;
  GObjectClass *gobject_class = G_OBJECT_CLASS(klass);

  alloc->alloc = GST_DEBUG_FUNCPTR(gst_tensor_dmabuf_alloc);
  alloc->free = GST_DEBUG_FUNCPTR(gst_tensor_dmabuf_free);
  GST_DEBUG_CATEGORY_INIT(dmabufallocator_debug, "dmabufallocator", 0, "GstDMABUFAllocator debug");

  gobject_class->set_property = gst_tensor_dmabuf_allocator_set_property;
  gobject_class->get_property = gst_tensor_dmabuf_allocator_get_property;

  obj_properties[PROP_DEVICE] = g_param_spec_string("device", "DMABUF device",
      "Set dmabuf device name", "/dev/udmabuf",
      static_cast<GParamFlags>(G_PARAM_CONSTRUCT_ONLY | G_PARAM_READWRITE));

  g_object_class_install_properties(gobject_class, N_PROPERTIES, obj_properties);
}

static void
gst_tensor_dmabuf_open(GstTensorDmaBufAllocator *self)
{
  self->uDmaBufDeviceFD = open(self->uDmaBufDeviceName, O_RDWR | O_CLOEXEC);
  if (self->uDmaBufDeviceFD < 0) {
    GST_ERROR_OBJECT(self, "Failed to open UDMABUF device: %s", self->uDmaBufDeviceName);
    return;
  }

  GST_INFO_OBJECT(self, "Initialization successfull: %d", self->uDmaBufDeviceFD);
}

/**
 * @brief initialzation for GstTensorDmaBufAllocator
 */
static void
gst_tensor_dmabuf_allocator_init(GstTensorDmaBufAllocator *self)
{

  GstAllocator *alloc = GST_ALLOCATOR_CAST(self);
  GstAllocator *sysmem_alloc;
  GObjectClass *object_class
      = G_OBJECT_CLASS(GST_TENSOR_DMABUF_ALLOCATOR_GET_CLASS(self));

  alloc->mem_type = GST_TENSOR_DMABUF_MEMORY_NAME;
  alloc->mem_map = gst_tensor_dmabuf_map;
  alloc->mem_unmap = gst_tensor_dmbuf_unmap;

  sysmem_alloc = gst_allocator_find(GST_ALLOCATOR_SYSMEM);

  alloc->mem_copy = sysmem_alloc->mem_copy;
  alloc->mem_share = sysmem_alloc->mem_share;
  alloc->mem_is_span = sysmem_alloc->mem_is_span;

  gst_object_unref(sysmem_alloc);

  object_class->finalize = gst_tensor_dmabuf_allocator_finalize;
}


void
gst_tensor_dmabuf_allocator_register(const char *device)
{
  auto alloc = static_cast<GstAllocator *>(
      g_object_new(GST_TYPE_TENSOR_DMABUF_ALLOCATOR, "device", device, NULL));
  gst_tensor_dmabuf_open(GST_TENSOR_DMABUF_ALLOCATOR_CAST(alloc));
  gst_allocator_register(GST_TENSOR_DMABUF_ALLOCATOR, alloc);
}

GstAllocator *
gst_tensor_dmabuf_allocator_get(const char *device)
{
  auto alloc = gst_allocator_find(GST_TENSOR_DMABUF_ALLOCATOR);
  if (!alloc) {
    gst_tensor_dmabuf_allocator_register(device);
    alloc = gst_allocator_find(GST_TENSOR_DMABUF_ALLOCATOR);
  }

  return alloc;
}

extern "C" gboolean
gst_is_tensor_dmabuf_memory(GstMemory *mem)
{
  return gst_memory_is_type(mem, GST_TENSOR_DMABUF_MEMORY_NAME);
}
