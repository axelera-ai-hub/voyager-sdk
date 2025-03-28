#include <glib.h>
#include <gst/allocators/allocators.h>

GST_EXPORT GstAllocator *
gst_tensor_dmabuf_allocator_get(const char *device)
{
  return NULL;
}

GST_EXPORT gboolean
gst_is_tensor_dmabuf_memory(GstMemory *mem)
{
  return FALSE;
}
