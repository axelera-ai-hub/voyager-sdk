#pragma once

#include <gst/gst.h>

G_BEGIN_DECLS

#define GST_TYPE_AXTRANSFORM (gst_axtransform_get_type())
#define GST_AXTRANSFORM(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_AXTRANSFORM, GstAxtransform))
#define GST_AXTRANSFORM_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass), GST_TYPE_AXTRANSFORM, GstAxtransformClass))
#define GST_IS_AXTRANSFORM(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj), GST_TYPE_AXTRANSFORM))
#define GST_IS_AXTRANSFORM_CLASS(obj) \
  (G_TYPE_CHECK_CLASS_TYPE((klass), GST_TYPE_AXTRANSFORM))

typedef struct _GstAxtransform GstAxtransform;
typedef struct _GstAxtransformClass GstAxtransformClass;

typedef struct _GstAxtransformData GstAxtransformData;

struct _GstAxtransform {
  GstElement parent;

  GstPad *srcpad;
  GstPad *sinkpad;

  GstBufferPool *pool;
  GstAllocator *allocator;
  gsize outsize;

  GstAxtransformData *data;
};

struct _GstAxtransformClass {
  GstElementClass parent_class;
};

GType gst_axtransform_get_type(void);

G_END_DECLS
