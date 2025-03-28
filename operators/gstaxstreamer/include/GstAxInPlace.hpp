#pragma once

#include <gst/base/gstbasetransform.h>

G_BEGIN_DECLS

#define GST_TYPE_AXINPLACE (gst_axinplace_get_type())
#define GST_AXINPLACE(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_AXINPLACE, GstAxinplace))
#define GST_AXINPLACE_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass), GST_TYPE_AXINPLACE, GstAxinplaceClass))
#define GST_IS_AXINPLACE(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj), GST_TYPE_AXINPLACE))
#define GST_IS_AXINPLACE_CLASS(obj) \
  (G_TYPE_CHECK_CLASS_TYPE((klass), GST_TYPE_AXINPLACE))

typedef struct _GstAxinplace GstAxinplace;
typedef struct _GstAxinplaceClass GstAxinplaceClass;

typedef struct _GstAxinplaceData GstAxinplaceData;

struct _GstAxinplace {
  GstBaseTransform parent;

  GstAxinplaceData *data;
};

struct _GstAxinplaceClass {
  GstBaseTransformClass parent_class;
};

GType gst_axinplace_get_type(void);

G_END_DECLS
