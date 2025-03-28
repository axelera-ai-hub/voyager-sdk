#pragma once

#include <gst/base/gstbasetransform.h>

G_BEGIN_DECLS

#define GST_TYPE_AXMETATOHAILO (gst_axmetatohailo_get_type())
#define GST_AXMETATOHAILO(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_AXMETATOHAILO, GstAxmetatohailo))
#define GST_AXMETATOHAILO_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass), GST_TYPE_AXMETATOHAILO, GstAxmetatohailoClass))
#define GST_IS_AXMETATOHAILO(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj), GST_TYPE_AXMETATOHAILO))
#define GST_IS_AXMETATOHAILO_CLASS(obj) \
  (G_TYPE_CHECK_CLASS_TYPE((klass), GST_TYPE_AXMETATOHAILO))

typedef struct _GstAxmetatohailo GstAxmetatohailo;
typedef struct _GstAxmetatohailoClass GstAxmetatohailoClass;

typedef struct _GstAxmetatohailoData GstAxmetatohailoData;

struct _GstAxmetatohailo {
  GstBaseTransform parent;
};

struct _GstAxmetatohailoClass {
  GstBaseTransformClass parent_class;
};

GType gst_axmetatohailo_get_type(void);

GST_ELEMENT_REGISTER_DECLARE(axmetatohailo)

G_END_DECLS
