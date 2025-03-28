#pragma once

#include <gst/base/gstaggregator.h>
#include <set>
G_BEGIN_DECLS

#define GST_TYPE_AXFUNNEL (gst_axfunnel_get_type())
#define GST_AXFUNNEL(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_AXFUNNEL, GstAxfunnel))
#define GST_AXFUNNEL_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass), GST_TYPE_AXFUNNEL, GstAxfunnelClass))
#define GST_IS_AXFUNNEL(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj), GST_TYPE_AXFUNNEL))
#define GST_IS_AXFUNNEL_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass), GST_TYPE_AXFUNNEL))

typedef struct _GstAxfunnel GstAxfunnel;
typedef struct _GstAxfunnelClass GstAxfunnelClass;

typedef struct _GstAxfunnelData GstAxfunnelData;

struct _GstAxfunnel {
  GstAggregator parent;
  std::set<int> stream_set;
};

struct _GstAxfunnelClass {
  GstAggregatorClass parent_class;
};

G_GNUC_INTERNAL GType gst_axfunnel_get_type(void);

G_END_DECLS
