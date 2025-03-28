#pragma once

#include <gst/gst.h>

G_BEGIN_DECLS

#define GST_TYPE_DISTRIBUTOR (gst_distributor_get_type())
#define GST_DISTRIBUTOR(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_DISTRIBUTOR, GstDistributor))
#define GST_DISTRIBUTOR_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass), GST_TYPE_DISTRIBUTOR, GstDistributorClass))
#define GST_IS_DISTRIBUTOR(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj), GST_TYPE_DISTRIBUTOR))
#define GST_IS_DISTRIBUTOR_CLASS(obj) \
  (G_TYPE_CHECK_CLASS_TYPE((klass), GST_TYPE_DISTRIBUTOR))

typedef struct _GstDistributor GstDistributor;
typedef struct _GstDistributorClass GstDistributorClass;

typedef struct _GstDistributorData GstDistributorData;

struct _GstDistributor {
  GstElement parent;

  GstPad *srcpad;
  GstPad *sinkpad;

  GstDistributorData *data;
};

struct _GstDistributorClass {
  GstElementClass parent_class;
};

GType gst_distributor_get_type(void);

G_END_DECLS
