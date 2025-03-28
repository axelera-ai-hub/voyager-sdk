#pragma once

#include <gst/base/gstaggregator.h>

G_BEGIN_DECLS

#define GST_TYPE_AXDECODER (gst_axdecoder_get_type())
#define GST_AXDECODER(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_AXDECODER, GstAxdecoder))
#define GST_AXDECODER_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass), GST_TYPE_AXDECODER, GstAxdecoderClass))
#define GST_IS_AXDECODER(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj), GST_TYPE_AXDECODER))
#define GST_IS_AXDECODER_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass), GST_TYPE_AXDECODER))

typedef struct _GstAxdecoder GstAxdecoder;
typedef struct _GstAxdecoderClass GstAxdecoderClass;

typedef struct _GstAxdecoderData GstAxdecoderData;

struct _GstAxdecoder {
  GstAggregator parent;

  GstPad *main_sinkpad;
  GstPad *tensor_sinkpad;

  GstAxdecoderData *data;
};

struct _GstAxdecoderClass {
  GstAggregatorClass parent_class;
};

G_GNUC_INTERNAL GType gst_axdecoder_get_type(void);

G_END_DECLS
