// Copyright Axelera AI, 2024

#pragma once
#include <glib.h>
#include <gst/base/gstadapter.h>
#include <gst/base/gstbasetransform.h>
#include <gst/base/gstcollectpads.h>
#include <gst/gst.h>
#include <stdint.h>


G_BEGIN_DECLS

#define GST_TYPE_AXINFERENCE (gst_axinference_get_type())
#define GST_AXINFERENCE(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_AXINFERENCE, GstAxInference))
#define GST_AXINFERENCE_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass), GST_TYPE_AXINFERENCE, GstAxInferenceClass))
#define GST_IS_AXINFERENCE(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj), GST_TYPE_AXINFERENCE))
#define GST_IS_AXINFERENCE_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass), GST_TYPE_AXINFERENCE))

typedef struct _GstAxInference GstAxInference;
typedef struct _GstAxInferenceClass GstAxInferenceClass;

typedef struct _GstAxInferenceData GstAxInferenceData;

typedef enum {
  GET_IN_OUT_INFO,
  SET_INPUT_INFO,
} model_info_ops;

struct GstAxInferenceImpl;
struct _GstAxInference {
  GstBaseTransform element;
  GstAxInferenceImpl *impl;
};

struct _GstAxInferenceClass {
  GstBaseTransformClass parent_class;
};

G_GNUC_INTERNAL GType gst_axinference_get_type(void);

G_END_DECLS
