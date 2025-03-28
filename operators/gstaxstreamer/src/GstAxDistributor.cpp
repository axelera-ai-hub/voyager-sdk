#include "GstAxDistributor.hpp"
#include <gst/gst.h>
#include "GstAxMeta.hpp"
#include "GstAxStreamerUtils.hpp"

GST_DEBUG_CATEGORY_STATIC(gst_distributor_debug_category);
#define GST_CAT_DEFAULT gst_distributor_debug_category

struct _GstDistributorData {
  std::string meta_string;
  Ax::Logger logger{ Ax::Severity::trace, nullptr, gst_distributor_debug_category };
};

G_DEFINE_TYPE_WITH_CODE(GstDistributor, gst_distributor, GST_TYPE_ELEMENT,
    GST_DEBUG_CATEGORY_INIT(gst_distributor_debug_category, "distributor", 0,
        "debug category for distributor element"));

enum {
  PROP_0,
  PROP_META_STRING,
};

static GstStaticPadTemplate sink_template
    = GST_STATIC_PAD_TEMPLATE("sink", GST_PAD_SINK, GST_PAD_ALWAYS, { GST_CAPS_ANY });

static GstStaticPadTemplate src_template
    = GST_STATIC_PAD_TEMPLATE("src", GST_PAD_SRC, GST_PAD_ALWAYS, { GST_CAPS_ANY });

static gboolean
gst_distributor_setcaps(GstDistributor *distributor, GstCaps *from)
{
  GstCaps *to = gst_caps_new_empty();
  for (int i = 0; i < gst_caps_get_size(from); ++i) {
    GstStructure *structure = gst_structure_copy(gst_caps_get_structure(from, i));

    if (gst_structure_has_field(structure, "framerate")) {
      gst_structure_remove_field(structure, "framerate");
      gst_structure_set(structure, "framerate", GST_TYPE_FRACTION, 0, 1, NULL);
    }

    gst_caps_append_structure(to, structure);
  }

  gboolean ret = gst_pad_set_caps(distributor->srcpad, to);
  gst_caps_unref(to);

  return ret;
}

static gboolean
gst_distributor_sink_event(GstPad *pad, GstObject *parent, GstEvent *event)
{
  gboolean ret;
  GstDistributor *distributor = GST_DISTRIBUTOR(parent);
  GST_DEBUG_OBJECT(distributor, "sink_event");

  switch (GST_EVENT_TYPE(event)) {
    case GST_EVENT_CAPS:
      {
        GstCaps *caps;
        gst_event_parse_caps(event, &caps);
        ret = gst_distributor_setcaps(distributor, caps);
        break;
      }
    default:
      ret = gst_pad_event_default(pad, parent, event);
      break;
  }
  return ret;
}

static GstFlowReturn
gst_distributor_sink_chain(GstPad *pad, GstObject *parent, GstBuffer *buffer)
{
  GstDistributor *distributor = GST_DISTRIBUTOR(parent);
  GST_DEBUG_OBJECT(distributor, "sink_chain");

  if (distributor->data->meta_string.empty()) {
    return gst_pad_push(distributor->srcpad, buffer);
  }

  if (!gst_buffer_has_general_meta(buffer)) {
    GST_ERROR_OBJECT(G_OBJECT(distributor), "buffer has no general meta");
    throw std::runtime_error("distributor : buffer has no ax metadata");
  }

  if (!gst_buffer_get_general_meta(buffer)->meta_map_ptr->count(
          distributor->data->meta_string)) {
    GST_ERROR_OBJECT(G_OBJECT(distributor), "key not found in meta map");
    throw std::runtime_error("distributor : key not found in meta map: "
                             + distributor->data->meta_string);
  }

  const int num = gst_buffer_get_general_meta(buffer)
                      ->meta_map_ptr->at(distributor->data->meta_string)
                      ->get_number_of_subframes();
  GST_INFO_OBJECT(G_OBJECT(distributor), "Number of buffers to be pushed: %d", num);

  if (num == 0) {
    GstEvent *gap_event
        = gst_event_new_gap(GST_BUFFER_PTS(buffer), GST_BUFFER_DURATION(buffer));
    if (!gst_pad_push_event(distributor->srcpad, gap_event)) {
      throw std::runtime_error("distributor : Failed to push gap event");
    }
    gst_buffer_unref(buffer);
    return GST_FLOW_OK;
  }

  for (int i = 0; i < num; ++i) {
    GstBuffer *out_buffer = gst_buffer_copy(buffer);

    GstMetaGeneral *meta = gst_buffer_get_general_meta(out_buffer);
    meta->subframe_index = i;
    meta->subframe_number = num;

    GstFlowReturn ret = gst_pad_push(distributor->srcpad, out_buffer);
    if (ret == GST_FLOW_ERROR) {
      throw std::runtime_error("distributor : Failed to push buffer");
    }

    if (ret == GST_FLOW_NOT_NEGOTIATED) {
      throw std::runtime_error("distributor : not negotiated");
    }
  }

  gst_buffer_unref(buffer);
  return GST_FLOW_OK;
}

static void
gst_distributor_init(GstDistributor *distributor)
{
  distributor->data = new GstDistributorData;
  distributor->data->logger
      = Ax::Logger(Ax::extract_severity_from_category(gst_distributor_debug_category),
          distributor, gst_distributor_debug_category);
  Ax::init_logger(distributor->data->logger);

  distributor->sinkpad = gst_pad_new_from_static_template(&sink_template, "sink");
  gst_element_add_pad(GST_ELEMENT(distributor), distributor->sinkpad);

  gst_pad_set_chain_function(
      distributor->sinkpad, GST_DEBUG_FUNCPTR(gst_distributor_sink_chain));
  gst_pad_set_event_function(
      distributor->sinkpad, GST_DEBUG_FUNCPTR(gst_distributor_sink_event));

  distributor->srcpad = gst_pad_new_from_static_template(&src_template, "src");
  gst_element_add_pad(GST_ELEMENT(distributor), distributor->srcpad);
}

static void
gst_distributor_set_property(
    GObject *object, guint prop_id, const GValue *value, GParamSpec *pspec)
{
  GstDistributor *distributor = GST_DISTRIBUTOR(object);
  GST_DEBUG_OBJECT(distributor, "set_property");

  switch (prop_id) {
    case PROP_META_STRING:
      distributor->data->meta_string = g_value_get_string(value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
      break;
  }
}

static void
gst_distributor_get_property(GObject *object, guint prop_id, GValue *value, GParamSpec *pspec)
{
  GstDistributor *distributor = GST_DISTRIBUTOR(object);
  GST_DEBUG_OBJECT(distributor, "get_property");

  switch (prop_id) {
    case PROP_META_STRING:
      g_value_set_string(value, distributor->data->meta_string.c_str());
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
      break;
  }
}

static void
gst_distributor_finalize(GObject *object)
{
  GstDistributor *distributor = GST_DISTRIBUTOR(object);
  GST_DEBUG_OBJECT(distributor, "finalize");

  if (distributor->data) {
    delete distributor->data;
    distributor->data = nullptr;
  }

  G_OBJECT_CLASS(gst_distributor_parent_class)->finalize(object);
}

static void
gst_distributor_class_init(GstDistributorClass *klass)
{
  gst_element_class_set_static_metadata(GST_ELEMENT_CLASS(klass), "distributor",
      "Effect", "description", "axelera.ai");

  G_OBJECT_CLASS(klass)->set_property = GST_DEBUG_FUNCPTR(gst_distributor_set_property);
  G_OBJECT_CLASS(klass)->get_property = GST_DEBUG_FUNCPTR(gst_distributor_get_property);
  G_OBJECT_CLASS(klass)->finalize = GST_DEBUG_FUNCPTR(gst_distributor_finalize);

  g_object_class_install_property(G_OBJECT_CLASS(klass), PROP_META_STRING,
      g_param_spec_string("meta", "meta key", "String with key to metadata", "",
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  gst_element_class_add_pad_template(
      GST_ELEMENT_CLASS(klass), gst_static_pad_template_get(&src_template));
  gst_element_class_add_pad_template(
      GST_ELEMENT_CLASS(klass), gst_static_pad_template_get(&sink_template));
}
