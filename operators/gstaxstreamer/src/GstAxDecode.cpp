#include "GstAxDecode.hpp"
#include <gmodule.h>
#include <gst/gst.h>
#include <memory>
#include <unordered_map>
#include "AxDataInterface.h"
#include "AxMeta.hpp"
#include "AxStreamerUtils.hpp"
#include "GstAxDataUtils.hpp"
#include "GstAxMeta.hpp"
#include "GstAxStreamerUtils.hpp"

GST_DEBUG_CATEGORY_STATIC(gst_axdecoder_debug);
#define GST_CAT_DEFAULT gst_axdecoder_debug

enum {
  PROP_0,
  PROP_SHARED_LIB_PATH,
  PROP_OPTIONS,
  PROP_MODE,
};

static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE(
    "sink", GST_PAD_SINK, GST_PAD_ALWAYS, GST_STATIC_CAPS_ANY);

static GstStaticPadTemplate sink_tensor_template = GST_STATIC_PAD_TEMPLATE("sink_tensor",
    GST_PAD_SINK, GST_PAD_ALWAYS, GST_STATIC_CAPS(GST_TENSORS_CAP_DEFAULT));

static GstStaticPadTemplate src_template
    = GST_STATIC_PAD_TEMPLATE("src", GST_PAD_SRC, GST_PAD_ALWAYS, GST_STATIC_CAPS_ANY);


struct _GstAxdecoderData {
  std::unique_ptr<Ax::SharedLib> shared;
  std::shared_ptr<void> subplugin_data;
  std::string shared_lib_path;
  std::string mode;
  std::string options;
  bool options_initialised = false;
  Ax::Logger logger{ Ax::Severity::trace, nullptr, GST_CAT_DEFAULT };

  Ax::V1Plugin::Decoder fns;
};

G_DEFINE_TYPE_WITH_CODE(GstAxdecoder, gst_axdecoder, GST_TYPE_AGGREGATOR,
    GST_DEBUG_CATEGORY_INIT(gst_axdecoder_debug, "axdecoder", 0, "axdecoder element"));

static void
gst_axdecoder_set_property(
    GObject *object, guint property_id, const GValue *value, GParamSpec *pspec)
{
  GstAxdecoder *muxer = GST_AXDECODER(object);
  GST_DEBUG_OBJECT(muxer, "set_property");

  switch (property_id) {
    case PROP_SHARED_LIB_PATH:
      muxer->data->shared_lib_path = Ax::libname(g_value_get_string(value));
      muxer->data->shared = std::make_unique<Ax::SharedLib>(
          muxer->data->logger, muxer->data->shared_lib_path);
      Ax::load_v1_plugin(*muxer->data->shared, muxer->data->fns);
      break;
    case PROP_MODE:
      muxer->data->mode = g_value_get_string(value);
      break;
    case PROP_OPTIONS:
      muxer->data->options = g_value_get_string(value);
      update_options(object, muxer->data->options, muxer->data->fns, muxer->data->logger,
          muxer->data->subplugin_data, muxer->data->options_initialised);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID(object, property_id, pspec);
      break;
  }
}

static void
gst_axdecoder_get_property(GObject *object, guint property_id, GValue *value, GParamSpec *pspec)
{
  GstAxdecoder *muxer = GST_AXDECODER(object);
  GST_DEBUG_OBJECT(muxer, "get_property");

  switch (property_id) {
    case PROP_SHARED_LIB_PATH:
      g_value_set_string(value, muxer->data->shared_lib_path.c_str());
      break;
    case PROP_OPTIONS:
      g_value_set_string(value, muxer->data->options.c_str());
      break;
    case PROP_MODE:
      g_value_set_string(value, muxer->data->mode.c_str());
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID(object, property_id, pspec);
      break;
  }
}

static void
gst_axdecoder_finalize(GObject *object)
{
  GstAxdecoder *muxer = GST_AXDECODER(object);
  GST_DEBUG_OBJECT(muxer, "finalize");

  if (muxer->data) {
    delete muxer->data;
    muxer->data = nullptr;
  }

  G_OBJECT_CLASS(gst_axdecoder_parent_class)->finalize(object);
}

static gboolean
gst_axdecoder_sink_event(GstAggregator *aggregator, GstAggregatorPad *pad, GstEvent *event)
{
  GstAxdecoder *muxer = GST_AXDECODER(aggregator);
  GST_DEBUG_OBJECT(muxer, "sink_event");

  if (GST_EVENT_CAPS == GST_EVENT_TYPE(event)) {
    if (pad == GST_AGGREGATOR_PAD_CAST(muxer->main_sinkpad)) {
      GstCaps *caps;
      gst_event_parse_caps(event, &caps);
      gst_caps_ref(caps);
      gst_aggregator_set_src_caps(aggregator, caps);
    }
    gst_event_unref(event);
    return TRUE;
  }

  return GST_AGGREGATOR_CLASS(gst_axdecoder_parent_class)->sink_event(aggregator, pad, event);
}

static gboolean
gst_axdecoder_src_query(GstAggregator *aggregator, GstQuery *query)
{
  switch (GST_QUERY_TYPE(query)) {
    case GST_QUERY_POSITION:
    case GST_QUERY_DURATION:
    case GST_QUERY_URI:
    case GST_QUERY_CAPS:
    case GST_QUERY_ACCEPT_CAPS:
    case GST_QUERY_ALLOCATION:
      {
        GstAxdecoder *muxer = GST_AXDECODER(aggregator);
        return gst_pad_peer_query(muxer->main_sinkpad, query);
      }
    default:
      return GST_AGGREGATOR_CLASS(gst_axdecoder_parent_class)->src_query(aggregator, query);
  }
}

static GstFlowReturn
gst_axdecoder_update_src_caps(
    GstAggregator *aggregator, GstCaps *downstream_caps, GstCaps **ret)
{
  GstAxdecoder *muxer = GST_AXDECODER(aggregator);
  GST_DEBUG_OBJECT(muxer, "update_src_caps");

  GstCaps *current_caps = gst_pad_get_current_caps(GST_AGGREGATOR_SRC_PAD(muxer));
  *ret = gst_caps_intersect(current_caps, downstream_caps);
  gst_caps_unref(current_caps);
  return GST_FLOW_OK;
}

static GstFlowReturn
gst_axdecoder_aggregate(GstAggregator *aggregator, gboolean timeout)
{
  GstAxdecoder *muxer = GST_AXDECODER(aggregator);
  GST_DEBUG_OBJECT(muxer, "aggregate");

  init_options(G_OBJECT(aggregator), muxer->data->options, muxer->data->fns,
      muxer->data->logger, muxer->data->subplugin_data, muxer->data->options_initialised);

  bool main_sinkpad_has_buffer
      = gst_aggregator_pad_has_buffer(GST_AGGREGATOR_PAD_CAST(muxer->main_sinkpad));
  bool main_sinkpad_is_eos
      = gst_aggregator_pad_is_eos(GST_AGGREGATOR_PAD_CAST(muxer->main_sinkpad));
  bool tensor_sinkpad_has_buffer
      = gst_aggregator_pad_has_buffer(GST_AGGREGATOR_PAD_CAST(muxer->tensor_sinkpad));
  bool tensor_sinkpad_is_eos
      = gst_aggregator_pad_is_eos(GST_AGGREGATOR_PAD_CAST(muxer->tensor_sinkpad));

  if (!main_sinkpad_has_buffer != main_sinkpad_is_eos) {
    throw std::logic_error("In decoder, main sinkpad -EOS- does not correspond to -no buffer-");
  }
  if (!tensor_sinkpad_has_buffer != tensor_sinkpad_is_eos) {
    throw std::logic_error("In decoder, tensor sinkpad -EOS- does not correspond to -no buffer-");
  }

  if (main_sinkpad_is_eos) {
    if (tensor_sinkpad_is_eos) {
      return GST_FLOW_EOS;
    }
    gst_aggregator_pad_drop_buffer(GST_AGGREGATOR_PAD_CAST(muxer->tensor_sinkpad));
    return GST_FLOW_OK;
  }

  if (tensor_sinkpad_is_eos) {
    return gst_aggregator_finish_buffer(GST_AGGREGATOR(muxer),
        gst_aggregator_pad_pop_buffer(GST_AGGREGATOR_PAD_CAST(muxer->main_sinkpad)));
  }

  GstBuffer *buffer2
      = gst_aggregator_pad_pop_buffer(GST_AGGREGATOR_PAD_CAST(muxer->tensor_sinkpad));

  if (GST_BUFFER_FLAG_IS_SET(buffer2, GST_BUFFER_FLAG_GAP)) {
    gst_buffer_unref(buffer2);
    return gst_aggregator_finish_buffer(GST_AGGREGATOR(muxer),
        gst_aggregator_pad_pop_buffer(GST_AGGREGATOR_PAD_CAST(muxer->main_sinkpad)));
  }

  unsigned int subframe_index = 0;
  unsigned int subframe_number = 1;
  if (gst_buffer_has_general_meta(buffer2)) {
    GstMetaGeneral *buffer2_meta = gst_buffer_get_general_meta(buffer2);
    subframe_index = buffer2_meta->subframe_index;
    subframe_number = buffer2_meta->subframe_number;
  }

  GstBuffer *buffer1;
  if (subframe_number == 1) {
    buffer1 = gst_aggregator_pad_pop_buffer(GST_AGGREGATOR_PAD_CAST(muxer->main_sinkpad));
    if (!gst_buffer_has_general_meta(buffer1)) {
      buffer1 = gst_buffer_make_writable(buffer1);
    }
  } else {
    buffer1 = gst_aggregator_pad_peek_buffer(GST_AGGREGATOR_PAD_CAST(muxer->main_sinkpad));
    g_assert(gst_buffer_has_general_meta(buffer1));
  }

  GstCaps *tensor_caps = gst_pad_get_current_caps(muxer->tensor_sinkpad);
  AxDataInterface tensors = interface_from_caps_and_meta(tensor_caps, nullptr);
  gst_caps_unref(tensor_caps);

  std::vector<GstMapInfo> tensors_memmap;
  if (muxer->data->mode == "read") {
    tensors_memmap = get_mem_map(buffer2, GST_MAP_READ, G_OBJECT(muxer));
  } else {
    tensors_memmap = get_mem_map(buffer2, GST_MAP_READWRITE, G_OBJECT(muxer));
  }
  assign_data_ptrs_to_interface(tensors_memmap, tensors);

  GstCaps *srcpad_caps = gst_pad_get_current_caps(GST_AGGREGATOR_SRC_PAD(muxer));
  AxDataInterface srcpad_info = interface_from_caps_and_meta(srcpad_caps, nullptr);
  gst_caps_unref(srcpad_caps);

  if (muxer->data->fns.decode_to_meta) {
    muxer->data->fns.decode_to_meta(std::get<AxTensorsInterface>(tensors),
        muxer->data->subplugin_data.get(), subframe_index, subframe_number,
        *gst_buffer_get_general_meta(buffer1)->meta_map_ptr, srcpad_info,
        muxer->data->logger);
  }

  unmap_mem(tensors_memmap);
  gst_buffer_unref(buffer2);

  bool last_subframe = (subframe_index + 1 == subframe_number); // TODO this might change
  if (subframe_number == 1) {
    return gst_aggregator_finish_buffer(GST_AGGREGATOR(muxer), buffer1);
  }
  if (last_subframe) {
    gst_aggregator_pad_drop_buffer(GST_AGGREGATOR_PAD_CAST(muxer->main_sinkpad));
    return gst_aggregator_finish_buffer(GST_AGGREGATOR(muxer), buffer1);
  }
  gst_buffer_unref(buffer1);
  return GST_FLOW_OK;
}

static void
gst_axdecoder_class_init(GstAxdecoderClass *klass)
{
  gst_element_class_set_static_metadata(GST_ELEMENT_CLASS(klass), "axdecode",
      "Effect", "description", "axelera.ai");

  G_OBJECT_CLASS(klass)->set_property = GST_DEBUG_FUNCPTR(gst_axdecoder_set_property);
  G_OBJECT_CLASS(klass)->get_property = GST_DEBUG_FUNCPTR(gst_axdecoder_get_property);
  G_OBJECT_CLASS(klass)->finalize = GST_DEBUG_FUNCPTR(gst_axdecoder_finalize);

  GST_AGGREGATOR_CLASS(klass)->aggregate = GST_DEBUG_FUNCPTR(gst_axdecoder_aggregate);
  GST_AGGREGATOR_CLASS(klass)->sink_event = GST_DEBUG_FUNCPTR(gst_axdecoder_sink_event);
  GST_AGGREGATOR_CLASS(klass)->src_query = GST_DEBUG_FUNCPTR(gst_axdecoder_src_query);
  GST_AGGREGATOR_CLASS(klass)->update_src_caps
      = GST_DEBUG_FUNCPTR(gst_axdecoder_update_src_caps);

  gst_element_class_add_static_pad_template_with_gtype(
      GST_ELEMENT_CLASS(klass), &sink_template, GST_TYPE_AGGREGATOR_PAD);
  gst_element_class_add_static_pad_template_with_gtype(
      GST_ELEMENT_CLASS(klass), &sink_tensor_template, GST_TYPE_AGGREGATOR_PAD);
  gst_element_class_add_static_pad_template_with_gtype(
      GST_ELEMENT_CLASS(klass), &src_template, GST_TYPE_AGGREGATOR_PAD);

  g_object_class_install_property(G_OBJECT_CLASS(klass), PROP_SHARED_LIB_PATH,
      g_param_spec_string("lib", "lib path", "String containing lib path", "",
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property(G_OBJECT_CLASS(klass), PROP_OPTIONS,
      g_param_spec_string("options", "options string", "Subplugin dependent options",
          "", (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property(G_OBJECT_CLASS(klass), PROP_MODE,
      g_param_spec_string("mode", "mode string",
          "Specify if buffer is read only by keyword read, if leaving out the property, the buffer is writable as well",
          "", (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
}

static void
gst_axdecoder_init(GstAxdecoder *muxer)
{
  muxer->data = new GstAxdecoderData;
  gst_debug_category_get_threshold(gst_axdecoder_debug);
  muxer->data->logger = Ax::Logger(
      Ax::extract_severity_from_category(gst_axdecoder_debug), muxer, gst_axdecoder_debug);
  Ax::init_logger(muxer->data->logger);

  GstPadTemplate *sink_templ = gst_static_pad_template_get(&sink_template);
  GstPadTemplate *sink_tensor_templ = gst_static_pad_template_get(&sink_tensor_template);

  muxer->main_sinkpad = GST_PAD_CAST(g_object_new(GST_TYPE_AGGREGATOR_PAD, "name",
      "sink_0", "direction", sink_templ->direction, "template", sink_templ, NULL));
  muxer->tensor_sinkpad
      = GST_PAD_CAST(g_object_new(GST_TYPE_AGGREGATOR_PAD, "name", "sink_1", "direction",
          sink_tensor_templ->direction, "template", sink_tensor_templ, NULL));

  gst_element_add_pad(GST_ELEMENT(muxer), muxer->main_sinkpad);
  gst_element_add_pad(GST_ELEMENT(muxer), muxer->tensor_sinkpad);
}
