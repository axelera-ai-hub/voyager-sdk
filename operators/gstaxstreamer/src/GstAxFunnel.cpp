#include "GstAxFunnel.hpp"
#include <gmodule.h>
#include <gst/gst.h>
#include <memory>
#include <set>
#include <sstream>
#include <unordered_map>
#include "AxDataInterface.h"
#include "AxMeta.hpp"
#include "AxStreamerUtils.hpp"
#include "GstAxDataUtils.hpp"
#include "GstAxMeta.hpp"
#include "GstAxStreamerUtils.hpp"

GST_DEBUG_CATEGORY_STATIC(gst_axfunnel_debug);
#define GST_CAT_DEFAULT gst_axfunnel_debug

enum {
  PROP_0,
  PROP_STREAM_SELECT,
};

static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE(
    "sink_%u", GST_PAD_SINK, GST_PAD_REQUEST, GST_STATIC_CAPS_ANY);

static GstStaticPadTemplate src_template
    = GST_STATIC_PAD_TEMPLATE("src", GST_PAD_SRC, GST_PAD_ALWAYS, GST_STATIC_CAPS_ANY);

G_DEFINE_TYPE_WITH_CODE(GstAxfunnel, gst_axfunnel, GST_TYPE_AGGREGATOR,
    GST_DEBUG_CATEGORY_INIT(gst_axfunnel_debug, "axfunnel", 0, "axfunnel element"));

static void
gst_axfunnel_set_property(
    GObject *object, guint property_id, const GValue *value, GParamSpec *pspec)
{
  GstAxfunnel *muxer = GST_AXFUNNEL(object);
  GST_DEBUG_OBJECT(muxer, "set_property");
  std::string str;
  switch (property_id) {
    case PROP_STREAM_SELECT:
      str = g_value_get_string(value);
      muxer->stream_set = Ax::create_stream_set(str);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID(object, property_id, pspec);
      break;
  }
}

static void
gst_axfunnel_get_property(GObject *object, guint property_id, GValue *value, GParamSpec *pspec)
{
  GstAxfunnel *muxer = GST_AXFUNNEL(object);
  GST_DEBUG_OBJECT(muxer, "get_property");
  std::string str;
  switch (property_id) {
    case PROP_STREAM_SELECT:
      str = Ax::to_string(muxer->stream_set);
      g_value_set_string(value, str.c_str());
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID(object, property_id, pspec);
      break;
  }
}

GstCaps *
retain_specific_fields(GstCaps *caps, const std::set<std::string> &fields_to_keep)
{
  for (guint i = 0; i < gst_caps_get_size(caps); i++) {
    GstStructure *structure = gst_caps_get_structure(caps, i);
    GstStructure *new_structure = gst_structure_copy(structure);

    for (guint j = 0; j < gst_structure_n_fields(structure); j++) {
      const gchar *field_name = gst_structure_nth_field_name(structure, j);
      if (!fields_to_keep.contains(field_name)) {
        gst_structure_remove_field(new_structure, field_name);
      }
    }
    GstCaps *new_caps = gst_caps_new_empty();
    gst_caps_append_structure(new_caps, new_structure);

    gst_caps_take(&caps, new_caps);
  }
  return caps;
}

static gboolean
gst_axfunnel_sink_event(GstAggregator *aggregator, GstAggregatorPad *pad, GstEvent *event)
{
  GstAxfunnel *muxer = GST_AXFUNNEL(aggregator);
  GST_DEBUG_OBJECT(muxer, "sink_event");

  if (GST_EVENT_CAPS == GST_EVENT_TYPE(event)) {
    GstCaps *caps;
    gst_event_parse_caps(event, &caps);
    gst_caps_ref(caps);
    gst_event_unref(event);
    GstCaps *current_caps = gst_pad_get_current_caps(GST_AGGREGATOR_SRC_PAD(muxer));
    // Define the fields to retain (optionally add framerate)
    std::set<std::string> fields_to_keep = { "width", "height", "format" };

    // Retain only specific fields
    caps = retain_specific_fields(caps, fields_to_keep);
    if (current_caps == NULL) {
      gst_aggregator_set_src_caps(aggregator, caps);
      return TRUE;
    }
    auto *intersection = gst_caps_intersect(current_caps, caps);

    if (gst_caps_is_empty(intersection)) {
      std::stringstream ss;
      auto *curcaps_str = gst_caps_to_string(current_caps);
      auto *caps_str = gst_caps_to_string(caps);

      ss << "In axfunnel, caps are not the same: " << curcaps_str << std::endl
         << " AND: " << caps_str;
      g_free(curcaps_str);
      g_free(caps_str);
      throw std::runtime_error(ss.str());
    }
    gst_caps_unref(caps);
    gst_caps_unref(intersection);
    gst_caps_unref(current_caps);
    return TRUE;
  }
  return GST_AGGREGATOR_CLASS(gst_axfunnel_parent_class)->sink_event(aggregator, pad, event);
}

static gboolean
gst_axfunnel_src_query(GstAggregator *aggregator, GstQuery *query)
{
  switch (GST_QUERY_TYPE(query)) {
    case GST_QUERY_POSITION:
    case GST_QUERY_DURATION:
    case GST_QUERY_URI:
    case GST_QUERY_CAPS:
    case GST_QUERY_ACCEPT_CAPS:
    case GST_QUERY_ALLOCATION:
      {
        GstAxfunnel *muxer = GST_AXFUNNEL(aggregator);
        GList *item = GST_ELEMENT(aggregator)->sinkpads;
        if (item == NULL) {
          throw std::runtime_error("Axfunnel has no sinkpads");
        }
        return gst_pad_peer_query(GST_PAD(item->data), query);
      }
    default:
      return GST_AGGREGATOR_CLASS(gst_axfunnel_parent_class)->src_query(aggregator, query);
  }
}

static GstFlowReturn
gst_axfunnel_update_src_caps(
    GstAggregator *aggregator, GstCaps *downstream_caps, GstCaps **ret)
{
  GstAxfunnel *muxer = GST_AXFUNNEL(aggregator);
  GST_DEBUG_OBJECT(muxer, "update_src_caps");

  GstCaps *current_caps = gst_pad_get_current_caps(GST_AGGREGATOR_SRC_PAD(muxer));
  *ret = gst_caps_intersect(current_caps, downstream_caps);
  gst_caps_unref(current_caps);
  return GST_FLOW_OK;
}
static std::string
get_pad_name(GstPad *pad)
{
  auto *psz = gst_pad_get_name(pad);
  const auto n = std::string{ psz ? psz : "" };
  g_free(psz);
  return n;
}

static GstFlowReturn
gst_axfunnel_aggregate(GstAggregator *aggregator, gboolean timeout)
{
  GstAxfunnel *muxer = GST_AXFUNNEL(aggregator);
  GST_DEBUG_OBJECT(muxer, "aggregate");

  bool at_least_one_playing = false;
  for (GList *item = GST_ELEMENT(aggregator)->sinkpads; item != NULL; item = item->next) {
    GstAggregatorPad *sinkpad = GST_AGGREGATOR_PAD(item->data);
    bool sinkpad_has_buffer = gst_aggregator_pad_has_buffer(sinkpad);
    bool sinkpad_is_eos = gst_aggregator_pad_is_eos(sinkpad);
    if (sinkpad_has_buffer && sinkpad_is_eos) {
      throw std::logic_error("In funnel, sinkpad -EOS- does not correspond to -no buffer-");
    }
    if (sinkpad_has_buffer) {
      at_least_one_playing = true;

      if (!muxer->stream_set.empty()) {
        const auto pad_name = get_pad_name(GST_PAD(sinkpad));
        auto tokens = Ax::Internal::split(pad_name, '_');
        if (tokens.empty()) {
          throw std::runtime_error("Axfunnel invalid sink pad name");
        }
        auto token = std::string{ tokens[1] };

        if (muxer->stream_set.contains(std::stoi(token))) {
          if (GST_FLOW_ERROR
              == gst_aggregator_finish_buffer(GST_AGGREGATOR(muxer),
                  gst_aggregator_pad_pop_buffer(sinkpad))) {
            return GST_FLOW_ERROR;
          }
        } else {
          gst_aggregator_pad_drop_buffer(sinkpad);
        }

      } else {
        if (GST_FLOW_ERROR
            == gst_aggregator_finish_buffer(
                GST_AGGREGATOR(muxer), gst_aggregator_pad_pop_buffer(sinkpad))) {
          return GST_FLOW_ERROR;
        }
      }
    }
  }

  if (!at_least_one_playing) {
    return GST_FLOW_EOS;
  }
  return GST_FLOW_OK;
}

static void
gst_axfunnel_class_init(GstAxfunnelClass *klass)
{
  gst_element_class_set_static_metadata(GST_ELEMENT_CLASS(klass), "axfunnel",
      "Effect", "description", "axelera.ai");

  G_OBJECT_CLASS(klass)->set_property = GST_DEBUG_FUNCPTR(gst_axfunnel_set_property);
  G_OBJECT_CLASS(klass)->get_property = GST_DEBUG_FUNCPTR(gst_axfunnel_get_property);


  GST_AGGREGATOR_CLASS(klass)->aggregate = GST_DEBUG_FUNCPTR(gst_axfunnel_aggregate);
  GST_AGGREGATOR_CLASS(klass)->sink_event = GST_DEBUG_FUNCPTR(gst_axfunnel_sink_event);
  GST_AGGREGATOR_CLASS(klass)->src_query = GST_DEBUG_FUNCPTR(gst_axfunnel_src_query);
  GST_AGGREGATOR_CLASS(klass)->update_src_caps
      = GST_DEBUG_FUNCPTR(gst_axfunnel_update_src_caps);

  gst_element_class_add_static_pad_template_with_gtype(
      GST_ELEMENT_CLASS(klass), &sink_template, GST_TYPE_AGGREGATOR_PAD);
  gst_element_class_add_static_pad_template_with_gtype(
      GST_ELEMENT_CLASS(klass), &src_template, GST_TYPE_AGGREGATOR_PAD);

  g_object_class_install_property(G_OBJECT_CLASS(klass), PROP_STREAM_SELECT,
      g_param_spec_string("stream_select", "Stream Select",
          "Select streams to pass", "", G_PARAM_READWRITE));
}

static void
gst_axfunnel_init(GstAxfunnel *muxer)
{
}
