// Copyright Axelera AI, 2024
#include "GstAxInPlace.hpp"
#include <gmodule.h>
#include <gst/gst.h>
#include <gst/video/video.h>

#include <memory>
#include <unordered_map>
#include "AxDataInterface.h"
#include "AxMeta.hpp"
#include "AxStreamerUtils.hpp"
#include "GstAxDataUtils.hpp"
#include "GstAxMeta.hpp"
#include "GstAxStreamerUtils.hpp"

GST_DEBUG_CATEGORY_STATIC(gst_axinplace_debug_category);
#define GST_CAT_DEFAULT gst_axinplace_debug_category

struct _GstAxinplaceData {
  std::string shared_lib_path;
  std::string mode;
  std::string options;
  Ax::Logger logger{ Ax::Severity::trace, nullptr, gst_axinplace_debug_category };

  std::unique_ptr<Ax::SharedLib> shared;
  std::unique_ptr<Ax::InPlace> plugin;

  std::string dmabuf_device{ "/dev/dma_heap/system" };
  GstAllocator *allocator{};
  GstBufferPool *pool{};
};

G_DEFINE_TYPE_WITH_CODE(GstAxinplace, gst_axinplace, GST_TYPE_BASE_TRANSFORM,
    GST_DEBUG_CATEGORY_INIT(gst_axinplace_debug_category, "axinplace", 0,
        "debug category for axinplace element"));

enum {
  PROP_0,
  PROP_SHARED_LIB_PATH,
  PROP_MODE,
  PROP_USE_ALIGNED,
  PROP_OPTIONS,
};

static void
gst_axinplace_init(GstAxinplace *axinplace)
{
  axinplace->data = new GstAxinplaceData;
  gst_debug_category_get_threshold(gst_axinplace_debug_category);
  axinplace->data->logger
      = Ax::Logger(Ax::extract_severity_from_category(gst_axinplace_debug_category),
          axinplace, gst_axinplace_debug_category);
  Ax::init_logger(axinplace->data->logger);
}

static void
gst_axinplace_finalize(GObject *object)
{
  GstAxinplace *axinplace = GST_AXINPLACE(object);
  GST_DEBUG_OBJECT(axinplace, "finalize");

  if (axinplace->data) {
    if (axinplace->data->allocator) {
      gst_object_unref(axinplace->data->allocator);
    }

    if (axinplace->data->pool) {
      gst_object_unref(axinplace->data->pool);
    }

    delete axinplace->data;
    axinplace->data = nullptr;
  }

  G_OBJECT_CLASS(gst_axinplace_parent_class)->finalize(object);
}

static void
gst_axinplace_set_property(
    GObject *object, guint property_id, const GValue *value, GParamSpec *pspec)
{
  GstAxinplace *axinplace = GST_AXINPLACE(object);
  GST_DEBUG_OBJECT(axinplace, "set_property");
  auto &data = *axinplace->data;

  switch (property_id) {
    case PROP_SHARED_LIB_PATH:
      data.shared_lib_path = Ax::libname(g_value_get_string(value));
      // load it immediately so we get early errors, we can't initialise the main plugin
      // yet because we don't know in what order the options will be set
      data.shared = std::make_unique<Ax::SharedLib>(data.logger, data.shared_lib_path);
      break;

    case PROP_MODE:
      data.mode = g_value_get_string(value);
      break;

    case PROP_OPTIONS:
      data.options = g_value_get_string(value);
      if (data.plugin) {
        auto opts = Ax::parse_and_validate_plugin_options(
            data.logger, data.options, data.plugin->allowed_properties());
        data.plugin->set_dynamic_properties(opts);
      }
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID(object, property_id, pspec);
      break;
  }
}

static void
gst_axinplace_get_property(GObject *object, guint property_id, GValue *value, GParamSpec *pspec)
{
  GstAxinplace *axinplace = GST_AXINPLACE(object);
  GST_DEBUG_OBJECT(axinplace, "get_property");

  switch (property_id) {
    case PROP_SHARED_LIB_PATH:
      g_value_set_string(value, axinplace->data->shared_lib_path.c_str());
      break;
    case PROP_MODE:
      g_value_set_string(value, axinplace->data->mode.c_str());
      break;

    case PROP_OPTIONS:
      g_value_set_string(value, axinplace->data->options.c_str());
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID(object, property_id, pspec);
      break;
  }
}

static gboolean
gst_axinplace_propose_allocation(
    GstBaseTransform *trans, GstQuery *decide_query, GstQuery *query)
{
  //  We pass the query to the downstream peer, and copy all the results back into our query
  GstPad *srcpad = GST_BASE_TRANSFORM_SRC_PAD(trans);
  GstPad *peer = gst_pad_get_peer(srcpad);
  if (!peer) {
    GST_DEBUG_OBJECT(trans, "No downstream peer for allocation query");
    return FALSE;
  }
  gst_object_unref(peer);

  GstQuery *down_q = gst_query_copy(query);
  if (!down_q)
    return FALSE;

  gboolean ret = gst_pad_peer_query(srcpad, down_q);
  if (!ret) {
    GST_DEBUG_OBJECT(trans, "Downstream allocation query failed");
    gst_query_unref(down_q);
    return FALSE;
  }

  auto n_params = gst_query_get_n_allocation_params(down_q);
  for (guint i = 0; i < n_params; i++) {
    GstAllocationParams params;
    gst_allocation_params_init(&params);

    GstAllocator *alloc = nullptr;
    gst_query_parse_nth_allocation_param(down_q, i, &alloc, &params);
    gst_query_add_allocation_param(query, alloc, &params);
    if (alloc)
      gst_object_unref(alloc);
  }

  auto n_pools = gst_query_get_n_allocation_pools(down_q);
  for (guint i = 0; i < n_pools; i++) {
    guint size = 0;
    guint min = 0;
    guint max = 0;
    GstBufferPool *pool = NULL;
    gst_query_parse_nth_allocation_pool(down_q, i, &pool, &size, &min, &max);
    gst_query_add_allocation_pool(query, pool, size, min, max);
    if (pool)
      gst_object_unref(pool);
  }
  guint n_metas = gst_query_get_n_allocation_metas(down_q);
  for (guint i = 0; i < n_metas; i++) {
    const GstStructure *params;
    GType api = gst_query_parse_nth_allocation_meta(down_q, i, &params);
    gst_query_add_allocation_meta(query, api, params);
  }
  gst_query_unref(down_q);
  return TRUE;
}


static GstFlowReturn
gst_axinplace_transform_ip(GstBaseTransform *trans, GstBuffer *buffer)
{
  bool success = true;
  GstAxinplace *axinplace = GST_AXINPLACE(trans);
  GST_DEBUG_OBJECT(axinplace, "transform_ip");
  auto &data = *axinplace->data;

  if (!data.plugin && data.shared) {
    data.plugin = std::make_unique<Ax::LoadedInPlace>(
        data.logger, std::move(*data.shared), data.options, nullptr, data.mode);
  }

  if (!data.plugin || !data.plugin->has_inplace()) {
    return GST_FLOW_OK;
  }

  AxDataInterface data_template;
  std::vector<GstMapInfo> map;
  if (data.mode != "meta") {
    GstCaps *in_caps = gst_pad_get_current_caps(trans->sinkpad);
    data_template = interface_from_caps_and_meta(in_caps, buffer);

    if (data.mode.empty()) {
      GST_INFO_OBJECT(axinplace, "No mode specified, not mapping memory.");
    } else {
      if (data.mode == "read") {
        map = get_mem_map(buffer, GST_MAP_READ, G_OBJECT(trans));
      } else {
        map = get_mem_map(buffer, GST_MAP_READWRITE, G_OBJECT(trans));
      }
      assign_data_ptrs_to_interface(map, data_template);
    }
    if (in_caps) {
      gst_caps_unref(in_caps);
    }
  }

  GstMetaGeneral *meta = gst_buffer_get_general_meta(buffer);
  data.plugin->inplace(data_template, meta->subframe_index,
      meta->subframe_number, *meta->meta_map_ptr);

  unmap_mem(map);
  return success ? GST_FLOW_OK : GST_FLOW_ERROR;
}

static void
gst_axinplace_class_init(GstAxinplaceClass *klass)
{
  gst_element_class_set_static_metadata(GST_ELEMENT_CLASS(klass), "axinplace",
      "Effect", "description", "axelera.ai");

  G_OBJECT_CLASS(klass)->set_property = GST_DEBUG_FUNCPTR(gst_axinplace_set_property);
  G_OBJECT_CLASS(klass)->get_property = GST_DEBUG_FUNCPTR(gst_axinplace_get_property);
  G_OBJECT_CLASS(klass)->finalize = GST_DEBUG_FUNCPTR(gst_axinplace_finalize);
  GST_BASE_TRANSFORM_CLASS(klass)->transform_ip
      = GST_DEBUG_FUNCPTR(gst_axinplace_transform_ip);
  GST_BASE_TRANSFORM_CLASS(klass)->propose_allocation
      = GST_DEBUG_FUNCPTR(gst_axinplace_propose_allocation);

  GST_BASE_TRANSFORM_CLASS(klass)->propose_allocation
      = GST_DEBUG_FUNCPTR(gst_axinplace_propose_allocation);

  g_object_class_install_property(G_OBJECT_CLASS(klass), PROP_SHARED_LIB_PATH,
      g_param_spec_string("lib", "lib path", "String containing lib path", "",
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property(G_OBJECT_CLASS(klass), PROP_MODE,
      g_param_spec_string("mode", "mode string",
          "Specify if buffer is read only by keyword read, is read and write by write, leave out property if data ppointer is not accessed, or use keyword meta when only the meta map is accessed.",
          "", (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property(G_OBJECT_CLASS(klass), PROP_OPTIONS,
      g_param_spec_string("options", "options string", "Subplugin dependent options",
          "", (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  gst_element_class_add_pad_template(GST_ELEMENT_CLASS(klass),
      gst_pad_template_new("src", GST_PAD_SRC, GST_PAD_ALWAYS, GST_CAPS_ANY));
  gst_element_class_add_pad_template(GST_ELEMENT_CLASS(klass),
      gst_pad_template_new("sink", GST_PAD_SINK, GST_PAD_ALWAYS, GST_CAPS_ANY));
}
