#include "GstAxInPlace.hpp"
#include <gmodule.h>
#include <gst/allocators/gstfdmemory.h>
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
  bool options_initialised = false;
  Ax::Logger logger{ Ax::Severity::trace, nullptr, gst_axinplace_debug_category };

  std::unique_ptr<Ax::SharedLib> shared;
  Ax::V1Plugin::InPlace fns;

  std::shared_ptr<void> subplugin_data;
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

  switch (property_id) {
    case PROP_SHARED_LIB_PATH:
      axinplace->data->shared_lib_path = Ax::libname(g_value_get_string(value));
      axinplace->data->shared = std::make_unique<Ax::SharedLib>(
          axinplace->data->logger, axinplace->data->shared_lib_path);
      Ax::load_v1_plugin(*axinplace->data->shared, axinplace->data->fns);
      break;
    case PROP_MODE:
      axinplace->data->mode = g_value_get_string(value);
      break;

    case PROP_OPTIONS:
      axinplace->data->options = g_value_get_string(value);
      update_options(object, axinplace->data->options, axinplace->data->fns,
          axinplace->data->logger, axinplace->data->subplugin_data,
          axinplace->data->options_initialised);
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

static size_t
ax_size_from_caps(GstCaps *caps)
{
  return size_from_interface(interface_from_caps_and_meta(caps, nullptr));
}

static gboolean
gst_axinplace_propose_allocation(GstBaseTransform *sink, GstQuery *decide_query, GstQuery *query)
{
  //  Tell the upstream element that we support GstVideoMeta. This allows it
  //  to give us buffers with "unusual" strides and offsets.
  gst_query_add_allocation_meta(query, GST_VIDEO_META_API_TYPE, NULL);

  auto *self = GST_AXINPLACE(sink);
  GstCaps *caps;
  gboolean need_pool;
  GstStructure *config;
  guint size;

  gst_query_parse_allocation(query, &caps, &need_pool);

  if (!caps) {
    GST_ERROR_OBJECT(self, "Allocation query has no caps");
    return TRUE;
  }

  if (self->data->allocator) {
    gst_object_unref(self->data->allocator);
  }
  self->data->allocator = gst_aligned_allocator_get();
  if (!self->data->allocator) {
    GST_ERROR_OBJECT(self, "Unable to get aligned allocator");
    return TRUE;
  }

  if (need_pool) {
    if (self->data->pool) {
      gst_object_unref(self->data->pool);
    }
    self->data->pool = gst_buffer_pool_new();
    config = gst_buffer_pool_get_config(self->data->pool);
    size = ax_size_from_caps(caps);

    /* we need at least 2 buffer because we hold on to the last one */
    gst_buffer_pool_config_set_params(config, caps, size, 4, 0);
    gst_buffer_pool_config_set_allocator(config, self->data->allocator, NULL);
    if (!gst_buffer_pool_set_config(self->data->pool, config)) {
      gst_object_unref(self->data->allocator);
      gst_object_unref(self->data->pool);
      GST_ERROR_OBJECT(self, "Failed to set pool configuration");
      return TRUE;
    }
    /* we need at least 2 buffer because we hold on to the last one */
    gst_query_add_allocation_pool(query, self->data->pool, size, 4, 0);
  }

  gst_query_add_allocation_param(query, self->data->allocator, NULL);

  return TRUE;
}

static GstFlowReturn
gst_axinplace_transform_ip(GstBaseTransform *trans, GstBuffer *buffer)
{
  bool success = true;
  GstAxinplace *axinplace = GST_AXINPLACE(trans);
  GST_DEBUG_OBJECT(axinplace, "transform_ip");

  init_options(G_OBJECT(trans), axinplace->data->options, axinplace->data->fns,
      axinplace->data->logger, axinplace->data->subplugin_data,
      axinplace->data->options_initialised);

  if (!axinplace->data->fns.inplace) {
    return GST_FLOW_OK;
  }

  AxDataInterface data_template;
  std::vector<GstMapInfo> map;
  if (axinplace->data->mode != "meta") {
    GstCaps *in_caps = gst_pad_get_current_caps(trans->sinkpad);
    data_template = interface_from_caps_and_meta(in_caps, buffer);

    if (axinplace->data->mode.empty()) {
      GST_INFO_OBJECT(axinplace, "No mode specified, not mapping memory.");
    } else {
      if (axinplace->data->mode == "read") {
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
  axinplace->data->fns.inplace(data_template,
      axinplace->data->subplugin_data.get(), meta->subframe_index,
      meta->subframe_number, *meta->meta_map_ptr, axinplace->data->logger);

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
