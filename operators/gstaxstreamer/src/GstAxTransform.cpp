// Copyright Axelera AI, 2026
#include "GstAxTransform.hpp"
#include <cstring>
#include <gmodule.h>
#include <gst/allocators/gstdmabuf.h>
#include <gst/gst.h>
#include <gst/video/video.h>
#include <memory>
#include <queue>
#include <sstream>
#include <unordered_map>
#include "AxDataInterface.h"
#include "AxMeta.hpp"
#include "AxPlugin.hpp"
#include "AxStreamerUtils.hpp"
#include "GstAxBufferPool.hpp"
#include "GstAxDataUtils.hpp"
#include "GstAxMeta.hpp"
#include "GstAxStreamerUtils.hpp"


GST_DEBUG_CATEGORY_STATIC(gst_axtransform_debug_category);
#define GST_CAT_DEFAULT gst_axtransform_debug_category

extern "C" {
gboolean gst_is_tensor_dmabuf_memory(GstMemory *mem);
}

struct EventDetails {
  GstPad *pad;
  GstEvent *event;
};

struct _GstAxtransformData {
  std::string shared_lib_path;
  std::string options;
  AxDataInterface output_template;
  std::unique_ptr<Ax::SharedLib> shared;
  std::unique_ptr<Ax::Transform> plugin;
  Ax::Logger logger{ Ax::Severity::trace, nullptr, gst_axtransform_debug_category };
  unsigned int batch = 1;
  std::string which_cl;
  unsigned int current_batch = 0;
  std::vector<int> tensor_size{};
  GstBuffer *outbuf = nullptr;
  Ax::GstHandle<GstAllocator> allocator{};
  Ax::GstHandle<GstBufferPool> pool{};
  bool downstream_supports_crop = false;
  std::queue<EventDetails> event_queue;
  bool block_on_pool_empty = true;
};

G_DEFINE_TYPE_WITH_CODE(GstAxtransform, gst_axtransform, GST_TYPE_ELEMENT,
    GST_DEBUG_CATEGORY_INIT(gst_axtransform_debug_category, "axtransform", 0,
        "debug category for axtransform element"));

enum {
  PROP_0,
  PROP_SHARED_LIB_PATH,
  PROP_OPTIONS,
  PROP_BATCH,
  PROP_WHICH_CL,
};

static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE(
    "sink", GST_PAD_SINK, GST_PAD_ALWAYS, GST_STATIC_CAPS_ANY);

static GstStaticPadTemplate src_template
    = GST_STATIC_PAD_TEMPLATE("src", GST_PAD_SRC, GST_PAD_ALWAYS,
        GST_STATIC_CAPS(GST_TENSORS_CAP_DEFAULT ";" AX_GST_VIDEO_FORMATS_CAPS));

static bool
plugin_supports_cropmeta(_GstAxtransformData *data)
{
  return data->plugin->query_supports(Ax::PluginFeature::crop_meta);
}

static bool
plugin_supports_gst_video_meta(GstAxtransform *sink)
{
  return sink->data->plugin->query_supports(Ax::PluginFeature::video_meta);
}

const auto *dmabuf_device = "/dev/dma_heap/system";

static bool
is_suitable(GstAllocator *allocator, GstAxtransform *self)
{
  auto &logger = self->data->logger;
  return !allocator || allocator == gst_aligned_allocator_get()
         || allocator == gst_tensor_dmabuf_allocator_get(dmabuf_device)
         || allocator == gst_opencl_allocator_get(self->data->which_cl.c_str(), &logger);
}

static GstContext *
get_va_context(GstAxtransform *axtransform, Ax::Logger &logger)
{
  GstContext *context = NULL;
  GstQuery *query = gst_query_new_context("gst.vaapi.Display"); // Request VAAPI context

  if (gst_element_query((GstElement *) axtransform, query)) { // Send query upstream
    gst_query_parse_context(query, &context);
  }
  if (context) {
    gst_context_ref(context);
  }
  gst_query_unref(query);
  return context;
}

struct GstVaapiDisplayPrivate {
  void *parent;
  GRecMutex mutex;
  char *display_name;
  void *display;
  //  Lots of other stuff we don't care about
};

struct GstVaapiDisplay {
  GstObject parent_instance;

  GstVaapiDisplayPrivate *priv;
};


static void *
get_va_display(GstAxtransform *axtransform, Ax::Logger &logger)
{
  GstContext *context = get_va_context(axtransform, logger);

  void *display = nullptr;
  if (context) {
    const GstStructure *structure = gst_context_get_structure(context);
    if (gst_structure_has_field(structure, "gst.vaapi.Display")) {
      GstVaapiDisplay *va_display = nullptr;
      if (gst_structure_get(structure, "gst.vaapi.Display.GObject",
              GST_TYPE_OBJECT, &va_display, NULL)) {
        display = va_display->priv->display;
      }
    }
    gst_context_unref(context);
  } else {
    logger(AX_WARN) << "Failed to get context from query." << std::endl;
  }
  return display;
}

bool
supports_opencl_buffers(GstAxtransform *self)
{
  return self->data->plugin->query_supports(Ax::PluginFeature::opencl_buffers);
}


static void
gst_axtransform_do_bufferpool(GstAxtransform *axtransform, GstCaps *caps)
{
  if (axtransform->pool) {
    GstStructure *config = gst_buffer_pool_get_config(axtransform->pool);
    GstCaps *pool_caps;
    guint size;
    guint min_buffers;
    guint max_buffers;
    if (!gst_buffer_pool_config_get_params(
            config, &pool_caps, &size, &min_buffers, &max_buffers)) {
      throw std::runtime_error("Cannot get pool config in axtransform");
    }
    bool caps_did_not_change = gst_caps_is_equal_fixed(caps, pool_caps);
    gst_structure_free(config);
    if (caps_did_not_change) {
      return;
    }

    gst_buffer_pool_set_active(axtransform->pool, FALSE);
    gst_object_unref(axtransform->pool);
    axtransform->pool = NULL;
  }
  GstQuery *query = gst_query_new_allocation(caps, TRUE);
  if (gst_pad_peer_query(axtransform->srcpad, query)) {
    //  Check if downstrean accepts GstVideCropMeta and if so enable
    axtransform->data->downstream_supports_crop
        = gst_query_find_allocation_meta(query, GST_VIDEO_CROP_META_API_TYPE, NULL)
          && gst_query_find_allocation_meta(query, GST_VIDEO_META_API_TYPE, NULL);
    if (gst_query_get_n_allocation_params(query) > 0) {
      GstAllocator *allocator = nullptr;
      gst_query_parse_nth_allocation_param(query, 0, &allocator, NULL);
      if (is_suitable(allocator, axtransform)) {
        if (axtransform->allocator) {
          gst_object_unref(axtransform->allocator);
        }
        axtransform->allocator = allocator;
        if (gst_query_get_n_allocation_pools(query) > 0) {
          gst_query_parse_nth_allocation_pool(query, 0, &axtransform->pool, NULL, NULL, NULL);
          gst_buffer_pool_set_active(axtransform->pool, TRUE);
        }
      }
    }
  }
  gst_query_unref(query);

  if (supports_opencl_buffers(axtransform)) {
    //  We shoud try to use opencl buffers
    //  Get the opencl allocator
    GstAllocator *opencl_allocator = gst_opencl_allocator_get(
        axtransform->data->which_cl.c_str(), &axtransform->data->logger);
    if (opencl_allocator && opencl_allocator == axtransform->allocator) {
      //  Already have OpenCL allocator
      return;
    }
    if (opencl_allocator) {
      if (axtransform->allocator) {
        gst_object_unref(axtransform->allocator);
      }
      axtransform->allocator = opencl_allocator;
      if (axtransform->pool) {
        gst_buffer_pool_set_active(axtransform->pool, FALSE);
        gst_object_unref(axtransform->pool);
        axtransform->pool = NULL;
      }
    } else {
      GST_DEBUG_OBJECT(axtransform,
          "OpenCL allocator is not suitable for this plugin, using default allocator");
    }
  }

  if (axtransform->pool) {
    return;
  }
  axtransform->pool = gst_ax_buffer_pool_new();
  if (axtransform->pool) {
    GstStructure *config = gst_buffer_pool_get_config(axtransform->pool);
    gst_buffer_pool_config_set_params(config, caps, axtransform->outsize, 0, 0);
    if (axtransform->allocator) {
      gst_buffer_pool_config_set_allocator(config, axtransform->allocator, NULL);
    }
    if (!gst_buffer_pool_set_config(axtransform->pool, config)) {
      throw std::runtime_error("Pool config cannot be set in axtransform");
    }


    if (axtransform->pool && !gst_buffer_pool_set_active(axtransform->pool, TRUE)) {
      throw std::runtime_error("Pool cannot be activated in axtransform");
    }
  }
}

static bool
plugin_has_set_output_interface(GstAxtransform *axtransform)
{
  return axtransform->data->plugin->has_set_output_interface()
         || axtransform->data->plugin->has_set_output_interface_from_meta();
}

static Ax::GstHandle<GstCaps>
gst_axtransform_outcaps(GstAxtransform *axtransform, GstCaps *from_event, GstBuffer *buffer)
{
  /*
  gst_pad_get_allowed_caps calls gst_pad_peer_query
  and intersects with the template caps
  other/tensors
              format: flexible
           framerate: [ 0/1, 2147483647/1 ]
   other/tensor
           framerate: [ 0/1, 2147483647/1 ]
           dimension: 160:160:3:1
                type: float32
   other/tensors
              format: static
         num_tensors: 1
           framerate: [ 0/1, 2147483647/1 ]
          dimensions: 160:160:3:1
               types: float32
  */
  auto from_srctemplate_and_peer
      = Ax::as_handle(gst_pad_get_allowed_caps(axtransform->srcpad));

  if (!plugin_has_set_output_interface(axtransform)
      && axtransform->data->plugin->has_transform()) {
    return from_srctemplate_and_peer;
  }

  AxDataInterface template_for_incaps = interface_from_caps_and_meta(from_event, nullptr);

  AxDataInterface template_for_outcaps = template_for_incaps;
  if (axtransform->data->plugin->has_set_output_interface()) {
    template_for_outcaps
        = axtransform->data->plugin->set_output_interface(template_for_incaps);
  } else if (axtransform->data->plugin->has_set_output_interface_from_meta() && buffer) {
    GstMetaGeneral *meta = gst_buffer_get_general_meta((buffer));
    template_for_outcaps = axtransform->data->plugin->set_output_interface_from_meta(
        template_for_incaps, meta->subframe_index, meta->subframe_number, *meta->meta_map_ptr);
  }

  if (axtransform->data->batch != 1) {
    if (!std::holds_alternative<AxTensorsInterface>(template_for_outcaps)) {
      throw std::runtime_error("Batching is only supported for tensors output");
    }
    auto &tensors = std::get<AxTensorsInterface>(template_for_outcaps);
    if (tensors.size() != 1) {
      throw std::runtime_error("Batching is only supported for outputs of a single tensor");
    }
    auto &tensor = tensors[0];
    if (tensor.sizes.empty()) {
      throw std::runtime_error("Batching is only supported for non-empty tensors");
    }
    if (axtransform->data->tensor_size.empty()) {
      axtransform->data->tensor_size = tensor.sizes;
    } else if (axtransform->data->tensor_size != tensor.sizes) {
      throw std::runtime_error("Batching is only supported for tensors of the same dimensions");
    }
    if (tensor.sizes[0] == 1) {
      tensor.sizes[0] = axtransform->data->batch;
    } else {
      throw std::runtime_error("Batching is only supported for not previously batched tensors");
    }
  }

  auto from_event_and_properties
      = Ax::as_handle(caps_from_interface(template_for_outcaps));
  auto to = Ax::as_handle(gst_caps_intersect(
      from_srctemplate_and_peer.get(), from_event_and_properties.get()));

  if (gst_caps_is_empty(to.get())) {
    GST_ERROR_OBJECT(axtransform, "Caps compatible to the following elements %" GST_PTR_FORMAT,
        from_srctemplate_and_peer.get());
    GST_ERROR_OBJECT(axtransform, "Caps compatible to -set_output_interface- %" GST_PTR_FORMAT,
        from_event_and_properties.get());
    auto *in_caps = gst_caps_to_string(from_event_and_properties.get());
    auto *out_caps = gst_caps_to_string(from_srctemplate_and_peer.get());
    std::stringstream s;
    s << "Output caps provided by -set_output_interface- do not intersect with allowed caps\n"
      << in_caps << "\n"
      << out_caps;
    g_free(in_caps);
    g_free(out_caps);
    throw std::runtime_error(s.str());
  }

  return to;
}

void
initialise_options(GstAxtransform *axtransform)
{
  auto &data = *axtransform->data;
  if (!data.plugin) {
    (void) get_va_display; // prevent warning for unused function
    auto context = AxAllocationContextHandle(gst_opencl_allocator_get_context(
        axtransform->data->which_cl.c_str(), &data.logger));

    data.plugin = std::make_unique<Ax::LoadedTransform>(
        data.logger, std::move(*data.shared), data.options, context.get());
  }
}

static gboolean
gst_axtransform_setcaps(GstAxtransform *axtransform, GstCaps *from_event, GstBuffer *buffer)
{
  initialise_options(axtransform);
  auto to = gst_axtransform_outcaps(axtransform, from_event, buffer);

  copy_or_fixate_framerate(from_event, to.get());
  to = Ax::as_handle(gst_caps_fixate(to.release()));
  auto current_caps = Ax::as_handle(gst_pad_get_current_caps(axtransform->srcpad));
  gboolean ret = TRUE;
  //  Only set caps if they have changed
  if (!current_caps || !gst_caps_is_equal(to.get(), current_caps.get())) {
    ret = gst_pad_set_caps(axtransform->srcpad, to.get());
  }
  axtransform->data->output_template = interface_from_caps_and_meta(to.get(), nullptr);
  axtransform->outsize = size_from_interface(axtransform->data->output_template);
  if (axtransform->data->plugin->has_transform()) {
    gst_axtransform_do_bufferpool(axtransform, to.get());
  }
  return ret;
}

static gboolean
process_sink_event(GstPad *pad, GstAxtransform *axtransform, GstEvent *event, GstBuffer *buffer)
{
  gboolean ret = FALSE;
  GST_DEBUG_OBJECT(axtransform, "sink_event");

  switch (GST_EVENT_TYPE(event)) {
    case GST_EVENT_CAPS:
      GstCaps *caps;
      gst_event_parse_caps(event, &caps);
      ret = gst_axtransform_setcaps(axtransform, caps, buffer);
      gst_event_unref(event);
      break;
    default:
      ret = gst_pad_event_default(pad, GST_OBJECT(axtransform), event);
      break;
  }
  return ret;
}

void
process_queued_events(GstAxtransform *axtransform, GstBuffer *buffer)
{
  while (!axtransform->data->event_queue.empty()) {
    auto [pad, event] = axtransform->data->event_queue.front();
    axtransform->data->event_queue.pop();
    process_sink_event(pad, axtransform, event, buffer);
  }
}

static gboolean
gst_axtransform_sink_event(GstPad *pad, GstObject *parent, GstEvent *event)
{
  GstAxtransform *axtransform = GST_AXTRANSFORM(parent);

  if (GST_EVENT_TYPE(event) == GST_EVENT_EOS) {
    process_queued_events(axtransform, nullptr);
    if (axtransform->data->current_batch != 0) {
      gst_pad_push(axtransform->srcpad, axtransform->data->outbuf);
      axtransform->data->current_batch = 0;
      axtransform->data->outbuf = nullptr;
    }
    return gst_pad_event_default(pad, parent, event);
  }
  if (GST_EVENT_IS_SERIALIZED(event)) {
    //  Serialized events should only be handled *after* the previous buffer has been processed
    //  ad before the next. So we put thme in a queue and handle them in the chain function
    axtransform->data->event_queue.push({ pad, event });
    return TRUE;
  }
  return process_sink_event(pad, axtransform, event, nullptr);
}

static GstBuffer *
acquire_buffer(GstAxtransform *axtransform, GstBuffer *in_buffer)
{
  if (axtransform->pool && gst_buffer_pool_is_active(axtransform->pool)) {
    GstBuffer *outbuf;
    GstBufferPoolAcquireParams params{
      .format = GST_FORMAT_UNDEFINED,
      .start = 0,
      .stop = 0,
      .flags = GST_BUFFER_POOL_ACQUIRE_FLAG_DONTWAIT,
    };
    GstBufferPoolAcquireParams *p
        = axtransform->data->block_on_pool_empty ? nullptr : &params;
    if (gst_buffer_pool_acquire_buffer(axtransform->pool, &outbuf, p) == GST_FLOW_OK) {
      return outbuf;
    }
  }
  return nullptr;
}

static GstBuffer *
get_output_buffer(GstAxtransform *axtransform, const AxDataInterface &output, GstBuffer *in_buffer)
{
  GstBuffer *outbuf;
  if (std::holds_alternative<AxTensorsInterface>(output)
      && std::get<AxTensorsInterface>(output).size() > 1) {
    outbuf = gst_buffer_new();
    for (const auto &tensor : std::get<AxTensorsInterface>(output)) {
      gst_buffer_append_memory(outbuf,
          gst_allocator_alloc(axtransform->allocator, tensor.total_bytes(), NULL));
    }
  } else if (outbuf = acquire_buffer(axtransform, in_buffer); !outbuf) {
    outbuf = gst_buffer_new_allocate(
        axtransform->allocator, size_from_interface(output), NULL);
  }
  gst_buffer_copy_into(outbuf, in_buffer,
      (GstBufferCopyFlags) (GST_BUFFER_COPY_FLAGS | GST_BUFFER_COPY_TIMESTAMPS | GST_BUFFER_COPY_METADATA),
      0, -1);
  return outbuf;
}

bool
can_passthrough(GstAxtransform *axtransform, const AxDataInterface &input)
{
  auto &data = *axtransform->data;
  return !data.plugin->has_transform()
         || data.plugin->can_passthrough(input, data.output_template);
}

bool
can_use_dmabuf(GstAxtransform *self)
{
  return self->data->plugin->query_supports(Ax::PluginFeature::dmabuf_buffers);
}

bool
can_use_vaapi(GstAxtransform *self)
{
  //  Not currently supported
  return false;
}

bool
is_dmabuf(GstBuffer *buffer)
{
  auto *mem = gst_buffer_peek_memory(buffer, 0);
  return gst_is_dmabuf_memory(mem);
}

bool
should_pass_fds(GstAxtransform *self, GstBuffer *buffer)
{
  return can_use_dmabuf(self) && is_dmabuf(buffer);
}

bool
should_pass_vaapi(GstAxtransform *self, GstBuffer *buffer)
{
  //  Determine if we should pass vaapi pointers to the plugin
  //  Check if plugin understands vaapi pointers
  //  Check if buffer is vaapi
  auto api_type = gst_vaapi_video_meta_api_get_type();
  if (!api_type) {
    return false;
  }
  auto *meta = gst_buffer_get_meta(buffer, api_type);
  return meta && can_use_vaapi(self);
}

static GstFlowReturn
add_meta_and_push_buffer(GstAxtransform *axtransform, GstBuffer *buffer,
    GstPad *pad, const AxDataInterface &out)
{
  if (auto *video = std::get_if<AxVideoInterface>(&out)) {
    if (video->info.cropped) {
      //  We have crop info, so we can pass the buffer through unchanged
      //  and just add GstVideoCropMeta to it (and GstVideoMeta if it's not there)
      buffer = gst_buffer_make_writable(buffer);
      //  Now we need to set new src caps
      add_video_meta_from_interface_to_buffer(buffer, out);
      gst_pad_push(axtransform->srcpad, buffer);
      return GST_FLOW_OK;
    }
  }
  return GST_FLOW_ERROR;
}

bool
has_width_and_height(const AxDataInterface &interface)
{
  if (auto *video = std::get_if<AxVideoInterface>(&interface)) {
    return video->info.width > 0 && video->info.height > 0;
  }
  return true;
}

static GstFlowReturn
gst_axtransform_sink_chain(GstPad *pad, GstObject *parent, GstBuffer *buffer)
{
  GstAxtransform *axtransform = GST_AXTRANSFORM(parent);
  auto *self = axtransform->data;
  GST_DEBUG_OBJECT(axtransform, "sink_chain");
  if (!self->event_queue.empty()) {
    process_queued_events(axtransform, buffer);
  }

  auto in_caps = Ax::as_handle(gst_pad_get_current_caps(pad));
  AxDataInterface input = interface_from_caps_and_meta(in_caps.get(), buffer);

  //  If no changes are required to the data, simply pass the buffer through
  if (can_passthrough(axtransform, input)) {
    axtransform->data->logger(AX_DEBUG) << "Passthtough unchanged buffer\n";
    return gst_pad_push(axtransform->srcpad, buffer);
  }

  //  If the output caps depends on metadata, we need to set the caps first
  if (self->plugin->has_set_output_interface_from_meta()) {
    GstMetaGeneral *meta = gst_buffer_get_general_meta(buffer); // input buffer
    auto out = self->plugin->set_output_interface_from_meta(
        input, meta->subframe_index, meta->subframe_number, *meta->meta_map_ptr);

    if (!has_width_and_height(out)) {
      throw(std::runtime_error("Bounding box width or height is 0."));
    }
    auto caps = Ax::as_handle(caps_from_interface(out));
    auto current_caps = Ax::as_handle(gst_pad_get_current_caps(axtransform->srcpad));
    bool need_bufferpool = false;
    if (!current_caps || !gst_caps_is_equal(caps.get(), current_caps.get())) {
      gst_pad_set_caps(axtransform->srcpad, caps.get());
      need_bufferpool = true;
    }
    current_caps = std::move(caps);

    if (self->downstream_supports_crop) {
      axtransform->outsize = size_from_interface(out);
      if (add_meta_and_push_buffer(axtransform, buffer, pad, out) == GST_FLOW_OK) {
        return GST_FLOW_OK;
      }
    }
    if (need_bufferpool) {
      gst_axtransform_do_bufferpool(axtransform, current_caps.get());
    }
  }

  if (self->downstream_supports_crop && self->plugin->has_set_output_interface()) {
    auto out = self->plugin->set_output_interface(input);
    if (add_meta_and_push_buffer(axtransform, buffer, pad, out) == GST_FLOW_OK) {
      return GST_FLOW_OK;
    }
  }
  auto current_caps = Ax::as_handle(gst_pad_get_current_caps(axtransform->srcpad));
  if (!current_caps || !GST_IS_BUFFER(buffer)) {
    //  Caps failed to set, typically happens when stream is aborted
    return GST_FLOW_OK;
  }
  AxDataInterface output = interface_from_caps_and_meta(current_caps.get(), nullptr);

  std::vector<GstMapInfo> inmap;
  if (should_pass_fds(axtransform, buffer)) {
    assign_fds_to_interface(input, buffer);
  } else if (should_pass_vaapi(axtransform, buffer)) {
    //  Need to add
    inmap = get_mem_map(buffer, GstMapFlags(0 | GST_MAP_VAAPI), G_OBJECT(parent));
    assign_vaapi_ptrs_to_interface(inmap, input);
  } else if (supports_opencl_buffers(axtransform)
             && gst_is_opencl_memory(gst_buffer_peek_memory(buffer, 0))) {
    assign_opencl_ptrs_to_interface(input, buffer);
  } else {
    inmap = get_mem_map(buffer, GST_MAP_READ, G_OBJECT(parent));
    assign_data_ptrs_to_interface(inmap, input);
  }
  GstBuffer *outbuf = axtransform->data->current_batch == 0 ?
                          get_output_buffer(axtransform, output, buffer) :
                          axtransform->data->outbuf;

  GstMetaGeneral *meta = gst_buffer_get_general_meta(outbuf); // input buffer
  std::vector<GstMapInfo> outmap;
  if (supports_opencl_buffers(axtransform)
      && gst_is_opencl_memory(gst_buffer_peek_memory(outbuf, 0))) {
    assign_opencl_ptrs_to_interface(output, outbuf);
    gst_opencl_memory_add_dependency(gst_buffer_peek_memory(outbuf, 0), buffer);
  } else {
    outmap = get_mem_map(outbuf, GST_MAP_WRITE, G_OBJECT(parent));
    assign_data_ptrs_to_interface(outmap, output);
  }

  if (axtransform->data->batch != 1) {
    auto &tensor = std::get<AxTensorsInterface>(output)[0];
    auto offset = axtransform->data->current_batch * tensor.total() / tensor.sizes[0];
    tensor.data = static_cast<char *>(tensor.data) + offset * tensor.bytes;
  }

  self->plugin->transform(input, output, meta->subframe_index,
      meta->subframe_number, *meta->meta_map_ptr);

  unmap_mem(inmap);
  unmap_mem(outmap);
  gst_buffer_unref(buffer);
  ++axtransform->data->current_batch;
  if (axtransform->data->current_batch == axtransform->data->batch) {
    axtransform->data->current_batch = 0;
    axtransform->data->outbuf = nullptr;
    return gst_pad_push(axtransform->srcpad, outbuf);
  }
  axtransform->data->outbuf = outbuf;
  return GST_FLOW_OK;
}

static size_t
ax_size_from_caps(GstCaps *caps)
{
  return size_from_interface(interface_from_caps_and_meta(caps, nullptr));
}


static gboolean
add_allocation_proposal(GstAxtransform *sink, GstQuery *query)
{
  //  Tell the upstream element that we support GstVideoMeta. This allows it
  //  to give us buffers with "unusual" strides and offsets.
  if (plugin_supports_gst_video_meta(sink)) {
    gst_query_add_allocation_meta(query, GST_VIDEO_META_API_TYPE, NULL);
    if (plugin_supports_cropmeta(sink->data)) {
      // If the plugin supports GstVideoCropMeta, tell the upstream element that we support it.
      gst_query_add_allocation_meta(query, GST_VIDEO_CROP_META_API_TYPE, NULL);
    }
  }
  auto *self = sink;
  GstCaps *caps = NULL;
  gboolean need_pool = FALSE;
  gst_query_parse_allocation(query, &caps, &need_pool);
  if (!caps) {
    return TRUE;
  }

  //  Only offer our pool if upstream is using system memory
  auto *features = gst_caps_get_features(caps, 0);
  if (features && !gst_caps_features_contains(features, GST_CAPS_FEATURE_MEMORY_SYSTEM_MEMORY)) {
    return TRUE;
  }
  //  Test caps only for system memory

  auto subplugin_use_dmabuf = can_use_dmabuf(self);

  self->data->allocator = Ax::as_handle(
      subplugin_use_dmabuf ?
          gst_tensor_dmabuf_allocator_get(dmabuf_device) :
          gst_opencl_allocator_get(sink->data->which_cl.c_str(), &self->data->logger));
  if (!self->data->allocator) {
    GST_ERROR_OBJECT(self, "Unable to get aligned allocator");
    return TRUE;
  }

  if (need_pool) {
    const int min_buffers = 4;
    //  Now that inplace no longer provides a pool we need to provide sufficient
    const int max_buffers = 0;
    self->data->pool = Ax::as_handle(gst_ax_buffer_pool_new());
    GstStructure *config = gst_buffer_pool_get_config(self->data->pool.get());
    guint size = ax_size_from_caps(caps);

    gst_buffer_pool_config_set_params(config, caps, size, min_buffers, max_buffers);
    gst_buffer_pool_config_set_allocator(config, self->data->allocator.get(), NULL);
    if (!gst_buffer_pool_set_config(self->data->pool.get(), config)) {
      self->data->allocator.reset();
      self->data->pool.reset();
      GST_ERROR_OBJECT(self, "Failed to set pool configuration");
      return TRUE;
    }
    gst_query_add_allocation_pool(query, self->data->pool.get(), size, min_buffers, max_buffers);
  }

  gst_query_add_allocation_param(query, self->data->allocator.get(), NULL);

  return TRUE;
}

#define STRINGIFY(x) #x
#define STRINGIZE(x) STRINGIFY(x)

#define GENERATE_VIDEO_CAPS(x) STRINGIZE(x) ,
#define VIDEO_FORMATS_CAPS GST_AX_VIDEO_FORMATS(GENERATE_VIDEO_CAPS)


std::string_view supported_video_formats[] = { VIDEO_FORMATS_CAPS };

static int
gst_axtransform_sink_query(GstPad *pad, GstObject *parent, GstQuery *query)
{
  auto *axtransform = GST_AXTRANSFORM(parent);
  GST_DEBUG_OBJECT(axtransform, "sink_query");

  switch (GST_QUERY_TYPE(query)) {
    case GST_QUERY_ALLOCATION:
      return add_allocation_proposal(axtransform, query);

    case GST_QUERY_CAPS:
      {
        initialise_options(axtransform);
        GstCaps *filter = NULL;
        gst_query_parse_caps(query, &filter);

        GstCaps *tmp_caps = gst_caps_from_string(GST_TENSORS_CAP_DEFAULT);

        /* System memory variant */
        const auto uses_dmabuf = can_use_dmabuf(axtransform);
        for (auto format : supported_video_formats) {
          GstStructure *s = gst_structure_new(
              "video/x-raw", "format", G_TYPE_STRING, format.data(), NULL);
          gst_caps_append_structure_full(tmp_caps, s,
              gst_caps_features_new(GST_CAPS_FEATURE_MEMORY_SYSTEM_MEMORY, NULL));
          if (uses_dmabuf) {
            GstStructure *s2 = gst_structure_new(
                "video/x-raw", "format", G_TYPE_STRING, format.data(), NULL);
            gst_caps_append_structure_full(tmp_caps, s2,
                gst_caps_features_new(GST_CAPS_FEATURE_MEMORY_DMABUF, NULL));
          }
        }

        auto *caps = filter ? gst_caps_intersect_full(tmp_caps, filter, GST_CAPS_INTERSECT_FIRST) :
                              gst_caps_ref(tmp_caps);
        gst_caps_unref(tmp_caps);
        gst_query_set_caps_result(query, caps);
        gst_caps_unref(caps);
        return TRUE;
      }
    default:
      return gst_pad_query_default(pad, parent, query);
  }
}

static void
gst_axtransform_init(GstAxtransform *axtransform)
{
  axtransform->data = new GstAxtransformData;
  gst_debug_category_get_threshold(gst_axtransform_debug_category);
  axtransform->data->logger
      = Ax::Logger(Ax::extract_severity_from_category(gst_axtransform_debug_category),
          axtransform, gst_axtransform_debug_category);
  Ax::init_logger(axtransform->data->logger);

  axtransform->data->allocator = nullptr;
  axtransform->data->pool = nullptr;
  axtransform->outsize = 0;

  axtransform->sinkpad = gst_pad_new_from_static_template(&sink_template, "sink");
  gst_element_add_pad(GST_ELEMENT(axtransform), axtransform->sinkpad);

  gst_pad_set_chain_function(
      axtransform->sinkpad, GST_DEBUG_FUNCPTR(gst_axtransform_sink_chain));
  gst_pad_set_event_function(
      axtransform->sinkpad, GST_DEBUG_FUNCPTR(gst_axtransform_sink_event));

  gst_pad_set_query_function(
      axtransform->sinkpad, GST_DEBUG_FUNCPTR(gst_axtransform_sink_query));

  axtransform->srcpad = gst_pad_new_from_static_template(&src_template, "src");
  gst_element_add_pad(GST_ELEMENT(axtransform), axtransform->srcpad);
}

static void
gst_axtransform_set_property(
    GObject *object, guint prop_id, const GValue *value, GParamSpec *pspec)
{
  GstAxtransform *axtransform = GST_AXTRANSFORM(object);
  GST_DEBUG_OBJECT(axtransform, "set_property");
  auto &data = *axtransform->data;

  switch (prop_id) {
    case PROP_SHARED_LIB_PATH:
      data.shared_lib_path = Ax::libname(g_value_get_string(value));
      data.shared = std::make_unique<Ax::SharedLib>(data.logger, data.shared_lib_path);
      break;
    case PROP_OPTIONS:
      data.options = g_value_get_string(value);
      if (data.plugin) {
        auto opts = Ax::parse_and_validate_plugin_options(
            data.logger, data.options, data.plugin->allowed_properties());
        data.plugin->set_dynamic_properties(opts);
      }
      break;
    case PROP_BATCH:
      data.batch = g_value_get_uint(value);
      data.current_batch = 0;
      break;
    case PROP_WHICH_CL:
      data.which_cl = g_value_get_string(value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
      break;
  }
}

static void
gst_axtransform_get_property(GObject *object, guint prop_id, GValue *value, GParamSpec *pspec)
{
  GstAxtransform *axtransform = GST_AXTRANSFORM(object);
  GST_DEBUG_OBJECT(axtransform, "get_property");

  switch (prop_id) {
    case PROP_SHARED_LIB_PATH:
      g_value_set_string(value, axtransform->data->shared_lib_path.c_str());
      break;
    case PROP_OPTIONS:
      g_value_set_string(value, axtransform->data->options.c_str());
      break;
    case PROP_BATCH:
      g_value_set_uint(value, axtransform->data->batch);
      break;
    case PROP_WHICH_CL:
      g_value_set_string(value, axtransform->data->which_cl.c_str());
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
      break;
  }
}

static void
gst_axtransform_finalize(GObject *object)
{
  GstAxtransform *axtransform = GST_AXTRANSFORM(object);
  GST_DEBUG_OBJECT(axtransform, "finalize");

  if (axtransform->pool) {
    gst_buffer_pool_set_active(axtransform->pool, FALSE);
    gst_object_unref(axtransform->pool);
    axtransform->pool = NULL;
  }
  if (axtransform->allocator) {
    gst_object_unref(axtransform->allocator);
    axtransform->allocator = NULL;
  }

  if (axtransform->data) {
    delete axtransform->data;
    axtransform->data = nullptr;
  }

  G_OBJECT_CLASS(gst_axtransform_parent_class)->finalize(object);
}

static void
gst_axtransform_class_init(GstAxtransformClass *klass)
{
  gst_element_class_set_static_metadata(GST_ELEMENT_CLASS(klass), "axtransform",
      "Effect", "description", "axelera.ai");

  G_OBJECT_CLASS(klass)->set_property = GST_DEBUG_FUNCPTR(gst_axtransform_set_property);
  G_OBJECT_CLASS(klass)->get_property = GST_DEBUG_FUNCPTR(gst_axtransform_get_property);
  G_OBJECT_CLASS(klass)->finalize = GST_DEBUG_FUNCPTR(gst_axtransform_finalize);

  g_object_class_install_property(G_OBJECT_CLASS(klass), PROP_SHARED_LIB_PATH,
      g_param_spec_string("lib", "lib path", "String containing lib path", "",
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property(G_OBJECT_CLASS(klass), PROP_OPTIONS,
      g_param_spec_string("options", "options string", "Subplugin dependent options",
          "", (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property(G_OBJECT_CLASS(klass), PROP_BATCH,
      g_param_spec_uint("batch", "size to batch", "Number of inputs to batch together",
          1, 16, 1, (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property(G_OBJECT_CLASS(klass), PROP_WHICH_CL,
      g_param_spec_string("cl-platform", "CL platform", "OpenCL platform to use",
          "auto", (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  gst_element_class_add_pad_template(
      GST_ELEMENT_CLASS(klass), gst_static_pad_template_get(&src_template));
  gst_element_class_add_pad_template(
      GST_ELEMENT_CLASS(klass), gst_static_pad_template_get(&sink_template));
}
