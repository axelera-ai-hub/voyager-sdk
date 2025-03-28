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
#include "AxStreamerUtils.hpp"
#include "GstAxDataUtils.hpp"
#include "GstAxMeta.hpp"
#include "GstAxStreamerUtils.hpp"

GST_DEBUG_CATEGORY_STATIC(gst_axtransform_debug_category);
#define GST_CAT_DEFAULT gst_axtransform_debug_category

extern "C" {
gboolean gst_is_tensor_dmabuf_memory(GstMemory *mem);
}

struct double_buffer_details {
  GstBuffer *inbuffer{};
  std::vector<GstMapInfo> inmap{};
  GstBuffer *outbuffer{};
  std::vector<GstMapInfo> outmap{};
  Ax::AsyncCompleter complete_last_xform{};
};

struct EventDetails {
  GstPad *pad;
  GstEvent *event;
};

struct _GstAxtransformData {
  std::string shared_lib_path;
  std::string options;
  bool options_initialised = false;
  AxDataInterface output_template;
  std::shared_ptr<void> subplugin_data;
  std::unique_ptr<Ax::SharedLib> shared;
  Ax::Logger logger{ Ax::Severity::trace, nullptr, gst_axtransform_debug_category };
  unsigned int batch = 1;
  unsigned int current_batch = 0;
  std::vector<int> tensor_size{};
  GstBuffer *outbuf = nullptr;
  Ax::GstHandle<GstAllocator> allocator{};
  Ax::GstHandle<GstBufferPool> pool{};
  double_buffer_details dbl_buffer;
  bool downstream_supports_crop = false;
  std::queue<EventDetails> event_queue;

  Ax::V1Plugin::Transform fns;
};

G_DEFINE_TYPE_WITH_CODE(GstAxtransform, gst_axtransform, GST_TYPE_ELEMENT,
    GST_DEBUG_CATEGORY_INIT(gst_axtransform_debug_category, "axtransform", 0,
        "debug category for axtransform element"));

enum {
  PROP_0,
  PROP_SHARED_LIB_PATH,
  PROP_OPTIONS,
  PROP_BATCH,
};

static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE(
    "sink", GST_PAD_SINK, GST_PAD_ALWAYS, GST_STATIC_CAPS_ANY);

static GstStaticPadTemplate src_template
    = GST_STATIC_PAD_TEMPLATE("src", GST_PAD_SRC, GST_PAD_ALWAYS,
        GST_STATIC_CAPS(GST_TENSORS_CAP_DEFAULT ";" AX_GST_VIDEO_FORMATS_CAPS));

static bool
plugin_supports_cropmeta(GstAxtransform *axtransform)
{
  return axtransform->data->fns.handles_crop_meta
         && axtransform->data->fns.handles_crop_meta();
}

const auto *dmabuf_device = "/dev/dma_heap/system";

static bool
is_suitable(GstAllocator *allocator)
{
  return !allocator || allocator == gst_aligned_allocator_get()
         || allocator == gst_tensor_dmabuf_allocator_get(dmabuf_device);
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
      if (is_suitable(allocator)) {
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
  if (axtransform->pool) {
    return;
  }
  if (std::holds_alternative<AxVideoInterface>(axtransform->data->output_template)) {
    axtransform->pool = gst_video_buffer_pool_new();
  } else {
    axtransform->pool = gst_buffer_pool_new();
  }
  if (!axtransform->pool) {
    return;
  }
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

bool
plugin_has_transform(GstAxtransform *axtransform)
{
  return axtransform->data->fns.transform || axtransform->data->fns.transform_async;
}


static bool
plugin_has_set_output_interface(GstAxtransform *axtransform)
{
  return axtransform->data->fns.set_output_interface
         || axtransform->data->fns.set_output_interface_from_meta;
}

static GstCaps *
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
  GstCaps *from_srctemplate_and_peer = gst_pad_get_allowed_caps(axtransform->srcpad);

  if (!plugin_has_set_output_interface(axtransform) && plugin_has_transform(axtransform)) {
    return from_srctemplate_and_peer;
  }

  AxDataInterface template_for_incaps = interface_from_caps_and_meta(from_event, nullptr);

  AxDataInterface template_for_outcaps = template_for_incaps;
  if (axtransform->data->fns.set_output_interface) {
    template_for_outcaps = axtransform->data->fns.set_output_interface(template_for_incaps,
        axtransform->data->subplugin_data.get(), axtransform->data->logger);
  } else if (axtransform->data->fns.set_output_interface_from_meta && buffer) {
    GstMetaGeneral *meta = gst_buffer_get_general_meta((buffer));
    template_for_outcaps = axtransform->data->fns.set_output_interface_from_meta(
        template_for_incaps, axtransform->data->subplugin_data.get(), meta->subframe_index,
        meta->subframe_number, *meta->meta_map_ptr, axtransform->data->logger);
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

  GstCaps *from_event_and_properties = caps_from_interface(template_for_outcaps);
  GstCaps *to = gst_caps_intersect(from_srctemplate_and_peer, from_event_and_properties);

  if (gst_caps_is_empty(to)) {
    GST_ERROR_OBJECT(axtransform, "Caps compatible to the following elements %" GST_PTR_FORMAT,
        from_srctemplate_and_peer);
    GST_ERROR_OBJECT(axtransform, "Caps compatible to -set_output_interface- %" GST_PTR_FORMAT,
        from_event_and_properties);
    auto *in_caps = gst_caps_to_string(from_event_and_properties);
    auto *out_caps = gst_caps_to_string(from_srctemplate_and_peer);
    std::stringstream s;
    s << "Output caps provided by -set_output_interface- do not intersect with allowed caps\n"
      << in_caps << "\n"
      << out_caps;
    g_free(in_caps);
    g_free(out_caps);
    throw std::runtime_error(s.str());
  }

  gst_caps_unref(from_srctemplate_and_peer);
  gst_caps_unref(from_event_and_properties);

  return to;
}

static gboolean
gst_axtransform_setcaps(GstAxtransform *axtransform, GstCaps *from_event, GstBuffer *buffer)
{
  init_options(G_OBJECT(axtransform), axtransform->data->options,
      axtransform->data->fns, axtransform->data->logger,
      axtransform->data->subplugin_data, axtransform->data->options_initialised);

  GstCaps *to = gst_axtransform_outcaps(axtransform, from_event, buffer);

  copy_or_fixate_framerate(from_event, to);
  to = gst_caps_fixate(to);
  auto *current_caps = gst_pad_get_current_caps(axtransform->srcpad);
  gboolean ret = TRUE;
  //  Only set caps if they have changed
  if (!current_caps || !gst_caps_is_equal(to, current_caps)) {
    ret = gst_pad_set_caps(axtransform->srcpad, to);
  }
  axtransform->data->output_template = interface_from_caps_and_meta(to, nullptr);
  axtransform->outsize = size_from_interface(axtransform->data->output_template);
  if (plugin_has_transform(axtransform)) {
    gst_axtransform_do_bufferpool(axtransform, to);
  }

  if (current_caps) {
    gst_caps_unref(current_caps);
  }
  gst_caps_unref(to);
  return ret;
}

static GstFlowReturn
finish_previous_transform(GstAxtransform *axtransform, GstBuffer *in_buffer,
    GstBuffer *out_buffer, std::vector<GstMapInfo> &inmap,
    std::vector<GstMapInfo> &outmap, Ax::AsyncCompleter complete_last_xform)
{
  if (axtransform->data->dbl_buffer.inbuffer) {
    //  We have a pending buffer
    axtransform->data->outbuf = out_buffer;
    std::swap(axtransform->data->dbl_buffer.complete_last_xform, complete_last_xform);
    std::swap(axtransform->data->dbl_buffer.inbuffer, in_buffer);
    std::swap(axtransform->data->dbl_buffer.inmap, inmap);
    std::swap(axtransform->data->dbl_buffer.outbuffer, out_buffer);
    std::swap(axtransform->data->dbl_buffer.outmap, outmap);
    complete_last_xform();
    unmap_mem(inmap);
    unmap_mem(outmap);
    gst_buffer_unref(in_buffer);
    ++axtransform->data->current_batch;
    if (axtransform->data->current_batch == axtransform->data->batch) {
      axtransform->data->current_batch = 0;
      axtransform->data->outbuf = nullptr;
      return gst_pad_push(axtransform->srcpad, out_buffer);
    }
    return GST_FLOW_OK;
  } else {
    //  No pending buffer, just set up next buffer
    axtransform->data->dbl_buffer.complete_last_xform = complete_last_xform;
    axtransform->data->dbl_buffer.inbuffer = in_buffer;
    axtransform->data->dbl_buffer.inmap = inmap;
    axtransform->data->dbl_buffer.outbuffer = out_buffer;
    axtransform->data->dbl_buffer.outmap = outmap;
    return GST_FLOW_OK;
  }
}

static GstFlowReturn
finish_previous_transform(GstAxtransform *axtransform)
{
  auto inmap = std::vector<GstMapInfo>{};
  auto outmap = std::vector<GstMapInfo>{};
  return finish_previous_transform(axtransform, nullptr, nullptr, inmap, outmap, {});
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
    finish_previous_transform(axtransform);
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
    if (gst_buffer_pool_acquire_buffer(axtransform->pool, &outbuf, NULL) == GST_FLOW_OK) {
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
  return !plugin_has_transform(axtransform)
         || (axtransform->data->fns.can_passthrough
             && axtransform->data->fns.can_passthrough(input, axtransform->data->output_template,
                 axtransform->data->subplugin_data.get(), axtransform->data->logger));
}

bool
can_use_dmabuf(GstAxtransform *self)
{
  return self->data->fns.transform && self->data->fns.can_use_dmabuf
         && self->data->fns.can_use_dmabuf(
             self->data->subplugin_data.get(), self->data->logger);
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

static GstFlowReturn
add_meta_and_push_buffer(GstAxtransform *axtransform, GstBuffer *buffer,
    GstPad *pad, const AxDataInterface &out)
{
  if (auto *video = std::get_if<AxVideoInterface>(&out)) {
    if (video->info.cropped) {
      //  We have crop info, so we can pass the buffer through unchanged
      //  and just add GstVideoCropMeta to it (and GstVideoMeta if it's not there)
      buffer = gst_buffer_make_writable(buffer);
      finish_previous_transform(axtransform);
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
    finish_previous_transform(axtransform);
    process_queued_events(axtransform, buffer);
  }
  GstCaps *in_caps = gst_pad_get_current_caps(pad);
  AxDataInterface input = interface_from_caps_and_meta(in_caps, buffer);
  gst_caps_unref(in_caps);

  //  If no changes are required to the data, simply pass the buffer through
  if (can_passthrough(axtransform, input)) {
    axtransform->data->logger(AX_DEBUG) << "Passthtough unchanged buffer\n";
    return gst_pad_push(axtransform->srcpad, buffer);
  }

  //  If the output capps depends on metadata, we need to set the caps first
  if (self->fns.set_output_interface_from_meta) {
    GstMetaGeneral *meta = gst_buffer_get_general_meta(buffer); // input buffer
    auto out = self->fns.set_output_interface_from_meta(input,
        axtransform->data->subplugin_data.get(), meta->subframe_index,
        meta->subframe_number, *meta->meta_map_ptr, self->logger);

    if (!has_width_and_height(out)) {
      throw(std::runtime_error("Bounding box width or height is 0."));
    }
    auto *caps = caps_from_interface(out);
    auto *current_caps = gst_pad_get_current_caps(axtransform->srcpad);
    bool need_bufferpool = false;
    if (!current_caps || !gst_caps_is_equal(caps, current_caps)) {
      gst_pad_set_caps(axtransform->srcpad, caps);
      need_bufferpool = true;
    }
    if (current_caps) {
      gst_caps_unref(current_caps);
    }
    current_caps = caps;

    if (self->downstream_supports_crop) {
      axtransform->outsize = size_from_interface(out);
      if (add_meta_and_push_buffer(axtransform, buffer, pad, out) == GST_FLOW_OK) {
        return GST_FLOW_OK;
      }
    }
    if (need_bufferpool) {
      gst_axtransform_do_bufferpool(axtransform, current_caps);
    }
    gst_caps_unref(current_caps);
  }

  if (self->downstream_supports_crop && self->fns.set_output_interface) {
    auto out = self->fns.set_output_interface(
        input, axtransform->data->subplugin_data.get(), self->logger);
    if (add_meta_and_push_buffer(axtransform, buffer, pad, out) == GST_FLOW_OK) {
      return GST_FLOW_OK;
    }
  }
  auto *current_caps = gst_pad_get_current_caps(axtransform->srcpad);
  if (!current_caps || !GST_IS_BUFFER(buffer)) {
    //  Caps failed to set, typically happens when stream is aborted
    return GST_FLOW_OK;
  }
  AxDataInterface output = interface_from_caps_and_meta(current_caps, nullptr);
  gst_caps_unref(current_caps);

  std::vector<GstMapInfo> inmap;
  if (should_pass_fds(axtransform, buffer)) {
    assign_fds_to_interface(input, buffer);
  } else {
    inmap = get_mem_map(buffer, GST_MAP_READ, G_OBJECT(parent));
    assign_data_ptrs_to_interface(inmap, input);
  }
  GstBuffer *outbuf = axtransform->data->current_batch == 0 ?
                          get_output_buffer(axtransform, output, buffer) :
                          axtransform->data->outbuf;

  GstMetaGeneral *meta = gst_buffer_get_general_meta(outbuf); // input buffer
  std::vector<GstMapInfo> outmap = get_mem_map(outbuf, GST_MAP_WRITE, G_OBJECT(parent));
  assign_data_ptrs_to_interface(outmap, output);

  if (axtransform->data->batch != 1) {
    auto &tensor = std::get<AxTensorsInterface>(output)[0];
    auto offset = axtransform->data->current_batch * tensor.total() / tensor.sizes[0];
    tensor.data = static_cast<char *>(tensor.data) + offset * tensor.bytes;
  }

  if (axtransform->data->fns.transform_async) {
    auto complete = axtransform->data->fns.transform_async(input, output,
        axtransform->data->subplugin_data.get(), meta->subframe_index,
        meta->subframe_number, *meta->meta_map_ptr, axtransform->data->logger);
    auto use_double_buffer = Ax::get_env("AXELERA_USE_CL_DOUBLE_BUFFER", "1");
    if (use_double_buffer == "1") {
      return finish_previous_transform(axtransform, buffer, outbuf, inmap, outmap, complete);
    }
    complete();
  } else {
    axtransform->data->fns.transform(input, output,
        axtransform->data->subplugin_data.get(), meta->subframe_index,
        meta->subframe_number, *meta->meta_map_ptr, axtransform->data->logger);
  }

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
  init_options(G_OBJECT(sink), sink->data->options, sink->data->fns, sink->data->logger,
      sink->data->subplugin_data, sink->data->options_initialised);

  //  Tell the upstream element that we support GstVideoMeta. This allows it
  //  to give us buffers with "unusual" strides and offsets.
  gst_query_add_allocation_meta(query, GST_VIDEO_META_API_TYPE, NULL);
  if (plugin_supports_cropmeta(sink)) {
    // If the plugin supports GstVideoCropMeta, tell the upstream element that we support it.
    gst_query_add_allocation_meta(query, GST_VIDEO_CROP_META_API_TYPE, NULL);
  }

  auto *self = sink;
  GstCaps *caps;
  gboolean need_pool;
  gst_query_parse_allocation(query, &caps, &need_pool);

  if (!caps) {
    GST_ERROR_OBJECT(self, "Allocation query without caps");
    return TRUE;
  }

  auto subplugin_use_dmabuf = can_use_dmabuf(self);

  self->data->allocator = Ax::as_handle(
      subplugin_use_dmabuf ? gst_tensor_dmabuf_allocator_get(dmabuf_device) :
                             gst_aligned_allocator_get());
  if (!self->data->allocator) {
    GST_ERROR_OBJECT(self, "Unable to get aligned allocator");
    return TRUE;
  }

  if (need_pool) {
    constexpr int min_buffers = 4;
    self->data->pool = Ax::as_handle(gst_buffer_pool_new());
    GstStructure *config = gst_buffer_pool_get_config(self->data->pool.get());
    guint size = ax_size_from_caps(caps);

    gst_buffer_pool_config_set_params(config, caps, size, min_buffers, 0);
    gst_buffer_pool_config_set_allocator(config, self->data->allocator.get(), NULL);
    if (!gst_buffer_pool_set_config(self->data->pool.get(), config)) {
      self->data->allocator.reset();
      self->data->pool.reset();
      GST_ERROR_OBJECT(self, "Failed to set pool configuration");
      return TRUE;
    }
    gst_query_add_allocation_pool(query, self->data->pool.get(), size, min_buffers, 0);
  }

  gst_query_add_allocation_param(query, self->data->allocator.get(), NULL);

  return TRUE;
}


static int
gst_axtransform_sink_query(GstPad *pad, GstObject *parent, GstQuery *query)
{
  auto *axtransform = GST_AXTRANSFORM(parent);
  GST_DEBUG_OBJECT(axtransform, "sink_query");

  switch (GST_QUERY_TYPE(query)) {
    case GST_QUERY_ALLOCATION:
      return add_allocation_proposal(axtransform, query);

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

  switch (prop_id) {
    case PROP_SHARED_LIB_PATH:
      axtransform->data->shared_lib_path = Ax::libname(g_value_get_string(value));
      axtransform->data->shared = std::make_unique<Ax::SharedLib>(
          axtransform->data->logger, axtransform->data->shared_lib_path);
      Ax::load_v1_plugin(*axtransform->data->shared, axtransform->data->fns);
      break;
    case PROP_OPTIONS:
      axtransform->data->options = g_value_get_string(value);
      update_options(object, axtransform->data->options, axtransform->data->fns,
          axtransform->data->logger, axtransform->data->subplugin_data,
          axtransform->data->options_initialised);
      break;
    case PROP_BATCH:
      axtransform->data->batch = g_value_get_uint(value);
      axtransform->data->current_batch = 0;
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
  gst_element_class_add_pad_template(
      GST_ELEMENT_CLASS(klass), gst_static_pad_template_get(&src_template));
  gst_element_class_add_pad_template(
      GST_ELEMENT_CLASS(klass), gst_static_pad_template_get(&sink_template));
}
