// Copyright Axelera AI, 2026
#include <cstdlib>
#include <cstring>
#include <gst/gst.h>
#include <gst/video/video.h>
#include <memory>
#include <unordered_map>
#include "AxDataInterface.h"
#include "AxInference.hpp"
#include "AxInferenceNet.hpp"
#include "AxLog.hpp"
#include "AxMeta.hpp"
#include "AxMetaStreamId.hpp"
#include "AxStreamerUtils.hpp"
#include "GstAxBufferPool.hpp"
#include "GstAxDataUtils.hpp"
#include "GstAxInPlace.hpp"
#include "GstAxInferenceNet.hpp"
#include "GstAxMeta.hpp"
#include "GstAxStreamerUtils.hpp"

#include <atomic>
#include <cinttypes>
#include <mutex>
#include <queue>

constexpr int PROPERTY_STRIDE = 4; // 4 properties per operator

GST_DEBUG_CATEGORY_STATIC(gst_axinferencenet_debug);
#define GST_CAT_DEFAULT gst_axinferencenet_debug

GST_ELEMENT_REGISTER_DEFINE(
    axinferencenet, "axinferencenet", GST_RANK_NONE, GST_TYPE_AXINFERENCENET);

static GstStaticPadTemplate sink_template
    = GST_STATIC_PAD_TEMPLATE("sink_%u", GST_PAD_SINK, GST_PAD_REQUEST,
        GST_STATIC_CAPS("video/x-raw,format={RGBA,BGRA,RGB,BGR,GRAY8}"));

static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE("src", GST_PAD_SRC,
    GST_PAD_ALWAYS, GST_STATIC_CAPS("video/x-raw,format={RGBA,BGRA,RGB,BGR,GRAY8}"));


G_DEFINE_TYPE_WITH_CODE(GstAxInferenceNet, gst_axinferencenet, GST_TYPE_ELEMENT,
    GST_DEBUG_CATEGORY_INIT(gst_axinferencenet_debug, "axinferencenet", 0,
        "axinferencenet element"));

// Custom query implementation for buffer requirements
GstQuery *
gst_query_new_ax_buffer_requirements(void)
{
  GstStructure *structure = gst_structure_new(
      "GstQueryAxBufferRequirements", "num_buffers", G_TYPE_UINT, 0, NULL);
  return gst_query_new_custom(GST_QUERY_AX_BUFFER_REQUIREMENTS, structure);
}

void
gst_query_set_ax_buffer_requirements(GstQuery *query, guint num_buffers)
{
  g_return_if_fail(GST_QUERY_TYPE(query) == GST_QUERY_AX_BUFFER_REQUIREMENTS);

  if (GstStructure *structure = gst_query_writable_structure(query)) {
    gst_structure_set(structure, "num_buffers", G_TYPE_UINT, num_buffers, NULL);
  }
}

guint
gst_query_parse_ax_buffer_requirements(GstQuery *query)
{
  if (GST_QUERY_TYPE(query) != GST_QUERY_AX_BUFFER_REQUIREMENTS) {
    return 0;
  }
  guint num_buffers = 0;
  const GstStructure *structure = gst_query_get_structure(query);
  return structure && gst_structure_get_uint(structure, "num_buffers", &num_buffers) ?
             num_buffers :
             0;
}

guint
query_downstream_buffers(GstPad *pad)
{
  auto *query = gst_query_new_ax_buffer_requirements();
  auto required_buffers = gst_pad_peer_query(pad, query) ?
                              gst_query_parse_ax_buffer_requirements(query) :
                              0;
  gst_query_unref(query);
  return required_buffers;
}

enum {
  PROP_PREPROC0_SHARED_LIB_PATH = Ax::AXINFERENCE_PROP_NEXT_AVAILABLE,
  PROP_PREPROC0_OPTIONS,
  PROP_PREPROC0_MODE,
  PROP_PREPROC0_BATCH,
  PROP_PREPROC_END = PROP_PREPROC0_SHARED_LIB_PATH + Ax::MAX_OPERATORS * PROPERTY_STRIDE,

  PROP_POSTPROC0_SHARED_LIB_PATH = PROP_PREPROC_END,
  PROP_POSTPROC0_OPTIONS,
  PROP_POSTPROC0_MODE,
  PROP_POSTPROC_END = PROP_POSTPROC0_SHARED_LIB_PATH + Ax::MAX_OPERATORS * PROPERTY_STRIDE,
  PROP_STREAM_SELECT,
  PROP_MAX_POOL_BUFFERS,
  PROP_LOOP,
};

class GstAxStreamSelect
{
  public:
  void set_property(const std::string &stream_select)
  {
    // this is a comma separated list of stream ids that should be enabled
    // so the omission of the stream id means disable it.
    // easiest way to handle this is set all to disabled and then enable them.
    for (auto &[sid, enabled] : ids_) {
      enabled = false;
    }
    for (auto &token : Ax::Internal::split(stream_select, ',')) {
      auto stoken = std::string(token);
      try {
        auto sid = std::stoi(stoken);
        ids_[sid] = true;
      } catch (const std::invalid_argument &e) {
        // TODO        GST_ERROR_OBJECT(inf, "Invalid number '%s' in stream_select", stoken.c_str());
      }
    }
  }

  std::string get_property() const
  {
    std::string stream_select;
    for (auto &[sid, enabled] : ids_) {
      if (enabled) {
        if (!stream_select.empty()) {
          stream_select += ",";
        }
        stream_select += std::to_string(sid);
      }
    }
    return stream_select;
  }

  bool is_pad_enabled(const std::string &pad_name, int stream_id)
  {
    // check if the stream has been explicity disabled
    // we also maintain a padname->stream_id mapping so that on pad removal we can remove the stream id
    // and not have removed streams returned in get_stream_select
    // (for belt and braces we also add it enabled if never seen before,
    //  in case the pipeline does not have axinplace:addstreamid)
    pads_.try_emplace(pad_name, stream_id);
    return ids_.try_emplace(stream_id, true).first->second;
  }

  void add_pad(const std::string &pad_name, int stream_id)
  {
    pads_.try_emplace(pad_name, stream_id);
    ids_.try_emplace(stream_id, true);
  }

  void remove_pad(const std::string &pad_name)
  {
    auto id = pads_.find(pad_name);
    if (id != pads_.end()) {
      ids_.erase(id->second);
      pads_.erase(id);
    }
  }

  private:
  std::map<std::string, int> pads_;
  std::map<int, bool> ids_;
};

static std::string *
find_property(Ax::InferenceNetProperties &properties, int property_id)
{
  if ((property_id >= PROP_PREPROC0_SHARED_LIB_PATH && property_id < PROP_PREPROC_END)
      || (property_id >= PROP_POSTPROC0_SHARED_LIB_PATH && property_id < PROP_POSTPROC_END)) {
    const bool pre = property_id < PROP_PREPROC_END;
    const int base = pre ? PROP_PREPROC0_SHARED_LIB_PATH : PROP_POSTPROC0_SHARED_LIB_PATH;
    const int op_num = (property_id - base) / PROPERTY_STRIDE;
    const int prop_num = (property_id - base) % PROPERTY_STRIDE;
    auto &ops = pre ? properties.preproc : properties.postproc;
    switch (prop_num) {
      case 0:
        return &ops[op_num].lib;
      case 1:
        return &ops[op_num].options;
      case 2:
        return &ops[op_num].mode;
      default: // note cannot happen
      case 3:
        return &ops[op_num].batch;
    }
  }
  return nullptr;
}

static void
gst_axinferencenet_set_property(
    GObject *object, guint property_id, const GValue *value, GParamSpec *pspec)
{
  auto *inf = GST_AXINFERENCENET(object);
  GST_DEBUG_OBJECT(inf, "set_property");

  if (auto *prop_string = find_property(*inf->properties, property_id)) {
    *prop_string = g_value_get_string(value);
    GST_DEBUG_OBJECT(inf, "set_property %s", prop_string->c_str());
    return;
  }

  if (Ax::set_inference_property(*inf->properties, property_id, value)) {
    return;
  }

  if (property_id == PROP_STREAM_SELECT) {
    // this is a comma separated list of stream ids that should be enabled
    // so the omission of the stream id means disable it.
    // easiest way to handle this is set all to disabled and then enable them.
    auto stream_select = Ax::get_string(value, "stream_select");
    inf->stream_select->set_property(stream_select);
    return;
  }

  if (property_id == PROP_MAX_POOL_BUFFERS) {
    const auto max_buffers = g_value_get_uint(value);
    if (max_buffers > 0) {
      inf->properties->max_buffers = max_buffers;
    }
    return;
  }

  if (property_id == PROP_LOOP) {
    inf->loop = g_value_get_boolean(value);
    return;
  }

  G_OBJECT_WARN_INVALID_PROPERTY_ID(object, property_id, pspec);
}

static void
gst_axinferencenet_get_property(
    GObject *object, guint property_id, GValue *value, GParamSpec *pspec)
{
  auto *inf = GST_AXINFERENCENET(object);
  GST_DEBUG_OBJECT(inf, "get_property");

  if (Ax::get_inference_property(*inf->properties, property_id, value)) {
    return;
  }

  if (const auto *prop_string = find_property(*inf->properties, property_id)) {
    g_value_set_string(value, prop_string->c_str());
    return;
  }
  if (property_id == PROP_STREAM_SELECT) {
    const auto stream_select = inf->stream_select->get_property();
    g_value_set_string(value, stream_select.c_str());
    return;
  }

  if (property_id == PROP_MAX_POOL_BUFFERS) {
    g_value_set_uint(value, inf->properties->max_buffers);
    return;
  }

  if (property_id == PROP_LOOP) {
    g_value_set_boolean(value, inf->loop);
    return;
  }

  G_OBJECT_WARN_INVALID_PROPERTY_ID(object, property_id, pspec);
}

struct GstBufferHandle {
  explicit GstBufferHandle(GstBuffer *buffer) : buffer(buffer)
  {
    map_info.memory = nullptr;
  }
  GstBufferHandle(const GstBufferHandle &) = delete;
  GstBufferHandle(GstBufferHandle &&) = delete;
  GstBufferHandle &operator=(const GstBufferHandle &) = delete;
  GstBufferHandle &operator=(GstBufferHandle &&) = delete;
  ~GstBufferHandle()
  {
    if (buffer && map_info.memory) {
      gst_buffer_unmap(buffer, &map_info);
    }
  }

  GstBuffer *buffer;
  GstMapInfo map_info;
};

static AxVideoInterface
video_interface_from_caps_and_meta(GstCaps *caps, GstBuffer *buffer)
{
  auto info = interface_from_caps_and_meta(caps, buffer);
  if (!std::holds_alternative<AxVideoInterface>(info)) {
    throw std::runtime_error(
        "Cannot get video info from caps in video_interface_from_caps_and_meta");
  }
  return std::get<AxVideoInterface>(info);
}

void
insert_event_end_marker(GstAxInferenceNet *inf)
{
  std::lock_guard<std::mutex> lock(inf->event_queue->mutex);
  inf->event_queue->queue.push({ nullptr, Ax::GstHandle<GstEvent>() });
}

static std::string
get_pad_name(GstPad *pad)
{
  auto *pname = gst_pad_get_name(pad);
  std::string padname = std::string(pname);
  g_free(pname);
  return padname;
}

static Ax::InferenceNet &net(_GstAxInferenceNet *inf);

static GstFlowReturn
gst_axinferencenet_sink_chain(GstPad *sinkpad, GstObject *parent, GstBuffer *buf)
{
  auto *inf = GST_AXINFERENCENET(parent);

  //  Once we are at eos, accept no more
  if (inf->at_eos) {
    return GST_FLOW_EOS;
  }

  const auto sinkpad_is_eos = GST_PAD_IS_EOS(sinkpad);
  const auto sinkpad_has_buffer = !sinkpad_is_eos; // gst_pad_has_buffer(sinkpad);
  if (sinkpad_has_buffer == sinkpad_is_eos) {
    throw std::logic_error("In funnel, sinkpad -EOS- does not correspond to -no buffer-");
  }
  if (sinkpad_is_eos) {
    return GST_FLOW_EOS;
  }
  if (sinkpad_has_buffer) {
    auto sinkpad_caps = Ax::as_handle(gst_pad_get_current_caps(sinkpad));
    if (!sinkpad_caps) {
      return GST_FLOW_EOS;
    }

    auto &axmetamap = *gst_buffer_get_general_meta(buf)->meta_map_ptr;
    auto stream_id = 0;
    if (auto *sid = dynamic_cast<AxMetaStreamId *>(axmetamap["stream_id"].get())) {
      stream_id = sid->stream_id;
    }

    if (!inf->stream_select->is_pad_enabled(get_pad_name(sinkpad), stream_id)) {
      gst_buffer_unref(buf);
      return GST_FLOW_OK;
    }

    auto video = video_interface_from_caps_and_meta(sinkpad_caps.get(), buf);
    if (gst_buffer_n_memory(buf) != 1) {
      throw std::runtime_error("Buffer must have exactly one memory (for now)");
    }
    auto handle = std::make_shared<GstBufferHandle>(buf);

    auto &infnet = net(inf);
    //  If the first element supports opencl buffers, we just use the buffer as is
    if (infnet.supports_opencl_buffers(video)
        && gst_is_opencl_memory(gst_buffer_peek_memory(buf, 0))) {
      video.ocl_buffer
          = gst_opencl_mem_get_opencl_buffer(gst_buffer_peek_memory(buf, 0));
      video.data = nullptr;
      video.vaapi = nullptr;
      video.fd = -1;
      //  Here we assign
    } else {
      if (FALSE == gst_buffer_map(buf, &handle->map_info, GST_MAP_READ)) {
        throw std::runtime_error("Unable to map GstBuffer");
      }
      video.data = handle->map_info.data;
    }
    insert_event_end_marker(inf);
    infnet.push_new_frame(std::move(handle), video, axmetamap, stream_id);
  }
  return GST_FLOW_OK;
}

static gboolean
process_sink_event(GstPad *pad, GstAxInferenceNet *parent, GstEvent *event)
{
  if (!GST_IS_PAD(pad)) {
    return TRUE;
  }
  return gst_pad_event_default(pad, GST_OBJECT(parent), event);
}

void
process_queued_events(GstAxInferenceNet *inf)
{
  std::lock_guard<std::mutex> lock(inf->event_queue->mutex);
  while (!inf->event_queue->queue.empty()) {
    auto [pad, event] = std::move(inf->event_queue->queue.front());
    inf->event_queue->queue.pop();
    if (!pad) {
      break;
    }
    process_sink_event(pad, inf, event.release());
  }
}

static GstElement *
find_pipeline(GstElement *element)
{
  GstElement *parent = nullptr;
  GstObject *obj = GST_OBJECT_PARENT(element);
  while (obj != nullptr) {
    if (GST_IS_PIPELINE(obj)) {
      parent = GST_ELEMENT(obj);
      break;
    }
    obj = GST_OBJECT_PARENT(obj);
  }
  return parent;
}

static int
set_flushing(GstAxInferenceNet *inf, GstPad *pad, bool flushing)
{
  std::lock_guard<std::mutex> lock(*inf->flushing_mutex);
  if (flushing) {
    inf->flushing_pads->insert(pad);
  } else {
    inf->flushing_pads->erase(pad);
  }
  return inf->flushing_pads->size();
}

bool
initiate_flush(GstAxInferenceNet *inf, GstPad *pad)
{
  std::lock_guard<std::mutex> lock(*inf->flushing_mutex);
  if (!inf->flushing_pads->empty()) {
    // Pad is already flushing
    return false;
  }
  inf->flushing_pads->insert(pad);
  return true;
}

static bool
is_flushing(GstAxInferenceNet *inf, GstPad *pad)
{
  std::lock_guard<std::mutex> lock(*inf->flushing_mutex);
  return inf->flushing_pads->count(pad) > 0;
}

static gboolean
gst_axinferencenet_sink_event(GstPad *pad, GstObject *parent, GstEvent *event)
{
  auto *inf = GST_AXINFERENCENET(parent);
  if (GST_EVENT_TYPE(event) == GST_EVENT_FLUSH_START
      || GST_EVENT_TYPE(event) == GST_EVENT_FLUSH_STOP) {
    set_flushing(inf, pad, GST_EVENT_TYPE(event) == GST_EVENT_FLUSH_START);
    gst_event_unref(event);
    return true;
  }
  if (GST_EVENT_TYPE(event) == GST_EVENT_EOS) {
    if (inf->loop) {
      if (initiate_flush(inf, pad)) {
        //  If we're the first pad to receive EOS, rewind
        if (GstElement *pipeline = find_pipeline(GST_ELEMENT(parent))) {
          auto seek_flags = static_cast<GstSeekFlags>(
              GST_SEEK_FLAG_FLUSH | GST_SEEK_FLAG_KEY_UNIT | GST_SEEK_FLAG_ACCURATE);
          GstEvent *seek_event = gst_event_new_seek(1.0, GST_FORMAT_TIME,
              seek_flags, GST_SEEK_TYPE_SET, 0, GST_SEEK_TYPE_NONE, -1);
          if (gst_element_send_event(pipeline, seek_event)) {
            // Clear the event queue since we're flushing
            std::lock_guard<std::mutex> lock(inf->event_queue->mutex);
            while (!inf->event_queue->queue.empty()) {
              inf->event_queue->queue.pop();
            }
          } else {
            GST_WARNING_OBJECT(inf, "looping - failed to send seek event to pipeline");
          }
        } else {
          GST_WARNING_OBJECT(inf, "looping - could not find pipeline element");
        }
      }
      set_flushing(inf, pad, true);
    } else {
      inf->at_eos = true;
      net(inf).end_of_input();
    }
    gst_event_unref(event);
    return TRUE;
  }
  if (is_flushing(inf, pad)) {
    gst_event_unref(event);
    return TRUE;
  }
  if (GST_EVENT_IS_SERIALIZED(event)) {
    //  Serialized events should only be handled *after* the previous buffer has been processed
    //  and before the next. So we put them in a queue and handle them in the chain function
    std::lock_guard<std::mutex> lock(inf->event_queue->mutex);
    inf->event_queue->queue.push({ pad, Ax::as_handle(event) });
    return TRUE;
  }
  return process_sink_event(pad, inf, event);
}

static void
gst_axinferencenet_push_done(GstAxInferenceNet *inf, Ax::CompletedFrame &frame)
{
  if (frame.end_of_input) {
    gst_pad_push_event(GST_PAD_CAST(inf->parent.srcpads->data), gst_event_new_eos());
    return;
  }
  auto handle = std::exchange(frame.buffer_handle, {});
  auto *gst_handle = static_cast<GstBufferHandle *>(handle.get());
  auto buf = gst_handle->buffer;
  handle.reset();
  auto *srcpads = inf->parent.srcpads;
  if (!srcpads || !srcpads->data) {
    GST_ERROR_OBJECT(inf, "AxInference has lost the source pad: %s", g_module_error());
  } else {
    process_queued_events(inf);
    auto out_caps = Ax::as_handle(caps_from_interface(frame.video));
    auto current_caps
        = Ax::as_handle(gst_pad_get_current_caps(GST_PAD_CAST(srcpads->data)));
    if (!current_caps || !gst_caps_is_equal(out_caps.get(), current_caps.get())) {
      gst_pad_set_caps(GST_PAD_CAST(srcpads->data), out_caps.get());
    }
    if (GST_FLOW_ERROR == gst_pad_push(GST_PAD_CAST(srcpads->data), buf)) {
      GST_ERROR_OBJECT(inf, "Failed pushing buffer to pad %p (s%d:f%" PRIu64 ") (%s)\n",
          srcpads->data, frame.stream_id, frame.frame_id, g_module_error());
    }
  }
}

static guint
count_sink_pads(GstElement *element)
{
  guint count = 0;
  GValue item = G_VALUE_INIT;
  GstIterator *it = gst_element_iterate_sink_pads(element);

  while (gst_iterator_next(it, &item) == GST_ITERATOR_OK) {
    count++;
    g_value_reset(&item);
  }

  gst_iterator_free(it);
  return count;
}

static int
determine_required_buffers(GstAxInferenceNet *sink)
{
  auto *properties = sink->properties.get();
  auto pre_fill = Ax::pipeline_pre_fill(*properties);
  auto output_drop = Ax::output_drop(*properties);
  return sink->net->frames_required_for_inference() * (pre_fill + output_drop);
}

static int
determine_max_pool_buffers(GstAxInferenceNet *sink)
{
  auto *properties = sink->properties.get();
  auto max_buffers = properties->max_buffers;
  auto required_buffers = determine_required_buffers(sink);
  if (max_buffers < required_buffers) {
    if (max_buffers > 0) {
      GST_WARNING_OBJECT(sink, "Max buffers %d is less than min required %d, using default",
          max_buffers, required_buffers);
    }
    max_buffers = required_buffers;
  }
  //  Query downstream axinferencenets for their requirements
  //  Get reuqired buffers from downstream
  auto *srcpad = GST_PAD_CAST(sink->parent.srcpads->data);
  auto downstream_buffers = query_downstream_buffers(srcpad);
  auto num_sinks = count_sink_pads(GST_ELEMENT(sink));
  auto other_buffers = 8 + 50 / (num_sinks > 0 ? num_sinks : 1);
  return max_buffers + other_buffers + downstream_buffers;
}

static gboolean
add_allocation_proposal(GstAxInferenceNet *sink, GstQuery *query)
{
  //  Tell the upstream element that we support GstVideoMeta. This allows it
  //  to give us buffers with "unusual" strides and offsets.
  gst_query_add_allocation_meta(query, GST_VIDEO_META_API_TYPE, NULL);

  auto *self = sink;
  GstCaps *caps;
  gboolean need_pool;
  gst_query_parse_allocation(query, &caps, &need_pool);

  if (!caps) {
    GST_ERROR_OBJECT(self, "Allocation query without caps");
    return TRUE;
  }
  self->allocator = Ax::as_handle(gst_opencl_allocator_get(
      sink->properties->which_cl.c_str(), self->logger.get()));
  if (!self->allocator) {
    GST_ERROR_OBJECT(self, "Unable to get aligned allocator");
    return TRUE;
  }

  if (need_pool) {
    const auto min_buffers = 4;
    const auto max_buffers = determine_max_pool_buffers(self);
    self->pool = Ax::as_handle(gst_ax_buffer_pool_new());
    GstStructure *config = gst_buffer_pool_get_config(self->pool.get());
    guint size = size_from_interface(interface_from_caps_and_meta(caps, nullptr));

    gst_buffer_pool_config_set_params(config, caps, size, min_buffers, max_buffers);
    gst_buffer_pool_config_set_allocator(config, self->allocator.get(), NULL);
    if (!gst_buffer_pool_set_config(self->pool.get(), config)) {
      self->allocator.reset();
      self->pool.reset();
      GST_ERROR_OBJECT(self, "Failed to set pool configuration");
      return TRUE;
    }
    gst_query_add_allocation_pool(query, self->pool.get(), size, min_buffers, max_buffers);
  }

  gst_query_add_allocation_param(query, self->allocator.get(), NULL);

  return TRUE;
}

static int
gst_axinferencenet_sink_query(GstPad *pad, GstObject *parent, GstQuery *query)
{
  auto *inf = GST_AXINFERENCENET(parent);
  GST_DEBUG_OBJECT(inf, "sink_query");

  if (GST_QUERY_TYPE(query) == GST_QUERY_AX_BUFFER_REQUIREMENTS) {
    guint required_buffers = determine_required_buffers(inf);
    auto *srcpad = GST_PAD_CAST(inf->parent.srcpads->data);
    if (srcpad) {
      guint downstream_buffers = query_downstream_buffers(srcpad);
      required_buffers += downstream_buffers;
      GST_DEBUG_OBJECT(inf, "Downstream requires %u buffers", downstream_buffers);
    }
    gst_query_set_ax_buffer_requirements(query, required_buffers);
    return TRUE;
  }

  return GST_QUERY_TYPE(query) == GST_QUERY_ALLOCATION ?
             add_allocation_proposal(inf, query) :
             gst_pad_query_default(pad, parent, query);
}


static GstPad *
gst_axinferencenet_request_new_pad(GstElement *element, GstPadTemplate *templ,
    const gchar *name, const GstCaps *caps)
{
  auto *inf = GST_AXINFERENCENET(element);
  GST_DEBUG_OBJECT(element, "requesting pad\n");
  (void) net(inf); // start the construction of the inference net and start loading models

  auto sinkpad = GST_PAD_CAST(g_object_new(GST_TYPE_PAD, "name", name,
      "direction", templ->direction, "template", templ, NULL));
  gst_pad_set_chain_function(sinkpad, GST_DEBUG_FUNCPTR(gst_axinferencenet_sink_chain));
  gst_pad_set_event_function(sinkpad, GST_DEBUG_FUNCPTR(gst_axinferencenet_sink_event));
  GST_OBJECT_FLAG_SET(sinkpad, GST_PAD_FLAG_PROXY_CAPS);
  GST_OBJECT_FLAG_SET(sinkpad, GST_PAD_FLAG_PROXY_ALLOCATION);
  gst_pad_set_query_function(sinkpad, GST_DEBUG_FUNCPTR(gst_axinferencenet_sink_query));
  gst_pad_set_active(sinkpad, TRUE);
  gst_element_add_pad(element, sinkpad);
  GST_DEBUG_OBJECT(element, "requested pad %s:%s", GST_DEBUG_PAD_NAME(sinkpad));
  return sinkpad;
}

static GstPad *
find_sink(GstElement *elem)
{
  GstPad *pad = nullptr;
  GValue value = G_VALUE_INIT;
  if (auto *iter = gst_element_iterate_sink_pads(elem)) {
    gst_iterator_next(iter, &value);
    // todo is this a problem for cascade?
    pad = GST_PAD(g_value_get_object(&value));
    gst_iterator_free(iter);
  }
  return pad;
}

static GstElement *
next_upstream(GstElement *elem)
{
  // iterate upstream, going peer to parent.  elem is unref'd and the return elem must be unref'd
  // by the caller. always takes the first sink pad, so will ont do the right thing on a funnel/net
  auto *sink = find_sink(elem);
  gst_object_unref(elem);
  elem = nullptr;
  if (sink) {
    auto *peer = gst_pad_get_peer(sink);
    gst_object_unref(sink);
    if (peer) {
      elem = gst_pad_get_parent_element(peer);
      gst_object_unref(peer);
    }
  }
  return elem;
}

static int
get_axinplace_stream_id(GstElement *elem)
{
  int stream_id = -1;
  if (GST_IS_AXINPLACE(elem)) {
    char *options = nullptr;
    g_object_get(elem, "options", &options, NULL);
    if (options) {
      if (auto sid = std::strstr(options, "stream_id:")) {
        stream_id = std::atoi(sid + 10);
      }
      g_free(options);
    }
  }
  return stream_id;
}

static void
axnet_on_pad_linked(GstPad *pad, GstPad *peer, gpointer user_data)
{
  auto *inf = GST_AXINFERENCENET(gst_pad_get_parent_element(pad));
  GST_INFO_OBJECT(inf, "pad connected %s:%s", GST_DEBUG_PAD_NAME(pad));
  auto stream_id = -1;
  // find axinplace stream_id element by walking the sink pads upstream
  auto *upstream = gst_pad_get_parent_element(peer);
  while (upstream && stream_id == -1) {
    GST_DEBUG_OBJECT(upstream, "searching for axinplace:addstreamid");
    stream_id = get_axinplace_stream_id(upstream);
    upstream = next_upstream(upstream);
    if (upstream && GST_IS_AXINFERENCENET(upstream)) {
      GST_INFO_OBJECT(upstream, "found axinferencenet whilst looking for stream_id");
      break;
    }
  }
  if (upstream) {
    gst_object_unref(upstream);
  }
  if (stream_id != -1) {
    GST_INFO_OBJECT(inf, "new sink pad added %s:%s with stream_id %d",
        GST_DEBUG_PAD_NAME(pad), stream_id);
    inf->stream_select->add_pad(get_pad_name(pad), stream_id);
  } else {
    GST_WARNING_OBJECT(inf, "new sink pad added %s:%s but could not find stream_id",
        GST_DEBUG_PAD_NAME(pad));
  }
  gst_object_unref(inf);
}


static void
axnet_on_pad_added(GstElement *element, GstPad *pad, gpointer user_data)
{
  if (pad->direction == GST_PAD_SINK) {
    g_signal_connect(pad, "linked", G_CALLBACK(axnet_on_pad_linked), nullptr);
  }
}

static void
axnet_on_pad_removed(GstElement *element, GstPad *pad, gpointer user_data)
{
  GST_INFO_OBJECT(element, "pad removed %s:%s", GST_DEBUG_PAD_NAME(pad));
  auto *inf = GST_AXINFERENCENET(element);
  if (pad->direction == GST_PAD_SINK) {
    inf->stream_select->remove_pad(get_pad_name(pad));
  }
}

static Ax::InferenceNet &
net(_GstAxInferenceNet *inf)
{
  if (!inf->net) {
    auto log_latency = [inf](const std::string &elem_name, uint64_t throughput,
                           uint64_t latency) {
      auto *n = gst_element_get_name(GST_ELEMENT(inf));
      auto name = std::string(n);
      g_free(n);
      auto s = name + std::string(":") + elem_name;
      gst_tracer_record_log(inf->element_latency, s.c_str(), throughput, latency);
    };
    auto frame_done
        = [inf](auto &frame) { gst_axinferencenet_push_done(inf, frame); };
    auto context = AxAllocationContextHandle(gst_opencl_allocator_get_context(
        inf->properties->which_cl.c_str(), inf->logger.get()));
    inf->net = Ax::create_inference_net(
        *inf->properties, *inf->logger, frame_done, log_latency, context.get());
  }
  return *inf->net;
}

static void
gst_axinferencenet_init(GstAxInferenceNet *inf)
{
  GST_DEBUG_OBJECT(inf, "init\n");

  inf->loop = false;

  inf->flushing_mutex = std::make_unique<std::mutex>();
  inf->flushing_pads = std::make_unique<std::unordered_set<GstPad *>>();

  inf->element_latency = gst_tracer_record_new("element-latency.class",
      "element", GST_TYPE_STRUCTURE,
      gst_structure_new("scope", "type", G_TYPE_GTYPE, G_TYPE_STRING, "related-to",
          GST_TYPE_TRACER_VALUE_SCOPE, GST_TRACER_VALUE_SCOPE_ELEMENT, NULL),
      "time", GST_TYPE_STRUCTURE,
      gst_structure_new("value", "type", G_TYPE_GTYPE, G_TYPE_UINT64, "description",
          G_TYPE_STRING, "time spent in the element in ns", "min", G_TYPE_UINT64,
          G_GUINT64_CONSTANT(0), "max", G_TYPE_UINT64, G_MAXUINT64, NULL),
      "latency", GST_TYPE_STRUCTURE,
      gst_structure_new("value", "type", G_TYPE_GTYPE, G_TYPE_UINT64, "description", G_TYPE_STRING,
          "time it took for the buffer to exit the element after entering", "min",
          G_TYPE_UINT64, G_GUINT64_CONSTANT(0), "max", G_TYPE_UINT64, G_MAXUINT64, NULL),
      NULL);

  inf->properties = std::make_unique<Ax::InferenceNetProperties>();
  inf->event_queue = std::make_unique<event_queue>();
  inf->logger = std::make_unique<Ax::Logger>(
      Ax::Severity::trace, nullptr, gst_axinferencenet_debug);
  inf->stream_select = std::make_unique<GstAxStreamSelect>();
  Ax::init_logger(*inf->logger);

  auto srcpad = gst_pad_new_from_static_template(&src_template, "src");
  gst_pad_use_fixed_caps(srcpad);
  // gst_pad_set_query_function(srcpad, GST_DEBUG_FUNCPTR(gst_axinferencenet_src_query));
  gst_pad_set_active(srcpad, TRUE);
  gst_element_add_pad(&inf->parent, srcpad);
  g_signal_connect(&inf->parent, "pad-removed", G_CALLBACK(axnet_on_pad_removed), nullptr);
  g_signal_connect(&inf->parent, "pad-added", G_CALLBACK(axnet_on_pad_added), nullptr);
}

static void
gst_axinferencenet_finalize(GObject *object)
{
  auto *inf = GST_AXINFERENCENET(object);
  GST_DEBUG_OBJECT(object, "finalizing");
  net(inf).stop();
  inf->logger.reset();
  inf->stream_select.reset();
  inf->flushing_pads.reset();
  inf->flushing_mutex.reset();
  inf->pool.reset();
  inf->allocator.reset();
  inf->event_queue.reset();
  inf->properties.reset();
  inf->net.reset();
  gst_object_unref(inf->element_latency);
  G_OBJECT_CLASS(gst_axinferencenet_parent_class)->finalize(object);
  GST_DEBUG_OBJECT(object, "disposed");
}

static void
gst_axinferencenet_class_init(GstAxInferenceNetClass *klass)
{
  auto object_klass = G_OBJECT_CLASS(klass);
  auto element_klass = GST_ELEMENT_CLASS(klass);
  gst_element_class_set_static_metadata(
      element_klass, "axinferencenet", "Effect", "description", "axelera.ai");

  object_klass->set_property = GST_DEBUG_FUNCPTR(gst_axinferencenet_set_property);
  object_klass->get_property = GST_DEBUG_FUNCPTR(gst_axinferencenet_get_property);
  object_klass->finalize = GST_DEBUG_FUNCPTR(gst_axinferencenet_finalize);
  element_klass->request_new_pad = GST_DEBUG_FUNCPTR(gst_axinferencenet_request_new_pad);

  gst_element_class_add_static_pad_template(element_klass, &sink_template);
  gst_element_class_add_static_pad_template(element_klass, &src_template);

  for (int n = 0; n != Ax::MAX_OPERATORS; ++n) {
    int off = n * PROPERTY_STRIDE;
    auto N = std::to_string(n);
    Ax::add_string_property(object_klass, PROP_PREPROC0_SHARED_LIB_PATH + off,
        "preprocess" + N + "_lib", "String containing lib path");
    Ax::add_string_property(object_klass, PROP_PREPROC0_OPTIONS + off,
        "preprocess" + N + "_options", "Subplugin dependent options");
    Ax::add_string_property(object_klass, PROP_PREPROC0_MODE + off, "preprocess" + N + "_mode",
        "Set to 'read' to specify the buffer is read only, if omitteed the buffer is read/write");
    Ax::add_string_property(object_klass, PROP_PREPROC0_BATCH + off,
        "preprocess" + N + "_batch",
        "Set the batch size this element outpputs, defaults to 1");
    Ax::add_string_property(object_klass, PROP_POSTPROC0_SHARED_LIB_PATH + off,
        "postprocess" + N + "_lib", "String containing lib path");
    Ax::add_string_property(object_klass, PROP_POSTPROC0_OPTIONS + off,
        "postprocess" + N + "_options", "Subplugin dependent options");
    Ax::add_string_property(object_klass, PROP_POSTPROC0_MODE + off,
        "postprocess" + N + "_mode",
        "Set to 'read' to specify the buffer is read only, if omitteed the buffer is read/write");
  }
  Ax::add_string_property(object_klass, Ax::AXINFERENCE_PROP_META_STRING,
      "meta", "String with key to metadata");
  Ax::add_string_property(object_klass, Ax::AXINFERENCE_PROP_WHICH_CL,
      "cl_platform", "String with key to OpenCL platform");

  Ax::add_inference_properties(object_klass, true, true);

  Ax::add_string_property(object_klass, PROP_STREAM_SELECT, "stream_select",
      "Select stream to output");

  Ax::add_uint_property(object_klass, PROP_MAX_POOL_BUFFERS, "max_pool_buffers",
      "Maximum number of buffers in the pool. 0 means application decides (default).",
      0, 1024, 0);
  Ax::add_boolean_property(object_klass, PROP_LOOP, "loop",
      "Whether to loop video input when EOS is received");
}
