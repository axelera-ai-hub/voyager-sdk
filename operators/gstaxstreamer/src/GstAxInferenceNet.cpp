#include <cstdlib>
#include <cstring>
#include <gst/gst.h>
#include <gst/video/video.h>
#include <memory>
#include <unordered_map>
#include "AxDataInterface.h"
#include "AxInferenceNet.hpp"
#include "AxLog.hpp"
#include "AxMeta.hpp"
#include "AxMetaStreamId.hpp"
#include "AxStreamerUtils.hpp"
#include "GstAxDataUtils.hpp"
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

static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE("sink_%u",
    GST_PAD_SINK, GST_PAD_REQUEST, GST_STATIC_CAPS("video/x-raw,format={RGBA,BGRA}"));

static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE("src",
    GST_PAD_SRC, GST_PAD_ALWAYS, GST_STATIC_CAPS("video/x-raw,format={RGBA,BGRA}"));


G_DEFINE_TYPE_WITH_CODE(GstAxInferenceNet, gst_axinferencenet, GST_TYPE_ELEMENT,
    GST_DEBUG_CATEGORY_INIT(gst_axinferencenet_debug, "axinferencenet", 0,
        "axinferencenet element"));

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
    auto stream_select = Ax::get_string(value, "stream_select");
    *inf->stream_select = Ax::create_stream_set(stream_select);
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
    g_value_set_string(value, Ax::to_string(*inf->stream_select).c_str());
    return;
  }
  G_OBJECT_WARN_INVALID_PROPERTY_ID(object, property_id, pspec);
}

struct GstBufferHandle {
  explicit GstBufferHandle(GstBuffer *buffer) : buffer(buffer)
  {
  }
  GstBufferHandle(const GstBufferHandle &) = delete;
  GstBufferHandle(GstBufferHandle &&) = delete;
  GstBufferHandle &operator=(const GstBufferHandle &) = delete;
  GstBufferHandle &operator=(GstBufferHandle &&) = delete;
  ~GstBufferHandle()
  {
    if (buffer) {
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

Ax::GstHandle<GstCaps>
as_handle(GstCaps *p)
{
  return { p, ::gst_caps_unref };
}

static bool
stream_filter(const std::set<int> &stream_select, GstPad *pad)
{
  if (stream_select.empty()) {
    return false;
  }
  auto *pname = gst_pad_get_name(pad);
  std::string padname = std::string(pname);
  g_free(pname);
  auto pos = padname.find(std::string("pad"));
  if (pos == std::string::npos) {
    throw std::runtime_error("Invalid pad name");
  }
  const auto id = std::atoi(padname.erase(pos, 3).c_str());
  if (stream_select.find(id) != stream_select.end()) {
    return false;
  }
  return true;
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
    auto sinkpad_caps = as_handle(gst_pad_get_current_caps(sinkpad));
    if (!sinkpad_caps) {
      return GST_FLOW_EOS;
    }

    if (stream_filter(*inf->stream_select, sinkpad)) {
      gst_buffer_unref(buf);
      return GST_FLOW_OK;
    }

    auto &axmetamap = *gst_buffer_get_general_meta(buf)->meta_map_ptr;
    auto video = video_interface_from_caps_and_meta(sinkpad_caps.get(), buf);
    if (gst_buffer_n_memory(buf) != 1) {
      throw std::runtime_error("Buffer must have exactly one memory (for now)");
    }
    auto handle = std::make_shared<GstBufferHandle>(buf);
    if (FALSE == gst_buffer_map(buf, &handle->map_info, GST_MAP_READ)) {
      throw std::runtime_error("Unable to map GstBuffer");
    }
    video.data = handle->map_info.data;
    insert_event_end_marker(inf);

    auto stream_id = 0;
    if (auto *sid = dynamic_cast<AxMetaStreamId *>(axmetamap["stream_id"].get())) {
      stream_id = sid->stream_id;
    }
    net(inf).push_new_frame(std::move(handle), video, axmetamap, stream_id);
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

Ax::GstHandle<GstEvent>
as_handle(GstEvent *event)
{
  return Ax::GstHandle<GstEvent>(
      event, [](auto *event) { gst_event_unref(event); });
}

static gboolean
gst_axinferencenet_sink_event(GstPad *pad, GstObject *parent, GstEvent *event)
{
  auto *inf = GST_AXINFERENCENET(parent);
  GST_DEBUG_OBJECT(inf, "sink_event");

  if (GST_EVENT_TYPE(event) == GST_EVENT_EOS) {
    inf->at_eos = true;
    net(inf).end_of_input();
    gst_event_unref(event);
    return true;
  }
  if (GST_EVENT_IS_SERIALIZED(event)) {
    //  Serialized events should only be handled *after* the previous buffer has been processed
    //  ad before the next. So we put thme in a queue and han§dle them in the chain function
    std::lock_guard<std::mutex> lock(inf->event_queue->mutex);
    inf->event_queue->queue.push({ pad, as_handle(event) });
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
    auto out_caps = as_handle(caps_from_interface(frame.video));
    auto current_caps = as_handle(gst_pad_get_current_caps(GST_PAD_CAST(srcpads->data)));
    if (!current_caps || !gst_caps_is_equal(out_caps.get(), current_caps.get())) {
      gst_pad_set_caps(GST_PAD_CAST(srcpads->data), out_caps.get());
    }
    if (GST_FLOW_ERROR == gst_pad_push(GST_PAD_CAST(srcpads->data), buf)) {
      GST_ERROR_OBJECT(inf, "Failed pushing buffer to pad %p (s%d:f%" PRIu64 ") (%s)\n",
          srcpads->data, frame.stream_id, frame.frame_id, g_module_error());
    }
  }
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

  self->allocator = Ax::as_handle(gst_aligned_allocator_get());
  if (!self->allocator) {
    GST_ERROR_OBJECT(self, "Unable to get aligned allocator");
    return TRUE;
  }

  if (need_pool) {
    constexpr int min_buffers = 4;
    self->pool = Ax::as_handle(gst_buffer_pool_new());
    GstStructure *config = gst_buffer_pool_get_config(self->pool.get());
    guint size = size_from_interface(interface_from_caps_and_meta(caps, nullptr));

    gst_buffer_pool_config_set_params(config, caps, size, min_buffers, 0);
    gst_buffer_pool_config_set_allocator(config, self->allocator.get(), NULL);
    if (!gst_buffer_pool_set_config(self->pool.get(), config)) {
      self->allocator.reset();
      self->pool.reset();
      GST_ERROR_OBJECT(self, "Failed to set pool configuration");
      return TRUE;
    }
    gst_query_add_allocation_pool(query, self->pool.get(), size, min_buffers, 0);
  }

  gst_query_add_allocation_param(query, self->allocator.get(), NULL);

  return TRUE;
}


static int
gst_axinferencenet_sink_query(GstPad *pad, GstObject *parent, GstQuery *query)
{
  auto *inf = GST_AXINFERENCENET(parent);
  GST_DEBUG_OBJECT(inf, "sink_query");

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
    inf->net = Ax::create_inference_net(*inf->properties, *inf->logger, frame_done, log_latency);
  }
  return *inf->net;
}

static void
gst_axinferencenet_init(GstAxInferenceNet *inf)
{
  GST_DEBUG_OBJECT(inf, "init\n");

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
  inf->stream_select = std::make_unique<std::set<int>>();
  Ax::init_logger(*inf->logger);

  auto srcpad = gst_pad_new_from_static_template(&src_template, "src");
  gst_pad_use_fixed_caps(srcpad);
  gst_pad_set_active(srcpad, TRUE);
  gst_element_add_pad(&inf->parent, srcpad);
}

static void
gst_axinferencenet_finalize(GObject *object)
{
  auto *inf = GST_AXINFERENCENET(object);
  GST_DEBUG_OBJECT(object, "dispose\n");
  net(inf).stop();
  inf->net.reset();
  inf->properties.reset();
  gst_object_unref(inf->element_latency);
  inf->event_queue.reset();
  G_OBJECT_CLASS(gst_axinferencenet_parent_class)->dispose(object);
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

  Ax::add_inference_properties(object_klass, true, true);

  Ax::add_string_property(object_klass, PROP_STREAM_SELECT, "stream_select",
      "Select stream to output");
}
