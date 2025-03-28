// Copyright Axelera AI, 2024

#include <gst/allocators/gstfdmemory.h>
#include <gst/gst.h>
// #include <gst/video/video.h>

#include <memory>
#include <stdio.h>

#include "AxInference.hpp"
#include "AxStreamerUtils.hpp"
#include "GstAxStreamerUtils.hpp"

#include "AxInference.hpp"
#include "GstAxDataUtils.hpp"
#include "GstAxInference.hpp"

constexpr auto max_tensor_count = AX_TENSOR_SIZE_LIMIT;
const auto *const dmabuf_device = "/dev/dma_heap/system";

GST_ELEMENT_REGISTER_DEFINE(axinference, "axinference", GST_RANK_NONE, GST_TYPE_AXINFERENCE);
GST_DEBUG_CATEGORY_STATIC(gst_axinference_debug);

#define GST_CAT_DEFAULT gst_axinference_debug
#define CAPS_STRING \
  GST_TENSOR_CAP_DEFAULT ";" GST_TENSORS_CAP_MAKE("{ static, flexible }")

extern "C" {
gboolean gst_is_tensor_dmabuf_memory(GstMemory *mem);
}

enum {
  PROP_0,
  PROP_UNBATCH = Ax::AXINFERENCE_PROP_NEXT_AVAILABLE,
};

static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE(
    "sink", GST_PAD_SINK, GST_PAD_ALWAYS, GST_STATIC_CAPS(CAPS_STRING));

static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE(
    "src", GST_PAD_SRC, GST_PAD_ALWAYS, GST_STATIC_CAPS(CAPS_STRING));

#define gst_axinference_parent_class parent_class
G_DEFINE_TYPE(GstAxInference, gst_axinference, GST_TYPE_BASE_TRANSFORM);

auto
as_buffer_handle(GstBuffer *p)
{
  return Ax::GstHandle<GstBuffer>(p, ::gst_buffer_unref);
}

auto
as_event_handle(GstEvent *p)
{
  return Ax::GstHandle<GstEvent>(p, ::gst_event_unref);
}


namespace
{
struct GstDmaBufHandle : public Ax::DmaBufHandle {
  explicit GstDmaBufHandle(GstMemory *mem)
      : Ax::DmaBufHandle(gst_fd_memory_get_fd(mem), false), mem(gst_memory_ref(mem))
  {
  }
  ~GstDmaBufHandle()
  {
    gst_memory_unref(mem);
  }

  private:
  GstMemory *const mem;
};

struct Params {
  std::vector<Ax::GstHandle<GstMemory>> mems;
  std::vector<std::shared_ptr<void>> mapped;
  std::vector<Ax::SharedFD> fds;
  Params() = default;
  Params(Params &&) = default;
  Params &operator=(Params &&) = default;
  Params(const Params &) = delete;
  Params &operator=(const Params &) = delete;
};
using input_event = std::variant<Ax::GstHandle<GstEvent>, Ax::GstHandle<GstBuffer>>;
} // namespace


struct GstAxInferenceImpl {
  Ax::InferenceProperties props;
  Ax::GstHandle<GstAllocator> allocator;
  Ax::GstHandle<GstAllocator> out_allocator;
  Ax::GstHandle<GstBufferPool> pool;
  Ax::Logger logger{ Ax::Severity::trace, nullptr, gst_axinference_debug };
  std::queue<Params> output_fifo;
  std::queue<input_event> input_fifo;
  std::unique_ptr<Ax::Inference> inference; // ensure inference is destroyed before the fifo

  int pipeline_pre_fill = 0;
  int output_drop = 0;
  bool remove_batch = true;
  std::set<int> stream_select;
};

static size_t
ax_size_from_caps(GstCaps *caps)
{
  return size_from_interface(interface_from_caps_and_meta(caps, nullptr));
}

static void
gst_axinference_set_property(
    GObject *object, guint property_id, const GValue *value, GParamSpec *pspec)
{
  auto self = GST_AXINFERENCE(object);
  GST_DEBUG_OBJECT(self, "set_propertyi%d", property_id);

  if (!Ax::set_inference_property(self->impl->props, property_id, value)) {

    switch (property_id) {
      case PROP_UNBATCH:
        self->impl->remove_batch = g_value_get_boolean(value);
        break;
      default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, property_id, pspec);
        break;
    }
  }
  GST_DEBUG_OBJECT(self, "set_propertyi%d end", property_id);
}

static void
gst_axinference_get_property(
    GObject *object, guint property_id, GValue *value, GParamSpec *pspec)
{
  auto self = GST_AXINFERENCE(object);
  if (!Ax::get_inference_property(self->impl->props, property_id, value)) {

    switch (property_id) {
      case PROP_UNBATCH:
        g_value_set_boolean(value, self->impl->remove_batch);
        break;
      default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, property_id, pspec);
        break;
    }
  }
}

//  Given a tensor, return the size of the unbatched tensor in bytes.
static size_t
unbatched_tensor_size(const AxTensorInterface &tensor)
{
  return tensor.total_bytes() / tensor.sizes[0];
}


static gboolean
gst_axinference_propose_allocation(
    GstBaseTransform *trans, GstQuery *decide_query, GstQuery *query)
{
  auto self = GST_AXINFERENCE(trans);
  GstCaps *caps;
  gboolean need_pool;
  gst_query_parse_allocation(query, &caps, &need_pool);

  if (!self->impl->props.dmabuf_inputs) {
    GST_DEBUG_OBJECT(self, "DMABUF disabled");
    return FALSE;
  }

  if (!caps) {
    GST_ERROR_OBJECT(self, "Allocation query without caps");
    return FALSE;
  }

  GST_DEBUG_OBJECT(self, "dmabuf device name: %s", dmabuf_device);
  self->impl->allocator = Ax::as_handle(gst_tensor_dmabuf_allocator_get(dmabuf_device));
  if (!self->impl->allocator) {
    GST_ERROR_OBJECT(self, "Unable to get dmabuf allocator");
    return FALSE;
  }

  if (need_pool) {
    self->impl->pool = Ax::as_handle(gst_buffer_pool_new());
    auto config = gst_buffer_pool_get_config(self->impl->pool.get());
    auto size = ax_size_from_caps(caps);

    /* we need at least 2 buffer because we hold on to the last one */
    gst_buffer_pool_config_set_params(config, caps, size, 4, 0);
    gst_buffer_pool_config_set_allocator(config, self->impl->allocator.get(), NULL);
    if (!gst_buffer_pool_set_config(self->impl->pool.get(), config)) {
      GST_ERROR_OBJECT(self, "Failed to set pool configuration");
      return FALSE;
    }
    /* we need at least 2 buffer because we hold on to the last one */
    gst_query_add_allocation_pool(query, self->impl->pool.get(), size, 4, 0);
  }

  gst_query_add_allocation_param(query, self->impl->allocator.get(), NULL);

  return TRUE;
}

static void
single_tensor_from_batch(GstBuffer *buffer, const AxTensorsInterface &model_outputs,
    std::vector<Ax::GstHandle<GstMemory>> &out_mems, int batch)
{
  for (auto i = 0; i < model_outputs.size(); i++) {
    auto tensor_size = unbatched_tensor_size(model_outputs[i]);
    auto offset = tensor_size * batch;
    auto *mem = gst_memory_share(out_mems[i].get(), offset, tensor_size);
    assert(mem != nullptr);
    gst_buffer_append_memory(buffer, mem);
  }
}


static Ax::GstHandle<GstMemory>
mem_handle(GstMemory *mem)
{
  return { mem, [](GstMemory *m) { gst_memory_unref(m); } };
}


static std::shared_ptr<void>
map_memory(GstMemory *mem, size_t expected_size, GstMapFlags flags)
{
  const auto input_output = flags == GST_MAP_READ ? "input" : "output";
  const auto read_write = flags == GST_MAP_READ ? "read" : "write";
  GstMapInfo minfo;
  if (!gst_memory_map(mem, &minfo, flags)) {
    throw std::runtime_error(
        std::string("Failed to map ") + input_output + " tensor for " + read_write);
  }
  gst_memory_ref(mem);
  auto p = std::shared_ptr<void>(minfo.data, [minfo](void *p) {
    GstMapInfo cpy = minfo;
    gst_memory_unmap(minfo.memory, &cpy);
    gst_memory_unref(minfo.memory);
  });
  if (expected_size && expected_size != minfo.size) {
    throw std::runtime_error(std::string(input_output) + " tensor invalid size ("
                             + std::to_string(minfo.size) + ") != expected ("
                             + std::to_string(expected_size) + ")");
  }
  return p;
}

static gboolean
process_sink_event(GstAxInference *self, GstEvent *event)
{
  auto *trans = (GstBaseTransform *) (self);
  return GST_BASE_TRANSFORM_CLASS(parent_class)->sink_event(trans, event);
}


static Params
push_inference(GstAxInference *self, GstBuffer *inbuf)
{
  Params inputs;
  for (auto i = 0; i < self->impl->inference->input_shapes().size(); i++) {
    auto mem = gst_buffer_get_memory(inbuf, i);
    inputs.mems.push_back(mem_handle(mem));
    if (gst_is_tensor_dmabuf_memory(mem)) {
      inputs.fds.emplace_back(std::make_shared<GstDmaBufHandle>(mem));
    } else {
      const auto expected = self->impl->inference->input_shapes()[i].total_bytes();
      inputs.mapped.push_back(map_memory(mem, expected, GST_MAP_READ));
    }
  }

  Params outputs;

  for (auto i = 0; i < self->impl->inference->output_shapes().size(); i++) {
    const auto sz = self->impl->inference->output_shapes()[i].total_bytes();
    auto *mem = gst_allocator_alloc(self->impl->out_allocator.get(), sz, NULL);
    if (!mem) {
      throw std::runtime_error("Cannot allocate memory for output buffer "
                               + std::to_string(i) + " (" + std::to_string(sz) + " bytes)");
    }
    outputs.mems.push_back(mem_handle(mem));
    if (self->impl->props.dmabuf_outputs) {
      outputs.fds.emplace_back(std::make_shared<GstDmaBufHandle>(mem));
    } else {
      outputs.mapped.push_back(map_memory(mem, 0, GST_MAP_WRITE));
    }
  }

  self->impl->inference->dispatch(
      inputs.mapped, inputs.fds, outputs.mapped, outputs.fds);
  return outputs;
}

static void
pop_inference(GstAxInference *self, GstBuffer *inbuf, GstBuffer *outbuf, Params &&outputs)
{
  gst_buffer_remove_all_memory(outbuf);
  if (!GST_BASE_TRANSFORM_GET_CLASS(self)->copy_metadata(
          GST_BASE_TRANSFORM(self), inbuf, outbuf)) {
    throw std::runtime_error("Failed to copy metadata to output buffer");
  }

  self->impl->inference->collect(outputs.mapped);
  // by this time the data should have been written to the output tensors so unmap them:
  outputs.mapped.clear();

  if (self->impl->remove_batch && self->impl->inference->batch_size() > 1) {
    auto *pad = GST_BASE_TRANSFORM_SRC_PAD(self);
    const auto batch_size = self->impl->inference->batch_size();
    for (auto batch = 0; batch != batch_size - 1; ++batch) {
      auto *buffer = gst_buffer_new();
      single_tensor_from_batch(
          buffer, self->impl->inference->output_shapes(), outputs.mems, batch);
      gst_pad_push(pad, buffer);
    }
    single_tensor_from_batch(outbuf, self->impl->inference->output_shapes(),
        outputs.mems, batch_size - 1);
  } else {
    for (auto i = 0; i < self->impl->inference->output_shapes().size(); i++) {
      // pass ownership of the memory to the buffer
      gst_buffer_append_memory(outbuf, outputs.mems[i].release());
    }
  }
}

Ax::GstHandle<GstBuffer>
process_input_events(GstAxInference *self)
{
  while (!self->impl->input_fifo.empty()) {
    auto input = Ax::pop_queue(self->impl->input_fifo);
    if (std::holds_alternative<Ax::GstHandle<GstBuffer>>(input)) {
      return std::move(std::get<Ax::GstHandle<GstBuffer>>(input));
    }
    auto &event = std::get<Ax::GstHandle<GstEvent>>(input);
    process_sink_event(self, event.release());
  }
  return {};
}

static GstFlowReturn
gst_axinference_transform(GstBaseTransform *trans, GstBuffer *inbuf, GstBuffer *outbuf)
{
  auto self = GST_AXINFERENCE(trans);
  try {
    const auto num_tensors = get_buffer_n_tensor(inbuf);
    if (num_tensors != self->impl->inference->input_shapes().size()) {
      throw std::runtime_error(
          "Expected " + std::to_string(self->impl->inference->input_shapes().size())
          + " input tensors but got " + std::to_string(num_tensors));
    }

    self->impl->input_fifo.push(as_buffer_handle(gst_buffer_ref(inbuf)));
    self->impl->output_fifo.push(push_inference(self, inbuf));

    if (self->impl->pipeline_pre_fill != 0) {
      GST_DEBUG_OBJECT(self, "pipeline_pre_fill %d", self->impl->pipeline_pre_fill);
      --self->impl->pipeline_pre_fill;
      return GST_BASE_TRANSFORM_FLOW_DROPPED;
    }
    auto outputs = Ax::pop_queue(self->impl->output_fifo);

    if (self->impl->output_drop != 0) {
      --self->impl->output_drop;
      // if are double buffered then we need to drop the output buffer because
      // actually the output for frame n is copied to the outputs given in frame n + 2
      self->impl->inference->collect(outputs.mapped);
      return GST_BASE_TRANSFORM_FLOW_DROPPED;
    }

    auto input = process_input_events(self);
    if (!input) {
      return GST_BASE_TRANSFORM_FLOW_DROPPED;
    }
    pop_inference(self, input.get(), outbuf, std::move(outputs));
    return GST_FLOW_OK;
  } catch (const std::exception &e) {
    GST_ERROR_OBJECT(trans, "Error: %s", e.what());
    return GST_FLOW_ERROR;
  }
}

static void
gst_axinference_init(GstAxInference *axinference)
{
  axinference->impl = new GstAxInferenceImpl;
}

static void
gst_axinference_finalize(GObject *object)
{
  auto self = GST_AXINFERENCE(object);
  delete self->impl;
  self->impl = nullptr;
  G_OBJECT_CLASS(gst_axinference_parent_class)->finalize(object);
}

static bool
validate_props(GstAxInference *self)
{
  if (self->impl->props.model.empty()) {
    GST_ERROR_OBJECT(self, "Missing model path");
    return false;
  }
  if (!g_file_test(self->impl->props.model.c_str(), G_FILE_TEST_IS_REGULAR)) {
    GST_ERROR_OBJECT(self, "Given file %s is not valid", self->impl->props.model.c_str());
    return false;
  }

  self->impl->pipeline_pre_fill = pipeline_pre_fill(self->impl->props);
  GST_DEBUG_OBJECT(self, "pipeline_pre_fill count: %d", self->impl->pipeline_pre_fill);
  self->impl->output_drop = output_drop(self->impl->props);
  GST_DEBUG_OBJECT(self, "output_drop count: %d", self->impl->output_drop);

  return true;
}

static bool
configure_instance(GstAxInference *self)
{
  if (self->impl->inference) {
    return true;
  }
  auto &props = self->impl->props;
  self->impl->inference = create_inference(self->impl->logger, props);
  if (auto alloc = props.dmabuf_outputs ? gst_tensor_dmabuf_allocator_get(dmabuf_device) :
                                          gst_aligned_allocator_get()) {
    self->impl->out_allocator = Ax::as_handle(alloc);
    return true;
  }
  GST_ERROR_OBJECT(self, "Unable to get %s allocator",
      self->impl->props.dmabuf_outputs ? "dmabuf" : "aligned");
  return true;
}

static gboolean
gst_axinference_transform_size(GstBaseTransform *trans, GstPadDirection direction,
    GstCaps *caps, gsize size, GstCaps *othercaps, gsize *othersize)
{
  *othersize = 0;
  return TRUE;
}

static AxTensorsInterface
remove_batch(const AxTensorsInterface &tensors, gboolean remove_batch)
{
  auto result = tensors;
  if (remove_batch) {
    std::transform(result.begin(), result.end(), result.begin(), [](auto t) {
      t.sizes[0] = 1;
      return t;
    });
  }
  return result;
}

static GstCaps *
gst_axinference_transform_caps(GstBaseTransform *trans,
    GstPadDirection direction, GstCaps *caps, GstCaps *filter)
{
  GstTensorsConfig in_config, out_config;
  auto self = GST_AXINFERENCE(trans);
  auto structure = gst_caps_get_structure(caps, 0);

  GST_DEBUG_OBJECT(self, "transform_caps %d", direction);

  auto *pad = direction == GST_PAD_SINK ? GST_BASE_TRANSFORM_SRC_PAD(trans) :
                                          GST_BASE_TRANSFORM_SINK_PAD(trans);

  if (!configure_instance(self)) {
    GST_ERROR_OBJECT(self, "Unable to configure instance");
    return NULL;
  }

  gst_tensors_config_from_structure(in_config, structure);

  /* set framerate from input config */
  out_config.rate_n = in_config.rate_n;
  out_config.rate_d = in_config.rate_d;

  const auto tensors = (direction == GST_PAD_SRC) ?
                           self->impl->inference->input_shapes() :
                           remove_batch(self->impl->inference->output_shapes(),
                               self->impl->remove_batch);

  for (auto it = tensors.begin(); it != tensors.end(); ++it) {
    const auto &tensor = *it;
    size_t i = std::distance(tensors.begin(), it);

    out_config.info.num_tensors = tensors.size();
    std::fill(out_config.info.info[i].dimension.begin(),
        out_config.info.info[i].dimension.end(), 1);
    std::copy(tensor.sizes.rbegin(), tensor.sizes.rend(),
        out_config.info.info[i].dimension.begin());

    // TODO: Support other types
    out_config.info.info[i].type
        = (tensor.bytes == 1) ? tensor_type::INT8 : tensor_type::FLOAT32;
  };

  auto *result = get_possible_pad_caps_from_config(pad, out_config);
  /* Update caps dimension for src pad cap */
  if (direction == GST_PAD_SINK) {
    if (auto *peer_caps = gst_pad_peer_query_caps(pad, NULL)) {
      update_tensor_dimensions(result, peer_caps);
      gst_caps_unref(peer_caps);
    }
  }

  if (filter && gst_caps_get_size(filter) > 0) {
    GstCaps *intersection;

    intersection = gst_caps_intersect_full(result, filter, GST_CAPS_INTERSECT_FIRST);

    gst_caps_unref(result);
    result = intersection;
  }

  return result;
}

/**
 * @brief set caps. required vmethod of GstBaseTransform.
 */
static gboolean
gst_axinference_set_caps(GstBaseTransform *trans, GstCaps *incaps, GstCaps *outcaps)
{
  auto self = GST_AXINFERENCE(trans);
  auto structure = gst_caps_get_structure(incaps, 0);
  GstTensorsConfig in_config;

  if (FALSE == validate_props(self)) {
    GST_ERROR_OBJECT(self, "Invalid props configuration");
    return FALSE;
  }

  gst_tensors_config_from_structure(in_config, structure);
  const auto &in_tensors = self->impl->inference->input_shapes();
  try {
    Ax::ensure_input_tensors_compatible(in_config, in_tensors);
  } catch (const std::exception &e) {
    GST_ERROR_OBJECT(self, "%s", e.what());
    return FALSE;
  }


  // TODO: Check type

  if (FALSE == validate_props(self)) {
    GST_ERROR_OBJECT(self, "Invalid props configuration");
    return FALSE;
  }

  return TRUE;
}

static GstCaps *
gst_axinference_fixate_caps(GstBaseTransform *trans, GstPadDirection direction,
    GstCaps *caps, GstCaps *othercaps)
{
  auto result = gst_axinference_transform_caps(trans, direction, caps, othercaps);
  gst_caps_unref(othercaps);

  result = gst_caps_make_writable(result);
  result = gst_caps_fixate(result);

  return result;
}

GstFlowReturn
gst_flush_eos_buffers(GstBaseTransform *trans)
{
  auto self = GST_AXINFERENCE(trans);
  GstFlowReturn result = GST_FLOW_OK;
  GstBuffer *inbuf, *outbuf;
  GstMemory *inmemory[max_tensor_count] = {};
  GstMemory *outmemory[max_tensor_count] = {};

  if (self->impl->pool && gst_buffer_pool_is_active(self->impl->pool.get())) {
    GST_DEBUG_OBJECT(self, "pool active");
    result = gst_buffer_pool_acquire_buffer(self->impl->pool.get(), &inbuf, NULL);
    if (G_UNLIKELY(GST_FLOW_OK != result)) {
      GST_ERROR_OBJECT(self, "Unable to acquire buffer from pool, error: %s",
          gst_flow_get_name(result));
      return result;
    }
  } else {
    if (self->impl->props.dmabuf_inputs) {
      self->impl->allocator
          = Ax::as_handle(gst_tensor_dmabuf_allocator_get(dmabuf_device));
    }
    GST_DEBUG_OBJECT(self, "pool inactive");
    inbuf = gst_buffer_new();
    const auto &in_tensors = self->impl->inference->input_shapes();
    for (size_t i = 0; i < in_tensors.size(); ++i) {
      const auto size = in_tensors[i].total_bytes();
      inmemory[i] = gst_allocator_alloc(self->impl->allocator.get(), size, NULL);
      GST_DEBUG_OBJECT(self, "allocate input tensor %zu with size: %zu", i, size);
      gst_buffer_append_memory(inbuf, inmemory[i]);
    }
  }

  auto num_to_flush
      = pipeline_pre_fill(self->impl->props) + output_drop(self->impl->props);

  for (size_t i = 0; i < num_to_flush; ++i) {
    outbuf = gst_buffer_new();
    GST_DEBUG_OBJECT(self, "pushing buffer id: %zu", i);
    result = gst_axinference_transform(trans, inbuf, outbuf);
    if (result == GST_FLOW_OK) {
      result = gst_pad_push(trans->srcpad, outbuf);
      if (G_UNLIKELY(GST_FLOW_OK != result)) {
        GST_ERROR_OBJECT(self, "Unable to push buffer %zu on EOS: %s", i,
            gst_flow_get_name(result));
      }
    }
  }
  gst_buffer_unref(inbuf);
  //  This simply ensures that input queue is empty and all buffers are freed.

  while (!self->impl->output_fifo.empty()) {
    auto outputs = Ax::pop_queue(self->impl->output_fifo);
    self->impl->inference->collect(outputs.mapped);
  }

  while (!self->impl->input_fifo.empty()) {
    Ax::pop_queue(self->impl->input_fifo);
  }

  self->impl->pipeline_pre_fill = pipeline_pre_fill(self->impl->props);
  self->impl->output_drop = output_drop(self->impl->props);
  return result;
}

static gboolean
gst_axinference_sink_event(GstBaseTransform *trans, GstEvent *event)
{
  auto self = GST_AXINFERENCE(trans);

  if (GST_EVENT_TYPE(event) == GST_EVENT_EOS) {
    GST_DEBUG_OBJECT(self, "entering EOS");
    gst_flush_eos_buffers(GST_BASE_TRANSFORM(self));
    GST_DEBUG_OBJECT(self, "EOS handling finished");
  } else if (GST_EVENT_TYPE(event) == GST_EVENT_GAP) {
    self->impl->input_fifo.push(as_event_handle(event));
    return TRUE;
  }
  return process_sink_event(self, event);
}

static GstFlowReturn
gst_axinference_prepare_output_buffer(
    GstBaseTransform *trans, GstBuffer *inbuf, GstBuffer **outbuf)
{
  *outbuf = gst_buffer_new();
  return GST_FLOW_OK;
}


static void
gst_axinference_class_init(GstAxInferenceClass *klass)
{
  GST_DEBUG_CATEGORY_INIT(gst_axinference_debug, "axinference", 0,
      "axinference to invoke neural network");

  auto trans_class = (GstBaseTransformClass *) klass;
  auto gstelement_class = (GstElementClass *) trans_class;
  auto gobject_class = (GObjectClass *) gstelement_class;

  gobject_class->set_property = gst_axinference_set_property;
  gobject_class->get_property = gst_axinference_get_property;
  gobject_class->finalize = gst_axinference_finalize;

  Ax::add_inference_properties(gobject_class, true, false);

  g_object_class_install_property(gobject_class, PROP_UNBATCH,
      g_param_spec_boolean("unbatch", "Unbatch", "Outputs unbatched tensors",
          TRUE, (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));


  gst_element_class_set_static_metadata(
      gstelement_class, "axinference", "Effect", "description", "axelera.ai");

  gst_element_class_set_details_simple(gstelement_class, "AxInference", "AxInference",
      "Runs TVM inference from GST pipeline", "<deniz.hasan@axelera.ai>");

  gst_element_class_add_pad_template(
      GST_ELEMENT_CLASS(klass), gst_static_pad_template_get(&src_template));
  gst_element_class_add_pad_template(
      GST_ELEMENT_CLASS(klass), gst_static_pad_template_get(&sink_template));

  trans_class->passthrough_on_same_caps = FALSE;
  trans_class->transform = GST_DEBUG_FUNCPTR(gst_axinference_transform);
  trans_class->fixate_caps = GST_DEBUG_FUNCPTR(gst_axinference_fixate_caps);
  trans_class->transform_caps = GST_DEBUG_FUNCPTR(gst_axinference_transform_caps);
  trans_class->set_caps = GST_DEBUG_FUNCPTR(gst_axinference_set_caps);
  trans_class->prepare_output_buffer
      = GST_DEBUG_FUNCPTR(gst_axinference_prepare_output_buffer);

  /* Allocation units */
  trans_class->transform_size = GST_DEBUG_FUNCPTR(gst_axinference_transform_size);
  trans_class->propose_allocation = GST_DEBUG_FUNCPTR(gst_axinference_propose_allocation);

  trans_class->sink_event = GST_DEBUG_FUNCPTR(gst_axinference_sink_event);
}
