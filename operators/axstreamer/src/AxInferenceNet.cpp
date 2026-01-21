// Copyright Axelera AI, 2026
#include "AxInferenceNet.hpp"
#include "AxDataInterface.h"
#include "AxInference.hpp"
#include "AxLog.hpp"
#include "AxMeta.hpp"
#include "AxMetaStreamId.hpp"
#include "AxOpenCl.hpp"
#include "AxStreamerUtils.hpp"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <unordered_map>
#include <vector>


namespace Ax
{

struct Frame : public CompletedFrame {
  SharedBatchBufferView op_input;
  bool skip_inference = false;
  int stream_id;
  int subframe_index = 0;
  int number_of_subframes = 1;
  time_point timestamp;
  time_point latency_start;
  time_point inference_start;

  // these are used in the low latency inference mode only
  uint64_t global_frame_idx{ 0 };
  bool inference_ready{ false };
  std::shared_ptr<BatchedBuffer> inf_batched_input;
  std::shared_ptr<BatchedBuffer> inf_batched_output;
};

using Buffers = std::list<ManagedDataInterface>;

class Operator
{
  public:
  //  Note: input will be consumed by the operator
  //  It might be returned in the output if the operator is in-place
  virtual SharedBatchBufferView execute(const AxVideoInterface &video,
      SharedBatchBufferView input, unsigned int subframe_index,
      unsigned int number_of_subframes, MetaMap &meta_map)
      = 0;

  virtual std::string name() const = 0;

  virtual AxDataInterface allocate_output(const AxDataInterface &input) = 0;

  virtual void set_allocator(int batch_size, DataInterfaceAllocator &) = 0;

  virtual bool supports_crop() const = 0;

  virtual void downstream_supports_crop(bool) = 0;

  virtual bool supports_opencl() const = 0;

  virtual void downstream_supports_opencl(bool) = 0;

  virtual bool consumes_frames() const = 0;

  virtual bool supports_dmabuf() const = 0;

  virtual ~Operator() = default;
};

class OperatorList
{
  public:
  explicit OperatorList(Logger &logger, LatencyCallback log_latency,
      Ax::DataInterfaceAllocator &default_alloc, Ax::DataInterfaceAllocator &null_alloc)
      : logger(logger), log_latency(log_latency), default_alloc(default_alloc),
        null_alloc(null_alloc)
  {
  }

  void add_operator(std::string libname, std::string options, AxAllocationContext *context,
      std::string mode = "none", std::string batch_size = "1");

  AxDataInterface compile(const AxDataInterface &input,
      std::function<void(std::unique_ptr<Ax::Frame>)> release_frame);

  void set_last_allocator(int batch_size, DataInterfaceAllocator &allocator)
  {
    operators.back()->set_allocator(batch_size, allocator);
  }

  BlockingQueue<std::unique_ptr<Ax::Frame>> &input_queue()
  {
    return links.front();
  }

  BlockingQueue<std::unique_ptr<Ax::Frame>> &output_queue()
  {
    return links.back();
  }

  void stop()
  {
    for (auto &q : links) {
      q.stop();
    }
  }

  void initialise()
  {
    auto first = operators.rbegin();
    auto last = operators.rend();
    if (first == last)
      return;

    bool supports_crop = false;
    bool supports_opencl = false;
    for (; first != last; ++first) {
      (*first)->downstream_supports_crop(supports_crop);
      (*first)->downstream_supports_opencl(supports_opencl);
      supports_crop = (*first)->supports_crop();
      supports_opencl = (*first)->supports_opencl();
    }
  }

  bool supports_opencl_buffers()
  {
    return !operators.empty() && operators.front()->supports_opencl();
  }

  private:
  void log_throughput(const std::string &op, time_point throughput_start,
      time_point latency_start);

  void operator_thread(int which, std::function<void(std::unique_ptr<Ax::Frame>)> release_frame);

  Logger &logger;
  LatencyCallback log_latency;
  Ax::DataInterfaceAllocator &default_alloc;
  Ax::DataInterfaceAllocator &null_alloc;
  std::vector<BlockingQueue<std::unique_ptr<Ax::Frame>>> links;
  std::vector<std::unique_ptr<Operator>> operators;
  struct OpCallParams {
    Operator *op;
  };
  std::vector<std::jthread> threads;
};

struct supported_features {
  bool opencl_buffers = false;
  bool dmabuffers = false;
};

supported_features
get_supported_features(Ax::Operator &op)
{
  return supported_features{
    .opencl_buffers = op.supports_opencl(),
    .dmabuffers = op.supports_dmabuf(),
  };
}

bool
is_opencl_buffer(std::shared_ptr<Ax::BatchedBuffer> input)
{
  return input->is_opencl();
}

bool
is_dmabuf_buffer(std::shared_ptr<Ax::BatchedBuffer> &input)
{
  return input->is_dmabuf();
}

bool
map(std::shared_ptr<Ax::BatchedBuffer> &input, supported_features supports, Ax::MapType type)
{
  if (supports.opencl_buffers && is_opencl_buffer(input)) {
    //  If the input is an opencl buffer then we do not need to map it
    //  We can use it directly in the plugin
    return false;
  }
  if (supports.dmabuffers && is_dmabuf_buffer(input)) {
    //  If the input is a dmabuf then we do not need to map it
    //  We can use it directly in the plugin
    return false;
  }
  //  The plugin uses system memory so let's map to the CPU
  input->map(type);
  return true;
}

bool
unmap(std::shared_ptr<Ax::BatchedBuffer> &input, supported_features supports)
{
  if (supports.opencl_buffers && is_opencl_buffer(input)) {
    //  If the input is an opencl buffer then we do not need to map it
    //  We can use it directly in the plugin
    return false;
  }
  if (supports.dmabuffers && is_dmabuf_buffer(input)) {
    //  If the input is a dmabuf then we do not need to map it
    //  We can use it directly in the plugin
    return false;
  }
  //  The plugin uses system memory so let's map to the CPU
  input->unmap();
  return true;
}

class AxInferenceNet : public InferenceNet
{
  public:
  AxInferenceNet(const InferenceNetProperties &properties,
      AxAllocationContext *context, Ax::Logger &logger,
      InferenceDoneCallback done_callback, LatencyCallback latency_callback);

  void push_new_frame(std::shared_ptr<void> &&buffer,
      const AxVideoInterface &video, MetaMap &axmetamap, int stream_id) override;
  void stop() override;
  void end_of_input() override;
  void cascade_frame(CompletedFrame &frame) override;
  bool supports_opencl_buffers(const AxVideoInterface &video) override;
  int frames_required_for_inference() const override;

  private:
  void release_frame(std::unique_ptr<Ax::Frame> frame);
  void init_frame(Ax::Frame &frame);
  std::unique_ptr<Ax::Frame> build_eos_frame(
      Frame *last_frame, int batch, std::shared_ptr<BatchedBuffer> batched);
  bool unbatch(std::queue<std::unique_ptr<Ax::Frame>> &pending_frames,
      std::shared_ptr<BatchedBuffer> out, int batch_size, int current_batch,
      const ManagedDataInterface &output);
  std::unique_ptr<Ax::Frame> next_frame(
      bool at_eos, const SharedBatchBufferView &last_good_input);

  struct Stream {
    std::atomic_uint64_t frame_id{ 0 };
  };

  void inference_thread(const int batch_size);
  void inference_thread_low_latency();
  void inference_low_latency_ready(uint64_t idx);
  std::unique_ptr<Frame> new_frame();
  void initialise_pipeline(const AxVideoInterface &video);
  void update_stream_latency(int which, std::chrono::microseconds latency);
  void update_frame_latency(Frame &frame, const char *label);
  void log_latency(const std::string &op, std::chrono::high_resolution_clock::time_point start);
  void finalize_thread();

  const InferenceNetProperties properties;
  AxAllocationContextHandle context_;
  Logger &logger;
  std::vector<std::unique_ptr<Frame>> frame_pool;
  std::mutex frame_pool_mutex;

  // note the order of destruction is really important here. will need to tidy
  // this up but the queues must be destroyed before the pools
  std::unique_ptr<DataInterfaceAllocator> inf_input_allocator;
  std::unique_ptr<DataInterfaceAllocator> inf_output_allocator;
  std::unique_ptr<DataInterfaceAllocator> allocator;
  std::unique_ptr<NullDataInterfaceAllocator> null_allocator;
  std::unique_ptr<BatchedBufferPool> inf_input_pool;
  std::unique_ptr<BatchedBufferPool> inf_output_pool;
  std::unordered_map<int, Stream> streams;
  std::mutex streams_mutex;
  std::once_flag compile_once_flag;
  InferenceDoneCallback done_callback;
  LatencyCallback latency_callback;
  int num_to_flush = 0;

  OperatorList pre_ops;
  std::unique_ptr<Inference> inference;
  OperatorList post_ops;

  std::vector<std::jthread> threads;
  std::jthread push_thread;
  // used in low-latency inference mode only
  std::atomic_uint64_t global_frame_idx{ 0 };
  std::mutex reorder_mutex;
  std::deque<std::unique_ptr<Ax::Frame>> reorder_queue;
};

class TransformOp : public Ax::Operator
{
  public:
  TransformOp(Ax::Logger &logger, Ax::SharedLib &&shared_, std::string libname,
      std::string options, AxAllocationContext *context, std::string mode,
      int batch_size, Ax::DataInterfaceAllocator &alloc)
      : plugin(logger, std::move(shared_), options, context, mode), logger(logger),
        allocator(&alloc), pool{ std::make_unique<Ax::BatchedBufferPool>(
                               batch_size, AxDataInterface{}, *allocator) },
        batch{ batch_size }
  {
  }

  void remove_cropinfo(AxDataInterface &out)
  {
    if (auto *video = std::get_if<AxVideoInterface>(&out)) {
      video->info.stride
          = video->info.width * AxVideoFormatNumChannels(video->info.format);
      video->strides = { size_t(video->info.stride) };
      video->info.cropped = false;
      video->info.x_offset = 0;
      video->info.y_offset = 0;
      video->info.actual_height = video->info.height;
    }
  }

  Ax::SharedBatchBufferView batch_output(std::shared_ptr<Ax::BatchedBuffer> output_buffer)
  {
    if (++current_batch != batch) {
      out_buf = std::move(output_buffer);
      return get_shared_view_of_batch_buffer(out_buf, 0);
    }
    current_batch = 0;
    out_buf.reset();
    unmap(output_buffer, supported_features{
                             .opencl_buffers = supports_opencl(),
                             .dmabuffers = supports_dmabuf(),
                         });
    return get_shared_view_of_batch_buffer(output_buffer, 0);
  }


  //  Should only be called if passthrough is possible
  AxDataInterface assign_input_buffers_to_output_buffers(
      const AxDataInterface &input, const AxDataInterface &out)
  {
    auto output = out;
    if (std::holds_alternative<AxVideoInterface>(input)) {
      auto *data = std::get<AxVideoInterface>(input).data;
      if (std::holds_alternative<AxVideoInterface>(output)) {
        auto &out_video = std::get<AxVideoInterface>(output);
        out_video.data = data;
      } else if (std::holds_alternative<AxTensorsInterface>(output)) {
        auto &out_tensors = std::get<AxTensorsInterface>(output);
        out_tensors[0].data = data;
      }
      return output;
    }
    if (std::holds_alternative<AxVideoInterface>(output)) {
      auto &out_video = std::get<AxVideoInterface>(output);
      out_video.data = std::get<AxTensorsInterface>(input)[0].data;
    } else if (std::holds_alternative<AxTensorsInterface>(output)) {
      auto &out_tensors = std::get<AxTensorsInterface>(output);
      auto &in_tensors = std::get<AxTensorsInterface>(input);
      for (size_t i = 0; i < in_tensors.size(); ++i) {
        out_tensors[i].data = in_tensors[i].data;
      }
    }
    return output;
  }

  std::shared_ptr<Ax::BatchedBuffer> allocate_batched_buffer(const AxDataInterface &input)
  {
    if (out_buf) {
      return out_buf;
    }
    auto interface = input;
    if (auto *p = std::get_if<AxVideoInterface>(&interface)) {
      p->info.stride = p->info.width * AxVideoFormatNumChannels(p->info.format);
      p->strides = { size_t(p->info.stride) };
    }
    auto buffer = pool->new_batched_buffer(input);
    auto mapped = map(buffer,
        supported_features{
            .opencl_buffers = supports_opencl(),
            .dmabuffers = supports_dmabuf(),
        },
        Ax::MAP_WRITE_ONLY);
    return buffer;
  }

  Ax::SharedBatchBufferView execute(const AxVideoInterface &video,
      Ax::SharedBatchBufferView input, unsigned int subframe_index,
      unsigned int number_of_subframes, Ax::MetaMap &meta_map) override
  {
    (void) video;
    if (!input) {
      if (out_buf) {
        //  We have an output buffer from a previous batched operation
        //  We need to unmap it.
        unmap(out_buf, supported_features{
                           .opencl_buffers = supports_opencl(),
                           .dmabuffers = supports_dmabuf(),
                       });
      }
      return input;
    }

    auto out = plugin.set_output_interface(*input);
    if (number_of_subframes == 0) {
      if (plugin.has_set_output_interface_from_meta()) {
        out = plugin.set_output_interface_from_meta(
            *input, subframe_index, number_of_subframes, meta_map);
      }
      remove_cropinfo(out);
      auto output_buffer = out_buf ? out_buf : allocate_batched_buffer(out);
      auto output = get_shared_view_of_batch_buffer(output_buffer, current_batch);
      return batch_output(output.underlying());
    }
    if (plugin.has_set_output_interface_from_meta()) {
      out = plugin.set_output_interface_from_meta(
          *input, subframe_index, number_of_subframes, meta_map);
      if (std::holds_alternative<AxVideoInterface>(out)) {
        auto &out_video = std::get<AxVideoInterface>(out);
        if (downstream_supports_cropmeta && out_video.info.cropped) {
          auto in = input.underlying();
          in->set_iface(out);
          //  We either always add crop metadata or we don't
          //  In which case we do not need to worry about double buffering
          return input;
        }
      }
      remove_cropinfo(out);
    }

    if (batch == 1 && plugin.can_passthrough(*input, out)) {
      //  If we get here we can passthrough the input to the output with just moidified caps
      out = assign_input_buffers_to_output_buffers(*input, out);
      auto in = input.underlying();
      in->set_iface(out);
      auto output = input;
      return output;
    }

    //  Here we check if output comes from meta and if so get the output
    //  interface Otherwise it comes from set_output_interface
    if (std::holds_alternative<AxVideoInterface>(out)) {
      auto &out_video = std::get<AxVideoInterface>(out);
      if (downstream_supports_cropmeta && out_video.info.cropped) {
        auto in = input.underlying();
        in->set_iface(out);
        //  We either always add crop metadata or we don't
        //  In which case we do not need to worry about double buffering
        return input;
      }
      //  Remove cropping metadata if we need to physically crop here.
      remove_cropinfo(out);
    }
    //  Here we might need to create a new output buffer, if so we either need
    //  to create one from the pool or an OpenCL buffer if the plugin supports OpenCL
    auto input_base = input.underlying();
    map(input_base, get_supported_features(*this), Ax::MAP_READ_ONLY);
    auto output_buffer = allocate_batched_buffer(out);
    auto output = get_shared_view_of_batch_buffer(output_buffer, current_batch);
    plugin.transform(*input, *output, subframe_index, number_of_subframes, meta_map);
    return batch_output(std::move(output_buffer));
  }

  AxDataInterface allocate_output(const AxDataInterface &input) override
  {
    auto output = plugin.set_output_interface(input);
    if (batch != 1) {
      if (!std::holds_alternative<AxTensorsInterface>(output)) {
        throw std::runtime_error("Batched output must be tensor");
      }
      auto &tensors = std::get<AxTensorsInterface>(output);
      tensors[0].sizes[0] = batch;
    }
    return output;
  }

  void set_allocator(int batch_size, Ax::DataInterfaceAllocator &alloc) override
  {
    allocator = &alloc;
    pool = std::make_unique<Ax::BatchedBufferPool>(batch_size, AxDataInterface{}, *allocator);
  }

  std::string name() const override
  {
    return plugin.name();
  }

  bool supports_opencl() const
  {
    return plugin.query_supports(PluginFeature::opencl_buffers);
  }

  bool supports_dmabuf() const
  {
    return plugin.query_supports(PluginFeature::dmabuf_buffers);
  }

  bool supports_crop() const override
  {
    return plugin.query_supports(PluginFeature::crop_meta);
  }

  void downstream_supports_crop(bool supports) override
  {
    downstream_supports_cropmeta = supports;
  }

  virtual void downstream_supports_opencl(bool supports) override
  {
    plugin.set_dynamic_properties(
        { { "downstream_supports_opencl", supports ? "1" : "0" } });
  }

  bool consumes_frames() const override
  {
    return false;
  }

  private:
  LoadedTransform plugin;
  Ax::Logger &logger;
  Ax::DataInterfaceAllocator *allocator;
  std::unique_ptr<Ax::BatchedBufferPool> pool;
  bool downstream_supports_cropmeta = false;
  bool downstream_supports_opencl_buffers = false;
  std::shared_ptr<Ax::BatchedBuffer> out_buf;
  int batch = 1;
  int current_batch = 0;
};


class InplaceOp : public Ax::Operator
{
  public:
  InplaceOp(Ax::Logger &logger, Ax::SharedLib &&shared, std::string libname,
      std::string options, AxAllocationContext *context, std::string mode)
      : plugin(logger, std::move(shared), options, context, mode), logger(logger)
  {
  }

  Ax::SharedBatchBufferView execute(const AxVideoInterface &video,
      Ax::SharedBatchBufferView input, unsigned int subframe_index,
      unsigned int number_of_subframes, Ax::MetaMap &meta_map) override
  {
    (void) video; // not used
    if (!input || number_of_subframes == 0) {
      return input;
    }
    auto input_base = input.underlying();
    if (plugin.mode() != "meta") {
      map(input_base, get_supported_features(*this), Ax::MAP_READ_WRITE);
    }
    plugin.inplace(*input, subframe_index, number_of_subframes, meta_map);
    unmap(input_base, supported_features{
                          .opencl_buffers = supports_opencl(),
                          .dmabuffers = supports_dmabuf(),
                      });
    return input;
  }


  AxDataInterface allocate_output(const AxDataInterface &input) override
  {
    return input;
  }

  std::string name() const override
  {
    return plugin.name();
  }

  void set_allocator(int /*batch_size*/, Ax::DataInterfaceAllocator &) override
  {
  }

  bool supports_crop() const override
  {
    return false;
  }

  void downstream_supports_crop(bool) override
  {
  }

  void downstream_supports_opencl(bool) override
  {
  }

  bool consumes_frames() const override
  {
    return false;
  }

  bool supports_opencl() const
  {
    return false;
  }

  bool supports_dmabuf() const
  {
    return false;
  }

  private:
  LoadedInPlace plugin;
  Ax::Logger &logger;
};

class DecodeOp : public Ax::Operator
{
  public:
  DecodeOp(Ax::Logger &logger, Ax::SharedLib &&shared, std::string libname,
      std::string options, AxAllocationContext *context, std::string mode,
      Ax::DataInterfaceAllocator &alloc)
      : plugin(logger, std::move(shared), options, context, mode), logger(logger),
        allocator(alloc), pool{ std::make_unique<Ax::BatchedBufferPool>(
                              1, AxDataInterface{}, allocator) }
  {
  }

  Ax::SharedBatchBufferView execute(const AxVideoInterface &video,
      Ax::SharedBatchBufferView input, unsigned int subframe_index,
      unsigned int number_of_subframes, Ax::MetaMap &meta_map) override
  {
    if (!input || number_of_subframes == 0) {
      return input;
    }
    auto input_base = input.underlying();
    map(input_base, get_supported_features(*this), Ax::MAP_READ_ONLY);
    auto &tensor = std::get<AxTensorsInterface>(*input);
    auto out = pool->new_batched_buffer(video);
    if (number_of_subframes != 0) {
      plugin.decode_to_meta(tensor, subframe_index, number_of_subframes, meta_map, video);
    }
    return number_of_subframes == 0 || number_of_subframes == subframe_index + 1 ?
               get_shared_view_of_batch_buffer(out, 0) :
               Ax::SharedBatchBufferView{};
  }

  AxDataInterface allocate_output(const AxDataInterface &input) override
  {
    return {};
  }
  std::string name() const override
  {
    return plugin.name();
  }

  void set_allocator(int /*batch_size*/, Ax::DataInterfaceAllocator &) override
  {
  }

  bool supports_crop() const override
  {
    return false;
  }

  void downstream_supports_crop(bool) override
  {
  }

  void downstream_supports_opencl(bool) override
  {
  }

  bool consumes_frames() const override
  {
    return true;
  }

  bool supports_opencl() const
  {
    return false;
  }

  bool supports_dmabuf() const
  {
    return false;
  }

  private:
  LoadedDecode plugin;
  Ax::Logger &logger;
  Ax::DataInterfaceAllocator &allocator;
  std::unique_ptr<Ax::BatchedBufferPool> pool;
};

static AxDataInterface
strip_pointers(AxDataInterface in)
{
  // return a copy of the output interface without the ->data pointers, as
  // this represents the output 'shape', not the actual data interface
  if (auto *video = std::get_if<AxVideoInterface>(&in)) {
    video->data = nullptr;
  } else if (auto *tensors = std::get_if<AxTensorsInterface>(&in)) {
    for (auto &tensor : *tensors) {
      tensor.data = nullptr;
    }
  }
  return in;
}

using time_point = Ax::time_point;

auto
as_duration(time_point start, time_point end)
{
  return std::chrono::duration_cast<std::chrono::microseconds>(end - start);
}

auto
as_duration_ns(time_point start, time_point end)
{
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
}

auto
duration_since(time_point start)
{
  return as_duration(start, std::chrono::high_resolution_clock::now());
}

auto
duration_since_ns(time_point start)
{
  return as_duration_ns(start, std::chrono::high_resolution_clock::now());
}


void
Ax::OperatorList::operator_thread(
    int which, std::function<void(std::unique_ptr<Ax::Frame>)> release_frame)
{
  auto &input_queue = links[which];
  auto &output_queue = links[which + 1];
  auto &op = *operators[which];
  auto previous_frame = std::unique_ptr<Ax::Frame>{};
  while (true) {
    auto frame = input_queue.wait_one();
    if (!frame) {
      return;
    }
    auto now = std::chrono::high_resolution_clock::now();
    frame->latency_start = now;
    static Ax::MetaMap dummy_meta_map{};
    auto &meta_map = frame->meta_map ? *frame->meta_map : dummy_meta_map;
    if (frame->end_of_input) {
      //  Here we just flush any pending frames and then send on the
      //  EOS/Gap frame
      auto out = op.execute({}, {}, 0, 1, meta_map);
      log_throughput(op.name(), now, frame->latency_start);
      if (out) {
        previous_frame->op_input = std::move(out);
        output_queue.push({ std::move(previous_frame) });
      }
      frame->op_input = {};
      output_queue.push(std::move(frame));
    } else {
      const auto &video = frame->video;
      auto subframe_index = frame->subframe_index;
      auto number_of_subframes = frame->number_of_subframes;
      auto out = op.execute(video, std::move(frame->op_input), subframe_index,
          number_of_subframes, meta_map);
      log_throughput(op.name(), now, frame->latency_start);
      if (out) {
        if (op.consumes_frames()) {
          //  After the decoder we can reset subframes
          frame->subframe_index = 0;
          frame->number_of_subframes = 1;
        }
        if (previous_frame) {
          std::swap(frame, previous_frame);
        }
        frame->op_input = std::move(out);
        output_queue.push({ std::move(frame) });
      } else if (op.consumes_frames()) {
        release_frame(std::move(frame));
      } else {
        previous_frame = std::move(frame);
      }
    }
  }
}

void
OperatorList::add_operator(std::string libname, std::string options,
    AxAllocationContext *context, std::string mode /*= "none"*/, std::string batch_size)
{
  auto batch_size_int = batch_size.empty() ? 1 : std::stoi(batch_size);
  Ax::SharedLib shared(logger, libname);
  if (shared.has_symbol("transform")) {
    operators.push_back(std::make_unique<TransformOp>(logger, std::move(shared),
        libname, options, context, mode, batch_size_int, default_alloc));
  } else if (shared.has_symbol("inplace")) {
    operators.push_back(std::make_unique<InplaceOp>(
        logger, std::move(shared), libname, options, context, mode));
  } else if (shared.has_symbol("decode_to_meta")) {
    operators.push_back(std::make_unique<DecodeOp>(
        logger, std::move(shared), libname, options, context, mode, null_alloc));
  } else {
    throw std::runtime_error("Unknown module " + libname);
  }
}


AxDataInterface
Ax::OperatorList::compile(const AxDataInterface &input,
    std::function<void(std::unique_ptr<Ax::Frame>)> release_frame)
{
  auto size = operators.size();
  std::vector<BlockingQueue<std::unique_ptr<Ax::Frame>>> queues(size + 1);
  links = std::move(queues);
  AxDataInterface in = input;
  for (int i = 0; i != operators.size(); ++i) {
    auto out = operators[i]->allocate_output(in);
    threads.emplace_back(std::jthread(&OperatorList::operator_thread, this, i, release_frame));
    in = out;
  }
  return strip_pointers(in);
}

void
Ax::OperatorList::log_throughput(
    const std::string &op, time_point throughput_start, time_point latency_start)
{
  log_latency(op, duration_since_ns(throughput_start).count(),
      duration_since_ns(latency_start).count());
}

AxInferenceNet::AxInferenceNet(const Ax::InferenceNetProperties &properties,
    AxAllocationContext *context, Ax::Logger &logger,
    InferenceDoneCallback done_callback, LatencyCallback latency_callback)
    : properties(properties),
      context_(context ? ax_utils::clone_context(context) : AxAllocationContextHandle()),
      logger(logger),
      allocator(context_ ? create_opencl_allocator(context_.get(), logger) :
                           create_heap_allocator()),
      null_allocator(std::make_unique<NullDataInterfaceAllocator>()),
      done_callback(done_callback), latency_callback(latency_callback),
      pre_ops(logger, latency_callback, *allocator, *null_allocator),
      inference(create_inference(logger, properties,
          [this](uint64_t idx) { inference_low_latency_ready(idx); })),
      post_ops(logger, latency_callback, *allocator, *null_allocator)
{
  logger(AX_INFO) << "InferenceNet created, low_latency=" << inference->is_low_latency()
                  << std::endl;
}

void
AxInferenceNet::update_frame_latency(Ax::Frame &frame, const char *label)
{
  if (!frame.end_of_input) {
    auto now = std::chrono::high_resolution_clock::now();
    auto start = std::exchange(frame.latency_start, now);
    log_latency(label, start);
  }
}

bool
AxInferenceNet::unbatch(std::queue<std::unique_ptr<Frame>> &pending_frames,
    std::shared_ptr<Ax::BatchedBuffer> out, int batch_size, int current_batch,
    const Ax::ManagedDataInterface &output)
{
  auto eos = false;
  //  If we are here and have not collected a full batch we are at end of
  //  stream and need to forward whatever frames we have (and no more)
  out->map(Ax::MAP_READ_ONLY);
  auto num_frames = current_batch != 0 ? current_batch : batch_size;
  for (int n = 0; n != num_frames && !eos;) {
    auto out_frame = Ax::pop_queue(pending_frames);
    out_frame->op_input = get_shared_view_of_batch_buffer(out, n++);
    eos = out_frame->end_of_input;
    update_frame_latency(*out_frame, "Inference latency");
    if (out_frame->buffer_handle || eos) {
      //  Do not forward manufactured frames
      post_ops.input_queue().push(std::move(out_frame));
    }
  }
  return eos;
}

struct params {
  std::shared_ptr<Ax::BatchedBuffer> input;
  std::shared_ptr<Ax::BatchedBuffer> output;
};

// This delivers frames from the queue whilst we are not at EOS
//  Once at EOS it will deliver enough frames to flush any pending buffers out
//  of the end Of the queue. It will then block until the queue terminates.
//
std::unique_ptr<Ax::Frame>
AxInferenceNet::next_frame(bool at_eos, const Ax::SharedBatchBufferView &last_good_input)
{
  std::unique_ptr<Ax::Frame> fr;
  if (at_eos && num_to_flush != 0) {
    --num_to_flush;
    fr = new_frame();
    fr->op_input = last_good_input;
    fr->end_of_input = num_to_flush == 0;
    fr->frame_id = -2;
    fr->timestamp = std::chrono::high_resolution_clock::now();
    fr->latency_start = std::chrono::high_resolution_clock::now();
  } else {
    //  We are not at EOS so we can just get the next frame
    fr = pre_ops.output_queue().wait_one();
  }
  if (fr) {
    fr->global_frame_idx = global_frame_idx++;
    fr->inference_ready = false;
  }
  return fr;
}

void
AxInferenceNet::log_latency(
    const std::string &op, std::chrono::high_resolution_clock::time_point start)
{
  auto now = std::chrono::high_resolution_clock::now();
  auto latency = as_duration_ns(start, now);
  latency_callback(op, latency.count(), latency.count());
}

static void
bump_inferences_count(Frame &frame)
{
  if (frame.meta_map != nullptr) {
    auto it = frame.meta_map->find("stream_id");
    if (it != frame.meta_map->end() && it->second) {
      auto *stream_id_meta = dynamic_cast<AxMetaStreamId *>(it->second.get());
      stream_id_meta->inference_count++;
    }
  }
}

void
AxInferenceNet::inference_thread(const int batch_size)
{
  std::queue<std::unique_ptr<Frame>> pending_frames;
  std::queue<params> pending_params;

  auto num_to_drop = output_drop(properties);
  auto pre_fill = pipeline_pre_fill(properties);
  bool at_eos = false;
  auto this_batched_input = inf_input_pool->new_batched_buffer();
  auto last_good_input = get_shared_view_of_batch_buffer(this_batched_input, 0);
  int current_batch = 0;

  auto features_supported = supported_features{
    .opencl_buffers = false,
    .dmabuffers = true,
  };
  while (true) {
    auto current = std::chrono::high_resolution_clock::now();
    auto frame = next_frame(at_eos, last_good_input);
    if (!frame) {
      return;
    }
    if (frame->end_of_input) {
      if (!at_eos) {
        at_eos = true;
        //  We need to send enough frames so that all frames currently in the inference
        //  pipeline are flushed out + enough to fill the current batch
        num_to_flush = 1 + batch_size * (output_drop(properties) + pipeline_pre_fill(properties))
                       + (batch_size - current_batch);
        log_latency("inference", current);
        continue;
      }
      post_ops.input_queue().push(std::move(frame));
      log_latency("inference", current);
      continue;
    }
    if (current_batch == 0) {
      //  We use the input from the first frane of the batch
      last_good_input = frame->op_input;
    }
    auto batched_input = last_good_input.underlying();
    bump_inferences_count(*frame);
    pending_frames.push(std::move(frame));
    if (++current_batch != batched_input->batch_size()) {
      log_latency("inference", current);
      continue;
    }
    current_batch = 0;
    auto batched_output = inf_output_pool->new_batched_buffer();
    const auto &input = batched_input->get_batched();
    const auto &output = batched_output->get_batched();
    map(batched_input, features_supported, Ax::MAP_READ_ONLY);
    map(batched_output, features_supported, Ax::MAP_WRITE_ONLY);
    pending_params.push({ batched_input, batched_output });
    const auto gidx = uint64_t{ global_frame_idx++ }; // unused in high-latency mode
    inference->dispatch(
        { input.buffers(), input.fds(), output.buffers(), output.fds(), gidx });
    if (pre_fill) {
      --pre_fill;
      log_latency("inference", current);
      continue;
    }

    auto [op_input, inf_output] = Ax::pop_queue(pending_params);
    if (num_to_drop) {
      --num_to_drop;
      inference->collect();
      unmap(inf_output, features_supported);
      log_latency("inference", current);
      continue;
    }
    inference->collect();
    unmap(inf_output, features_supported);
    log_latency("inference", current);
    //  This can potentially block, so we do not want to include it in the latency
    auto eos = unbatch(pending_frames, inf_output, batch_size, current_batch,
        inf_output->get_batched());
    if (eos) {
      Ax::clear_queue(pending_frames);
      Ax::clear_queue(pending_params);
      break;
    }
  }
}

void
AxInferenceNet::inference_thread_low_latency()
{

  auto in_features_supported = supported_features{
    .opencl_buffers = false,
    .dmabuffers = true,
  };
  auto out_features_supported = supported_features{
    .opencl_buffers = false,
    .dmabuffers = true,
  };
  // This is an optimised version of the inference thread where we know that
  // the batch size is 1 and double buffering is disabled. This allows us to
  // avoid some of the overhead of the normal inference thread
  //
  // In this mode the approach is a bit different.  We will not do any prefill
  // and instead of checking for output being ready in this thread we will
  // receive the inference ready callback and stick it in the inf out queue
  while (true) {
    auto frame = pre_ops.output_queue().wait_one();
    if (!frame) {
      return;
    }
    const auto idx = frame->global_frame_idx = global_frame_idx++;
    frame->inference_ready = false;
    if (frame->end_of_input) {
      auto moved_frame = frame.get();
      post_ops.input_queue().push(std::move(frame));
      log_latency("inference", moved_frame->timestamp);
      break;
    }

    frame->inference_start = std::chrono::high_resolution_clock::now();
    frame->inf_batched_input = frame->op_input.underlying();
    frame->inf_batched_output = inf_output_pool->new_batched_buffer();
    const auto &input = frame->inf_batched_input->get_batched();
    const auto &output = frame->inf_batched_output->get_batched();
    map(frame->inf_batched_input, in_features_supported, Ax::MAP_READ_ONLY);
    map(frame->inf_batched_output, out_features_supported, Ax::MAP_WRITE_ONLY);
    {
      std::unique_lock lock(reorder_mutex);
      reorder_queue.push_back(std::move(frame));
    }
    inference->dispatch(
        { input.buffers(), input.fds(), output.buffers(), output.fds(), idx });
  }
}

void
AxInferenceNet::inference_low_latency_ready(uint64_t idx)
{
  // This is called whenever a frame's inference is done, but it is called from
  // any thread and we do not know the order that frames will arrive. So we need
  // to queue up out of order frames. (strictly speaking we could actually
  // continue with post ops out of order, but this might confuse some downstream
  // operators that expect frames to be in order) So we use a priority queue.
  // The queue is implicitly ordered by the global_frame_idx so the front item
  // in the queue is the one with the lowest idx. If it is ready we can emit it.
  // e.g. in our reorder_queue we might have:
  //     frame 0 not ready
  //     frame 1 ready
  //     frame 2 not ready
  //     frame 3 not ready
  // then if frame 3 ready arrives we do nothing
  // then if frame 0 ready arrives we emit 0 and 1

  std::vector<std::unique_ptr<Ax::Frame>> inferenced;
  {
    std::unique_lock lock(reorder_mutex);
    // First mark the now ready frame as ready.
    auto new_ready_frame = std::find_if(reorder_queue.begin(), reorder_queue.end(),
        [idx](const auto &frame) { return frame->global_frame_idx == idx; });
    if (new_ready_frame != reorder_queue.end()) {
      (*new_ready_frame)->inference_ready = true;
    } else {
      logger(AX_ERROR) << "New ready frame not found in reorder_queue list, idx: " << idx
                       << std::endl;
      assert(0);
      return;
    }
    // Then check repeatedly if the lowest id is ready
    while (!reorder_queue.empty() && reorder_queue.front()->inference_ready) {
      //  We have a frame that is ready, so we can process it
      auto frame = Ax::pop_queue(reorder_queue);
      // q insertion can potentially block, so we do not want to include it in
      // the latency log_latency("inference", current);
      frame->op_input = get_shared_view_of_batch_buffer(frame->inf_batched_output, 0);

      update_frame_latency(*frame, "Inference latency");
      inferenced.push_back(std::move(frame));
    }
  }
  for (auto &f : inferenced) {
    bump_inferences_count(*f);
    f->inf_batched_output->map(Ax::MAP_READ_ONLY);
    // note I don't call inference->collect here, because it is a no-op
    post_ops.input_queue().push(std::move(f));
  }
}


void
AxInferenceNet::finalize_thread()
{
  while (true) {
    auto frame = post_ops.output_queue().wait_one();
    if (!frame) {
      return;
    }
    frame->subframe_index = 0;
    frame->number_of_subframes = 1;
    update_frame_latency(*frame, "Postprocessing latency");
    if (!frame->end_of_input) {
      log_latency("Total latency", frame->timestamp);
    }
    done_callback(*frame);
    release_frame(std::move(frame));
  }
}

void
AxInferenceNet::stop()
{
  pre_ops.stop();
  post_ops.stop();
  threads.clear();
  push_thread.join();
  inference.reset();
  // all threads are joined now, so no need to lock reorder queue or frame pool
  // but it is important to ensure they are empty before destruction
  reorder_queue.clear();
  frame_pool.clear();
}

std::unique_ptr<Ax::Frame>
AxInferenceNet::new_frame()
{
  std::unique_lock<std::mutex> lock(frame_pool_mutex);
  if (frame_pool.empty()) {
    return std::make_unique<Frame>();
  }
  auto frame = std::move(frame_pool.back());
  frame_pool.pop_back();
  return frame;
}

void
AxInferenceNet::release_frame(std::unique_ptr<Ax::Frame> frame)
{
  frame->stream_id = 0;
  frame->frame_id = 0;
  frame->buffer_handle.reset();
  frame->video.data = nullptr;
  frame->meta_map = nullptr;
  frame->op_input.reset();
  frame->end_of_input = false;
  std::unique_lock<std::mutex> lock(frame_pool_mutex);
  frame_pool.push_back(std::move(frame));
}

void
AxInferenceNet::init_frame(Ax::Frame &frame)
{
  AxDataInterface in{ frame.video };
  auto input = std::make_shared<Ax::BatchedBuffer>(1, in, *null_allocator);
  frame.op_input = get_shared_view_of_batch_buffer(input, 0);
}

static AxMetaBase &
get_meta(const MetaMap &meta_map, const std::string &key,
    const std::string &source, const std::string &extra_err = {})
{
  auto meta_itr = meta_map.find(key);
  if (meta_itr == meta_map.end()) {
    std::string valid_keys;
    for (const auto &pair : meta_map) {
      if (!valid_keys.empty()) {
        valid_keys += ",";
      }
      valid_keys += pair.first;
    }
    throw std::runtime_error(
        source + ": " + key + " not found in meta map " + valid_keys + extra_err);
  }
  return *meta_itr->second;
}


static size_t
get_number_of_subframes(const MetaMap &axmetamap, const std::string &meta_to_distribute)
{
  if (!meta_to_distribute.empty()) {
    auto &meta = get_meta(axmetamap, meta_to_distribute, "filter", ", cannot distribute");
    return meta.get_number_of_subframes();
  }
  return 1;
}

void
AxInferenceNet::push_new_frame(std::shared_ptr<void> &&buffer_handle,
    const AxVideoInterface &video, MetaMap &axmetamap, int stream_id)
{
  std::atomic<uint64_t> *counter = nullptr;
  {
    std::unique_lock<std::mutex> lock(streams_mutex);
    counter = &streams[stream_id].frame_id;
  }
  auto frame_id = counter->fetch_add(1, std::memory_order_relaxed);

  std::call_once(compile_once_flag, [this, video] { initialise_pipeline(video); });
  auto frame = new_frame();
  frame->buffer_handle = buffer_handle;
  frame->video = video;
  frame->meta_map = &axmetamap;
  frame->stream_id = stream_id;
  frame->frame_id = frame_id;
  if (properties.skip_stride > 1 && properties.skip_count > 0) {
    const auto reverse_index_in_slice
        = properties.skip_stride - (frame->frame_id % properties.skip_stride) - 1;
    frame->skip_inference = reverse_index_in_slice < properties.skip_count;
  }
  frame->end_of_input = false;
  frame->timestamp = std::chrono::high_resolution_clock::now();
  frame->latency_start = frame->timestamp;
  frame->inference_start = frame->timestamp;
  auto &preq = pre_ops.input_queue();
  const auto num = get_number_of_subframes(axmetamap, properties.meta);
  //  It is one or more subframes
  frame->subframe_index = 0;
  frame->number_of_subframes = num;
  init_frame(*frame);
  preq.push(std::move(frame));
  for (int n = 1; n < num; ++n) {
    auto subframe = new_frame();
    subframe->buffer_handle = buffer_handle;
    subframe->video = video;
    subframe->meta_map = &axmetamap;
    subframe->stream_id = stream_id;
    subframe->frame_id = frame_id;
    subframe->subframe_index = n;
    subframe->number_of_subframes = num;
    subframe->end_of_input = false;
    subframe->timestamp = std::chrono::high_resolution_clock::now();
    subframe->latency_start = subframe->timestamp;
    subframe->inference_start = subframe->timestamp;
    init_frame(*subframe);
    preq.push(std::move(subframe));
  }
}

void
AxInferenceNet::cascade_frame(CompletedFrame &frame)
{
  if (frame.end_of_input) {
    end_of_input();
  } else {
    auto &video = frame.video;
    auto *meta_map = frame.meta_map;
    push_new_frame(std::move(frame.buffer_handle), video, *meta_map, frame.stream_id);
  }
}

void
AxInferenceNet::initialise_pipeline(const AxVideoInterface &video)
{
  auto compile_list
      = [this](const decltype(properties.preproc) &props, OperatorList &list) {
          for (int n = 0; n != MAX_OPERATORS && !props[n].lib.empty(); ++n) {
            logger(AX_INFO) << "Adding operator: " << props[n].lib << "("
                            << props[n].options << ", " << props[n].mode << ", "
                            << props[n].batch << ")" << std::endl;
            list.add_operator(props[n].lib, props[n].options, context_.get(),
                props[n].mode, props[n].batch);
          }
          list.initialise();
        };

  compile_list(properties.preproc, pre_ops);
  compile_list(properties.postproc, post_ops);
  ManagedDataInterfaces buffers;
  auto releaser = std::function<void(std::unique_ptr<Ax::Frame>)>(
      [this](std::unique_ptr<Ax::Frame> frame) {
        return release_frame(std::move(frame));
      });
  auto inf_input_template = pre_ops.compile(video, releaser);
  const auto exp_model_input
      = Ax::to_string(AxDataInterface(inference->input_shapes()));
  const auto got_model_input = Ax::to_string(inf_input_template);
  const auto batch_size = inference->batch_size();
  if (exp_model_input != got_model_input) {
    throw std::runtime_error("Expected model input=" + exp_model_input
                             + " but got input=" + got_model_input);
  }
  auto inf_output_template = inference->output_shapes();
  post_ops.compile(Ax::batch_view(inf_output_template, 0), releaser);

  inf_input_allocator = properties.dmabuf_inputs ?
                            create_dma_buf_allocator() :
                            create_opencl_allocator(context_.get(), logger);
  inf_output_allocator = properties.dmabuf_outputs ?
                             create_dma_buf_allocator() :
                             create_opencl_allocator(context_.get(), logger);
  inf_input_pool = std::make_unique<BatchedBufferPool>(
      batch_size, inf_input_template, *inf_input_allocator);
  inf_output_pool = std::make_unique<BatchedBufferPool>(
      batch_size, inf_output_template, *inf_output_allocator);

  if (inference->is_low_latency()) {
    assert(batch_size == 1);
    threads.emplace_back(&AxInferenceNet::inference_thread_low_latency, this);
  } else {
    threads.emplace_back(&AxInferenceNet::inference_thread, this, batch_size);
  }
  push_thread = std::jthread(&AxInferenceNet::finalize_thread, this);
  pre_ops.set_last_allocator(batch_size, *inf_input_allocator);
}

void
AxInferenceNet::end_of_input()
{
  auto frame = new_frame();
  frame->video = AxVideoInterface{};
  frame->meta_map = nullptr;
  frame->stream_id = 0;
  frame->frame_id = -2;
  frame->buffer_handle.reset();
  frame->op_input.reset();
  frame->end_of_input = true;
  frame->timestamp = std::chrono::high_resolution_clock::now();
  frame->latency_start = std::chrono::high_resolution_clock::now();
  pre_ops.input_queue().push(std::move(frame));
}

bool
AxInferenceNet::supports_opencl_buffers(const AxVideoInterface &video)
{
  std::call_once(compile_once_flag, [this, video] { initialise_pipeline(video); });
  return pre_ops.supports_opencl_buffers();
}

int
AxInferenceNet::frames_required_for_inference() const
{
  return inference->batch_size();
}

void
default_latency_callback(const std::string &, uint64_t, uint64_t)
{
}

} // namespace Ax

std::unique_ptr<Ax::InferenceNet>
Ax::create_inference_net(const InferenceNetProperties &properties,
    Ax::Logger &logger, InferenceDoneCallback done_callback)
{
  return create_inference_net(properties, logger, done_callback, {}, nullptr);
}

std::unique_ptr<Ax::InferenceNet>
Ax::create_inference_net(const InferenceNetProperties &properties, Ax::Logger &logger,
    InferenceDoneCallback done_callback, LatencyCallback latency_callback)
{
  return create_inference_net(properties, logger, done_callback, latency_callback, nullptr);
}

std::unique_ptr<Ax::InferenceNet>
Ax::create_inference_net(const InferenceNetProperties &properties,
    Ax::Logger &logger, InferenceDoneCallback done_callback,
    LatencyCallback latency_callback, AxAllocationContext *allocation_context)
{
  auto lcb = latency_callback ? latency_callback : default_latency_callback;
  return std::make_unique<AxInferenceNet>(
      properties, allocation_context, logger, done_callback, lcb);
}

static bool
as_bool(const std::string &s)
{
  return s == "1" || s == "true" || s == "True";
}

Ax::InferenceNetProperties
Ax::read_inferencenet_properties(std::istream &s, Ax::Logger &logger)
{
  InferenceNetProperties props;
  std::string line;
  while (std::getline(s, line)) {
    if (line.empty()) {
      continue;
    }
    auto pos = line.find('=');
    if (pos == std::string::npos) {
      logger(AX_WARN) << "Invalid line in InferenceNetProperties: " << line << std::endl;
      continue;
    }

    const auto key = std::string{ line.substr(0, pos) };
    const auto value = std::string{ line.substr(pos + 1) };
    try {
      if (key == "model") {
        props.model = value;
      } else if (key == "double_buffer") {
        props.double_buffer = as_bool(value);
      } else if (key == "dmabuf_inputs") {
        props.dmabuf_inputs = as_bool(value);
      } else if (key == "dmabuf_outputs") {
        props.dmabuf_outputs = as_bool(value);
      } else if (key == "num_children") {
        props.num_children = std::stoi(value);
      } else if (key == "inference_skip_rate") {
        const auto rate = Ax::parse_skip_rate(value);
        props.skip_stride = rate.stride;
        props.skip_count = rate.count;
      } else if (key == "options") {
        props.options = value;
      } else if (key == "meta") {
        props.meta = value;
      } else if (key == "devices") {
        props.devices = value;
      } else if (key.starts_with("preprocess") || key.starts_with("postprocess")) {
        auto *ops = key.starts_with("preprocess") ? props.preproc : props.postproc;
        const auto under = key.find("_");
        const auto prefixlen = key.starts_with("preprocess") ? 10 : 11;
        const auto num = std::stoi(key.substr(prefixlen, under - prefixlen));
        if (under == std::string::npos || num >= MAX_OPERATORS) {
          logger(AX_WARN) << "Invalid operator keyname in InferenceNetProperties: " << key
                          << std::endl;
          continue;
        }
        const auto subkey = key.substr(under + 1);
        if (subkey == "lib") {
          ops[num].lib = value;
        } else if (subkey == "options") {
          ops[num].options = value;
        } else if (subkey == "mode") {
          ops[num].mode = value;
        } else if (subkey == "batch") {
          ops[num].batch = value;
        } else {
          logger(AX_WARN) << "Invalid operator subkey in InferenceNetProperties: " << key
                          << std::endl;
        }
      } else {
        logger(AX_WARN) << "Invalid key in InferenceNetProperties: " << key << std::endl;
      }
    }

    catch (const std::exception &e) {
      logger(AX_ERROR) << "Failed to convert value for " << key << " : "
                       << e.what() << std::endl;
    }
  }
  return props;
}

Ax::InferenceNetProperties
Ax::read_inferencenet_properties(const std::string &path, Ax::Logger &logger)
{
  std::ifstream f(path);
  if (!f) {
    throw std::runtime_error("Failed to open file: " + path);
  }
  return read_inferencenet_properties(f, logger);
}
