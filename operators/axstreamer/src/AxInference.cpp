// Copyright Axelera AI, 2024
#include "AxInference.hpp"
#include <array>
#include <axruntime/axruntime.hpp>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <map>
#include <queue>
#include <regex>
#include <string>
#include "AxStreamerUtils.hpp"

using namespace std::string_literals;
using namespace std::chrono_literals;

namespace
{
constexpr std::array<uint32_t, 256>
init_crc_table()
{
  std::array<uint32_t, 256> table{};
  for (uint32_t i = 0; i < 256; ++i) {
    uint32_t crc = i;
    for (uint32_t j = 0; j < 8; ++j) {
      if (crc & 1) {
        crc = (crc >> 1) ^ 0xEDB88320;
      } else {
        crc >>= 1;
      }
    }
    table[i] = crc;
  }
  return table;
}
constexpr auto crc_table = init_crc_table();

uint32_t
start_crc()
{
  return 0xFFFFFFFF;
}

uint32_t
update_crc(const void *data, size_t size, uint32_t crc)
{
  const auto *buf = static_cast<const uint8_t *>(data);
  const auto *end = buf + size;
  for (; buf != end; ++buf) {
    crc = (crc >> 8) ^ crc_table[(crc ^ *buf) & 0xFF];
  }
  return crc ^ 0xFFFFFFFF;
}
uint32_t
finalize_crc(uint32_t crc)
{
  return crc ^ 0xFFFFFFFF;
}

std::vector<std::byte>
read_dmabuf(int fd, size_t size)
{
  auto *mapped_mem = mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
  if (mapped_mem == MAP_FAILED) {
    throw std::runtime_error("MockInference: Failed to mmap dmabuf: "s + strerror(errno));
  }
  const auto *data = static_cast<const std::byte *>(mapped_mem);
  auto buf = std::vector<std::byte>(data, data + size);
  ::munmap(mapped_mem, size);
  return buf;
}

uint32_t
calculate_input_crc(const AxTensorsInterface &inputs,
    const std::vector<std::shared_ptr<void>> &input_ptrs,
    const std::vector<Ax::SharedFD> &input_fds)
{
  assert(input_ptrs.size() == inputs.size() || input_fds.size() == inputs.size());

  auto crc = start_crc();
  for (auto &&[n, ptr] : Ax::Internal::enumerate(input_ptrs)) {
    crc = update_crc(ptr.get(), inputs[n].total_bytes(), crc);
  }
  for (auto &&[n, fd] : Ax::Internal::enumerate(input_fds)) {
    const auto size = inputs[n].total_bytes();
    const auto buf = read_dmabuf(fd->fd, size);
    crc = update_crc(buf.data(), size, crc);
  }
  return finalize_crc(crc);
}

std::string
output_name(uint32_t crc, int param_num)
{
  std::string name = "12345678_00.bin";
  std::snprintf(name.data(), name.size() + 1, "%08x_%02d.bin", crc, param_num);
  return name;
}
std::tuple<AxTensorsInterface, AxTensorsInterface> get_shapes_from_model(axrModel *model);

class MockInference : public Ax::BasicInference
{
  public:
  MockInference(Ax::Logger &logger, const std::string &model,
      const std::string &shapes, int fps)
      : logger_(logger),
        path_(model),
        frame_duration_(1000000us / std::max(fps, 1))
  {
    for (auto &&i : Ax::Internal::split(shapes, ',')) {
      AxTensorInterface tensor{};
      tensor.bytes = 1;
      for (auto &&n : Ax::Internal::split(i, 'x')) {
        tensor.sizes.push_back(std::stoi(std::string{ n }));
      }
      auto &dest = inputs_.empty() ? inputs_ : outputs_;
      dest.push_back(tensor);
    }
  }

  Ax::InferenceParams execute(Ax::InferenceParams p) override
  {
    ready_time_ = std::chrono::steady_clock::now() + frame_duration_;
    assert(p.output_ptrs.size() == outputs_.size());
    const auto crc = calculate_input_crc(inputs_, p.input_ptrs, p.input_fds);
    for (auto &&[n, ptr] : Ax::Internal::enumerate(p.output_ptrs)) {
      std::ifstream file(path_ + "/" + output_name(crc, n), std::ios::binary);
      const auto size = outputs_[n].total_bytes();
      if (file) {
        file.read(static_cast<char *>(ptr.get()), size);
      } else {
        logger_(AX_INFO) << "MOCK_LOAD cache miss: " << path_ << "/"
                         << output_name(crc, n) << std::endl;
        auto *data = static_cast<std::byte *>(ptr.get());
        std::fill(data, data + size, std::byte{ 0xa0 });
      }
    }

    std::this_thread::sleep_until(ready_time_);
    return p;
  }

  private:
  Ax::Logger &logger_;
  AxTensorsInterface inputs_;
  AxTensorsInterface outputs_;
  std::string path_;
  std::chrono::time_point<std::chrono::steady_clock> ready_time_;
  std::chrono::microseconds frame_duration_;
};

class SaveInference : public Ax::BasicInference
{
  public:
  SaveInference(const std::string &path, axrModel *model,
      std::unique_ptr<Ax::BasicInference> &&inference)
      : path_(path),
        inference_(std::move(inference))
  {
    std::tie(inputs_, outputs_) = get_shapes_from_model(model);
    std::string shapes;
    for (auto &&tensor : inputs_) {
      shapes += Ax::Internal::join(tensor.sizes, "x");
      shapes += ',';
    }
    for (auto &&tensor : outputs_) {
      shapes += Ax::Internal::join(tensor.sizes, "x");
      shapes += ',';
    }
    shapes.pop_back();
    std::filesystem::create_directories(path_);
    std::ofstream file(path_ + "/shapes.txt");
    file << shapes << std::endl;
  }

  Ax::InferenceParams execute(Ax::InferenceParams p) override
  {
    crcs_.push(calculate_input_crc(inputs_, p.input_ptrs, p.input_fds));

    auto q = inference_->execute(std::move(p));

    const auto crc = Ax::pop_queue(crcs_);
    // TODO output_fds not supported in mock
    for (auto &&[n, ptr] : Ax::Internal::enumerate(q.output_ptrs)) {
      const auto size = outputs_[n].total_bytes();
      std::ofstream file(path_ + "/" + output_name(crc, n), std::ios::binary);
      file.write(static_cast<char *>(ptr.get()), size);
    }
    return q;
  }

  private:
  std::string path_;
  AxTensorsInterface inputs_;
  AxTensorsInterface outputs_;
  std::unique_ptr<Ax::BasicInference> inference_;
  std::queue<uint32_t> crcs_;
};

std::map<std::string, std::string>
parse_options(Ax::Logger &logger, const std::string &options)
{
  std::map<std::string, std::string> m;
  for (const auto &opt : Ax::Internal::split(options, ';')) {
    const auto colon = opt.find(':');
    if (colon == std::string::npos) {
      logger(AX_ERROR) << "Badly formatted options - " << opt << "." << std::endl;
      continue;
    }
    m[std::string{ opt.substr(0, colon) }] = opt.substr(colon + 1);
  }
  return m;
}


std::unique_ptr<Ax::BasicInference>
maybe_create_mock_inference(Ax::Logger &logger, const Ax::InferenceProperties &props)
{
  auto opts = parse_options(logger, props.options);
  if (!opts.contains("mock-load")) {
    return {};
  }
  const auto path = opts["mock-load"];
  auto shapes = opts["mock-shapes"];
  if (shapes.empty()) {
    std::ifstream file(path + "/shapes.txt");
    std::getline(file, shapes);
  }
  const auto fps = std::stoi(opts.contains("mock-fps") ? opts["mock-fps"] : "500");
  return std::make_unique<MockInference>(logger, path, shapes, fps);
}

std::unique_ptr<Ax::BasicInference>
create_single_inference(Ax::Logger &logger, axrContext *ctx, axrModel *model,
    const Ax::InferenceProperties &props)
{
  auto opts = parse_options(logger, props.options);
  auto mock = maybe_create_mock_inference(logger, props);
  if (mock) {
    return mock;
  }
  if (opts.contains("mock-save")) {
    const auto path = opts["mock-save"];
    auto subprops = props;
    subprops.options.clear(); // strip any mock options
    auto inf = create_single_inference(logger, ctx, model, subprops);
    return std::make_unique<SaveInference>(path, model, std::move(inf));
  }
  return create_axruntime_inference(logger, ctx, model, props);
}

AxTensorInterface
to_axtensorinfo(const axrTensorInfo &info)
{
  AxTensorInterface tensor;
  tensor.sizes.assign(info.dims, info.dims + info.ndims);
  tensor.bytes = info.bits / 8;
  return tensor;
}

std::tuple<AxTensorsInterface, AxTensorsInterface>
get_shapes_from_model(axrModel *model)
{
  auto inputs = axr_num_model_inputs(model);
  AxTensorsInterface input_shapes;
  AxTensorsInterface output_shapes;
  for (int n = 0; n != inputs; ++n) {
    input_shapes.push_back(to_axtensorinfo(axr_get_model_input(model, n)));
  }
  auto outputs = axr_num_model_outputs(model);
  for (int n = 0; n != outputs; ++n) {
    output_shapes.push_back(to_axtensorinfo(axr_get_model_output(model, n)));
  }
  return { input_shapes, output_shapes };
}

void
log(void *arg, axrLogLevel level, const char *msg)
{
  auto &logger = *static_cast<Ax::Logger *>(arg);
  const auto ax_level = static_cast<Ax::Severity>(level);
  const auto tag = Ax::SeverityTag{ ax_level, {}, 0, {} };
  logger(tag) << msg << std::endl;
}

// Conversion from gst debug level to axr log level
static const axrLogLevel gst_levels[] = {
  AXR_LOG_ERROR, // (pad with error just to make array line up with gst values)
  AXR_LOG_ERROR, // 1=ERROR
  AXR_LOG_WARNING, // 2=WARNING
  AXR_LOG_FIXME, // 3=FIXME
  AXR_LOG_INFO, // 4=INFO
  AXR_LOG_DEBUG, // 5=DEBUG
  AXR_LOG_LOG, // 6=LOG
  AXR_LOG_TRACE, // 7=TRACE
};

axrLogLevel
read_gst_debug_level(const std::string &gst_debug)
{
  auto level = AXR_LOG_WARNING;
  for (auto &debug : Ax::Internal::split(gst_debug, ',')) {
    const auto colon = debug.find(':');
    if (colon == std::string::npos) {
      continue;
    }
    // convert wildcard to regex
    const auto expr = std::regex_replace(
        std::string(debug.substr(0, colon)), std::regex("\\*"), ".*");
    if (std::regex_match("axinference", std::regex(expr))) {
      const auto nlevel = std::stoi(std::string(debug.substr(colon + 1)));
      if (nlevel >= 0 && nlevel < std::size(gst_levels)) {
        level = gst_levels[nlevel];
      }
    }
  }
  return level;
}

axr::ptr<axrContext>
create_context(Ax::Logger &logger)
{
#if __SANITIZE_ADDRESS__
  // Disable implicit address sanitizer for subprocesses (e.g. the riscv toolchain)
  ::unsetenv("LD_PRELOAD");
#endif
  auto ctx = axr::to_ptr(axr_create_context());
  auto level = read_gst_debug_level(getenv("GST_DEBUG") ? getenv("GST_DEBUG") : "");
  axr_set_logger(ctx.get(), level, log, &logger);
  return ctx;
}

class Executor
{
  public:
  virtual void thread_func(size_t n, Ax::BasicInference &instance) = 0;
  virtual void stop() = 0;
  virtual void dispatch(Ax::InferenceParams params) = 0;
  virtual void collect() = 0;
  virtual ~Executor() = default;
};

class RoundRobinExecutor : public Executor
{
  public:
  explicit RoundRobinExecutor(size_t num_instances)
      : inqs_(num_instances),
        outqs_(num_instances)
  {
  }


  void dispatch(Ax::InferenceParams params) override
  {
    inqs_[next_available_child_].push(std::move(params));
    next_available_child_ = (next_available_child_ + 1) % inqs_.size();
  }

  void collect() override
  {
    outqs_[next_ready_child_].wait_one();
    next_ready_child_ = (next_ready_child_ + 1) % outqs_.size();
  }

  void stop() override
  {
    stop_queues(inqs_);
    stop_queues(outqs_);
  }

  private:
  void stop_queues(std::vector<Ax::BlockingQueue<Ax::InferenceParams>> &queues)
  {
    for (auto &&q : queues) {
      q.stop();
    }
  }

  void thread_func(size_t n, Ax::BasicInference &instance)
  {
    auto &inq = inqs_[n];
    auto &outq = outqs_[n];
    while (auto p = inq.wait_one()) {
      outq.push(instance.execute(std::move(p)));
    }
  }
  std::vector<Ax::BlockingQueue<Ax::InferenceParams>> inqs_;
  std::vector<Ax::BlockingQueue<Ax::InferenceParams>> outqs_;
  size_t next_available_child_ = 0;
  size_t next_ready_child_ = 0;
};


class LowLatencyExecutor : public Executor
{
  public:
  explicit LowLatencyExecutor(Ax::InferenceReadyCallback callback)
      : inq_(4),
        callback_(std::move(callback))
  {
  }


  void dispatch(Ax::InferenceParams params) override
  {
    inq_.push(std::move(params));
  }

  void collect() override
  {
  }

  void stop() override
  {
    inq_.stop();
  }

  void thread_func(size_t n, Ax::BasicInference &instance) override
  {
    while (auto p = inq_.wait_one()) {
      auto frame_id = p.frame_id;
      instance.execute(std::move(p));
      callback_(frame_id);
    }
  }

  // TODO we need to empirally measure some other queue sizes here:
  Ax::BlockingQueue<Ax::InferenceParams> inq_;
  Ax::InferenceReadyCallback callback_;
};

class MultiThreadedInference : public Ax::Inference
{
  public:
  MultiThreadedInference(Ax::Logger &logger,
      const Ax::InferenceProperties &props, Ax::InferenceReadyCallback callback)
      : logger_(logger),
        context_(create_context(logger)),
        model_(axr_load_model(context_.get(), props.model.c_str()))
  {
    if (!model_) {
      throw std::runtime_error("Failed to load model: "s
                               + axr_last_error_string(AXR_OBJECT(context_.get())));
    }
    std::tie(input_shapes_, output_shapes_) = get_shapes_from_model(model_.get());

    const auto num_cores = std::max(props.num_children, 1);
    auto devices = Ax::Internal::split(props.devices, ",");
    if (devices.empty()) {
      devices.push_back("");
    }
    for (auto n = 0; n != num_cores; ++n) {
      for (auto &&device : devices) {
        auto subprops = props;
        subprops.devices = device;
        subprops_.emplace_back(subprops);
      }
    }
    is_low_latency_ = callback && !props.double_buffer
                      && input_shapes_.front().sizes.front() == 1;
    if (is_low_latency_) {
      executor_ = std::make_unique<LowLatencyExecutor>(callback);
    } else {
      executor_ = std::make_unique<RoundRobinExecutor>(subprops_.size());
    }
    first_thread_ = std::jthread([this]() { init_instances(); });
  }

  void init_instances()
  {
    for (const auto &[n, subprop] : Ax::Internal::enumerate(subprops_)) {
      logger_(AX_INFO) << "Creating instance " << n << " on device "
                       << subprop.devices << std::endl;
      instances_.push_back(
          create_single_inference(logger_, context_.get(), model_.get(), subprop));
    }
    logger_(AX_INFO) << "Done creating runtime instances" << std::endl;
    for (auto n = size_t{ 1 }; n != instances_.size(); ++n) {
      jthreads_.emplace_back(
          [this, n]() { executor_->thread_func(n, *instances_[n]); });
    }
    executor_->thread_func(0, *instances_[0]);
  }

  bool is_low_latency() const override
  {
    return is_low_latency_;
  }

  int batch_size() const override
  {
    return input_shapes_.front().sizes[0];
  }

  const AxTensorsInterface &input_shapes() const override
  {
    return input_shapes_;
  }

  const AxTensorsInterface &output_shapes() const override
  {
    return output_shapes_;
  }

  void dispatch(Ax::InferenceParams params) override
  {
    executor_->dispatch(std::move(params));
  }

  void collect() override
  {
    executor_->collect();
  }

  ~MultiThreadedInference()
  {
    executor_->stop();
    first_thread_.join();
    jthreads_.clear();
    instances_.clear();
  }

  private:
  Ax::Logger &logger_;
  axr::ptr<axrContext> context_;
  axr::ptr<axrModel> model_;
  std::vector<Ax::InferenceProperties> subprops_;
  std::vector<std::unique_ptr<Ax::BasicInference>> instances_;
  std::jthread first_thread_;
  std::vector<std::jthread> jthreads_;
  AxTensorsInterface input_shapes_;
  AxTensorsInterface output_shapes_;
  std::unique_ptr<Executor> executor_;
  bool is_low_latency_ = false;
};
} // namespace

std::unique_ptr<Ax::Inference>
Ax::create_inference(Ax::Logger &logger, const InferenceProperties &props,
    InferenceReadyCallback callback)
{
  return std::make_unique<MultiThreadedInference>(logger, props, callback);
}
