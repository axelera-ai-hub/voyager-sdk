// Copyright Axelera AI, 2025
#include "AxInference.hpp"
#include <array>
#include <chrono>
#include <fstream>
#include <map>
#include <queue>
#include <string>
#include "AxStreamerUtils.hpp"

using namespace std::string_literals;
using namespace std::chrono_literals;

namespace
{
constexpr std::array<uint32_t, 256>
init_crc_table()
{
  std::array<uint32_t, 256> table;
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

class MockInference : public Ax::Inference
{
  public:
  MockInference(Ax::Logger &logger, const std::string &model,
      const std::string &shapes, int fps)
      : logger_(logger), path_(model), frame_duration_(1000000us / std::max(fps, 1))
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

  int batch_size() const override
  {
    return inputs_.front().sizes.front();
  }

  const AxTensorsInterface &input_shapes() const override
  {
    return inputs_;
  }

  const AxTensorsInterface &output_shapes() const override
  {
    return outputs_;
  }

  void dispatch(const std::vector<std::shared_ptr<void>> &input_ptrs,
      const std::vector<Ax::SharedFD> &input_fds,
      const std::vector<std::shared_ptr<void>> &output_ptrs,
      const std::vector<Ax::SharedFD> &output_fds) override
  {
    ready_time_ = std::chrono::steady_clock::now() + frame_duration_;
    assert(output_ptrs.size() == output_shapes().size());
    const auto crc = calculate_input_crc(inputs_, input_ptrs, input_fds);
    for (auto &&[n, ptr] : Ax::Internal::enumerate(output_ptrs)) {
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
  }

  void collect(const std::vector<std::shared_ptr<void>> &) override
  {
    std::this_thread::sleep_until(ready_time_);
  }

  private:
  Ax::Logger &logger_;
  AxTensorsInterface inputs_;
  AxTensorsInterface outputs_;
  std::string path_;
  std::chrono::time_point<std::chrono::steady_clock> ready_time_;
  std::chrono::microseconds frame_duration_;
};

class SaveInference : public Ax::Inference
{
  public:
  SaveInference(const std::string &path, std::unique_ptr<Ax::Inference> &&inference)
      : path_(path), inference_(std::move(inference))
  {
    std::string shapes;
    for (auto &&tensor : inference_->input_shapes()) {
      shapes += Ax::Internal::join(tensor.sizes, "x");
      shapes += ',';
    }
    for (auto &&tensor : inference_->output_shapes()) {
      shapes += Ax::Internal::join(tensor.sizes, "x");
      shapes += ',';
    }
    shapes.pop_back();
    mkdir(path_.c_str(), 0755);
    std::ofstream file(path_ + "/shapes.txt");
    file << shapes << std::endl;
  }

  int batch_size() const override
  {
    return inference_->input_shapes().front().sizes.front();
  }

  const AxTensorsInterface &input_shapes() const override
  {
    return inference_->input_shapes();
  }

  const AxTensorsInterface &output_shapes() const override
  {
    return inference_->output_shapes();
  }

  void dispatch(const std::vector<std::shared_ptr<void>> &input_ptrs,
      const std::vector<Ax::SharedFD> &input_fds,
      const std::vector<std::shared_ptr<void>> &output_ptrs,
      const std::vector<Ax::SharedFD> &output_fds) override
  {
    assert(output_ptrs.size() == output_shapes().size());
    crcs_.push(calculate_input_crc(input_shapes(), input_ptrs, input_fds));
    inference_->dispatch(input_ptrs, input_fds, output_ptrs, output_fds);
  }

  void collect(const std::vector<std::shared_ptr<void>> &output_ptrs) override
  {
    inference_->collect(output_ptrs);
    const auto crc = crcs_.front();
    crcs_.pop();
    // TODO output_fds not supported in mock
    for (auto &&[n, ptr] : Ax::Internal::enumerate(output_ptrs)) {
      const auto size = output_shapes()[n].total_bytes();
      std::ofstream file(path_ + "/" + output_name(crc, n), std::ios::binary);
      file.write(static_cast<char *>(ptr.get()), size);
    }
  }

  private:
  std::string path_;
  std::unique_ptr<Ax::Inference> inference_;
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

std::unique_ptr<Ax::Inference>
create_single_inference(Ax::Logger &logger, const Ax::InferenceProperties &props)
{
  auto opts = parse_options(logger, props.options);
  if (opts.contains("mock-load")) {
    const auto path = opts["mock-load"];
    auto shapes = opts["mock-shapes"];
    if (shapes.empty()) {
      std::ifstream file(path + "/shapes.txt");
      std::getline(file, shapes);
    }
    const auto fps = std::stoi(opts.contains("mock-fps") ? opts["mock-fps"] : "500");
    return std::make_unique<MockInference>(logger, path, shapes, fps);
  }
  if (opts.contains("mock-save")) {
    const auto path = opts["mock-save"];
    auto subprops = props;
    subprops.options.clear(); // strip any mock options
    auto inf = create_single_inference(logger, subprops);
    return std::make_unique<SaveInference>(path, std::move(inf));
  }
  return create_axruntime_inference(logger, props);
}


class MultiThreadedInference : public Ax::Inference
{
  public:
  MultiThreadedInference(Ax::Logger &logger, const Ax::InferenceProperties &props)
      : logger_(logger)
  {
    const auto num_threads = std::max(props.num_children, 1);
    auto devices = Ax::Internal::split(props.devices, ",");
    if (devices.empty()) {
      devices.push_back("");
    }
    for (auto n = 0; n != num_threads; ++n) {
      for (auto &&device : devices) {
        auto subprops = props;
        subprops.devices = device;
        logger_(AX_INFO)
            << "Creating instance " << n << " on device " << device << std::endl;
        inqs_.emplace_back(std::make_unique<Ax::BlockingQueue<Ax::InferenceParams>>());
        outqs_.emplace_back(std::make_unique<Ax::BlockingQueue<Ax::InferenceParams>>());
        instances_.push_back(create_single_inference(logger, subprops));
      }
    }
    logger_(AX_INFO) << "Done creating runtime instances, starting threads" << std::endl;
    for (auto n = size_t{ 0 }; n != instances_.size(); ++n) {
      threads_.emplace_back(&MultiThreadedInference::thread_func, this, n);
    }
  }

  int batch_size() const override
  {
    return instances_.front()->batch_size();
  }

  const AxTensorsInterface &input_shapes() const override
  {
    return instances_.front()->input_shapes();
  }

  const AxTensorsInterface &output_shapes() const override
  {
    return instances_.front()->output_shapes();
  }

  void dispatch(const std::vector<std::shared_ptr<void>> &input_ptrs,
      const std::vector<Ax::SharedFD> &input_fds,
      const std::vector<std::shared_ptr<void>> &output_ptrs,
      const std::vector<Ax::SharedFD> &output_fds) override
  {
    inqs_[next_available_child_]->push(
        Ax::InferenceParams(input_ptrs, input_fds, output_ptrs, output_fds));
    next_available_child_ = (next_available_child_ + 1) % threads_.size();
  }

  void collect(const std::vector<std::shared_ptr<void>> &) override
  {
    outqs_[next_ready_child_]->wait_one();
    next_ready_child_ = (next_ready_child_ + 1) % threads_.size();
  }

  ~MultiThreadedInference()
  {
    stop_queues(inqs_);
    stop_queues(outqs_);
    for (auto &thread : threads_) {
      thread.join();
    }
    logger_(AX_INFO) << "Inference threads joined" << std::endl;
  }

  private:
  void stop_queues(
      std::vector<std::unique_ptr<Ax::BlockingQueue<Ax::InferenceParams>>> &queues)
  {
    for (auto &&q : queues) {
      q->stop();
    }
  }

  void thread_func(size_t n)
  {
    auto &tvm = *instances_[n];
    auto &inq = *inqs_[n];
    auto &outq = *outqs_[n];

    while (1) {
      auto p = inq.wait_one();
      if (!p) {
        logger_(AX_INFO) << "Got shutdown event on thread  " << n << std::endl;
        return;
      }
      tvm.dispatch(p.input_ptrs, p.input_fds, p.output_ptrs, p.output_fds);
      tvm.collect(p.output_ptrs);
      outq.push(std::move(p));
    }
  }

  std::vector<std::unique_ptr<Ax::BlockingQueue<Ax::InferenceParams>>> inqs_;
  std::vector<std::unique_ptr<Ax::BlockingQueue<Ax::InferenceParams>>> outqs_;
  std::vector<std::unique_ptr<Ax::Inference>> instances_;
  std::vector<std::thread> threads_;
  Ax::Logger &logger_;
  size_t next_available_child_ = 0;
  size_t next_ready_child_ = 0;
};
} // namespace

Ax::InferenceParams
Ax::zero_params(const Ax::Inference &tvm, bool input_dmabuf, bool output_dmabuf)
{
  auto in_alloc = input_dmabuf ? Ax::create_dma_buf_allocator() :
                                 Ax::create_heap_allocator();
  auto out_alloc = output_dmabuf ? Ax::create_dma_buf_allocator() :
                                   Ax::create_heap_allocator();
  auto in = in_alloc->allocate(tvm.input_shapes());
  auto out = out_alloc->allocate(tvm.output_shapes());
  return InferenceParams(in.buffers(), in.fds(), out.buffers(), out.fds());
}

std::unique_ptr<Ax::Inference>
Ax::create_inference(Ax::Logger &logger, const InferenceProperties &props)
{
  return std::make_unique<MultiThreadedInference>(logger, props);
}
