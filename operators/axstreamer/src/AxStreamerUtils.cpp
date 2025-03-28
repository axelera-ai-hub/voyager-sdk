// Copyright Axelera AI, 2025
#include "AxStreamerUtils.hpp"

#include "AxOpUtils.hpp"

#include <algorithm>
#include <string>
#include <string_view>
#include <vector>
#include "AxDataInterface.h"
#include "AxInferenceNet.hpp"
#include "AxLog.hpp"

#define AX_VIDEO_FORMATS(AX_VIDEO_FORMAT_REGISTER) \
  AX_VIDEO_FORMAT_REGISTER(RGB, 3)                 \
  AX_VIDEO_FORMAT_REGISTER(RGBA, 4)                \
  AX_VIDEO_FORMAT_REGISTER(RGBx, 4)                \
  AX_VIDEO_FORMAT_REGISTER(BGR, 3)                 \
  AX_VIDEO_FORMAT_REGISTER(BGRA, 4)                \
  AX_VIDEO_FORMAT_REGISTER(BGRx, 4)                \
  AX_VIDEO_FORMAT_REGISTER(GRAY8, 1)               \
  AX_VIDEO_FORMAT_REGISTER(NV12, 3)                \
  AX_VIDEO_FORMAT_REGISTER(I420, 3)                \
  AX_VIDEO_FORMAT_REGISTER(YUY2, 3)

#define AX_GENERATE_NUM_VIDEO_CHANNELS(x, y) \
  case AxVideoFormat::x:                     \
    return y;

int
AxVideoFormatNumChannels(AxVideoFormat format)
{
  switch (format) {
    AX_VIDEO_FORMATS(AX_GENERATE_NUM_VIDEO_CHANNELS)
    case AxVideoFormat::UNDEFINED:
      throw std::runtime_error("Format undefined.");
  }
  throw std::runtime_error("Format not registered.");
}

#define MAP_STRING_TO_AX_VIDEO_FORMAT(x, y) \
  if (#x == format)                         \
    return AxVideoFormat::x;

AxVideoFormat
AxVideoFormatFromString(const std::string &format)
{
  AX_VIDEO_FORMATS(MAP_STRING_TO_AX_VIDEO_FORMAT)
  throw std::runtime_error("Format not registered.");
}

#define MAP_AX_VIDEO_FORMAT_TO_STRING(x, y) \
  case AxVideoFormat::x:                    \
    return #x;

std::string
AxVideoFormatToString(AxVideoFormat format)
{
  switch (format) {
    AX_VIDEO_FORMATS(MAP_AX_VIDEO_FORMAT_TO_STRING)
    case AxVideoFormat::UNDEFINED:
      return "UNDEFINED";
  }
  assert(!"The enum AxVideoFormat is not one of the known values");
  return "UNDEFINED";
}

std::vector<std::string_view>
Ax::Internal::split(std::string_view s, char delim)
{
  std::vector<std::string_view> result;

  auto first = std::begin(s);
  auto last = std::end(s);
  while (first != last) {
    auto pos = std::find(first, last, delim);
    result.emplace_back(std::string_view(&*first, std::distance(first, pos)));
    first = pos;
    if (first != last) {
      ++first;
      if (first == last) {
        result.emplace_back(std::string_view());
      }
    }
  }
  return result;
}

std::vector<std::string_view>
Ax::Internal::split(std::string_view s, const std::string &delims)
{
  std::vector<std::string_view> result;

  auto first = std::begin(s);
  auto last = std::end(s);
  while (first != last) {
    auto pos = std::find_first_of(first, last, std::begin(delims), std::end(delims));
    result.emplace_back(std::string_view(&*first, std::distance(first, pos)));
    first = pos;
    if (first != last) {
      ++first;
      if (first == last) {
        result.emplace_back(std::string_view());
      }
    }
  }
  return result;
}


std::string_view
Ax::Internal::trim(std::string_view s)
{
  auto first
      = std::find_if(s.begin(), s.end(), [](char c) { return !std::isspace(c); });
  auto last = std::find_if(s.rbegin(), std::make_reverse_iterator(first), [](char c) {
    return !std::isspace(c);
  }).base();
  return first != last ? std::string_view(&*first, std::distance(first, last)) :
                         std::string_view();
}

std::unordered_map<std::string, std::string>
Ax::Internal::extract_options(Ax::Logger &logger, const std::string &opts)
{
  std::unordered_map<std::string, std::string> properties;

  auto options = split(opts, ';');
  properties.reserve(options.size());
  for (const auto &op : options) {
    if (op.empty()) {
      continue;
    }
    auto option = split(op, ':');
    if (option.size() != 2) {
      logger(AX_ERROR)
          << "Options must be specified as a semicolon separated list of colon separated pairs.\n"
          << "'" << op << "' has no colon in '" << opts << "'" << std::endl;
    } else {
      auto key = std::string(trim(option[0]));
      if (properties.count(key) != 0) {
        logger(AX_ERROR) << "Option is specified more than once." << std::endl;
      }
      auto value = std::string(trim(option[1]));
      properties[key] = value;
    }
  }
  return properties;
}

Ax::SkipRate
Ax::parse_skip_rate(const std::string &s)
{
  if (s.empty()) {
    return { 0, 0 };
  }

  auto parts = Internal::split(s, '/');
  if (parts.size() != 2) {
    throw std::invalid_argument("Invalid skip rate format use <count>/<stride>");
  }
  auto count = std::stoi(std::string(parts[0]));
  auto stride = std::stoi(std::string(parts[1]));
  if (count > stride) {
    throw std::invalid_argument("Invalid skip rate format, count must be <= stride");
  }
  return { count, stride };
}


std::string
Ax::to_string(const AxVideoInterface &video)
{
  return AxVideoFormatToString(video.info.format) + "/"
         + std::to_string(video.info.width) + "x" + std::to_string(video.info.height);
}

std::string
Ax::to_string(const AxTensorInterface &tensor)
{
  std::string s;
  for (const auto &size : tensor.sizes) {
    s += std::to_string(size) + ",";
  }
  s.pop_back();
  s += "[" + std::to_string(tensor.bytes) + " byte]";
  return s;
}


std::string
Ax::to_string(const AxTensorsInterface &tensors)
{
  std::string s;
  if (tensors.empty()) {
    s += "empty;";
  }
  for (const auto &tensor : tensors) {
    s += to_string(tensor) + ";";
  }
  s.pop_back();
  return s;
}

std::string
Ax::to_string(const AxDataInterface &data)
{
  if (const auto *video = std::get_if<AxVideoInterface>(&data)) {
    return "video/" + to_string(*video);
  } else if (const auto *tensors = std::get_if<AxTensorsInterface>(&data)) {
    return "tensors/" + to_string(*tensors);
  }
  return "empty";
}

std::unordered_map<std::string, std::string>
Ax::parse_and_validate_plugin_options(Ax::Logger &logger, const std::string &options,
    const std::unordered_set<std::string> &allowed_properties)
{
  auto opts = Internal::extract_options(logger, options);
  for (const auto &opt : opts) {
    if (allowed_properties.count(opt.first) == 0) {
      logger(AX_ERROR) << "Property not allowed - " << opt.first << "." << std::endl;
    }
  }
  return opts;
}


Ax::ManagedDataInterface
batched_buffer_allocate(int batch_size, const AxTensorsInterface &in_tensors,
    Ax::DataInterfaceAllocator &allocator)
{
  AxTensorsInterface tensors;
  for (auto &input : in_tensors) {
    AxTensorInterface tensor;
    tensor.sizes = input.sizes;
    tensor.sizes[0] = batch_size;
    tensor.data = nullptr;
    tensor.bytes = input.bytes;
    tensors.push_back(tensor);
  }
  return allocator.allocate(tensors);
}

Ax::ManagedDataInterface
batched_buffer_allocate(int batch_size, const AxVideoInterface &input,
    Ax::DataInterfaceAllocator &allocator)
{
  return allocator.allocate(input);
}

Ax::ManagedDataInterface
batched_buffer_allocate(int batch_size, const std::monostate &input,
    Ax::DataInterfaceAllocator &allocator)
{
  throw std::runtime_error("Cannot allocate batched buffer for empty interface");
}


Ax::ManagedDataInterface
Ax::allocate_batched_buffer(int batch_size, const AxDataInterface &input,
    DataInterfaceAllocator &allocator)
{
  return std::visit(
      [&batch_size, &allocator](const auto &data) {
        return batched_buffer_allocate(batch_size, data, allocator);
      },
      input);
}

AxDataInterface
batch_view(const AxTensorsInterface &tensors, int n)
{
  AxTensorsInterface result;
  result.reserve(tensors.size());
  for (auto &t : tensors) {
    AxTensorInterface newt = t;
    newt.sizes[0] = 1;
    const auto offset = static_cast<ptrdiff_t>(n * t.total() / t.sizes[0] / t.bytes);
    newt.data = t.data ? static_cast<std::byte *>(t.data) + offset : nullptr;
    result.push_back(std::move(newt));
  }
  return result;
}

AxDataInterface
batch_view(const AxVideoInterface &video, int n)
{
  return video;
}

AxDataInterface
batch_view(const std::monostate &i, int n)
{
  return {};
}

AxDataInterface
Ax::batch_view(const AxDataInterface &i, int n)
{
  return std::visit([n](auto &&arg) { return batch_view(arg, n); }, i);
}

namespace
{

void *
heap_alloc(size_t size)
{
  auto page_size = 4096;
  auto adjusted_size = (size + page_size - 1) & ~(page_size - 1);
  auto *p = std::aligned_alloc(page_size, adjusted_size);
  return p;
}

class HeapDataInterfaceAllocator : public Ax::DataInterfaceAllocator
{
  public:
  Ax::ManagedDataInterface allocate(const AxDataInterface &data) override
  {
    Ax::ManagedDataInterface res{ data };
    res.allocate(heap_alloc);
    return res;
  }

  void map(Ax::ManagedDataInterface &) override
  {
  }
  void unmap(Ax::ManagedDataInterface &) override
  {
  }

  private:
};

class DmaBufDataInterfaceAllocator : public Ax::DataInterfaceAllocator
{
  public:
  DmaBufDataInterfaceAllocator()
      : device_name_("/dev/dma_heap/system"),
        device_fd_(::open(device_name_.c_str(), O_RDWR | O_CLOEXEC)),
        page_size_(static_cast<size_t>(getpagesize()))
  {
    if (device_fd_ < 0) {
      throw std::runtime_error(
          "Failed to open " + device_name_ + " : " + std::string(strerror(errno)));
    }
  }

  ~DmaBufDataInterfaceAllocator()
  {
    if (device_fd_ >= 0) {
      ::close(device_fd_);
    }
  }

  Ax::ManagedDataInterface allocate(const AxDataInterface &data) override
  {
    Ax::ManagedDataInterface buffer{ data };
    buffer.allocate([this](size_t size) { return alloc_buf(size); });
    return buffer;
  }

  void map(Ax::ManagedDataInterface &buffer) override
  {
    std::vector<std::shared_ptr<void>> buffers;
    auto &fds = buffer.fds();
    if (auto *video = std::get_if<AxVideoInterface>(&buffer.data())) {
      auto p = map_buf(fds[0]->fd, video->info.stride * video->info.height
                                       * AxVideoFormatNumChannels(video->info.format));
      buffers.push_back(std::move(p));
    } else if (auto *tensors = std::get_if<AxTensorsInterface>(&buffer.data())) {
      size_t n = 0;
      for (auto &tensor : *tensors) {
        auto p = map_buf(fds[n]->fd, tensor.total_bytes());
        buffers.push_back(std::move(p));
        ++n;
      }
    }
    buffer.set_buffers(std::move(buffers));
  }

  void unmap(Ax::ManagedDataInterface &buffer) override
  {
    buffer.set_buffers({});
  }

  private:
  struct dma_heap_allocation_data {
    std::uint64_t len;
    int fd;
    std::uint32_t fd_flags;
    std::uint64_t heap_flags;
  };

#define DMA_HEAP_IOC_MAGIC 'H'
#define DMA_HEAP_IOCTL_ALLOC \
  _IOWR(DMA_HEAP_IOC_MAGIC, 0x0, struct dma_heap_allocation_data)

  size_t align_to_page_size(size_t size) const
  {
    return (((size - 1) / page_size_) + 1) * page_size_;
  }

  public:
  Ax::SharedFD alloc_buf(size_t size)
  {
    const auto aligned_size = align_to_page_size(size);
    dma_heap_allocation_data data = { aligned_size, 0, O_RDWR | O_CLOEXEC, 0 };
    const auto ret = ::ioctl(device_fd_, DMA_HEAP_IOCTL_ALLOC, &data);
    if (ret < 0 || data.fd < 0) {
      throw std::runtime_error(
          "Failed to alloc dmabuf for " + std::to_string(aligned_size)
          + " bytes : got invalid fd : " + std::string(strerror(errno)));
    }
    return std::make_shared<Ax::DmaBufHandle>(data.fd);
  }

  std::shared_ptr<void> map_buf(int fd, size_t size)
  {
    const auto aligned_size = align_to_page_size(size);
    const int prot = PROT_READ | PROT_WRITE;
    const int flags = MAP_SHARED;
    void *const ptr = ::mmap(nullptr, aligned_size, prot, flags, fd, 0);
    if (ptr == MAP_FAILED) {
      throw std::runtime_error("Failed to mmap dmabuf for " + std::to_string(aligned_size)
                               + " bytes : " + std::string(strerror(errno)));
    }
    return { ptr, [aligned_size](void *p) { ::munmap(p, aligned_size); } };
  }

  std::string device_name_;
  int device_fd_;
  size_t page_size_;
};
} // namespace

std::unique_ptr<Ax::DataInterfaceAllocator>
Ax::create_heap_allocator()
{
  return std::make_unique<HeapDataInterfaceAllocator>();
}
std::unique_ptr<Ax::DataInterfaceAllocator>
Ax::create_dma_buf_allocator()
{
  return std::make_unique<DmaBufDataInterfaceAllocator>();
}

std::string
Ax::to_string(const std::set<int> &s)
{
  return Ax::Internal::join(s, ",");
}

std::set<int>
Ax::create_stream_set(std::string &input)
{
  std::set<int> result;
  for (auto &token : Ax::Internal::split(input, ',')) {
    try {
      result.insert(std::stoi(std::string(token)));
    } catch (const std::invalid_argument &e) {
      throw std::runtime_error("Invalid number '" + std::string(token) + "' in stream_select");
    }
  }
  return result;
}

static void
load_v1_base(Ax::SharedLib &lib, Ax::V1Plugin::Base &base)
{
  lib.initialise_function("init_and_set_static_properties", base.init_and_set_static_properties);
  lib.initialise_function("allowed_properties", base.allowed_properties);
  lib.initialise_function("set_dynamic_properties", base.set_dynamic_properties, false);
}

void
Ax::load_v1_plugin(SharedLib &lib, Ax::V1Plugin::InPlace &inplace)
{
  load_v1_base(lib, inplace);
  lib.initialise_function("inplace", inplace.inplace);
}

void
Ax::load_v1_plugin(SharedLib &lib, Ax::V1Plugin::Transform &xform)
{
  load_v1_base(lib, xform);
  lib.initialise_function("transform_async", xform.transform_async, false);
  lib.initialise_function("transform", xform.transform, !xform.transform_async);
  lib.initialise_function("set_output_interface", xform.set_output_interface, false);
  lib.initialise_function("can_passthrough", xform.can_passthrough, false);
  lib.initialise_function("handles_crop_meta", xform.handles_crop_meta, false);
  lib.initialise_function("set_output_interface_from_meta",
      xform.set_output_interface_from_meta, false);
  lib.initialise_function("can_use_dmabuf", xform.can_use_dmabuf, false);
}

void
Ax::load_v1_plugin(SharedLib &lib, Ax::V1Plugin::Decoder &decoder)
{
  load_v1_base(lib, decoder);
  lib.initialise_function("decode_to_meta", decoder.decode_to_meta);
}
