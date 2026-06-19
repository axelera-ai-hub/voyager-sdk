// Copyright Axelera AI, 2024
#include "AxStreamerUtils.hpp"

#include "AxOpUtils.hpp"

#include <algorithm>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>
#include "AxDataInterface.h"
#include "AxInferenceNet.hpp"
#include "AxLog.hpp"

#include <linux/dma-buf.h>

namespace fs = std::filesystem;
using namespace std::string_literals;

// TODO: We need to keep number of channels and number of planes for each format
#define AX_VIDEO_FORMATS(AX_VIDEO_FORMAT_REGISTER) \
  AX_VIDEO_FORMAT_REGISTER(RGB, 3)                 \
  AX_VIDEO_FORMAT_REGISTER(RGBA, 4)                \
  AX_VIDEO_FORMAT_REGISTER(RGBx, 4)                \
  AX_VIDEO_FORMAT_REGISTER(BGR, 3)                 \
  AX_VIDEO_FORMAT_REGISTER(BGRA, 4)                \
  AX_VIDEO_FORMAT_REGISTER(BGRx, 4)                \
  AX_VIDEO_FORMAT_REGISTER(GRAY8, 1)               \
  AX_VIDEO_FORMAT_REGISTER(NV12, 3)                \
  AX_VIDEO_FORMAT_REGISTER(NV16, 3)                \
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


#define MAP_STRING_TO_AX_VIDEO_FORMAT_LOWER(x, y) \
  if (AxToLower(#x) == format)                    \
    return AxVideoFormat::x;

AxVideoFormat
AxVideoFormatFromString(const std::string &format)
{
  AX_VIDEO_FORMATS(MAP_STRING_TO_AX_VIDEO_FORMAT_LOWER)
  return AxVideoFormat::UNDEFINED;
}

std::string
to_string(AxVideoFormat fmt)
{
  return AxToLower(AxVideoFormatToString(fmt));
}

#define MAP_AX_VIDEO_FORMAT_TO_STRING(x, y) \
  case AxVideoFormat::x:                    \
    return #x;

cv::ColorConversionCodes
Ax::Internal::format2format(AxVideoFormat in_format, AxVideoFormat out_format)
{
  static constexpr std::array<Ax::Internal::ColorConversionCodesTableEntry, 36> lut = { {
      { AxVideoFormat::RGB, AxVideoFormat::RGBA, cv::COLOR_RGB2RGBA },
      { AxVideoFormat::RGB, AxVideoFormat::BGRA, cv::COLOR_RGB2BGRA },
      { AxVideoFormat::RGB, AxVideoFormat::BGR, cv::COLOR_RGB2BGR },
      { AxVideoFormat::RGB, AxVideoFormat::GRAY8, cv::COLOR_RGB2GRAY },

      { AxVideoFormat::BGR, AxVideoFormat::RGBA, cv::COLOR_BGR2RGBA },
      { AxVideoFormat::BGR, AxVideoFormat::BGRA, cv::COLOR_BGR2BGRA },
      { AxVideoFormat::BGR, AxVideoFormat::RGB, cv::COLOR_BGR2RGB },
      { AxVideoFormat::BGR, AxVideoFormat::GRAY8, cv::COLOR_BGR2GRAY },

      { AxVideoFormat::RGBA, AxVideoFormat::RGB, cv::COLOR_RGBA2RGB },
      { AxVideoFormat::RGBA, AxVideoFormat::BGR, cv::COLOR_RGBA2BGR },
      { AxVideoFormat::RGBA, AxVideoFormat::BGRA, cv::COLOR_RGBA2BGRA },
      { AxVideoFormat::RGBA, AxVideoFormat::GRAY8, cv::COLOR_RGBA2GRAY },

      { AxVideoFormat::BGRA, AxVideoFormat::RGB, cv::COLOR_BGRA2RGB },
      { AxVideoFormat::BGRA, AxVideoFormat::BGR, cv::COLOR_BGRA2BGR },
      { AxVideoFormat::BGRA, AxVideoFormat::RGBA, cv::COLOR_BGRA2RGBA },
      { AxVideoFormat::BGRA, AxVideoFormat::GRAY8, cv::COLOR_BGRA2GRAY },

      { AxVideoFormat::GRAY8, AxVideoFormat::RGB, cv::COLOR_GRAY2RGB },
      { AxVideoFormat::GRAY8, AxVideoFormat::BGR, cv::COLOR_GRAY2BGR },
      { AxVideoFormat::GRAY8, AxVideoFormat::RGBA, cv::COLOR_GRAY2RGBA },
      { AxVideoFormat::GRAY8, AxVideoFormat::BGRA, cv::COLOR_GRAY2BGRA },

      { AxVideoFormat::YUY2, AxVideoFormat::RGB, cv::COLOR_YUV2RGB_YUY2 },
      { AxVideoFormat::YUY2, AxVideoFormat::BGR, cv::COLOR_YUV2BGR_YUY2 },
      { AxVideoFormat::YUY2, AxVideoFormat::RGBA, cv::COLOR_YUV2RGBA_YUY2 },
      { AxVideoFormat::YUY2, AxVideoFormat::BGRA, cv::COLOR_YUV2BGRA_YUY2 },
      { AxVideoFormat::YUY2, AxVideoFormat::GRAY8, cv::COLOR_YUV2GRAY_YUY2 },

      { AxVideoFormat::NV12, AxVideoFormat::RGB, cv::COLOR_YUV2RGB_NV12 },
      { AxVideoFormat::NV12, AxVideoFormat::BGR, cv::COLOR_YUV2BGR_NV12 },
      { AxVideoFormat::NV12, AxVideoFormat::RGBA, cv::COLOR_YUV2RGBA_NV12 },
      { AxVideoFormat::NV12, AxVideoFormat::BGRA, cv::COLOR_YUV2BGRA_NV12 },
      { AxVideoFormat::NV12, AxVideoFormat::GRAY8, cv::COLOR_YUV2GRAY_NV12 },

      { AxVideoFormat::I420, AxVideoFormat::RGB, cv::COLOR_YUV2RGB_I420 },
      { AxVideoFormat::I420, AxVideoFormat::BGR, cv::COLOR_YUV2BGR_I420 },
      { AxVideoFormat::I420, AxVideoFormat::RGBA, cv::COLOR_YUV2RGBA_I420 },
      { AxVideoFormat::I420, AxVideoFormat::BGRA, cv::COLOR_YUV2BGRA_I420 },
      { AxVideoFormat::I420, AxVideoFormat::GRAY8, cv::COLOR_YUV2GRAY_I420 },
  } };
  auto it = std::find_if(lut.begin(), lut.end(),
      [&](const Ax::Internal::ColorConversionCodesTableEntry &e) {
        return e.in_fmt == in_format && e.out_fmt == out_format;
      });
  if (it != lut.end()) {
    return it->code;
  }
  throw std::runtime_error("OpenCV color conversion not supported for "
                           + AxVideoFormatToString(in_format) + " to "
                           + AxVideoFormatToString(out_format));
}

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

void
Ax::validate_output_format(AxVideoFormat format, std::string_view prop_fmt,
    std::string_view label, std::span<const AxVideoFormat> valid_formats)
{
  if (std::none_of(valid_formats.begin(), valid_formats.end(),
          [format](auto fmt) { return fmt == format; })) {

    auto supported = Ax::Internal::join(valid_formats, ", ");
    std::string prop_fmt_str = prop_fmt.empty() ?
                                   AxToLower(AxVideoFormatToString(format)) :
                                   std::string{ prop_fmt };
    throw std::runtime_error(std::string(label) + ": " + prop_fmt_str + " is not suppported as an output format. Supported formats are: "
                             + supported);
  }
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

template <char element_delimiter = ';', char key_value_delimiter = ':'>
static std::unordered_map<std::string, std::string>
extract_options_impl(Ax::Logger &logger, const std::string &opts)
{
  std::unordered_map<std::string, std::string> properties;

  auto options = Ax::Internal::split(opts, element_delimiter);
  properties.reserve(options.size());
  for (const auto &op : options) {
    if (op.empty()) {
      continue;
    }
    auto option = Ax::Internal::split(op, key_value_delimiter);
    if (option.size() != 2) {
      logger(AX_ERROR)
          << "Options must be specified as a semicolon separated list of colon separated pairs.\n"
          << "'" << op << "' has no colon in '" << opts << "'" << std::endl;
    } else {
      auto key = std::string(Ax::Internal::trim(option[0]));
      if (properties.count(key) != 0) {
        logger(AX_ERROR) << "Option is specified more than once." << std::endl;
      }
      auto value = std::string(Ax::Internal::trim(option[1]));
      properties[key] = value;
    }
  }
  return properties;
}

std::unordered_map<std::string, std::string>
Ax::extract_options(Ax::Logger &logger, const std::string &opts)
{
  return extract_options_impl(logger, opts);
}

std::unordered_map<std::string, std::string>
Ax::extract_secondary_options(Ax::Logger &logger, const std::string &opts)
{
  return extract_options_impl<'&', '='>(logger, opts);
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
  s += std::string("[") + std::to_string(tensor.bytes) + " byte]";
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
  auto opts = extract_options(logger, options);
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

  void map(Ax::ManagedDataInterface &, Ax::MapType) override
  {
  }
  void unmap(Ax::ManagedDataInterface &) override
  {
  }
  void release(Ax::ManagedDataInterface &buffer) override
  {
    unmap(buffer);
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

  void map(Ax::ManagedDataInterface &buffer, Ax::MapType type) override
  {
    if (!buffer.is_mapped()) {
      std::vector<std::shared_ptr<void>> buffers;
      auto &fds = buffer.fds();
      if (auto *video = std::get_if<AxVideoInterface>(&buffer.data())) {
        auto p = map_buf(*fds[0], video->info.stride * video->info.height, type);
        buffers.push_back(std::move(p));
      } else if (auto *tensors = std::get_if<AxTensorsInterface>(&buffer.data())) {
        size_t n = 0;
        for (auto &tensor : *tensors) {
          auto p = map_buf(*fds[n], tensor.total_bytes(), type);
          buffers.push_back(std::move(p));
          ++n;
        }
      }
      buffer.set_buffers(std::move(buffers));
    }
  }

  void unmap(Ax::ManagedDataInterface &buffer) override
  {
    if (buffer.is_mapped()) {
      buffer.set_buffers(std::vector<std::shared_ptr<void>>{});
    }
  }

  void release(Ax::ManagedDataInterface &buffer) override
  {
    unmap(buffer);
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
    const int prot = PROT_READ | PROT_WRITE;
    const int flags = MAP_SHARED;
    void *const ptr = ::mmap(nullptr, aligned_size, prot, flags, data.fd, 0);
    if (ptr == MAP_FAILED) {
      throw std::runtime_error("Failed to mmap dmabuf for " + std::to_string(aligned_size)
                               + " bytes : " + std::string(strerror(errno)));
    }
    return std::make_shared<Ax::DmaBufHandle>(data.fd, aligned_size, ptr);
  }

  std::shared_ptr<void> map_buf(const Ax::DmaBufHandle &fd, size_t size, Ax::MapType type)
  {
    uint64_t flags = DMA_BUF_SYNC_RW;
    if (type == Ax::MAP_READ_ONLY) {
      flags = DMA_BUF_SYNC_READ;
    } else if (type == Ax::MAP_WRITE_ONLY) {
      flags = DMA_BUF_SYNC_WRITE;
    }
    struct dma_buf_sync sync = {
      .flags = DMA_BUF_SYNC_START | flags,
    };
    ioctl(fd.fd, DMA_BUF_IOCTL_SYNC, &sync);
    return { fd.mapped, [flags, fd = fd.fd](void *p) {
              struct dma_buf_sync end_sync = { .flags = DMA_BUF_SYNC_END | flags };
              ioctl(fd, DMA_BUF_IOCTL_SYNC, &end_sync);
            } };
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

Ax::slice_overlap
Ax::determine_overlap(size_t image_size, size_t slice_size, size_t overlap)
{
  if (image_size <= slice_size) {
    return { 1, 0 };
  }
  auto overlap_size = slice_size * overlap / 100;
  auto overlapped_slice = slice_size - overlap_size;
  auto num_slices = (image_size + overlapped_slice - 1) / overlapped_slice;
  auto total_overlap = num_slices * slice_size - image_size;
  auto adjusted_overlap = num_slices == 1 ? 0 : total_overlap / (num_slices - 1);
  return { num_slices, adjusted_overlap };
}

static void
load_v1_base(Ax::SharedLib &lib, Ax::V1Plugin::Base &base)
{
  lib.initialise_function("init_and_set_static_properties",
      base.init_and_set_static_properties, false);
  lib.initialise_function("init_and_set_static_properties_with_context",
      base.init_and_set_static_properties_with_context, false);
  lib.initialise_function("allowed_properties", base.allowed_properties, false);
  lib.initialise_function("set_dynamic_properties", base.set_dynamic_properties, false);
}


namespace Ax
{
void
load_v1_plugin(SharedLib &lib, Ax::V1Plugin::InPlace &inplace)
{
  load_v1_base(lib, inplace);
  lib.initialise_function("inplace", inplace.inplace);
}

void
load_v1_plugin(SharedLib &lib, Ax::V1Plugin::Transform &xform)
{
  load_v1_base(lib, xform);
  lib.initialise_function("transform", xform.transform, true);
  lib.initialise_function("set_output_interface", xform.set_output_interface, false);
  lib.initialise_function("can_passthrough", xform.can_passthrough, false);
  lib.initialise_function("set_output_interface_from_meta",
      xform.set_output_interface_from_meta, false);
  lib.initialise_function("query_supports", xform.query_supports, false);

  // New style plugins should implement query_supports
  // But if not and handles_crop_meta or can_use_dmabuf are present then fail
  // and tell the user how to upgrade, or not to mix and match new/old behaviour
  const auto has_legacy_supports
      = lib.has_symbol("handles_crop_meta") || lib.has_symbol("can_use_dmabuf");
  if (xform.query_supports && has_legacy_supports) {
    throw std::runtime_error(
        R"##(V1 Transform plugins should not implement can_use_dmabuf or handles_crop_meta, query_supports supercedes it)##");
  }
  if (!xform.query_supports && has_legacy_supports) {
    throw std::runtime_error(
        R"##(V1 Transform plugins should not implement can_use_dmabuf or handles_crop_meta, use query_supports instead\n"
For example replace:
  extern "C" bool can_use_dmabuf(const void *, Ax::Logger &) { return true; }
  extern "C" bool handles_crop_meta() { return true; }
with:
  extern "C" bool query_supports(Ax::PluginFeature feature, const void *, Ax::Logger &) {
    if (feature == Ax::PluginFeature::dmabuf_buffers || feature == Ax::PluginFeature::crop_meta) {
      return true;
    }
    return Ax::PluginFeatureDefaults(feature);
  })##");
  }
  if (!xform.query_supports) {
    xform.query_supports = [](Ax::PluginFeature f, const void *, Ax::Logger &) {
      return Ax::PluginFeatureDefaults(f);
    };
  }
}

void
load_v1_plugin(SharedLib &lib, Ax::V1Plugin::Decode &decode)
{
  load_v1_base(lib, decode);
  lib.initialise_function("decode_to_meta", decode.decode_to_meta);
}
} // namespace Ax

void
Ax::load_v1_plugin(SharedLib &lib, Ax::V1Plugin::DetermineObjectAttribute &determine_object_attribute)
{
  load_v1_base(lib, determine_object_attribute);
  lib.initialise_function("determine_object_attribute",
      determine_object_attribute.determine_object_attribute);
}

void
Ax::load_v1_plugin(SharedLib &lib, Ax::V1Plugin::TrackerFilter &tracker_filter)
{
  load_v1_base(lib, tracker_filter);
  lib.initialise_function("filter", tracker_filter.filter);
}

std::mutex plugin_names_count_mutex;
std::unordered_map<std::string, int> plugin_names_count{};

static std::string
make_plugin_name(std::string n)
{
  auto name = std::move(n);
  if (name.ends_with(".so")) {
    name = name.substr(0, name.size() - 3);
  }
  std::lock_guard lock(plugin_names_count_mutex);
  auto count = plugin_names_count[name]++;
  return name + "_" + std::to_string(count);
}

template <typename PluginType, typename PluginBase>
Ax::LoadedPlugin<PluginType, PluginBase>::LoadedPlugin(Ax::Logger &logger,
    Ax::SharedLib &&shared, std::string options, AxAllocationContext *context,
    std::string mode)
    : logger(logger),
      shared_(std::move(shared)),
      name_(make_plugin_name(shared_.libname())),
      mode_(mode)
{
  Ax::load_v1_plugin(shared, fns);
  if (fns.allowed_properties) {
    allowed_ = fns.allowed_properties();
  }
  auto opts = Ax::parse_and_validate_plugin_options(logger, options, allowed_);
  if (fns.init_and_set_static_properties_with_context) {
    subplugin_data = fns.init_and_set_static_properties_with_context(opts, context, logger);
  } else if (fns.init_and_set_static_properties) {
    subplugin_data = fns.init_and_set_static_properties(opts, logger);
  }
  set_dynamic_properties(opts);
}

// explicitly instantiate the templates we need
template class Ax::LoadedPlugin<Ax::V1Plugin::InPlace, Ax::InPlace>;
template class Ax::LoadedPlugin<Ax::V1Plugin::Transform, Ax::Transform>;
template class Ax::LoadedPlugin<Ax::V1Plugin::Decode, Ax::Decode>;

//  @brief - Takes an existing image (which itself may be an ROI) and creates a
//  new image which is a new ROI of the original image.
//  @param original - the original image
//  @param x - the x coordinate of the new ROI
//  @param y - the y coordinate of the new ROI
//  @param width - the width of the new ROI
//  @param height - the height of the new ROI
//
//  @return - the new ROI
//
//  Any new ROI will be clipped to the original image and a warning logged
AxVideoInterface
Ax::create_roi(const AxVideoInterface &original, int x, int y, int width, int height)
{
  AxVideoInterface roi = original;
  if (x + width > original.info.width) {
    width = original.info.width - x;
  }
  if (y + height > original.info.height) {
    height = original.info.height - y;
  }
  if (original.info.cropped) {
    x += original.info.x_offset;
    y += original.info.y_offset;
  }
  roi.info.x_offset = x;
  roi.info.y_offset = y;
  roi.info.width = width;
  roi.info.height = height;
  roi.info.cropped = true;
  return roi;
}

Ax::BatchedBuffer::BatchedBuffer(int batch_size, const AxDataInterface &iface,
    DataInterfaceAllocator &allocator)
    : batched(allocate_batched_buffer(batch_size, iface, allocator)),
      allocator(allocator)
{
  for (int i = 0; i < batch_size; ++i) {
    views.push_back(batch_view(batched.data(), i));
  }
}


void
Ax::BatchedBuffer::map(Ax::MapType flags)
{
  allocator.map(batched, flags);
  update_views();
}

void
Ax::BatchedBuffer::unmap()
{
  allocator.unmap(batched);
  update_views();
}

const Ax::ManagedDataInterface &
Ax::BatchedBuffer::get_batched()
{
  return batched;
}

void
Ax::BatchedBuffer::release()
{
  allocator.release(batched);
}

void
Ax::BatchedBuffer::set_iface(const AxDataInterface &iface)
{
  batched.set_data(iface);
  update_views();
}

AxDataInterface
Ax::BatchedBuffer::update_iface(
    const AxDataInterface &data, const AxDataInterface &iface, int batch_size)
{
  // Update iface data from current
  AxDataInterface output = iface;
  struct map_data {
    void *data{};
    int fd = -1;
    opencl_buffer *ocl_buffer{ nullptr };
  };
  std::vector<map_data> ptrs;
  if (auto *tensors = std::get_if<AxTensorsInterface>(&data)) {
    for (auto &tensor : *tensors) {
      ptrs.emplace_back(tensor.data, tensor.fd, tensor.ocl_buffer);
    }
  } else if (auto *video = std::get_if<AxVideoInterface>(&data)) {
    ptrs.emplace_back(video->data, video->fd, video->ocl_buffer);
  } else {
    return output; // nothing to update
  }

  if (auto *tensors = std::get_if<AxTensorsInterface>(&output)) {
    if (tensors->size() != ptrs.size()) {
      throw std::runtime_error("BatchedBuffer: output tensors size mismatch");
    }
    for (int i = 0; i < tensors->size(); ++i) {
      (*tensors)[i].data = ptrs[i].data;
      (*tensors)[i].fd = ptrs[i].fd;
      (*tensors)[i].sizes[0] = batch_size;
      (*tensors)[i].ocl_buffer = ptrs[i].ocl_buffer;
    }

  } else if (auto *video = std::get_if<AxVideoInterface>(&output)) {
    if (ptrs.size() != 1) {
      throw std::runtime_error("BatchedBuffer: output tensors size mismatch");
    }
    video->data = ptrs[0].data;
    video->fd = ptrs[0].fd;
    video->ocl_buffer = ptrs[0].ocl_buffer;
  }
  return output;
}

//  This updates the buffer description but does not change the data
void
Ax::BatchedBuffer::update_iface(const AxDataInterface &iface, int batch_size)
{
  auto new_iface = update_iface(batched.data(), iface, batch_size);
  batched.set_data(new_iface);
  update_views();
}

int
Ax::BatchedBuffer::batch_size() const
{
  return views.size();
}

bool
Ax::BatchedBuffer::is_dmabuf() const
{
  return !batched.fds().empty();
}

bool
Ax::BatchedBuffer::is_opencl() const
{
  return !batched.ocl_buffers().empty();
}

void
Ax::BatchedBuffer::update_views()
{
  size_t n = 0;
  for (auto &view : views) {
    // TODO this is inefficient, we just need to init the data param
    view = batch_view(batched.data(), n);
    ++n;
  }
}

std::unique_ptr<Ax::Plugin>
Ax::load_plugin(Logger &logger, const std::string &name, const std::string &options,
    AxAllocationContext *context, const std::string &mode)
{
  const auto plugin_path = Ax::get_env("AX_SUBPLUGIN_PATH", "");
  auto libname = fs::path{ name };
  if (libname.extension() != ".so") {
    libname = libname.replace_filename("lib" + libname.filename().string() + ".so");
  }
  if (!libname.is_absolute() && !plugin_path.empty()) {
    libname = fs::path{ plugin_path } / libname;
  }
  Ax::SharedLib shared(logger, Ax::libname(libname));
  if (shared.has_symbol("transform")) {
    return std::make_unique<LoadedTransform>(
        logger, std::move(shared), options, context, mode);
  } else if (shared.has_symbol("inplace")) {
    return std::make_unique<LoadedInPlace>(logger, std::move(shared), options, context, mode);
  } else if (shared.has_symbol("decode_to_meta")) {
    return std::make_unique<LoadedDecode>(logger, std::move(shared), options, context, mode);
  } else {
    throw std::runtime_error(
        "Unknown plugin type in library " + libname.string()
        + ". Must contain 'transform', 'inplace', or 'decode_to_meta' symbol.");
  }
}
