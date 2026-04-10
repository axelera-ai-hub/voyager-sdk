// Copyright Axelera AI, 2024
// collection of utils taken from axstreamer
#pragma once
#include <algorithm>
#include <cassert>
#include <condition_variable>
#include <cstddef>
#include <cstring>
#include <fcntl.h>
#ifdef __linux__
#include <linux/memfd.h>
#include <linux/udmabuf.h>
#endif
#include <dlfcn.h>
#include <list>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <queue>
#include <set>
#include <span>
#include <stdio.h>
#include <string>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <unordered_set>
#include <utility>
#include "AxDataInterface.h"
#include "AxLog.hpp"
#include "AxPlugin.hpp"

struct opencl_buffer_details;

namespace Ax
{

enum MapType {
  MAP_NONE,
  MAP_READ_ONLY,
  MAP_WRITE_ONLY,
  MAP_READ_WRITE,
};

template <typename T>
T
pop_queue(std::queue<T> &q)
{
  assert(!q.empty());
  T t = std::move(q.front());
  q.pop();
  return t;
}
template <typename T>
T
pop_queue(std::deque<T> &q)
{
  assert(!q.empty());
  T t = std::move(q.front());
  q.pop_front();
  return t;
}

template <typename T>
void
clear_queue(std::queue<T> &q)
{
  while (!q.empty()) {
    q.pop();
  }
}

struct DmaBufHandle {
  explicit DmaBufHandle(int fd, size_t size, void *p)
      : fd(fd),
        mapped(p),
        size(size),
        should_close(true)
  {
    //  Size should always be non-zero if we own the fd
    assert(size != 0);
  }
  explicit DmaBufHandle(int fd, size_t size)
      : fd(fd),
        size(size),
        should_close(false)
  {
    //  We should only close if we own the fd
    assert(should_close != (size == 0));
  }

  DmaBufHandle(const DmaBufHandle &) = delete;
  DmaBufHandle &operator=(const DmaBufHandle &) = delete;
  DmaBufHandle(DmaBufHandle &&) = delete;
  DmaBufHandle &operator=(DmaBufHandle &&) = delete;
  ~DmaBufHandle()
  {
    if (fd >= 0 && should_close) {
      ::munmap(mapped, size);
      ::close(fd);
    }
  }
  const int fd;
  void *mapped = nullptr;

  private:
  const size_t size;
  const bool should_close;
};
using SharedFD = std::shared_ptr<DmaBufHandle>;


namespace Internal
{

struct ColorConversionCodesTableEntry {
  AxVideoFormat in_fmt;
  AxVideoFormat out_fmt;
  cv::ColorConversionCodes code;
};

cv::ColorConversionCodes format2format(AxVideoFormat in_format, AxVideoFormat out_format);
std::vector<std::string_view> split(std::string_view s, char delim);
std::vector<std::string_view> split(std::string_view s, const std::string &delims);

std::string_view trim(std::string_view s);

inline std::string
to_string(const std::string s)
{
  return s;
}

template <typename It>
std::string
join(It begin, It end, const std::string &delim)
{
  using Internal::to_string;
  using std::to_string;
  std::string s;
  if (begin != end) {
    auto pen = std::prev(end);
    for (; begin != pen; ++begin) {
      s += to_string(*begin) + delim;
    }
    s += to_string(*begin);
  }
  return s;
}

template <typename Range>
std::string
join(const Range &range, const std::string &delim)
{
  return join(std::begin(range), std::end(range), delim);
}

template <typename Iterator> class Enumerator
{
  public:
  Enumerator(Iterator iter, std::size_t index)
      : iter_(iter),
        index_(index)
  {
  }

  auto operator*()
  {
    return std::make_pair(index_, *iter_);
  }

  Enumerator &operator++()
  {
    ++iter_;
    ++index_;
    return *this;
  }

  bool operator!=(const Enumerator &other) const
  {
    return index_ != other.index_;
  }

  private:
  Iterator iter_;
  std::size_t index_;
};

template <typename T> class IntegerIterator
{
  public:
  explicit IntegerIterator(T value)
      : value_(value)
  {
  }

  T operator*() const
  {
    return value_;
  }

  IntegerIterator &operator++()
  {
    ++value_;
    return *this;
  }

  bool operator!=(const IntegerIterator &other) const
  {
    return value_ != other.value_;
  }

  private:
  T value_;
};

template <typename Iterator> class IteratorRange
{
  public:
  explicit IteratorRange(Iterator begin, Iterator end)
      : begin_(begin),
        end_(end)
  {
  }

  Iterator begin()
  {
    return begin_;
  }
  Iterator end()
  {
    return end_;
  }

  private:
  Iterator begin_, end_;
};

// Like python enumerato:
// for(auto [index, item] : enumerate(container)) {
auto
enumerate(auto &container, std::size_t start = 0)
{
  return IteratorRange(Enumerator(container.begin(), start),
      Enumerator(container.end(), start + container.size()));
}


} // namespace Internal

struct SkipRate {
  int count;
  int stride;
};
inline bool
operator==(const SkipRate &a, const SkipRate &b)
{
  return a.count == b.count && a.stride == b.stride;
}

SkipRate parse_skip_rate(const std::string &s);

std::string to_string(const AxVideoInterface &video);
std::string to_string(const AxTensorInterface &tensor);
std::string to_string(const AxTensorsInterface &tensors);
std::string to_string(const AxDataInterface &data);


std::unordered_map<std::string, std::string> extract_options(
    Ax::Logger &logger, const std::string &opts);

std::unordered_map<std::string, std::string> extract_secondary_options(
    Ax::Logger &logger, const std::string &opts);

std::unordered_map<std::string, std::string> parse_and_validate_plugin_options(
    Ax::Logger &logger, const std::string &options,
    const std::unordered_set<std::string> &allowed_properties);

class ManagedDataInterface
{
  public:
  ManagedDataInterface(const AxDataInterface &data)
      : data_(data)
  {
  }
  ManagedDataInterface(const ManagedDataInterface &data) = delete;
  ManagedDataInterface &operator=(const ManagedDataInterface &data) = delete;
  ManagedDataInterface(ManagedDataInterface &&data) = default;
  ManagedDataInterface &operator=(ManagedDataInterface &&data) = default;


  void set_data(const AxDataInterface &data)
  {
    data_ = data;
  }

  const AxDataInterface &data() const
  {
    return data_;
  }

  const std::vector<std::shared_ptr<void>> &buffers() const
  {
    return buffers_;
  }

  const std::vector<SharedFD> &fds() const
  {
    return fds_;
  }

  const std::vector<std::shared_ptr<opencl_buffer>> &ocl_buffers() const
  {
    return ocl_buffers_;
  }

  template <typename T, typename U> void set_buffer_info(T p, U &interface)
  {
    interface.data = nullptr;
    interface.fd = -1;
    interface.ocl_buffer = nullptr;
    if constexpr (std::is_same_v<T, void *>) {
      interface.data = p;
      buffers_.emplace_back(p, std::free);
    } else if constexpr (std::is_same_v<T, SharedFD>) {
      interface.fd = p->fd;
      fds_.push_back(std::move(p));
    } else if constexpr (std::is_same_v<T, opencl_buffer_details *>) {
      interface.ocl_buffer = p;
      interface.data = p->data.data();
      ocl_buffers_.push_back(std::shared_ptr<opencl_buffer>(p));
    } else {
      throw std::runtime_error("ManagedDataInterface: unsupported type for tensor data");
    }
  }


  void allocate(auto &&allocator)
  {
    buffers_.clear();
    fds_.clear();
    if (auto *video = std::get_if<AxVideoInterface>(&data_)) {
      auto p = allocator(video->info.stride * video->info.height);
      set_buffer_info(p, *video);
    } else if (auto *tensors = std::get_if<AxTensorsInterface>(&data_)) {
      for (auto &tensor : *tensors) {
        auto p = allocator(tensor.total_bytes());
        set_buffer_info(p, tensor);
      }
    }
  }

  void set_buffers(std::vector<std::shared_ptr<void>> buffers)
  {
    buffers_ = std::move(buffers);
    if (auto *video = std::get_if<AxVideoInterface>(&data_)) {
      video->data = buffers_.empty() ? nullptr : buffers_[0].get();
    } else if (auto *tensors = std::get_if<AxTensorsInterface>(&data_)) {
      for (auto i = 0; i != tensors->size(); ++i) {
        auto &t = (*tensors)[i];
        t.data = buffers_.empty() ? nullptr : buffers_[i].get();
      }
    }
  }

  void set_buffers(const std::vector<std::shared_ptr<opencl_buffer>> &buffers)
  {
    if (auto *video = std::get_if<AxVideoInterface>(&data_)) {
      video->data = nullptr;
      video->ocl_buffer = ocl_buffers_.empty() ? nullptr : ocl_buffers_[0].get();
    } else if (auto *tensors = std::get_if<AxTensorsInterface>(&data_)) {
      for (auto i = 0; i != tensors->size(); ++i) {
        auto &t = (*tensors)[i];
        t.ocl_buffer = ocl_buffers_.empty() ? nullptr : ocl_buffers_[i].get();
        t.data = nullptr;
      }
    }
    buffers_.clear();
  }

  bool is_mapped() const
  {
    return !buffers_.empty();
  }

  private:
  AxDataInterface data_;
  std::vector<std::shared_ptr<void>> buffers_;
  std::vector<SharedFD> fds_;
  std::vector<std::shared_ptr<opencl_buffer>> ocl_buffers_;
};

using ManagedDataInterfaces = std::list<ManagedDataInterface>;

class DataInterfaceAllocator
{
  public:
  virtual ManagedDataInterface allocate(const AxDataInterface &data) = 0;
  virtual void map(ManagedDataInterface &data, MapType type) = 0;
  virtual void unmap(ManagedDataInterface &data) = 0;
  virtual void release(ManagedDataInterface &data) = 0;
  virtual ~DataInterfaceAllocator() = default;
};

class NullDataInterfaceAllocator : public DataInterfaceAllocator
{
  public:
  ManagedDataInterface allocate(const AxDataInterface &data) override
  {
    return ManagedDataInterface(data);
  }
  void map(ManagedDataInterface &, MapType) override
  {
    //  Map if it an OpenCL buffer
  }
  void unmap(ManagedDataInterface &) override
  {
    //  Unmap if it an OpenCL buffer
  }
  void release(ManagedDataInterface &) override
  {
  }
};

std::unique_ptr<DataInterfaceAllocator> create_heap_allocator();
std::unique_ptr<DataInterfaceAllocator> create_dma_buf_allocator();
std::unique_ptr<DataInterfaceAllocator> create_opencl_allocator(
    AxAllocationContext *context, Ax::Logger &logger);

ManagedDataInterface allocate_batched_buffer(int batch_size,
    const AxDataInterface &input, DataInterfaceAllocator &allocator);

AxDataInterface batch_view(const AxDataInterface &i, int n);

class BatchedBuffer;
class SharedBatchBufferView
{
  public:
  SharedBatchBufferView() = default;
  SharedBatchBufferView(std::shared_ptr<BatchedBuffer> self, AxDataInterface *view)
      : self_(self),
        view_(view)
  {
  }

  const AxDataInterface *get() const
  {
    return view_;
  }

  operator bool() const
  {
    return self_ != nullptr;
  }

  const AxDataInterface *operator->() const
  {
    return view_;
  }

  const AxDataInterface &operator*() const
  {
    return *view_;
  }

  void reset()
  {
    self_.reset();
    view_ = nullptr;
  }

  std::shared_ptr<BatchedBuffer> underlying() const
  {
    return self_;
  }

  private:
  std::shared_ptr<BatchedBuffer> self_;
  AxDataInterface *view_;
};

class BatchedBuffer
{
  public:
  BatchedBuffer(int batch_size, const AxDataInterface &iface,
      DataInterfaceAllocator &allocator);

  BatchedBuffer &operator=(const BatchedBuffer &other) = delete;
  BatchedBuffer &operator=(BatchedBuffer &&other) = delete;
  BatchedBuffer(BatchedBuffer &&other) = delete;
  BatchedBuffer(const BatchedBuffer &other) = delete;
  ~BatchedBuffer()
  {
    //  Ensure we unamp
    unmap();
  }

  void map(MapType type);

  void unmap();

  //  Called when a buffer is released back into a pool
  void release();

  const ManagedDataInterface &get_batched();

  // create a shared ptr to a view that will ensure the batch buffer is not
  // removed until all views of it are done with
  friend SharedBatchBufferView get_shared_view_of_batch_buffer(
      std::shared_ptr<BatchedBuffer> self, int batch)
  {
    return SharedBatchBufferView(self, &self->views[batch]);
  }

  void set_iface(const AxDataInterface &iface);

  AxDataInterface update_iface(
      const AxDataInterface &data, const AxDataInterface &iface, int batch_size);

  //  This updates the buffer description but does not change the data
  void update_iface(const AxDataInterface &iface, int batch_size);

  int batch_size() const;

  bool is_dmabuf() const;

  bool is_opencl() const;

  bool is_mapped() const
  {
    return !batched.buffers().empty();
  }

  private:
  void update_views();
  ManagedDataInterface batched;
  std::vector<AxDataInterface> views;
  DataInterfaceAllocator &allocator;
};

// Interfaces are considered equivalent if they have the same number of
// elements and each element has the same size
inline bool
are_equivalent(const AxDataInterface &a, const AxDataInterface &b)
{
  if (std::holds_alternative<AxTensorsInterface>(a)
      && std::holds_alternative<AxTensorsInterface>(b)) {
    auto &ta = std::get<AxTensorsInterface>(a);
    auto &tb = std::get<AxTensorsInterface>(b);
    return ta.size() == tb.size()
           && std::equal(ta.begin(), ta.end(), tb.begin(),
               [](const AxTensorInterface &a, const AxTensorInterface &b) {
                 return a.total_bytes() == b.total_bytes();
               });
  } else if (std::holds_alternative<AxVideoInterface>(a)
             && std::holds_alternative<AxVideoInterface>(b)) {
    auto &va = std::get<AxVideoInterface>(a);
    auto &vb = std::get<AxVideoInterface>(b);
    return va.info.width == vb.info.width && va.info.height == vb.info.height
           && AxVideoFormatNumChannels(va.info.format)
                  == AxVideoFormatNumChannels(vb.info.format);
  }
  return false;
}

class BatchedBufferPool
{
  public:
  BatchedBufferPool(int batch_size, const AxDataInterface &iface,
      DataInterfaceAllocator &allocator)
      : batch_size_(batch_size),
        iface_(iface),
        allocator_(allocator),
        mutex_(std::make_shared<std::mutex>()),
        free_(std::make_shared<FreeList>())
  {
  }

  int round_up(int value, int multiple)
  {
    return ((value + multiple - 1) / multiple) * multiple;
  }

  AxDataInterface create_new_iface(const AxDataInterface &iface)
  {
    auto new_iface = iface;
    if (std::holds_alternative<AxVideoInterface>(new_iface)) {
      auto &video = std::get<AxVideoInterface>(new_iface);
      video.strides[0] = round_up(
          video.info.width * AxVideoFormatNumChannels(video.info.format), 4);
      video.info.stride = video.strides[0];
    }
    return new_iface;
  }

  std::shared_ptr<BatchedBuffer> new_batched_buffer(const AxDataInterface &new_iface)
  {
    {
      std::lock_guard<std::mutex> lock(*mutex_);
      if (!are_equivalent(iface_, new_iface)) {
        iface_ = new_iface;
        free_ = std::make_shared<FreeList>();
      }
    }

    std::unique_ptr<BatchedBuffer> buffer;
    auto free = free_;
    if (!free->empty()) {
      std::lock_guard<std::mutex> lock(*mutex_);
      buffer = std::move(free->back());
      free->pop_back();
      //  Ensure we have the correct caps
      buffer->update_iface(iface_, batch_size_);
    } else {
      buffer.reset(new BatchedBuffer(batch_size_, iface_, allocator_));
    }
    //  The destructor captures the free list so that even if the member free_
    //  is destroyed the buffer will still be returned to the free list which
    //  will be finally destroyed once the last buffer that belongs to that free
    //  list is returned (Python-like garbage collection)
    return std::shared_ptr<BatchedBuffer>(buffer.release(),
        [this, free = std::move(free), mutex = mutex_](BatchedBuffer *buffer) {
          buffer->release();
          std::lock_guard<std::mutex> lock(*mutex);
          free->emplace_back(buffer);
        });
  }


  std::shared_ptr<BatchedBuffer> new_batched_buffer()
  {
    return new_batched_buffer(iface_);
  }

  private:
  int batch_size_;
  AxDataInterface iface_;
  DataInterfaceAllocator &allocator_;
  std::shared_ptr<std::mutex> mutex_;
  using FreeList = std::vector<std::unique_ptr<BatchedBuffer>>;
  std::shared_ptr<FreeList> free_;
};

template <typename T> struct BlockingQueue {
  explicit BlockingQueue(size_t queue_depth = 1)
      : queue_depth(queue_depth)
  {
  }

  void push_locked(T &&t)
  {
    std::unique_lock<std::mutex> lock(mutex);
    full_condition.wait(
        lock, [this] { return queue.size() < queue_depth || !running; });
    queue.push(std::move(t));
  }

  void push(T &&t)
  {
    push_locked(std::move(t));
    empty_condition.notify_one();
  }

  T wait_one_locked()
  {
    std::unique_lock<std::mutex> lock(mutex);
    empty_condition.wait(lock, [this] { return !queue.empty() || !running; });
    return running ? pop_queue(queue) : T{};
  }

  T wait_one()
  {
    auto t = wait_one_locked();
    full_condition.notify_one();
    return t;
  }

  void stop()
  {
    std::unique_lock<std::mutex> lock(mutex);
    running = false;
    empty_condition.notify_all();
    full_condition.notify_all();
    std::queue<T> null;
    queue.swap(null);
  }

  std::condition_variable empty_condition;
  std::condition_variable full_condition;
  std::mutex mutex;
  std::queue<T> queue;
  bool running = { true };
  size_t queue_depth{ 1 };
};

inline std::string
libname(std::string libname)
{
#ifdef __APPLE__
  if (libname.ends_with(".so")) {
    libname = libname.substr(0, libname.size() - 3) + ".dylib";
  }
#endif
  return libname;
}

class SharedLib
{
  public:
  SharedLib(const SharedLib &) = delete;
  SharedLib(SharedLib &&) = default;
  SharedLib &operator=(const SharedLib &) = delete;
  SharedLib &operator=(SharedLib &&) = delete;

  SharedLib(Ax::Logger &logger, const std::string &libname, bool close_on_destruct = false)
      : logger_(logger),
        module_(dlopen(libname.c_str(), RTLD_NOLOAD | RTLD_NOW)),
        libname_(libname),
        close_on_destruct_(close_on_destruct)
  {

    if (!module_) {
      module_ = dlopen(libname.c_str(), RTLD_LOCAL | RTLD_NOW);
      if (!module_) {
        throw std::runtime_error("Failed to load shared library " + libname
                                 + ": " + std::string(dlerror()));
      }
    }
  }

  ~SharedLib()
  {
    if (close_on_destruct_) {
      dlclose(module_);
    }
  }

  bool has_symbol(const std::string &symbol)
  {
    return get_symbol(symbol, false) != nullptr;
  }

  void *get_symbol(const std::string &symbol, bool required = true)
  {
    logger_(AX_DEBUG) << "Getting " << libname_ << ":" << symbol;
    auto p = dlsym(module_, symbol.c_str());
    if (!p && required) {
      logger_(AX_DEBUG) << " not found." << std::endl;
      throw std::runtime_error("Failed to load symbol " + symbol + " from "
                               + libname_ + ": " + std::string(dlerror()));
    }
    logger_(AX_DEBUG) << " == " << p << std::endl;
    return p;
  }

  template <typename FunctionType>
  bool initialise_function(const std::string &func_name, FunctionType &f, bool required = true)
  {
    f = reinterpret_cast<FunctionType>(get_symbol(func_name, required));
    return f != nullptr;
  }

  std::string libname() const
  {
    return libname_;
  }

  private:
  Ax::Logger &logger_;
  void *module_;
  const std::string libname_;
  const bool close_on_destruct_;
};


std::string to_string(const std::set<int> &s);

inline std::string
get_env(const std::string &name, const std::string &default_value)
{
  const char *env = std::getenv(name.c_str());
  return env ? env : default_value;
}

struct slice_overlap {
  size_t num_slices; //  Number of slices required in this dimension
  size_t overlap; //  Amount that they will actually overlap
};

slice_overlap determine_overlap(size_t image_size, size_t slice_size, size_t overlap);

void load_v1_plugin(SharedLib &lib, V1Plugin::DetermineObjectAttribute &plugin);
void load_v1_plugin(SharedLib &lib, V1Plugin::TrackerFilter &plugin);

AxVideoInterface create_roi(
    const AxVideoInterface &original, int x, int y, int width, int height);

void validate_output_format(AxVideoFormat format, std::string_view prop_fmt,
    std::string_view label, std::span<const AxVideoFormat> valid_formats);

template <typename PluginType, typename PluginBase>
class LoadedPlugin : public PluginBase
{
  public:
  LoadedPlugin(Ax::Logger &logger, Ax::SharedLib &&shared, std::string options,
      AxAllocationContext *context, std::string mode = "none");

  std::string name() const override
  {
    return name_;
  }

  std::string mode() const override
  {
    return mode_;
  }

  virtual ~LoadedPlugin() = default;

  const Ax::StringSet &allowed_properties() const override
  {
    return allowed_;
  }

  void set_dynamic_properties(const Ax::StringMap &options) override
  {
    if (fns.set_dynamic_properties) {
      fns.set_dynamic_properties(options, subplugin_data.get(), logger);
    }
  }

  protected:
  PluginType fns;
  std::shared_ptr<void> subplugin_data;
  Ax::Logger &logger;
  Ax::SharedLib shared_;
  std::string name_;
  std::string mode_;
  StringSet allowed_;
};

class LoadedInPlace : public LoadedPlugin<Ax::V1Plugin::InPlace, InPlace>
{
  public:
  using LoadedPlugin<Ax::V1Plugin::InPlace, InPlace>::LoadedPlugin;

  bool has_inplace() const override
  {
    return fns.inplace != nullptr;
  }

  void inplace(const AxDataInterface &interface, unsigned int subframe_index,
      unsigned int number_of_subframes, MetaMap &map) override
  {
    fns.inplace(interface, subplugin_data.get(), subframe_index,
        number_of_subframes, map, logger);
  }
};

class LoadedTransform : public LoadedPlugin<Ax::V1Plugin::Transform, Transform>
{
  public:
  using LoadedPlugin<Ax::V1Plugin::Transform, Transform>::LoadedPlugin;

  bool has_transform() const override
  {
    return fns.transform != nullptr;
  }

  void transform(const AxDataInterface &input, const AxDataInterface &output,
      unsigned int subframe_index, unsigned int subframe_count, MetaMap &meta_map) override
  {
    fns.transform(input, output, subplugin_data.get(), subframe_index,
        subframe_count, meta_map, logger);
  }

  bool has_set_output_interface() const override
  {
    return fns.set_output_interface != nullptr;
  }

  bool has_set_output_interface_from_meta() const override
  {
    return fns.set_output_interface_from_meta != nullptr;
  }

  AxDataInterface set_output_interface(const AxDataInterface &input) override
  {
    return fns.set_output_interface ?
               fns.set_output_interface(input, subplugin_data.get(), logger) :
               input;
  }

  AxDataInterface set_output_interface_from_meta(const AxDataInterface &input,
      unsigned int subframe_index, unsigned int number_of_subframes,
      std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &meta_map) override
  {
    return fns.set_output_interface_from_meta(input, subplugin_data.get(),
        subframe_index, number_of_subframes, meta_map, logger);
  }

  bool can_passthrough(const AxDataInterface &input, const AxDataInterface &output) const override
  {
    return fns.can_passthrough
           && fns.can_passthrough(input, output, subplugin_data.get(), logger);
  }

  bool query_supports(PluginFeature feature) const override
  {
    return fns.query_supports && fns.query_supports(feature, subplugin_data.get(), logger);
  }
};

class LoadedDecode : public LoadedPlugin<Ax::V1Plugin::Decode, Decode>
{
  public:
  using LoadedPlugin<Ax::V1Plugin::Decode, Decode>::LoadedPlugin;

  void decode_to_meta(const AxTensorsInterface &tensors, unsigned int subframe_index,
      unsigned int number_of_subframes, MetaMap &map, const AxDataInterface &video) override
  {
    return fns.decode_to_meta(tensors, subplugin_data.get(), subframe_index,
        number_of_subframes, map, video, logger);
  }
};


} // namespace Ax
