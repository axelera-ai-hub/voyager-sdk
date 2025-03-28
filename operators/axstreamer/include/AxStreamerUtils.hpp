// Copyright Axelera AI, 2025
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
#include <queue>
#include <set>
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

namespace Ax
{
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
void
clear_queue(std::queue<T> &q)
{
  while (!q.empty()) {
    q.pop();
  }
}

struct DmaBufHandle {
  explicit DmaBufHandle(int fd, bool should_close = true)
      : fd(fd), should_close(should_close)
  {
  }
  DmaBufHandle(const DmaBufHandle &) = delete;
  DmaBufHandle &operator=(const DmaBufHandle &) = delete;
  DmaBufHandle(DmaBufHandle &&) = delete;
  DmaBufHandle &operator=(DmaBufHandle &&) = delete;
  ~DmaBufHandle()
  {
    if (fd >= 0 && should_close) {
      ::close(fd);
    }
  }
  const int fd;

  private:
  const bool should_close;
};
using SharedFD = std::shared_ptr<DmaBufHandle>;


namespace Internal
{

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

std::unordered_map<std::string, std::string> extract_options(
    Ax::Logger &logger, const std::string &opts);

template <typename Iterator> class Enumerator
{
  public:
  Enumerator(Iterator iter, std::size_t index) : iter_(iter), index_(index)
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
  explicit IntegerIterator(T value) : value_(value)
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
      : begin_(begin), end_(end)
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

std::unordered_map<std::string, std::string> parse_and_validate_plugin_options(
    Ax::Logger &logger, const std::string &options,
    const std::unordered_set<std::string> &allowed_properties);

class ManagedDataInterface
{
  public:
  ManagedDataInterface(const AxDataInterface &data) : data_(data)
  {
  }
  ManagedDataInterface(const ManagedDataInterface &data) = delete;
  ManagedDataInterface &operator=(const ManagedDataInterface &data) = delete;
  ManagedDataInterface(ManagedDataInterface &&data) = default;
  ManagedDataInterface &operator=(ManagedDataInterface &&data) = default;

  ManagedDataInterface clone() const
  {
    ManagedDataInterface res(data_);
    res.buffers_ = buffers_;
    res.fds_ = fds_;
    return res;
  }

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


  void allocate(void *(allocator) (size_t))
  {
    buffers_.clear();
    fds_.clear();
    if (auto *video = std::get_if<AxVideoInterface>(&data_)) {

      auto *p = allocator(video->info.stride * video->info.height
                          * AxVideoFormatNumChannels(video->info.format));
      video->data = p;
      buffers_.emplace_back(p, std::free);
    } else if (auto *tensors = std::get_if<AxTensorsInterface>(&data_)) {
      for (auto &tensor : *tensors) {
        auto *p = allocator(tensor.total_bytes());
        tensor.data = p;
        buffers_.emplace_back(p, std::free);
      }
    }
  }

  template <typename F> void allocate(F &&allocator)
  {
    buffers_.clear();
    fds_.clear();
    if (auto *video = std::get_if<AxVideoInterface>(&data_)) {
      auto fd = allocator(video->info.stride * video->info.height
                          * AxVideoFormatNumChannels(video->info.format));
      video->data = nullptr;
      fds_.push_back(std::move(fd));
    } else if (auto *tensors = std::get_if<AxTensorsInterface>(&data_)) {
      for (auto &tensor : *tensors) {
        auto fd = allocator(tensor.total_bytes());
        tensor.data = nullptr;
        fds_.push_back(std::move(fd));
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

  private:
  AxDataInterface data_;
  std::vector<std::shared_ptr<void>> buffers_;
  std::vector<SharedFD> fds_;
};

using ManagedDataInterfaces = std::list<ManagedDataInterface>;

class DataInterfaceAllocator
{
  public:
  virtual ManagedDataInterface allocate(const AxDataInterface &data) = 0;
  virtual void map(ManagedDataInterface &data) = 0;
  virtual void unmap(ManagedDataInterface &data) = 0;
  virtual ~DataInterfaceAllocator() = default;
};

class NullDataInterfaceAllocator : public DataInterfaceAllocator
{
  public:
  ManagedDataInterface allocate(const AxDataInterface &data) override
  {
    return ManagedDataInterface(data);
  }
  void map(ManagedDataInterface &) override
  {
  }
  void unmap(ManagedDataInterface &) override
  {
  }
};

std::unique_ptr<DataInterfaceAllocator> create_heap_allocator();
std::unique_ptr<DataInterfaceAllocator> create_dma_buf_allocator();

ManagedDataInterface allocate_batched_buffer(int batch_size,
    const AxDataInterface &input, DataInterfaceAllocator &allocator);

AxDataInterface batch_view(const AxDataInterface &i, int n);

class BatchedBuffer;
class SharedBatchBufferView
{
  public:
  SharedBatchBufferView() = default;
  SharedBatchBufferView(std::shared_ptr<BatchedBuffer> self, AxDataInterface *view)
      : self_(self), view_(view)
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
  BatchedBuffer(int batch_size, const AxDataInterface &iface, DataInterfaceAllocator &allocator)
      : batched(allocate_batched_buffer(batch_size, iface, allocator)),
        allocator(allocator)
  {
    for (int i = 0; i < batch_size; ++i) {
      views.push_back(batch_view(batched.data(), i));
    }
  }

  BatchedBuffer &operator=(const BatchedBuffer &other) = delete;
  BatchedBuffer &operator=(BatchedBuffer &&other) = delete;
  BatchedBuffer(BatchedBuffer &&other) = delete;
  BatchedBuffer(const BatchedBuffer &other) = delete;
  ~BatchedBuffer() = default;

  void map()
  {
    allocator.map(batched);
    update_views();
  }

  const ManagedDataInterface &get_batched(bool unmap_ = false)
  {
    if (unmap_) {
      allocator.unmap(batched);
      update_views();
    }
    return batched;
  }

  // create a shared ptr to a view that will ensure the batch buffer is not
  // removed until all views of it are done with
  friend SharedBatchBufferView get_shared_view_of_batch_buffer(
      std::shared_ptr<BatchedBuffer> self, int batch)
  {
    return SharedBatchBufferView(self, &self->views[batch]);
  }

  void set_iface(const AxDataInterface &iface)
  {
    batched.set_data(iface);
    update_views();
  }

  int batch_size() const
  {
    return views.size();
  }

  private:
  void update_views()
  {
    size_t n = 0;
    for (auto &view : views) {
      // TODO this is inefficient, we just need to init the data param
      view = batch_view(batched.data(), n);
      ++n;
    }
  }
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
    return va.info.stride == vb.info.stride && va.info.height == vb.info.height
           && AxVideoFormatNumChannels(va.info.format)
                  == AxVideoFormatNumChannels(vb.info.format);
  } else if (std::holds_alternative<AxVideoInterface>(a)
             && std::holds_alternative<AxTensorsInterface>(b)) {
    auto &va = std::get<AxVideoInterface>(a);
    auto &tb = std::get<AxTensorsInterface>(b);
    return tb.size() == 1 && va.strides.size() == 1
           && tb[0].total_bytes()
                  == va.info.stride * va.info.height
                         * AxVideoFormatNumChannels(va.info.format);
  } else if (std::holds_alternative<AxTensorsInterface>(a)
             && std::holds_alternative<AxVideoInterface>(b)) {
    auto &ta = std::get<AxTensorsInterface>(a);
    auto &vb = std::get<AxVideoInterface>(b);
    return ta.size() == 1 && vb.strides.size() == 1
           && ta[0].total_bytes()
                  == vb.info.stride * vb.info.height
                         * AxVideoFormatNumChannels(vb.info.format);
  }

  return false;
}

class BatchedBufferPool
{
  public:
  BatchedBufferPool(int batch_size, const AxDataInterface &iface,
      DataInterfaceAllocator &allocator)
      : batch_size_(batch_size), iface_(iface), allocator_(allocator),
        free_(std::make_shared<std::vector<std::unique_ptr<BatchedBuffer>>>())
  {
  }


  std::shared_ptr<BatchedBuffer> new_batched_buffer(
      const AxDataInterface &new_iface, bool map = true)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!are_equivalent(iface_, new_iface)) {
      iface_ = new_iface;
      free_ = std::make_shared<std::vector<std::unique_ptr<BatchedBuffer>>>();
    }

    std::unique_ptr<BatchedBuffer> buffer;
    auto &free = *free_;
    if ((free).empty()) {
      buffer.reset(new BatchedBuffer(batch_size_, iface_, allocator_));
    } else {
      buffer = std::move(free.back());
      free.pop_back();
    }
    if (map) {
      buffer->map();
    }
    //  The destructor captures the free list so that even if the member free_
    //  is destroyed the buffer will still be returned to the free list which
    //  will be finally destroyed once the last buffer that belongs to that free
    //  list is returned (Python-like garbage collection)
    return std::shared_ptr<BatchedBuffer>(
        buffer.release(), [this, free = free_](BatchedBuffer *buffer) {
          std::lock_guard<std::mutex> lock(mutex_);
          (*free).emplace_back(buffer);
        });
  }

  std::shared_ptr<BatchedBuffer> new_batched_buffer(bool map = true)
  {
    return new_batched_buffer(iface_, map);
  }

  private:
  int batch_size_;
  AxDataInterface iface_;
  DataInterfaceAllocator &allocator_;
  std::mutex mutex_;
  std::shared_ptr<std::vector<std::unique_ptr<BatchedBuffer>>> free_;
};

template <typename T, int queue_size = 1> struct BlockingQueue {
  void push(T &&t)
  {
    std::unique_lock<std::mutex> lock(mutex);
    full_condition.wait(
        lock, [this] { return queue.size() < queue_depth || !running; });
    queue.push(std::move(t));
    empty_condition.notify_one();
  }

  T wait_one()
  {
    std::unique_lock<std::mutex> lock(mutex);
    empty_condition.wait(lock, [this] { return !queue.empty() || !running; });
    if (!running) {
      return {};
    }
    auto t = pop_queue(queue);
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
  size_t queue_depth = queue_size;
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

  SharedLib(Ax::Logger &logger, const std::string &libname)
      : logger_(logger), module_(dlopen(libname.c_str(), RTLD_LOCAL | RTLD_NOW)),
        libname_(libname)
  {
    if (!module_) {
      logger_(AX_ERROR) << "Failed to open shared library " << libname_ << std::endl;
      throw std::runtime_error("Shared library " + libname_ + " could not be opened. "
                               + std::string(dlerror()));
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
  std::string libname_;
};

std::string to_string(const std::set<int> &s);
std::set<int> create_stream_set(std::string &input);

inline std::string
get_env(const std::string &name, const std::string &default_value)
{
  const char *env = std::getenv(name.c_str());
  return env ? env : default_value;
}

void load_v1_plugin(SharedLib &lib, V1Plugin::InPlace &plugin);
void load_v1_plugin(SharedLib &lib, V1Plugin::Transform &plugin);
void load_v1_plugin(SharedLib &lib, V1Plugin::Decoder &plugin);

} // namespace Ax
