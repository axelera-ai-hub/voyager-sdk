// Copyright Axelera AI, 2026
#include "AxLog.hpp"
#include "AxOpenCl.hpp"
#include "AxStreamerUtils.hpp"


static Ax::Logger defaultLogger(Ax::Severity::warning);

extern "C" AxAllocationContext *
ax_create_allocation_context(const char *which_cl, Ax::Logger *logger)
{
  auto details = ax_utils::build_cl_details(
      logger ? *logger : defaultLogger, which_cl ? which_cl : "auto", nullptr);
  if (!details.context || !details.commands) {
    return nullptr;
  }
  return new AxAllocationContext{ details };
}

extern "C" void
ax_free_allocation_context(AxAllocationContext *context)
{
  delete context;
}

struct opencl_buffer_details : opencl_buffer {
  std::vector<std::shared_ptr<Ax::BatchedBuffer>> dependent_buffers{};

  ~opencl_buffer_details()
  {
    if (buffer) {
      clReleaseMemObject(buffer);
      buffer = nullptr;
    }
    if (event) {
      event.reset();
    }
    mapped = nullptr;
    dependent_buffers.clear();
    std::free(data.data());
  }
};

class OpenCLAllocator : public Ax::DataInterfaceAllocator
{
  public:
  OpenCLAllocator(AxAllocationContext *context, Ax::Logger &logger)
      : context_(ax_utils::copy_context_and_retain(context)), logger(logger)
  {
  }

  opencl_buffer_details *allocate_buffer(size_t size)
  {
    auto buffer = std::make_unique<opencl_buffer_details>();
    auto page_size = 4096;
    auto adjusted_size = (size + page_size - 1) & ~(page_size - 1);
    auto *data = std::aligned_alloc(page_size, adjusted_size);
    buffer->data = std::span<uint8_t>(reinterpret_cast<uint8_t *>(data), adjusted_size);
    return buffer.release();
  }

  Ax::ManagedDataInterface allocate(const AxDataInterface &data) override
  {
    Ax::ManagedDataInterface buffer{ data };
    //  Allocate the buffer along with
    buffer.allocate([this](size_t size) { return allocate_buffer(size); });
    return buffer;
  }

  void *map(opencl_buffer &ocl_buffer)
  {
    if (ocl_buffer.buffer) {
      if (ocl_buffer.event) {
        clWaitForEvents(1, &*ocl_buffer.event);
        ocl_buffer.event.reset();
      } else {
        cl_int error = CL_SUCCESS;
        auto *mapped = clEnqueueMapBuffer(context_.commands, ocl_buffer.buffer, CL_TRUE,
            CL_MAP_READ, 0, ocl_buffer.data.size(), 0, nullptr, nullptr, &error);
        if (!mapped) {
          logger(AX_ERROR) << "Failed to map OpenCL buffer, error = " << error << std::endl;
          return nullptr;
        }
        ocl_buffer.mapped = mapped;
      }
      return ocl_buffer.mapped;
    }
    return ocl_buffer.data.data();
  }

  void unmap(opencl_buffer &ocl_buffer)
  {
    cl_int error = clEnqueueUnmapMemObject(
        context_.commands, ocl_buffer.buffer, ocl_buffer.mapped, 0, NULL, NULL);
    ocl_buffer.mapped = nullptr;
  }

  void map(Ax::ManagedDataInterface &buffer, Ax::MapType /*unused*/) override
  {
    if (!buffer.is_mapped()) {
      std::vector<std::shared_ptr<void>> buffers;
      auto &fds = buffer.fds();
      auto &ocl_buffers = buffer.ocl_buffers();
      auto p = std::shared_ptr<void>{};
      if (auto *video = std::get_if<AxVideoInterface>(&buffer.data())) {
        auto ocl = ocl_buffers[0];
        auto *p = map(*ocl);
        buffers.emplace_back(p, [](void *p) {});
      } else if (auto *tensors = std::get_if<AxTensorsInterface>(&buffer.data())) {
        size_t n = 0;
        for (auto &tensor : *tensors) {
          auto ocl = ocl_buffers[n];
          auto *p = map(*ocl);
          buffers.emplace_back(p, [](void *p) {});
          ++n;
        }
      }
      buffer.set_buffers(std::move(buffers));
    }
  }

  void unmap(Ax::ManagedDataInterface &buffer) override
  {
    if (buffer.is_mapped()) {
      auto &ocl_buffers = buffer.ocl_buffers();
      if (std::get_if<AxVideoInterface>(&buffer.data())) {
        auto ocl = ocl_buffers[0];
        unmap(*ocl);
      } else if (auto *tensors = std::get_if<AxTensorsInterface>(&buffer.data())) {
        size_t n = 0;
        for (size_t n = 0; n != tensors->size(); ++n) {
          auto ocl = ocl_buffers[n];
          unmap(*ocl);
        }
      }
      buffer.set_buffers(std::vector<std::shared_ptr<void>>{});
    }
  }

  void release(Ax::ManagedDataInterface &buffer) override
  {
    //  Remove any dependent buffers
    unmap(buffer);
  }

  virtual ~OpenCLAllocator() = default;

  private:
  ax_utils::opencl_details context_;
  Ax::Logger &logger;
};

std::unique_ptr<Ax::DataInterfaceAllocator>
Ax::create_opencl_allocator(AxAllocationContext *context, Ax::Logger &logger)
{
  return context ? std::make_unique<OpenCLAllocator>(context, logger) :
                   create_heap_allocator();
}
