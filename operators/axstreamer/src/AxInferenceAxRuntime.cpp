// Copyright Axelera AI, 2024
#include "AxInference.hpp"
#include "AxStreamerUtils.hpp"

#include <algorithm>
#include <axruntime/axruntime.hpp>
#include <chrono>
#include <fstream>
#include <semaphore>
#include <thread>

using namespace std::string_literals;
using axr::to_ptr;

namespace
{

template <size_t N>
void
fill_char_array(char (&arr)[N], const std::string &str)
{
  std::fill(std::begin(arr), std::end(arr), 0);
  std::copy(str.begin(), str.begin() + std::min(N - 1, str.size()), arr);
}

axr::ptr<axrProperties>
create_properties(axrContext *context, bool input_dmabuf, bool output_dmabuf,
    bool double_buffer, int num_sub_devices, bool async_mode, int max_inflight, int max_pending)
{
  std::string s;
  s += "input_dmabuf=" + std::to_string(int(input_dmabuf)) + "\n";
  s += "output_dmabuf=" + std::to_string(int(output_dmabuf)) + "\n";
  s += "num_sub_devices=" + std::to_string(num_sub_devices) + "\n";
  s += "aipu_cores=" + std::to_string(num_sub_devices) + "\n";
  s += "double_buffer=" + std::to_string(int(double_buffer)) + "\n";
  s += "async_mode=" + std::to_string(int(async_mode)) + "\n";
  if (async_mode) {
    s += "max_inflight=" + std::to_string(max_inflight) + "\n";
    s += "max_pending=" + std::to_string(max_pending) + "\n";
  }
  return to_ptr(axr_create_properties(context, s.c_str()));
}

axr::ptr<axrProperties>
create_conn_properties(axrContext *context)
{
  std::string s;
  s += "device_firmware_check=0"; // AF checks this further up
  return to_ptr(axr_create_properties(context, s.c_str()));
}

static std::mutex second_slice_workaround_mutex;

// Fills input_args/output_args (already sized to the model's input/output
// count) from an InferenceParams. Shared by the blocking and async instance
// implementations below.
void
populate_args(std::vector<axrArgument> &input_args,
    std::vector<axrArgument> &output_args, const Ax::InferenceParams &p)
{
  if (p.input_ptrs.empty()) {
    assert(p.input_fds.size() == input_args.size());
    for (auto &&[i, shared_fd] : Ax::Internal::enumerate(p.input_fds)) {
      input_args[i].fd = shared_fd->fd;
      input_args[i].ptr = nullptr;
      input_args[i].offset = 0;
    }
  } else {
    assert(p.input_ptrs.size() == input_args.size());
    for (auto &&[i, ptr] : Ax::Internal::enumerate(p.input_ptrs)) {
      input_args[i].fd = 0;
      input_args[i].ptr = ptr.get();
      input_args[i].offset = 0;
    }
  }
  if (!p.output_ptrs.empty()) {
    assert(p.output_ptrs.size() == output_args.size());
    for (auto &&[i, ptr] : Ax::Internal::enumerate(p.output_ptrs)) {
      output_args[i].fd = 0;
      output_args[i].ptr = ptr.get();
      output_args[i].offset = 0;
    }
  } else if (!p.output_fds.empty()) {
    assert(p.output_fds.size() == output_args.size());
    for (auto &&[i, shared_fd] : Ax::Internal::enumerate(p.output_fds)) {
      output_args[i].fd = shared_fd->fd;
      output_args[i].ptr = nullptr;
      output_args[i].offset = 0;
    }
  }
}

// Shared model-connect/load-instance setup for both the blocking and async
// axruntime-backed BasicInference/AsyncBasicInference implementations.
class AxRuntimeConnectionBase
{
  protected:
  AxRuntimeConnectionBase(Ax::Logger &logger, axrContext *ctx, axrModel *model,
      const Ax::InferenceProperties &props, bool async_mode)
      : logger(logger)
  {
    // level-zero/triton/kmd has issues if we try to load the model from
    // multiple threads. So lock here to load a model at a time. This is a
    // workaround for the issue, and it needs fixing lower down.
    // Proper fix tracked here https://axeleraai.atlassian.net/browse/SDK-6708
    std::lock_guard lock(second_slice_workaround_mutex);
    num_inputs = axr_num_model_inputs(model);
    auto input0 = axr_get_model_input(model, 0);
    num_outputs = axr_num_model_outputs(model);
    logger(AX_INFO) << "Loaded model " << props.model << " with " << num_inputs
                    << " inputs and " << num_outputs << " outputs" << std::endl;

    auto device = axrDeviceInfo{};
    fill_char_array(device.name, props.devices);
    const auto *pdevice = props.devices.empty() ? nullptr : &device;
    const auto num_sub_devices = input0.dims[0];
    const auto conn_props = create_conn_properties(ctx);
    connection = to_ptr(
        axr_device_connect(ctx, pdevice, num_sub_devices, conn_props.get()));
    if (!connection) {
      throw std::runtime_error(
          "axr_device_connect failed : "s + axr_last_error_string(AXR_OBJECT(ctx)));
    }
    const auto load_props = create_properties(ctx, props.dmabuf_inputs,
        props.dmabuf_outputs, props.double_buffer, num_sub_devices, async_mode,
        props.max_inflight, props.max_pending);
    instance
        = to_ptr(axr_load_model_instance(connection.get(), model, load_props.get()));
    if (!instance) {
      throw std::runtime_error("axr_load_model_instance failed : "s
                               + axr_last_error_string(AXR_OBJECT(ctx)));
    }
  }

  Ax::Logger &logger;
  axr::ptr<axrConnection> connection;
  axr::ptr<axrModelInstance> instance;
  int num_inputs = 0;
  int num_outputs = 0;
};

class AxRuntimeInference : public AxRuntimeConnectionBase, public Ax::BasicInference
{
  public:
  AxRuntimeInference(Ax::Logger &logger, axrContext *ctx, axrModel *model,
      const Ax::InferenceProperties &props)
      : AxRuntimeConnectionBase(logger, ctx, model, props, /*async_mode=*/false)
  {
    input_args.resize(num_inputs);
    output_args.resize(num_outputs);
  }

  Ax::InferenceParams execute(Ax::InferenceParams p) override
  {
    populate_args(input_args, output_args, p);
    auto res = axr_run_model_instance(instance.get(), input_args.data(),
        input_args.size(), output_args.data(), output_args.size());
    if (res != AXR_SUCCESS) {
      throw std::runtime_error("axr_run_model failed with "s
                               + axr_last_error_string(AXR_OBJECT(instance.get())));
    }
    return p;
  }

  private:
  std::vector<axrArgument> input_args;
  std::vector<axrArgument> output_args;
};

// Async-executor-backed instance. submit() never blocks the caller except to
// respect this instance's own host_depth backlog; completion is reported via
// on_complete_ from whatever thread axruntime's async executor calls back on.
class AsyncAxRuntimeInference : public AxRuntimeConnectionBase, public Ax::AsyncBasicInference
{
  public:
  // Host-side outstanding-submission depth: deliberately kept much shallower
  // than the device's own max_pending/max_inflight (roughly one submission
  // running and one queued behind it), rather than matched to it. The
  // device's deeper pipelining is still configured via max_inflight/
  // max_pending and passed through unchanged; this is purely our own
  // client-side pacing. Keeping it shallow means we rarely push axruntime's
  // internal queue anywhere near full from the client side, which avoids the
  // race in axruntime's finalize_workload() (it invokes our completion
  // callback before requeuing the workload onto its own free list) under
  // normal load, rather than relying solely on the submit() retry below.
  // Clamped to max_pending itself in case that's configured smaller than
  // default_host_depth, so we never try to keep more requests outstanding
  // than the device can actually hold.
  static constexpr int default_host_depth = 2;

  // Retry budget for the transient "queue full" race in submit(): a short
  // sleep between attempts, ~2s total before giving up and reporting a real
  // failure.
  static constexpr int submit_retries = 10000;
  static constexpr std::chrono::microseconds submit_backoff{ 200 };

  AsyncAxRuntimeInference(Ax::Logger &logger, axrContext *ctx, axrModel *model,
      const Ax::InferenceProperties &props,
      std::function<void(Ax::InferenceParams, bool)> on_complete)
      : AxRuntimeConnectionBase(logger, ctx, model, props, /*async_mode=*/true),
        on_complete_(std::move(on_complete)),
        host_depth_(std::min(default_host_depth, std::max(1, props.max_pending))),
        slots_(host_depth_)
  {
  }

  void submit(Ax::InferenceParams params) override
  {
    auto req = std::make_unique<PendingRequest>();
    req->self = this;
    req->input_args.resize(num_inputs);
    req->output_args.resize(num_outputs);
    populate_args(req->input_args, req->output_args, params);
    req->params = std::move(params);
    auto *raw = req.release();
    // Acquire only once the above (which can throw bad_alloc) is done, so a
    // failure there can't leak a permit that would never be released.
    slots_.acquire();
    // axruntime's own finalize_workload() invokes our completion callback
    // (which releases slots_) before it requeues the workload onto its
    // internal free list. That leaves a brief window where slots_ says a
    // slot is free but axr_run_async_model_instance still reports "queue
    // full". host_depth_ being much shallower than max_pending makes this
    // rare, but retry with a short sleep-based backoff (~2s budget) to ride
    // out the window rather than surfacing a spurious failure and
    // over-releasing slots_.
    axrResult res;
    for (int attempt = 0; attempt < submit_retries; ++attempt) {
      res = axr_run_async_model_instance(instance.get(), raw->input_args.data(),
          raw->input_args.size(), raw->output_args.data(),
          raw->output_args.size(), &on_complete_trampoline, raw);
      if (res == AXR_SUCCESS) {
        break;
      }
      //  This should be a rare case, but if it happens we will retry a few times with a short sleep
      std::this_thread::sleep_for(submit_backoff);
    }
    if (res != AXR_SUCCESS) {
      logger(AX_ERROR)
          << "axr_run_async_model_instance failed with "
          << axr_last_error_string(AXR_OBJECT(instance.get())) << std::endl;
      std::unique_ptr<PendingRequest> reclaimed(raw);
      on_complete_(std::move(reclaimed->params), false);
      slots_.release();
    }
  }

  void drain() override
  {
    // Blocks until every outstanding submission's completion has run and
    // released its slot. Release the permits back afterwards so drain() can
    // be called more than once (e.g. by a caller that drains defensively in
    // more than one place) without deadlocking on a subsequent call.
    for (int i = 0; i != host_depth_; ++i) {
      slots_.acquire();
    }
    for (int i = 0; i != host_depth_; ++i) {
      slots_.release();
    }
  }

  private:
  struct PendingRequest {
    Ax::InferenceParams params;
    std::vector<axrArgument> input_args;
    std::vector<axrArgument> output_args;
    AsyncAxRuntimeInference *self = nullptr;
  };

  static void on_complete_trampoline(void *arg, axrResult result)
  {
    std::unique_ptr<PendingRequest> req(static_cast<PendingRequest *>(arg));
    auto *self = req->self;
    // Call the completion callback before releasing the slot: drain() only
    // waits on slots_, so releasing first would let drain() (and the
    // destructor it guards) return while this callback is still running,
    // racing with teardown of the object it was constructed to call back
    // into.
    self->on_complete_(std::move(req->params), result == AXR_SUCCESS);
    self->slots_.release();
  }

  std::function<void(Ax::InferenceParams, bool)> on_complete_;
  int host_depth_;
  std::counting_semaphore<64> slots_;
};
} // namespace

std::unique_ptr<Ax::BasicInference>
Ax::create_axruntime_inference(Ax::Logger &logger, axrContext *ctx,
    axrModel *model, const InferenceProperties &props)
{
  return std::make_unique<AxRuntimeInference>(logger, ctx, model, props);
}

std::unique_ptr<Ax::AsyncBasicInference>
Ax::create_async_axruntime_inference(Ax::Logger &logger, axrContext *ctx,
    axrModel *model, const InferenceProperties &props,
    std::function<void(InferenceParams, bool)> on_complete)
{
  return std::make_unique<AsyncAxRuntimeInference>(
      logger, ctx, model, props, std::move(on_complete));
}
