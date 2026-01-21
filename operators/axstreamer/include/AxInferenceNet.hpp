// Copyright Axelera AI, 2026
#pragma once

#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include "AxDataInterface.h"
#include "AxLog.hpp"
#include "AxMeta.hpp"
#include "AxStreamerUtils.hpp"


namespace Ax
{

constexpr int MAX_OPERATORS = 8; // arbirary number of operators

struct InferenceProperties {
  std::string model;
  bool double_buffer{ false };
  bool dmabuf_inputs{ false };
  bool dmabuf_outputs{ false };
  int skip_stride{ 1 };
  int skip_count{ 0 };
  int num_children{ 0 };
  std::string options;
  std::string meta;
  std::string devices;
  std::string which_cl;
  int max_buffers{ 0 };
};

struct OperatorProperties {
  std::string lib;
  std::string options;
  std::string mode;
  std::string batch;
};

struct InferenceNetProperties : InferenceProperties {
  OperatorProperties preproc[MAX_OPERATORS];
  OperatorProperties postproc[MAX_OPERATORS];
};

using MetaMap = std::unordered_map<std::string, std::unique_ptr<AxMetaBase>>;
using time_point = std::chrono::high_resolution_clock::time_point;

struct CompletedFrame {
  bool end_of_input = false;
  int stream_id;
  uint64_t frame_id;
  std::shared_ptr<void> buffer_handle{};
  AxVideoInterface video = {};
  MetaMap *meta_map = nullptr;
};

using LatencyCallback = std::function<void(const std::string &, uint64_t, uint64_t)>;
using InferenceDoneCallback = std::function<void(CompletedFrame &)>;
class InferenceNet
{
  public:
  // Enqueue a new input frame to be inferenced. When it is complete the passed
  // buffer will be returned via the done_callback.
  virtual void push_new_frame(std::shared_ptr<void> &&buffer,
      const AxVideoInterface &video, MetaMap &axmetamap, int stream_id = 0)
      = 0;

  // Take the result from a previous InferenceNet and pass it to the next in a
  // cascade.  This is called from the done_callback of the previous InferenceNet.
  // If the completed frame has end_of_input set, this InferenceNet will be flushed.
  virtual void cascade_frame(CompletedFrame &frame) = 0;

  // Signal that the end of input has been reached, remaining frames will be
  // flushed.  After end_of_input no new frames should be pushed.
  virtual void end_of_input() = 0;

  // Stop the inference pipeline, joins all threads and releases resources
  virtual void stop() = 0;

  virtual bool supports_opencl_buffers(const AxVideoInterface &video) = 0;

  //  Number of frames required before input can be sent for inference
  //  this is usually the same as batch_size
  virtual int frames_required_for_inference() const = 0;

  virtual ~InferenceNet() = default;
};

template <typename Frame>
void
forward(BlockingQueue<std::shared_ptr<Frame>> &queue, CompletedFrame &done)
{
  // We first check to see if inference is complete and in which case stop
  // the ready queue to indicate to the main loop to exit.
  if (done.end_of_input) {
    queue.stop();
  }
  // If we have a valid inference result, we cast the buffer handle to our own
  // Frame type and push it to the ready queue
  auto frame = std::exchange(done.buffer_handle, {});
  queue.push(std::static_pointer_cast<Frame>(frame));
}
inline void
forward(InferenceNet &net, CompletedFrame &done)
{
  net.cascade_frame(done);
}

// Create an InferenceDoneCallback that will cascade the completed frame to the next InferenceNet
inline InferenceDoneCallback
forward_to(InferenceNet &next)
{
  return [&next](auto &done) { forward(next, done); };
}

// Create an InferenceDoneCallback callback that will push the completed frame to a BlockingQueue
template <typename Frame>
InferenceDoneCallback
forward_to(BlockingQueue<std::shared_ptr<Frame>> &queue)
{
  return [&queue](auto &done) { forward(queue, done); };
}

/// Parse a stream for InferenceNetProperties. Properties are newline separated.
/// Other errors and warnings are reported to the logger.
InferenceNetProperties read_inferencenet_properties(std::istream &s, Ax::Logger &logger);

/// Read InferenceNetProperties from a file. If the file does not exist
/// Other errors and warnings are reported to the logger.
InferenceNetProperties read_inferencenet_properties(const std::string &p, Ax::Logger &logger);

/// @brief Create an InferenceNet with the specified properties.
/// @param properties - The properties that define the inference net, these can
/// be manually specified or loaded from a configuration file.
/// @param logger - The logger to use for logging messages.
/// @param done_callback - called when a frame has completed the pipeline.
/// @param latency_callback - called to report latency measurements.
/// @param allocation_context - If non-null then it is used to avoid copying
/// buffers that were created in axtransform operators to AxInferenceNet.
std::unique_ptr<InferenceNet> create_inference_net(const InferenceNetProperties &properties,
    Ax::Logger &logger, InferenceDoneCallback done_callback);
std::unique_ptr<InferenceNet> create_inference_net(
    const InferenceNetProperties &properties, Ax::Logger &logger,
    InferenceDoneCallback done_callback, LatencyCallback latency_callback);
std::unique_ptr<InferenceNet> create_inference_net(const InferenceNetProperties &properties,
    Ax::Logger &logger, InferenceDoneCallback done_callback,
    LatencyCallback latency_callback, AxAllocationContext *allocation_context);
} // namespace Ax
