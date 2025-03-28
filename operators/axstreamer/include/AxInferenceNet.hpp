// Copyright Axelera AI, 2025
#pragma once

#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include "AxDataInterface.h"
#include "AxLog.hpp"
#include "AxMeta.hpp"


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

  // Signal that the end of input has been reached, remaining frames will be
  // flushed.  After end_of_input no new frames should be pushed.
  virtual void end_of_input() = 0;

  // Stop the inference pipeline, joins all threads and releases resources
  virtual void stop() = 0;

  virtual ~InferenceNet() = default;
};

/// Parse a string of InferenceNetProperties. Properties are newline separated.
InferenceNetProperties properties_from_string(const std::string &s, Ax::Logger &logger);

std::unique_ptr<InferenceNet> create_inference_net(
    const InferenceNetProperties &properties, Ax::Logger &logger,
    InferenceDoneCallback done_callback, LatencyCallback latency_callback = {});

} // namespace Ax
