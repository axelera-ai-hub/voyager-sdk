// Copyright Axelera AI, 2025
#pragma once

#include <memory>
#include <set>
#include <string>
#include <vector>
#include "AxDataInterface.h"
#include "AxInferenceNet.hpp"
#include "AxMeta.hpp"
#include "AxStreamerUtils.hpp"

struct axrContext;
struct axrModel;

namespace Ax
{

inline int
num_devices(const InferenceProperties &props)
{
  return std::count(props.devices.begin(), props.devices.end(), ',') + 1;
}

inline int
pipeline_pre_fill(const InferenceProperties &props)
{
  const auto ndevices = num_devices(props);
  return std::max(1, props.num_children) * ndevices;
}

inline int
output_drop(const InferenceProperties &props)
{
  const auto ndevices = num_devices(props);
  const auto num_children = std::max(1, props.num_children) * ndevices;
  return 2 * num_children * props.double_buffer;
}


struct InferenceParams {
  InferenceParams() = default;
  InferenceParams(const std::vector<std::shared_ptr<void>> &input_ptrs,
      const std::vector<Ax::SharedFD> &input_fds,
      const std::vector<std::shared_ptr<void>> &output_ptrs,
      const std::vector<Ax::SharedFD> &output_fds, uint64_t frame_id)
      : input_ptrs(input_ptrs),
        input_fds(input_fds),
        output_ptrs(output_ptrs),
        output_fds(output_fds),
        frame_id(frame_id)
  {
  }
  InferenceParams(InferenceParams &&) = default;
  InferenceParams(const InferenceParams &) = delete;
  InferenceParams &operator=(InferenceParams &&) = default;
  InferenceParams &operator=(const InferenceParams &) = delete;

  explicit operator bool() const
  {
    return !(input_ptrs.empty() && input_fds.empty());
  }
  bool operator!() const
  {
    return !static_cast<bool>(*this);
  }
  std::vector<std::shared_ptr<void>> input_ptrs;
  std::vector<Ax::SharedFD> input_fds;
  std::vector<std::shared_ptr<void>> output_ptrs;
  std::vector<Ax::SharedFD> output_fds;
  uint64_t frame_id; // Used for order reconstruction in low-latency inference
};


class BasicInference
{
  public:
  virtual ~BasicInference() = default;
  virtual InferenceParams execute(InferenceParams params) = 0;
};

class Inference
{
  public:
  virtual ~Inference() = default;
  virtual bool is_low_latency() const = 0;
  virtual int batch_size() const = 0;
  virtual const AxTensorsInterface &input_shapes() const = 0;
  virtual const AxTensorsInterface &output_shapes() const = 0;
  virtual void dispatch(InferenceParams params) = 0;
  virtual void collect() = 0;
};

using InferenceReadyCallback = std::function<void(uint64_t frame_id)>;
std::unique_ptr<Inference> create_inference(Logger &logger,
    const InferenceProperties &props, InferenceReadyCallback callback);

std::unique_ptr<BasicInference> create_axruntime_inference(Logger &logger,
    axrContext *ctx, axrModel *model, const InferenceProperties &props);

} // namespace Ax
