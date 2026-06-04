// Copyright Axelera AI, 2024
// UNet decoder

#include <algorithm>
#include <chrono>
#include <cmath>
#include <span>
#include <unordered_set>
#include <vector>
#include "AxLog.hpp"
#include "AxMetaSemanticSegmentation.hpp"

namespace semantic_seg
{

// `threshold` is interpreted in probability space when `sigmoid` is true
// (and converted to a logit at init time so the per-pixel loop can compare
// raw logits directly), and in raw-logit space when `sigmoid` is false.
// `threshold` is only used on the single-class path; the multi-class path
// always emits the argmax class.
struct properties {
  std::string meta_name{};
  bool class_map_out{ true };
  std::string decoder_name;
  float threshold{ 0.00001f };
  bool sigmoid{ false };
};
} // namespace semantic_seg

extern "C" void
decode_to_meta(const AxTensorsInterface &in_tensors, const semantic_seg::properties *prop,
    unsigned int current_frame, unsigned int total_frames,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &map,
    const AxDataInterface &video_interface, Ax::Logger &logger)
{
  auto start_time = std::chrono::high_resolution_clock::now();

  auto &tensor = in_tensors[0];

  std::vector<int> size{ tensor.sizes[1], tensor.sizes[2], tensor.sizes[3] };
  auto *fdata = static_cast<float *>(tensor.data);
  if (prop->class_map_out) {
    std::vector<int> max_indices(tensor.sizes[1] * tensor.sizes[2]);
    auto out_it = max_indices.data();
    auto total_size = tensor.sizes[1] * tensor.sizes[2] * tensor.sizes[3];
    if (tensor.sizes[3] == 1) {
      for (int offset = 0; offset != total_size; ++offset) {
        *out_it++ = fdata[offset] > prop->threshold ? 1 : 0;
      }
    } else {
      for (int offset = 0; offset != total_size; offset += tensor.sizes[3]) {
        std::span<float> vec(fdata + offset, tensor.sizes[3]);
        auto max_it = std::max_element(vec.begin(), vec.end());
        *out_it++ = std::distance(vec.begin(), max_it);
      }
    }
    map[prop->meta_name] = std::make_unique<AxMetaSemanticSegmentation>(
        std::move(max_indices), size, prop->decoder_name);


  } else {
    std::vector<float> data(fdata, fdata + tensor.total());
    map[prop->meta_name] = std::make_unique<AxMetaSemanticSegmentation>(
        std::move(data), size, prop->decoder_name);
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
  logger(AX_DEBUG) << "decode_to_meta : Decoding semantic_seg"
                   << duration.count() << " microseconds" << std::endl;
}

extern "C" const std::unordered_set<std::string> &
allowed_properties()
{
  static const std::unordered_set<std::string> allowed_properties{
    "meta_key",
    "class_map_out",
    "decoder_name",
    "threshold",
    "sigmoid",
  };
  return allowed_properties;
}

extern "C" std::shared_ptr<void>
init_and_set_static_properties(
    const std::unordered_map<std::string, std::string> &input, Ax::Logger &logger)
{
  auto props = std::make_shared<semantic_seg::properties>();
  props->meta_name = Ax::get_property(
      input, "meta_key", "decode_static_properties", props->meta_name);

  props->decoder_name = Ax::get_property(
      input, "decoder_name", "decode_static_properties", props->decoder_name);

  props->threshold = Ax::get_property(
      input, "threshold", "decode_static_properties", props->threshold);

  props->class_map_out = Ax::get_property(
      input, "class_map_out", "decode_static_properties", props->class_map_out);

  props->sigmoid
      = Ax::get_property(input, "sigmoid", "decode_static_properties", props->sigmoid);

  if (props->sigmoid) {
    // Avoid log(0) and log(1) by clamping threshold to a reasonable range, then convert to logit
    props->threshold = std::clamp(props->threshold, 0.00001F, 0.99999F);
    props->threshold = std::log(props->threshold / (1.0f - props->threshold));
  }

  return props;
}

extern "C" void
set_dynamic_properties(const std::unordered_map<std::string, std::string> &input,
    semantic_seg::properties *prop, Ax::Logger &logger)
{
}
