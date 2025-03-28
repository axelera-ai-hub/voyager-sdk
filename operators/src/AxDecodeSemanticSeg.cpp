// Copyright Axelera AI, 2024
// UNet decoder

#include <chrono>
#include <span>
#include <unordered_set>
#include <vector>
#include "AxLog.hpp"
#include "AxMetaSemanticSegmentation.hpp"

namespace semantic_seg
{

struct properties {
  std::string meta_name{};
  bool class_map_out{ true };
  std::string decoder_name;
  float threshold{ 0.0f };
  bool sigmoid{ true };
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
    auto out_it = max_indices.begin();
    int offset = 0;
    for (int i = 0; i < tensor.sizes[1]; ++i) {
      for (int j = 0; j < tensor.sizes[2]; ++j) {
        if (tensor.sizes[3] == 1) {
          float value = prop->sigmoid ? ax_utils::to_sigmoid(fdata[offset]) :
                                        fdata[offset];
          *out_it++ = value > prop->threshold ? 1 : 0;
        } else {
          std::span<float> vec(fdata + offset, tensor.sizes[3]);
          auto max_it = std::max_element(vec.begin(), vec.end());
          *out_it++ = *max_it > prop->threshold ? std::distance(vec.begin(), max_it) : -1;
        }
        offset += tensor.sizes[3];
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
  logger(AX_INFO) << "decode_to_meta : Decoding semantic_seg"
                  << duration.count() << " microseconds" << std::endl;
}

extern "C" const std::unordered_set<std::string> &
allowed_properties()
{
  static const std::unordered_set<std::string> allowed_properties{
    "meta_key", "class_map_out", "decoder_name", "threshold"

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

  return props;
}

extern "C" void
set_dynamic_properties(const std::unordered_map<std::string, std::string> &input,
    semantic_seg::properties *prop, Ax::Logger &logger)
{
}
