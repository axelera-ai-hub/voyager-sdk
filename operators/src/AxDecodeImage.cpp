// Copyright Axelera AI, 2025
// Tasks such as super resolution, image enhancement, depth estimation, etc.
// can use this decoder to decode the tensor to image meta
#include <unordered_set>

#include "AxMetaImage.hpp"
#include "AxOpUtils.hpp"

extern "C" const std::unordered_set<std::string> &
allowed_properties()
{
  static const std::unordered_set<std::string> allowed_properties{
    "meta_key",
    "master_meta",
    "output_datatype",
    "scale",
  };
  return allowed_properties;
}

struct depth_properties {
  std::string meta_key{};
  std::string master_meta{};
  std::string output_datatype{ "float32" };
  bool scale{ false };
};

extern "C" std::shared_ptr<void>
init_and_set_static_properties(
    const std::unordered_map<std::string, std::string> &input, Ax::Logger &logger)
{
  std::shared_ptr<depth_properties> prop = std::make_shared<depth_properties>();

  prop->meta_key = Ax::get_property(input, "meta_key", "static_properties", prop->meta_key);

  prop->output_datatype = Ax::get_property(
      input, "output_datatype", "static_properties", prop->output_datatype);

  prop->scale = Ax::get_property(input, "scale", "static_properties", prop->scale);

  prop->master_meta
      = Ax::get_property(input, "master_meta", "static_properties", prop->master_meta);

  return prop;
}

extern "C" void
set_dynamic_properties(const std::unordered_map<std::string, std::string> &input,
    depth_properties *prop, Ax::Logger &logger)
{
}

extern "C" void
decode_to_meta(const AxTensorsInterface &tensors, const depth_properties *prop,
    unsigned int current_frame, unsigned int total_frames,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &map,
    const AxDataInterface &, Ax::Logger &logger)
{
  if (total_frames <= current_frame) {
    throw std::runtime_error("image_decode_to_meta: Current frame is out of bounds");
  }

  if (1 != tensors.size()) {
    throw std::runtime_error("image_decode_to_meta: Number of tensors must be 1");
  }

  auto &tensor = tensors[0];

  if (4 != tensor.bytes) {
    throw std::runtime_error("image_decode_to_meta: NN must return float");
  }

  if (4 != tensor.sizes.size()) {
    throw std::runtime_error("image_decode_to_meta: Tensor must be 4-D (NCHW)");
  }

  // Extract NCHW dimensions explicitly
  const int c = tensor.sizes[1];
  const int h = tensor.sizes[2];
  const int w = tensor.sizes[3];

  const auto *data = static_cast<const float *>(tensors[0].data);
  const auto size = c * h * w;

  if (prop->output_datatype == "float32" && prop->scale == false) {
    VectorType output(std::vector<float>(data, data + size));
    ax_utils::insert_meta<AxMetaImage>(map, prop->meta_key, prop->master_meta,
        current_frame, total_frames, std::move(output), c, w, h);
    return;
  }

  VectorType output;
  if (prop->output_datatype == "float32") {
    output = VectorType(std::vector<float>(size));
  } else if (prop->output_datatype == "uint8") {
    output = VectorType(std::vector<uint8_t>(size));
  } else {

    throw std::runtime_error("image_decode_to_meta: Invalid variant type");
  }

  if (auto *outvec = std::get_if<std::vector<uint8_t>>(&output)) {
    std::transform(data, data + size, outvec->begin(),
        [scale = prop->scale](float val) -> uint8_t {
          return scale ? static_cast<uint8_t>(std::clamp(val * 255, 0.0f, 255.0f)) :
                         static_cast<uint8_t>(val);
        });
  } else if (auto *outvec = std::get_if<std::vector<float>>(&output)) {
    std::transform(data, data + size, outvec->begin(),
        [scale = prop->scale](float val) -> float {
          return scale ? std::clamp(val * 255, 0.0f, 255.0f) : val;
        });
  } else {
    throw std::runtime_error("image_decode_to_meta: Invalid variant type, internal error");
  }

  ax_utils::insert_meta<AxMetaImage>(map, prop->meta_key, prop->master_meta,
      current_frame, total_frames, std::move(output), c, w, h);
}
