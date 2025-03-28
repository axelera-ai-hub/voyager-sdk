// Copyright Axelera AI, 2023
#include <array>
#include <unordered_map>
#include <unordered_set>
#include "AxDataInterface.h"
#include "AxLog.hpp"
#include "AxMeta.hpp"
#include "AxUtils.hpp"

struct dequant_properties {
  std::vector<float> dequant_scale{};
  std::vector<float> dequant_zeropoint{};
  bool do_transpose = false;
};

extern "C" const std::unordered_set<std::string> &
allowed_properties()
{
  static const std::unordered_set<std::string> allowed_properties{ "dequant_scale",
    "dequant_zeropoint", "transpose" };
  return allowed_properties;
}

extern "C" std::shared_ptr<void>
init_and_set_static_properties(
    const std::unordered_map<std::string, std::string> &input, Ax::Logger &logger)
{
  std::shared_ptr<dequant_properties> prop = std::make_shared<dequant_properties>();
  prop->dequant_scale = Ax::get_property(
      input, "dequant_scale", "transform_dequantize", prop->dequant_scale);
  prop->dequant_zeropoint = Ax::get_property(input, "dequant_zeropoint",
      "transform_dequantize", prop->dequant_zeropoint);
  if (prop->dequant_scale.empty() && prop->dequant_zeropoint.empty()) {
    throw std::runtime_error(
        "Either dequant_scale, dequant_zeropoint or both must be specified in transform_dequantize");
  }
  if (prop->dequant_scale.empty()) {
    prop->dequant_scale = std::vector<float>(prop->dequant_zeropoint.size(), 1.0);
  }
  if (prop->dequant_zeropoint.empty()) {
    prop->dequant_zeropoint = std::vector<float>(prop->dequant_scale.size(), 0.0);
  }
  if (prop->dequant_scale.size() != prop->dequant_zeropoint.size()) {
    throw std::logic_error(
        "dequant_scale and dequant_zeropoint must be the same size in transform_dequantize");
  }
  prop->do_transpose = Ax::get_property(input, "transpose", "transform_dequantize", false);
  return prop;
}

extern "C" AxDataInterface
set_output_interface(const AxDataInterface &interface,
    const dequant_properties *prop, Ax::Logger &logger)
{
  if (!std::holds_alternative<AxTensorsInterface>(interface)) {
    throw std::runtime_error("dequantize works on tensor input only");
  }
  int num_tensors = std::get<AxTensorsInterface>(interface).size();
  if (prop->dequant_scale.size() != num_tensors || prop->dequant_zeropoint.size() != num_tensors) {
    throw std::runtime_error(
        "dequant_scale and dequant_zeropoint must be the same size as the number of tensors");
  }

  AxDataInterface output = interface;
  for (auto &tensor : std::get<AxTensorsInterface>(output)) {
    if (tensor.bytes != 1) {
      throw std::runtime_error("dequantize must transform int8 to float32");
    }
    tensor.bytes = 4;
    if (prop->do_transpose) {
      if (tensor.sizes.size() != 4) {
        throw std::runtime_error("dequantize with transpose must tranform 4 dimensional tensor");
      }
      std::swap(tensor.sizes[3], tensor.sizes[2]); // NHWC -> NHCW
      std::swap(tensor.sizes[2], tensor.sizes[1]); // NHCW -> NCHW
    }
  }

  return output;
}

extern "C" void
transform(const AxDataInterface &input, const AxDataInterface &output,
    const dequant_properties *prop, unsigned int, unsigned int,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &, Ax::Logger &logger)
{
  auto &input_tensors = std::get<AxTensorsInterface>(input);
  auto &output_tensors = std::get<AxTensorsInterface>(output);

  for (int i = 0; i < input_tensors.size(); ++i) {
    auto &input_tensor = input_tensors[i];
    auto &output_tensor = output_tensors[i];

    const int N = output_tensor.sizes[0];
    const int C = output_tensor.sizes.size() > 1 ? output_tensor.sizes[1] : 1;
    const int H = output_tensor.sizes.size() > 2 ? output_tensor.sizes[2] : 1;
    const int W = output_tensor.sizes.size() > 3 ? output_tensor.sizes[3] : 1;
    const int in0 = input_tensor.sizes[0];
    const int in1 = input_tensor.sizes.size() > 1 ? input_tensor.sizes[1] : 1;
    const int in2 = input_tensor.sizes.size() > 2 ? input_tensor.sizes[2] : 1;
    const int in3 = input_tensor.sizes.size() > 3 ? input_tensor.sizes[3] : 1;

    std::function<int(int, int, int, int)> compute_index;
    if (prop->do_transpose && N == in0 && H == in1 && W == in2 && C == in3) {
      compute_index = [=](int iN, int iC, int iH, int iW) {
        return iN * H * W * C + iH * W * C + iW * C + iC;
      };
    } else if (!prop->do_transpose && N == in0 && C == in1 && H == in2 && W == in3) {
      compute_index = [=](int iN, int iC, int iH, int iW) {
        return iN * C * H * W + iC * H * W + iH * W + iW;
      };
    } else {
      throw std::runtime_error("dequantize input and output sizes do not correspond");
    }

    float s = prop->dequant_scale[i];
    int z = static_cast<int>(prop->dequant_zeropoint[i]);

    float *outptr = static_cast<float *>(output_tensor.data);
    int8_t *inptr = static_cast<int8_t *>(input_tensor.data);
    for (int iN = 0; iN < N; ++iN) {
      for (int iC = 0; iC < C; ++iC) {
        for (int iH = 0; iH < H; ++iH) {
          for (int iW = 0; iW < W; ++iW) {
            *outptr++ = s * (inptr[compute_index(iN, iC, iH, iW)] - z);
          }
        }
      }
    }
  }
}
