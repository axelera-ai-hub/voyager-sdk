// Copyright Axelera AI, 2024
#pragma once

#include <onnxruntime_cxx_api.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace ax_onnxruntime
{

class OnnxRuntimeInference
{
  public:
  explicit OnnxRuntimeInference(const std::string &model_path);
  std::vector<Ort::Value> operator()(const std::vector<Ort::Value> &input_tensors);

  std::vector<std::vector<int64_t>> get_input_node_dims() const
  {
    return input_node_dims;
  }
  std::vector<std::vector<int64_t>> get_output_node_dims() const
  {
    return output_node_dims;
  }


  private:
  Ort::Env env;
  Ort::Session session;
  std::vector<std::string> input_node_names;
  std::vector<const char *> input_node_names_c;
  std::vector<std::vector<int64_t>> input_node_dims;
  std::vector<std::string> output_node_names;
  std::vector<const char *> output_node_names_c;
  std::vector<std::vector<int64_t>> output_node_dims;
  bool first_call = true;

  std::string inspect_format(const std::vector<int64_t> &providedDims,
      const std::vector<int64_t> &expectedDims, size_t inputIndex);
  std::vector<const char *> convert_to_c_str_vector(
      const std::vector<std::string> &string_vector);
};

} // namespace ax_onnxruntime
