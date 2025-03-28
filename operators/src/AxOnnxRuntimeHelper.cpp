// Copyright Axelera AI, 2024
#include "AxOnnxRuntimeHelper.hpp"
#include <filesystem>
#include <sstream>

namespace ax_onnxruntime
{

std::string
print_shape(const std::vector<std::int64_t> &v)
{
  std::ostringstream ss("");
  for (std::size_t i = 0; i < v.size() - 1; i++)
    ss << v[i] << "x";
  ss << v[v.size() - 1];
  return ss.str();
}

OnnxRuntimeInference::OnnxRuntimeInference(const std::string &model_path)
    : env(nullptr), session(nullptr), first_call(true)
{
  std::string env_name = std::filesystem::path(model_path).extension().string() + "_onnxruntime";
  env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, env_name.c_str());
  Ort::SessionOptions session_options;
  session_options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
  session_options.EnableCpuMemArena();
  session_options.EnableMemPattern();
  // session_options.SetIntraOpNumThreads(1);
  // session_options.SetInterOpNumThreads(1);

  try {
    session = Ort::Session(env, model_path.c_str(), session_options);

    for (size_t i = 0; i < session.GetInputCount(); i++) {
      Ort::AllocatorWithDefaultOptions allocator;
      auto input_name = session.GetInputNameAllocated(i, allocator);
      input_node_names.push_back(std::string(input_name.get()));

      // Cache input node dimensions
      Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
      auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
      std::vector<int64_t> dims = tensor_info.GetShape();
      input_node_dims.emplace_back(dims);
    }

    for (size_t i = 0; i < session.GetOutputCount(); i++) {
      Ort::AllocatorWithDefaultOptions allocator;
      auto output_name = session.GetOutputNameAllocated(i, allocator);
      output_node_names.push_back(std::string(output_name.get()));

      // Cache output node dimensions
      Ort::TypeInfo type_info = session.GetOutputTypeInfo(i);
      auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
      std::vector<int64_t> dims = tensor_info.GetShape();
      output_node_dims.emplace_back(dims);
    }
  } catch (const Ort::Exception &e) {
    throw std::runtime_error(
        "Failed to create ONNX Runtime session: " + std::string(e.what()));
  } catch (const std::exception &e) {
    throw std::runtime_error(
        "Failed to create ONNX Runtime session: " + std::string(e.what()));
  }
}


std::vector<Ort::Value>
OnnxRuntimeInference::operator()(const std::vector<Ort::Value> &input_tensors)
{
  if (input_tensors.size() != input_node_names.size()) {
    std::string error_msg = "Mismatch between provided inputs ("
                            + std::to_string(input_tensors.size()) + ") and model expected inputs ("
                            + std::to_string(input_node_names.size()) + ").";
    throw std::runtime_error(error_msg);
  }
  // Validate input tensor dimensions against expected dimensions on the first call
  if (first_call) {
    for (size_t i = 0; i < input_tensors.size(); ++i) {
      auto tensor_info = input_tensors[i].GetTensorTypeAndShapeInfo();
      std::vector<int64_t> dims = tensor_info.GetShape();

      if (dims != input_node_dims[i]) {
        throw std::runtime_error(inspect_format(dims, input_node_dims[i], i));
      }
    }
    // Convert input and output node names to C strings for ONNX Runtime API
    input_node_names_c = convert_to_c_str_vector(input_node_names);
    output_node_names_c = convert_to_c_str_vector(output_node_names);
    first_call = false;
  }

  std::vector<Ort::Value> output_tensors;
  try {
    output_tensors = session.Run(Ort::RunOptions{ nullptr },
        input_node_names_c.data(), input_tensors.data(), input_tensors.size(),
        output_node_names_c.data(), output_node_names_c.size());
  } catch (const Ort::Exception &e) {
    throw std::runtime_error(
        "Failed to run ONNX Runtime session: " + std::string(e.what()));
  } catch (const std::exception &e) {
    throw std::runtime_error(
        "Failed to run ONNX Runtime session: " + std::string(e.what()));
  }
  return output_tensors;
}

std::vector<const char *>
OnnxRuntimeInference::convert_to_c_str_vector(const std::vector<std::string> &string_vector)
{
  std::vector<const char *> c_str_vector;
  for (const auto &str : string_vector) {
    c_str_vector.push_back(str.c_str());
  }
  return c_str_vector;
}

std::string
OnnxRuntimeInference::inspect_format(const std::vector<int64_t> &providedDims,
    const std::vector<int64_t> &expectedDims, size_t inputIndex)
{
  std::string error_message = "Mismatch between provided input dimensions (";
  for (const auto &dim : providedDims) {
    error_message += std::to_string(dim) + ",";
  }
  error_message.pop_back();
  error_message += ") and model expected input dimensions (";
  for (const auto &model_dim : expectedDims) {
    error_message += std::to_string(model_dim) + ",";
  }
  error_message.pop_back();
  error_message += ") for input " + std::to_string(inputIndex) + ".";
  return error_message;
}


} // namespace ax_onnxruntime
