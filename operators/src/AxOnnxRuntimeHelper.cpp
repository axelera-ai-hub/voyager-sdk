// Copyright Axelera AI, 2024

#include "AxOnnxRuntimeHelper.hpp"
#include <filesystem>
#include <iostream>
#include <numeric> // For std::accumulate
#include <sstream>
#include <string>
#include <vector>

namespace ax_onnxruntime
{

std::string
print_shape(const std::vector<std::int64_t> &v)
{
  std::ostringstream ss("");
  if (v.empty()) {
    ss << "[]";
  } else {
    for (std::size_t i = 0; i < v.size() - 1; i++)
      ss << v[i] << "x";
    ss << v[v.size() - 1];
  }
  return ss.str();
}

OnnxRuntimeInference::OnnxRuntimeInference(const std::string &model_path,
    Ax::Logger &logger, int intra_op_num_threads, int inter_op_num_threads)
    : env(nullptr),
      session(nullptr),
      first_call(true),
      logger_(logger)
{
  logger_(AX_INFO) << "Initializing ONNX Runtime for model: " << model_path << std::endl;
  std::string env_name = std::filesystem::path(model_path).extension().string() + "_onnxruntime";
  env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, env_name.c_str());
  Ort::SessionOptions session_options;
  session_options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
  session_options.EnableCpuMemArena();
  session_options.EnableMemPattern();
  // Set thread counts based on parameters
  session_options.SetIntraOpNumThreads(intra_op_num_threads); // threads for intra-op parallelism
  session_options.SetInterOpNumThreads(inter_op_num_threads); // threads for inter-op parallelism

  logger_(AX_INFO) << "ONNX Runtime configured with " << intra_op_num_threads
                   << " intra-op threads and " << inter_op_num_threads
                   << " inter-op threads" << std::endl;

#ifdef CUDA_AVAILABLE
  // Append CUDA execution provider
  OrtCUDAProviderOptions cuda_options;
  session_options.AppendExecutionProvider_CUDA(cuda_options);

  logger_(AX_INFO) << "CUDA execution provider enabled" << std::endl;
#endif

  // TODO: Add configuration for Execution Providers here

  try {
    session = Ort::Session(env, model_path.c_str(), session_options);
    logger_(AX_INFO) << "ONNX session created." << std::endl;

    // --- Input Query Logging ---
    size_t input_count = session.GetInputCount();
    logger_(AX_DEBUG) << "Model Input Count: " << input_count << std::endl;
    input_node_names.clear(); // Ensure vectors are empty before filling
    input_node_dims.clear();
    Ort::AllocatorWithDefaultOptions allocator_in; // Use one allocator for inputs
    for (size_t i = 0; i < input_count; i++) {
      auto input_name_ptr = session.GetInputNameAllocated(i, allocator_in);
      std::string input_name = std::string(input_name_ptr.get());
      input_node_names.push_back(input_name);

      Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
      auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
      std::vector<int64_t> dims = tensor_info.GetShape();
      // Handle dynamic dimensions (represented as -1 or 0)
      for (auto &dim : dims) {
        if (dim <= 0)
          dim = 1; // Replace dynamic dim with 1 for logging/default shape
                   // Note: This assumption might not be correct for all models!
      }
      input_node_dims.emplace_back(dims);
      logger_(AX_DEBUG) << "  Input[" << i << "]: Name='" << input_name
                        << "', Shape=" << print_shape(dims) << std::endl;
    }

    // --- Output Query Logging ---
    size_t output_count = session.GetOutputCount();
    logger_(AX_DEBUG) << "Model Output Count reported by ONNX Runtime: " << output_count
                      << std::endl; // Log the count!
    output_node_names.clear(); // Ensure vectors are empty before filling
    output_node_dims.clear();
    Ort::AllocatorWithDefaultOptions allocator_out; // Use one allocator for outputs
    for (size_t i = 0; i < output_count; i++) { // Loop using the reported count
      auto output_name_ptr = session.GetOutputNameAllocated(i, allocator_out);
      std::string output_name = std::string(output_name_ptr.get());
      output_node_names.push_back(output_name);

      Ort::TypeInfo type_info = session.GetOutputTypeInfo(i);
      auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
      std::vector<int64_t> dims = tensor_info.GetShape();
      // Handle dynamic dimensions (represented as -1 or 0)
      for (auto &dim : dims) {
        if (dim <= 0)
          dim = 1; // Replace dynamic dim with 1 for logging/default shape
                   // Note: This assumption might not be correct for all models!
                   // I/O Binding may require fixed shapes or specific handling.
      }
      output_node_dims.emplace_back(dims);
      // **Log details for each output found**
      logger_(AX_DEBUG) << "  Output[" << i << "]: Name='" << output_name
                        << "', Shape=" << print_shape(dims) << std::endl;
    }

    // Convert node names to C strings once during initialization
    input_node_names_c = convert_to_c_str_vector(input_node_names);
    output_node_names_c = convert_to_c_str_vector(output_node_names);

  } catch (const Ort::Exception &e) {
    logger_(AX_ERROR)
        << "Failed to create ONNX Runtime session (Ort::Exception): " << e.what()
        << std::endl;
    throw std::runtime_error(
        "Failed to create ONNX Runtime session: " + std::string(e.what()));
  } catch (const std::exception &e) {
    logger_(AX_ERROR)
        << "Failed to create ONNX Runtime session (std::exception): " << e.what()
        << std::endl;
    throw std::runtime_error(
        "Failed to create ONNX Runtime session: " + std::string(e.what()));
  }
}

// --- Implementation for old operator() ---
std::vector<Ort::Value>
OnnxRuntimeInference::operator()(const std::vector<Ort::Value> &input_tensors)
{
  if (input_tensors.size() != input_node_names.size()) {
    std::string error_msg = "Mismatch between provided inputs ("
                            + std::to_string(input_tensors.size()) + ") and model expected inputs ("
                            + std::to_string(input_node_names.size()) + ").";
    logger_(AX_ERROR) << error_msg << std::endl;
    throw std::runtime_error(error_msg);
  }
  // Validate input tensor dimensions against expected dimensions on the first call
  if (first_call) {
    for (size_t i = 0; i < input_tensors.size(); ++i) {
      auto tensor_info = input_tensors[i].GetTensorTypeAndShapeInfo();
      std::vector<int64_t> dims = tensor_info.GetShape();
      // Skip check if expected dims are dynamic (contain <= 0)
      bool has_dynamic_expected = false;
      for (int64_t d : input_node_dims[i]) {
        if (d <= 0) {
          has_dynamic_expected = true;
          break;
        }
      }
      if (!has_dynamic_expected && dims != input_node_dims[i]) {
        std::string error_msg = inspect_format(dims, input_node_dims[i], i);
        logger_(AX_ERROR) << error_msg << std::endl;
        throw std::runtime_error(error_msg);
      }
    }
    first_call = false;
  }

  std::vector<Ort::Value> output_tensors;
  try {
    logger_(AX_DEBUG) << "Running ONNX session (old method)..." << std::endl;
    output_tensors = session.Run(Ort::RunOptions{ nullptr },
        input_node_names_c.data(), input_tensors.data(), input_tensors.size(),
        output_node_names_c.data(), output_node_names_c.size());
    logger_(AX_DEBUG) << "ONNX session Run() completed. Received "
                      << output_tensors.size() << " tensors." << std::endl;
  } catch (const Ort::Exception &e) {
    std::string error_msg
        = "Failed to run ONNX Runtime session: " + std::string(e.what());
    logger_(AX_ERROR) << error_msg << std::endl;
    throw std::runtime_error(error_msg);
  } catch (const std::exception &e) {
    std::string error_msg
        = "Failed to run ONNX Runtime session: " + std::string(e.what());
    logger_(AX_ERROR) << error_msg << std::endl;
    throw std::runtime_error(error_msg);
  }
  return output_tensors;
}


// --- Implementation for run_with_io_binding ---
Ort::Value
OnnxRuntimeInference::create_output_ort_value(
    AxTensorInterface *ax_tensor, const std::vector<int64_t> &expected_dims)
{
  // Ensure tensor is float for now (common case)
  if (ax_tensor->bytes != sizeof(float)) {
    logger_(AX_ERROR) << "I/O Binding requires output tensor buffer to be float (4 bytes). "
                      << "Found " << ax_tensor->bytes << " bytes.";
    throw std::runtime_error("Invalid output buffer type for I/O Binding");
  }

  // Check if buffer is large enough for expected dimensions
  size_t expected_elements = 1;
  for (int64_t dim : expected_dims) {
    if (dim <= 0) { // Should not happen if we handled dynamic dims in constructor
      logger_(AX_ERROR) << "Dynamic output dimensions not fully supported with current I/O binding setup.";
      throw std::runtime_error("Dynamic output dimensions not handled for I/O binding");
    }
    expected_elements *= dim;
  }
  size_t expected_bytes = expected_elements * sizeof(float);
  size_t actual_bytes = ax_tensor->total() * ax_tensor->bytes; // Based on current AxTensor size

  // It's better if the provided buffer is exactly the expected size.
  // We allow it to be larger, but it must be at least the expected size.
  if (actual_bytes < expected_bytes) {
    logger_(AX_ERROR) << "Output buffer for I/O Binding is too small. Expected at least "
                      << expected_bytes << " bytes, but buffer has " << actual_bytes
                      << " bytes. Expected shape: " << print_shape(expected_dims);
    throw std::runtime_error("Output buffer too small for I/O Binding");
  }

  // Create Ort::Value wrapping the existing buffer
  Ort::MemoryInfo memory_info
      = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  float *data_ptr = static_cast<float *>(ax_tensor->data);

  // Create Ort::Value using the expected dimensions from the model
  return Ort::Value::CreateTensor<float>(memory_info, data_ptr, expected_bytes,
      expected_dims.data(), expected_dims.size());
}


void
OnnxRuntimeInference::run_with_io_binding(const std::vector<Ort::Value> &input_tensors,
    const std::vector<AxTensorInterface *> &output_ax_tensors)
{
  if (input_tensors.size() != input_node_names.size()) {
    std::string error_msg = "I/O Binding: Mismatch between provided inputs ("
                            + std::to_string(input_tensors.size()) + ") and model expected inputs ("
                            + std::to_string(input_node_names.size()) + ").";
    logger_(AX_ERROR) << error_msg << std::endl;
    throw std::runtime_error(error_msg);
  }
  if (output_ax_tensors.size() != output_node_names.size()) {
    std::string error_msg = "I/O Binding: Mismatch between provided output tensors ("
                            + std::to_string(output_ax_tensors.size()) + ") and model expected outputs ("
                            + std::to_string(output_node_names.size()) + ").";
    logger_(AX_ERROR) << error_msg << std::endl;
    throw std::runtime_error(error_msg);
  }

  try {
    Ort::IoBinding io_binding(session);

    for (size_t i = 0; i < input_tensors.size(); ++i) {
      io_binding.BindInput(input_node_names[i].c_str(), input_tensors[i]);
    }

    // Bind outputs (using pre-allocated buffers from output_ax_tensors)
    std::vector<Ort::Value> output_ort_values; // Keep Ort::Value objects alive
    output_ort_values.reserve(output_ax_tensors.size());

    for (size_t i = 0; i < output_ax_tensors.size(); ++i) {
      // Create Ort::Value wrapping the AxTensorInterface buffer
      // Use the expected dims stored during initialization
      output_ort_values.push_back(
          create_output_ort_value(output_ax_tensors[i], output_node_dims[i]));
      // Bind this Ort::Value to the corresponding output name
      io_binding.BindOutput(output_node_names[i].c_str(), output_ort_values.back());
    }

    // Run inference
    session.Run(Ort::RunOptions{ nullptr }, io_binding);
    logger_(AX_DEBUG) << "ONNX session Run() with I/O Binding completed." << std::endl;

    // Results are now directly in the buffers pointed to by output_ax_tensors.
    // IMPORTANT: We need to update the 'sizes' field of the AxTensorInterface objects
    // if the model could potentially output dynamic shapes different from what was expected.
    // For now, we assume the output dimensions match output_node_dims.
    // If dynamic output shapes are possible, we would need to get the actual
    // shape from Ort::IoBinding::GetOutputValues() and update AxTensorInterface::sizes.

    // Example (if dynamic shapes needed handling):
    // std::vector<Ort::Value> actual_outputs = io_binding.GetOutputValues();
    // for(size_t i=0; i < actual_outputs.size(); ++i) {
    //   auto tensor_info = actual_outputs[i].GetTensorTypeAndShapeInfo();
    //   auto actual_dims_int64 = tensor_info.GetShape();
    //   // Convert int64 dims to int and update output_ax_tensors[i]->sizes
    //   output_ax_tensors[i]->sizes.clear();
    //   output_ax_tensors[i]->sizes.reserve(actual_dims_int64.size());
    //   for(int64_t dim : actual_dims_int64) {
    //      output_ax_tensors[i]->sizes.push_back(static_cast<int>(dim));
    //   }
    // }

  } catch (const Ort::Exception &e) {
    std::string error_msg = "Failed to run ONNX Runtime session with I/O Binding (Ort::Exception): "
                            + std::string(e.what());
    logger_(AX_ERROR) << error_msg << std::endl;
    throw std::runtime_error(error_msg);
  } catch (const std::exception &e) {
    std::string error_msg = "Failed to run ONNX Runtime session with I/O Binding (std::exception): "
                            + std::string(e.what());
    logger_(AX_ERROR) << error_msg << std::endl;
    throw std::runtime_error(error_msg);
  }
}

// --- Helper function implementations ---

std::vector<const char *>
OnnxRuntimeInference::convert_to_c_str_vector(const std::vector<std::string> &string_vector)
{
  std::vector<const char *> c_str_vector;
  c_str_vector.reserve(string_vector.size());
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
  error_message += print_shape(providedDims);
  error_message += ") and model expected input dimensions (";
  error_message += print_shape(expectedDims);
  error_message += ") for input " + std::to_string(inputIndex) + ".";
  return error_message;
}

std::vector<ONNXTensorElementDataType>
OnnxRuntimeInference::get_input_node_types() const
{
  std::vector<ONNXTensorElementDataType> types;
  types.reserve(input_node_names.size());
  for (size_t i = 0; i < input_node_names.size(); ++i) {
    auto type_info = session.GetInputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    types.push_back(tensor_info.GetElementType());
  }
  return types;
}

std::vector<size_t>
OnnxRuntimeInference::get_input_node_ranks() const
{
  std::vector<size_t> ranks(input_node_names.size());
  for (size_t i = 0; i < input_node_names.size(); ++i) {
    ranks[i] = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetDimensionsCount();
  }
  return ranks;
}

std::vector<ONNXTensorElementDataType>
OnnxRuntimeInference::get_output_node_types() const
{
  std::vector<ONNXTensorElementDataType> types;
  types.reserve(output_node_names.size());
  for (size_t i = 0; i < output_node_names.size(); ++i) {
    auto type_info = session.GetOutputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    types.push_back(tensor_info.GetElementType());
  }
  return types;
}

} // namespace ax_onnxruntime
