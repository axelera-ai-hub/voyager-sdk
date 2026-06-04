// Copyright Axelera AI, 2025
#include <algorithm>
#include <functional>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "AxDataInterface.h"
#include "AxLog.hpp"
#include "AxMeta.hpp"
#include "AxOnnxRuntimeHelper.hpp"
#include "AxOpUtils.hpp"
#include "AxUtils.hpp"

using lookups = std::array<float, 256>;
// Properties for the postamble processor
struct postamble_properties {
  std::string onnx_path{}; // Path to ONNX model for postamble processing
  std::vector<int> tensor_selection_plan{}; // Pre-calculated indices of which tensors to use as ONNX inputs
  std::vector<bool> transpose{}; // Whether to transpose each input tensor
  bool dequant_lut{ true }; // Whether to use dequantization lookup tables
  std::vector<float> dequant_scale{}; // Dequantization scale factors
  std::vector<float> dequant_zeropoint{}; // Dequantization zero points
  std::vector<lookups> dequantize_tables{}; // Dequantization lookup tables for each tensor
  std::vector<std::vector<int>> paddings{}; // Padding configurations for each tensor
  int ort_intra_op_num_threads{ 4 }; // Number of threads for intra-op parallelism
  int ort_inter_op_num_threads{ 4 }; // Number of threads for inter-op parallelism
  std::unique_ptr<ax_onnxruntime::OnnxRuntimeInference> onnx_runtime_; // ONNX runtime engine
  mutable std::vector<AxTensorInterface> input_tensors{}; // Input tensors for ONNX inference
  mutable std::vector<std::vector<float>> input_datas{}; // Pre-allocated buffers for dequantized data
};

float
dequantize(int8_t value, const float *the_table)
{
  int index = value + 128;
  return the_table[index];
}

// Define allowed properties that can be set from Python
extern "C" const std::unordered_set<std::string> &
allowed_properties()
{
  static const std::unordered_set<std::string> allowed_properties{ "onnx_path",
    "tensor_selection_plan", // Format: comma-separated integers "0,2,5"
    "padding", "transpose", "dequant_scale", "dequant_zeropoint", "dequant_lut",
    "ort_intra_op_num_threads", "ort_inter_op_num_threads" };
  return allowed_properties;
}

// TODO: move this to AxUtils.hpp
//  Helper function to convert a vector to a 4D shape
std::vector<int>
to_4d_shape(const std::vector<int> &shape)
{
  std::vector<int> result = shape;
  while (result.size() < 4) {
    result.insert(result.begin(), 1);
  }
  return result;
}

// TODO: move this to AxUtils.hpp
//  Helper for logging shapes
template <typename T>
std::string
shape_to_string(const std::vector<T> &shape)
{
  return std::string("[") + Ax::Internal::join(shape, ",") + "]";
}

void
apply_dequantize_transform(int8_t *inptr, float *outptr, const int N,
    const int C, const int H, const int W, bool do_transpose,
    const float *table, size_t total_elements, bool dequant_lut)
{
  if (!do_transpose) {
    std::transform(inptr, inptr + total_elements, outptr, [&table, dequant_lut](int8_t val) {
      return dequant_lut ? dequantize(val, table) :
                           table[0] * (static_cast<float>(val) - table[1]);
    });
  } else {
    const int sh = W * C;
    const int sn = H * sh;
    for (int iN = 0; iN < N; ++iN) {
      for (int iC = 0; iC < C; ++iC) {
        for (int iH = 0; iH < H; ++iH) {
          for (int iW = 0; iW < W; ++iW) {
            int input_index = iN * sn + iH * sh + iW * C + iC;
            *outptr++ = dequant_lut ?
                            dequantize(inptr[input_index], table) :
                            table[0] * (static_cast<float>(inptr[input_index]) - table[1]);
          }
        }
      }
    }
  }
}

// Function to process and dequantize input tensor
std::vector<int>
process_input_tensor(const AxTensorInterface &tensor, const std::vector<int> &in_shape,
    const std::vector<int> &padding, const postamble_properties *prop,
    size_t tensor_index, float *output_ptr, Ax::Logger &logger)
{
  const auto info = ax_utils::get_transfer_info(in_shape, padding);
  cv::Mat input_mat(info.in_sizes, CV_8SC1, tensor.data);
  auto cropped = std::all_of(padding.begin(), padding.end(),
                     [](int val) { return val == 0; }) ?
                     input_mat :
                     input_mat(info.ranges).clone();

  int N = cropped.size[0];
  int C = cropped.size[1];
  int H = cropped.size[2];
  int W = cropped.size[3];

  const int total_elements = N * C * H * W;
  bool should_transpose
      = tensor_index < prop->transpose.size() ? prop->transpose[tensor_index] : false;

  if (should_transpose) {
    std::swap(H, W); // NHWC -> NHCW
    std::swap(C, H); // NHCW -> NCHW
  }

  // Validate tensor_index is within bounds for dequant parameters
  if (tensor_index >= prop->dequant_scale.size()
      || tensor_index >= prop->dequant_zeropoint.size()) {
    logger(AX_ERROR) << "Tensor index " << tensor_index
                     << " is out of bounds for dequantization parameters. "
                     << "Scale size: " << prop->dequant_scale.size()
                     << ", Zeropoint size: " << prop->dequant_zeropoint.size();
    throw std::runtime_error("Tensor index out of bounds for dequantization parameters");
  }

  // Stack array for non-LUT dequantization parameters
  float d[2] = { prop->dequant_scale[tensor_index], prop->dequant_zeropoint[tensor_index] };

  const float *dequant_data
      = prop->dequant_lut ? prop->dequantize_tables[tensor_index].data() : d;

  // Write directly to output_ptr instead of intermediate buffer
  apply_dequantize_transform(cropped.ptr<int8_t>(), output_ptr, N, C, H, W,
      should_transpose, dequant_data, total_elements, prop->dequant_lut);

  return { N, C, H, W };
}

// Initialize properties from configuration
extern "C" std::shared_ptr<void>
init_and_set_static_properties(
    const std::unordered_map<std::string, std::string> &input, Ax::Logger &logger)
{
  std::shared_ptr<postamble_properties> prop = std::make_shared<postamble_properties>();

  // Get ONNX model path
  prop->onnx_path
      = Ax::get_property(input, "onnx_path", "transform_postamble", std::string{});

  // Get tensor selection plan (which tensors to use for ONNX inference)
  prop->tensor_selection_plan = Ax::get_property(
      input, "tensor_selection_plan", "transform_postamble", std::vector<int>{});

  prop->paddings = Ax::get_property(input, "padding",
      "padding_dequant_properties", std::vector<std::vector<int>>{});

  // For backward compatibility - if no paddings were specified but there's a single padding vector, convert it
  if (prop->paddings.empty()) {
    std::vector<int> single_padding = Ax::get_property(
        input, "padding", "padding_dequant_properties", std::vector<int>{});
    if (!single_padding.empty()) {
      prop->paddings.push_back(single_padding);
    }
  }

  prop->dequant_scale = Ax::get_property(
      input, "dequant_scale", "transform_postamble", std::vector<float>{});
  prop->dequant_zeropoint = Ax::get_property(
      input, "dequant_zeropoint", "transform_postamble", std::vector<float>{});

  // Validate dequantization parameters - if one is provided, both must be provided and same size
  bool has_dequant_scale = !prop->dequant_scale.empty();
  bool has_dequant_zeropoint = !prop->dequant_zeropoint.empty();

  if (has_dequant_scale != has_dequant_zeropoint) {
    throw std::logic_error(
        "dequant_scale and dequant_zeropoint must both be provided or both be empty in transform_postamble");
  }

  if (has_dequant_scale && prop->dequant_scale.size() != prop->dequant_zeropoint.size()) {
    throw std::logic_error(
        "dequant_scale and dequant_zeropoint must be the same size in transform_postamble");
  }

  if (has_dequant_scale) {
    logger(AX_INFO) << "Dequantization parameters provided for "
                    << prop->dequant_scale.size() << " tensors";
  } else {
    logger(AX_INFO) << "No dequantization parameters provided. Tensors will be passed through unchanged.";
  }

  prop->dequant_lut = Ax::get_property(
      input, "dequant_lut", "transform_postamble", prop->dequant_lut);

  if (has_dequant_scale && prop->dequant_lut) {
    prop->dequantize_tables = ax_utils::build_dequantization_tables(
        prop->dequant_zeropoint, prop->dequant_scale);
  }

  std::string transpose_str
      = Ax::get_property(input, "transpose", "transform_postamble", std::string{});
  if (!transpose_str.empty()) {
    // Split the comma-separated string and convert to booleans
    auto transpose_string_views = Ax::Internal::split(transpose_str, ',');
    prop->transpose.reserve(transpose_string_views.size());
    for (const auto &val_view : transpose_string_views) {
      std::string val(val_view); // Convert string_view to string
      prop->transpose.push_back(std::stoi(val) != 0);
    }
  }

  // Get thread count parameters (defaults to 4 if not specified)
  prop->ort_intra_op_num_threads = Ax::get_property(input, "ort_intra_op_num_threads",
      "transform_postamble", prop->ort_intra_op_num_threads);
  prop->ort_inter_op_num_threads = Ax::get_property(input, "ort_inter_op_num_threads",
      "transform_postamble", prop->ort_inter_op_num_threads);

  // Initialize ONNX runtime if path is provided
  if (!prop->onnx_path.empty()) {
    try {
      // Initialize ONNX Runtime
      prop->onnx_runtime_
          = std::make_unique<ax_onnxruntime::OnnxRuntimeInference>(prop->onnx_path,
              logger, prop->ort_intra_op_num_threads, prop->ort_inter_op_num_threads);
      if (!prop->onnx_runtime_) {
        logger(AX_ERROR) << "Failed to initialize ONNX Runtime. onnx_runtime_ is null.";
        throw std::runtime_error("ONNX Runtime initialization failed");
      }
      logger(AX_INFO) << "Initialized ONNX Runtime for postamble: " << prop->onnx_path;

      // Log ONNX model information
      const auto &input_names = prop->onnx_runtime_->get_input_node_names();
      const auto &input_dims = prop->onnx_runtime_->get_input_node_dims();
      const auto &output_names = prop->onnx_runtime_->get_output_node_names();
      const auto &output_dims = prop->onnx_runtime_->get_output_node_dims();

      logger(AX_INFO) << "ONNX model has " << input_names.size()
                      << " inputs and " << output_names.size() << " outputs";

      // If tensor selection plan is empty, use default (first N tensors)
      if (prop->tensor_selection_plan.empty()) {
        logger(AX_INFO) << "No tensor selection plan provided, will use first "
                        << input_names.size() << " tensors as input";
        prop->tensor_selection_plan.resize(input_names.size());
        std::iota(prop->tensor_selection_plan.begin(),
            prop->tensor_selection_plan.end(), 0);
      }

      // Validate tensor selection plan size matches ONNX model requirements
      if (prop->tensor_selection_plan.size() != input_names.size()) {
        logger(AX_ERROR)
            << "Tensor selection plan has " << prop->tensor_selection_plan.size()
            << " indices but ONNX model requires " << input_names.size() << " inputs.";
        throw std::runtime_error("Mismatch between tensor selection plan and ONNX model inputs");
      }

      // Validate that selected tensors have dequantization parameters if needed
      if (has_dequant_scale) {
        for (int tensor_idx : prop->tensor_selection_plan) {
          if (tensor_idx >= static_cast<int>(prop->dequant_scale.size())
              || tensor_idx >= static_cast<int>(prop->dequant_zeropoint.size())) {
            logger(AX_ERROR)
                << "Tensor " << tensor_idx
                << " is selected for ONNX but missing dequantization parameters. "
                << "Provided " << prop->dequant_scale.size() << " scale values and "
                << prop->dequant_zeropoint.size() << " zeropoint values.";
            throw std::runtime_error("Missing dequantization parameters for ONNX input tensor");
          }
        }
      }

      // Log input shapes for debugging
      for (size_t i = 0; i < input_names.size(); ++i) {
        const auto &dims = input_dims[i];
        logger(AX_INFO) << "ONNX input " << i << " (" << input_names[i]
                        << "): Expected Shape " << shape_to_string(dims);
      }

      // Log output shapes for debugging
      for (size_t i = 0; i < output_names.size(); ++i) {
        const auto &dims = output_dims[i];
        logger(AX_INFO) << "ONNX output " << i << " (" << output_names[i]
                        << "): Expected Shape " << shape_to_string(dims);
      }

    } catch (const std::exception &e) {
      logger(AX_ERROR) << "Failed to initialize ONNX Runtime: " << e.what();
      throw;
    }
  } else {
    logger(AX_WARN) << "No ONNX model path provided. This transform will pass through tensors unchanged.";
  }

  return prop;
}

// Set output interface based on ONNX model and input tensors
extern "C" AxDataInterface
set_output_interface(const AxDataInterface &interface,
    const postamble_properties *prop, Ax::Logger &logger)
{
  if (!std::holds_alternative<AxTensorsInterface>(interface)) {
    throw std::runtime_error("transform_postamble works on tensor input only");
  }

  const auto &input_tensors = std::get<AxTensorsInterface>(interface);

  // If we have no ONNX model, output interface is the same as input
  if (!prop->onnx_runtime_) {
    logger(AX_INFO) << "No ONNX runtime, output interface matches input.";

    AxTensorsInterface output_tensors = input_tensors; // Copy input interface

    // Update output interface for dequantization if needed
    for (size_t i = 0; i < output_tensors.size(); ++i) {
      // If we have dequant parameters and this tensor index is covered
      if (!prop->dequant_scale.empty() && i < prop->dequant_scale.size()) {
        output_tensors[i].bytes = sizeof(float);
      }

      // Handle transpose
      bool should_transpose = i < prop->transpose.size() ? prop->transpose[i] : false;
      if (should_transpose && output_tensors[i].sizes.size() >= 4) {
        std::swap(output_tensors[i].sizes[3], output_tensors[i].sizes[2]); // NHWC -> NHCW
        std::swap(output_tensors[i].sizes[2], output_tensors[i].sizes[1]); // NHCW -> NCHW
      }

      output_tensors[i].data = nullptr; // Framework will allocate
    }

    return AxDataInterface{ output_tensors };
  }

  try {
    // Get expected output shapes from the initialized ONNX runtime
    const auto &output_names = prop->onnx_runtime_->get_output_node_names();
    const auto &output_dims = prop->onnx_runtime_->get_output_node_dims();
    size_t num_onnx_outputs = output_names.size();

    // Create a set of used input tensor indices for quick lookup
    std::unordered_set<int> used_indices(
        prop->tensor_selection_plan.begin(), prop->tensor_selection_plan.end());

    // Count unused input tensors
    size_t num_unused_inputs = 0;
    for (size_t i = 0; i < input_tensors.size(); ++i) {
      if (used_indices.find(static_cast<int>(i)) == used_indices.end()) {
        num_unused_inputs++;
      }
    }

    // Create output interface with the right number of tensors
    AxTensorsInterface output_tensors;
    output_tensors.resize(num_onnx_outputs + num_unused_inputs);
    logger(AX_INFO) << "Setting output interface: " << num_onnx_outputs
                    << " ONNX outputs + " << num_unused_inputs
                    << " unused inputs = " << output_tensors.size() << " total outputs.";

    prop->input_datas.resize(input_tensors.size());
    prop->input_tensors.resize(input_tensors.size());

    // Configure ONNX output tensors based on model info
    for (size_t i = 0; i < num_onnx_outputs; ++i) {
      // Get expected dims (int64_t) and convert to int for AxTensorInterface
      auto dims_int64 = output_dims[i];

      if (dims_int64.size() < 4) {
        dims_int64.insert(dims_int64.begin(), 4 - dims_int64.size(), 1);
      } else if (dims_int64.size() > 4) {
        throw std::runtime_error("Output tensor rank exceeds 4. Unsupported.");
      }

      output_tensors[i].sizes.clear();
      output_tensors[i].sizes.reserve(dims_int64.size());
      for (const auto &dim : dims_int64) {
        if (dim <= 0) {
          logger(AX_WARN)
              << "Output tensor " << i << " has dynamic dimension (" << dim
              << "). I/O Binding might not work correctly. Defaulting dim to 1.";
          output_tensors[i].sizes.push_back(1);
        } else {
          output_tensors[i].sizes.push_back(static_cast<int>(dim));
        }
      }

      // ONNX outputs are assumed float (4 bytes) for this implementation
      output_tensors[i].bytes = sizeof(float);
      output_tensors[i].fd = -1; // Not file-based
      output_tensors[i].data = nullptr; // Data pointer will be set by framework allocator

      logger(AX_INFO)
          << "Output tensor " << i << " (from ONNX " << output_names[i]
          << ") configured with shape: " << shape_to_string(output_tensors[i].sizes)
          << ", bytes: " << output_tensors[i].bytes;
    }

    // Configure unused input tensors (passthrough)
    size_t output_idx = num_onnx_outputs;
    for (size_t i = 0; i < input_tensors.size(); ++i) {
      if (used_indices.find(static_cast<int>(i)) == used_indices.end()) {
        if (output_idx < output_tensors.size()) {
          output_tensors[output_idx] = input_tensors[i]; // Copy interface info
          output_tensors[output_idx].data
              = nullptr; // Data pointer will be set by framework allocator

          // Handle dequantization type change
          if (!prop->dequant_scale.empty() && i < prop->dequant_scale.size()) {
            output_tensors[output_idx].bytes = sizeof(float);
          }

          // Handle transpose
          bool should_transpose = i < prop->transpose.size() ? prop->transpose[i] : false;
          if (should_transpose && output_tensors[output_idx].sizes.size() >= 4) {
            std::swap(output_tensors[output_idx].sizes[3],
                output_tensors[output_idx].sizes[2]); // NHWC -> NHCW
            std::swap(output_tensors[output_idx].sizes[2],
                output_tensors[output_idx].sizes[1]); // NHCW -> NCHW
          }

          logger(AX_INFO) << "Output tensor " << output_idx << " (passthrough from input "
                          << i << ") configured with shape: "
                          << shape_to_string(output_tensors[output_idx].sizes)
                          << ", bytes: " << output_tensors[output_idx].bytes;
          output_idx++;
        } else {
          logger(AX_ERROR) << "Logic error: Not enough space allocated for unused input tensors in output interface.";
          break;
        }
      }
    }

    return AxDataInterface{ output_tensors };

  } catch (const std::exception &e) {
    logger(AX_ERROR) << "Error determining output interface: " << e.what();
    throw;
  }
}

/// @brief Converting Ax tensor to ONNX input format
/// @param ax_tensor Input tensor to convert
/// @param logger Logger for error reporting
/// @return ONNX input tensor as Ort::Value
Ort::Value
convert_tensor_to_onnx_input(
    const AxTensorInterface &ax_tensor, Ax::Logger &logger, size_t rank = 4)
{
  // Memory info for CPU allocation
  Ort::MemoryInfo memory_info
      = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  // Get data pointer as float
  const float *data_ptr = static_cast<const float *>(ax_tensor.data);

  // Convert sizes to int64_t for ONNX
  std::vector<int64_t> dims;
  dims.reserve(ax_tensor.sizes.size());
  for (int size : ax_tensor.sizes) {
    dims.push_back(static_cast<int64_t>(size));
  }
  if (dims.size() > rank) {
    dims = std::vector<int64_t>(dims.end() - rank, dims.end());
  } else if (dims.size() < rank) {
    dims.insert(dims.begin(), rank - dims.size(), 1);
  }

  // Calculate total size in bytes
  size_t total_bytes = ax_tensor.total() * sizeof(float);

  // Create ONNX tensor (non-owning - using original data)
  return Ort::Value::CreateTensor<float>(memory_info,
      const_cast<float *>(data_ptr), total_bytes, dims.data(), rank);
}

// Main transform function using I/O Binding
extern "C" void
transform(const AxDataInterface &input, const AxDataInterface &output,
    const postamble_properties *prop, unsigned int, unsigned int,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &, Ax::Logger &logger)
{
  auto &input_tensors = std::get<AxTensorsInterface>(input);
  auto &output_tensors = std::get<AxTensorsInterface>(output);

  // If no ONNX model, just copy input to output (passthrough)
  if (!prop->onnx_runtime_) {
    logger(AX_DEBUG) << "No ONNX model. Passing tensors through with potential processing.";

    for (size_t i = 0; i < std::min(input_tensors.size(), output_tensors.size()); ++i) {
      const auto &in_tensor = input_tensors[i];
      auto &out_tensor = const_cast<AxTensorInterface &>(output_tensors[i]);

      if (!in_tensor.data || !out_tensor.data) {
        logger(AX_ERROR)
            << "Null data pointer found during passthrough copy for tensor " << i;
        throw std::runtime_error("Null data pointer in passthrough");
      }

      const auto in_shape = to_4d_shape(in_tensor.sizes);
      auto padding = prop->paddings.empty() ?
                         std::vector<int>(in_shape.size() * 2, 0) :
                         (i < prop->paddings.size() ?
                                 prop->paddings[i] :
                                 std::vector<int>(in_shape.size() * 2, 0));

      // Validate padding dimensions match in_shape dimensions
      if ((padding.size() / 2) != in_shape.size()) {
        logger(AX_ERROR)
            << "Mismatch between padding dimensions (" << padding.size() / 2
            << ") and input shape dimensions (" << in_shape.size() << ").";
        throw std::runtime_error("Padding dimensions do not match input shape dimensions");
      }

      // Check if we need dequantization for this tensor
      if (!prop->dequant_scale.empty() && i < prop->dequant_scale.size()) {
        // Need dequantization
        if (in_tensor.bytes != 1) {
          logger(AX_ERROR)
              << "Tensor " << i << " is marked for dequantization but has "
              << in_tensor.bytes << " bytes per element (expected 1 for int8)";
          throw std::runtime_error("Invalid tensor type for dequantization");
        }

        std::vector<int> output_shape = process_input_tensor(in_tensor, in_shape,
            padding, prop, i, static_cast<float *>(out_tensor.data), logger);
        out_tensor.bytes = sizeof(float);
        out_tensor.sizes = output_shape;
      } else {
        // Just copy data (no dequantization)
        size_t copy_bytes = in_tensor.total() * in_tensor.bytes;
        std::memcpy(out_tensor.data, in_tensor.data, copy_bytes);
        out_tensor.bytes = in_tensor.bytes;
        out_tensor.sizes = in_tensor.sizes;
      }
    }
    return;
  }

  // --- ONNX Processing with I/O Binding ---

  // Get ONNX model information needed for run
  const auto &input_names = prop->onnx_runtime_->get_input_node_names();
  const auto &output_names = prop->onnx_runtime_->get_output_node_names();
  const auto &input_ranks = prop->onnx_runtime_->get_input_node_ranks();

  size_t num_onnx_inputs = input_names.size();
  size_t num_onnx_outputs = output_names.size();

  // Verify tensor selection plan validity (should match number of ONNX inputs)
  if (prop->tensor_selection_plan.size() != num_onnx_inputs) {
    logger(AX_ERROR)
        << "Invalid tensor selection plan size for ONNX model. Expected "
        << num_onnx_inputs << ", got " << prop->tensor_selection_plan.size();
    throw std::runtime_error("Invalid tensor selection plan size");
  }

  // Verify we have enough input tensors based on selection plan
  int max_index = -1;
  if (!prop->tensor_selection_plan.empty()) {
    max_index = *std::max_element(
        prop->tensor_selection_plan.begin(), prop->tensor_selection_plan.end());
  }
  if (input_tensors.size() <= static_cast<size_t>(max_index)) {
    logger(AX_ERROR) << "Not enough input tensors for tensor selection plan. "
                     << "Need at least " << (max_index + 1) << " but have "
                     << input_tensors.size();
    throw std::runtime_error("Not enough input tensors for selection plan");
  }

  // Prepare ONNX input tensors (Ort::Value wrappers around input data)
  std::vector<Ort::Value> onnx_inputs;
  onnx_inputs.reserve(num_onnx_inputs);
  try {
    for (size_t i = 0; i < num_onnx_inputs; ++i) {
      int tensor_idx = prop->tensor_selection_plan[i];
      auto &tensor = input_tensors[tensor_idx];
      if (!tensor.data) {
        logger(AX_ERROR) << "Input tensor " << tensor_idx
                         << " selected for ONNX has null data pointer.";
        throw std::runtime_error("Null data pointer in selected input tensor");
      }

      auto in_shape = to_4d_shape(tensor.sizes);
      // FIXED VERSION - uses correct tensor index (not ONNX input index)
      auto padding = prop->paddings.empty() ?
                         std::vector<int>(in_shape.size() * 2, 0) :
                         (static_cast<size_t>(tensor_idx) < prop->paddings.size() ?
                                 prop->paddings[tensor_idx] :
                                 std::vector<int>(in_shape.size() * 2, 0));

      // Validate that padding dimensions match in_shape dimensions
      if ((padding.size() / 2) != in_shape.size()) {
        logger(AX_ERROR)
            << "Mismatch between padding dimensions (" << padding.size() / 2
            << ") and input shape dimensions (" << in_shape.size() << ").";
        throw std::runtime_error("Padding dimensions do not match input shape dimensions");
      }

      if (!prop->dequant_scale.empty()) {
        // Check if this tensor needs dequantization
        if (tensor_idx >= static_cast<int>(prop->dequant_scale.size()) || tensor_idx < 0) {
          logger(AX_ERROR)
              << "Tensor " << tensor_idx
              << " is selected for ONNX but no dequantization parameters provided";
          throw std::runtime_error("Missing dequantization parameters for ONNX input tensor");
        }

        // Allocate buffer in the prop->input_datas for ONNX inputs if needed
        if (prop->input_datas[i].size() < tensor.total()) {
          prop->input_datas[i].resize(tensor.total());
        }

        std::vector<int> output_shape = process_input_tensor(tensor, in_shape,
            padding, prop, tensor_idx, prop->input_datas[i].data(), logger);
        prop->input_tensors[i].data = prop->input_datas[i].data();
        prop->input_tensors[i].bytes = sizeof(float);
        prop->input_tensors[i].sizes = output_shape;
      } else {
        // Validate that the original tensor format matches ONNX's expectations
        if (tensor.bytes != sizeof(float) || tensor.sizes.empty() || !tensor.data) {
          logger(AX_ERROR)
              << "Input tensor " << i
              << " is invalid. Data pointer is null or sizes are empty or datatype isn't float.";
          throw std::runtime_error("Invalid input tensor configuration for ONNX input");
        }
        prop->input_tensors[i] = tensor; // Use original tensor if no dequantization
      }
      onnx_inputs.push_back(convert_tensor_to_onnx_input(
          prop->input_tensors[i], logger, input_ranks[i]));
      logger(AX_DEBUG) << "Using input tensor " << tensor_idx << " ("
                       << shape_to_string(tensor.sizes) << ") as ONNX input "
                       << i << " (" << input_names[i] << ")";
    }
  } catch (const std::exception &e) {
    logger(AX_ERROR) << "Failed to convert input tensors to ONNX format: " << e.what();
    throw;
  }

  // Prepare pointers to the output AxTensorInterface objects for I/O Binding
  std::vector<AxTensorInterface *> onnx_output_ax_tensors;
  onnx_output_ax_tensors.reserve(num_onnx_outputs);
  if (output_tensors.size() < num_onnx_outputs) {
    logger(AX_ERROR)
        << "Framework did not provide enough output tensors for the ONNX model. "
        << "Expected " << num_onnx_outputs << ", got " << output_tensors.size();
    throw std::runtime_error("Insufficient output tensors allocated");
  }

  for (size_t i = 0; i < num_onnx_outputs; ++i) {
    // Need non-const pointer to modify the tensor interface (specifically its data buffer)
    auto &mutable_out_tensor = const_cast<AxTensorInterface &>(output_tensors[i]);
    if (!mutable_out_tensor.data) {
      logger(AX_ERROR) << "Output tensor " << i << " for ONNX binding has null data pointer.";
      throw std::runtime_error("Null data pointer in output tensor for binding");
    }
    onnx_output_ax_tensors.push_back(&mutable_out_tensor);
    logger(AX_DEBUG)
        << "Binding output tensor " << i << " (buffer for " << output_names[i]
        << ", shape " << shape_to_string(mutable_out_tensor.sizes) << ")";
  }

  // Run ONNX inference using I/O Binding
  try {
    logger(AX_DEBUG)
        << "Running ONNX inference with I/O Binding (" << onnx_inputs.size()
        << " inputs, " << onnx_output_ax_tensors.size() << " outputs).";

    prop->onnx_runtime_->run_with_io_binding(onnx_inputs, onnx_output_ax_tensors);

    logger(AX_INFO) << "ONNX inference completed using I/O Binding.";
    // Log final shapes placed in output buffers
    for (size_t i = 0; i < num_onnx_outputs; ++i) {
      logger(AX_INFO) << "Final data for ONNX output " << i << " ("
                      << output_names[i] << ") placed in output tensor " << i
                      << " with shape " << shape_to_string(output_tensors[i].sizes);
    }

  } catch (const std::exception &e) {
    logger(AX_ERROR) << "ONNX inference with I/O Binding failed: " << e.what();
    throw;
  }

  // --- Handle unused input tensors (passthrough) ---

  // Create a set of used input tensor indices for quick lookup
  std::unordered_set<int> used_indices(
      prop->tensor_selection_plan.begin(), prop->tensor_selection_plan.end());

  // Copy any unused input tensors to the remaining output tensor slots
  size_t output_idx = num_onnx_outputs; // Start filling after the ONNX outputs
  for (size_t i = 0; i < input_tensors.size(); ++i) {
    // Skip tensors used for ONNX input
    if (used_indices.find(static_cast<int>(i)) != used_indices.end()) {
      continue;
    }

    // Check if we have space in the output array
    if (output_idx < output_tensors.size()) {
      const auto &in_tensor = input_tensors[i];
      auto &mutable_out_tensor
          = const_cast<AxTensorInterface &>(output_tensors[output_idx]);

      if (!in_tensor.data || !mutable_out_tensor.data) {
        logger(AX_ERROR) << "Null data pointer found during passthrough copy for unused input "
                         << i << " to output " << output_idx;
        throw std::runtime_error("Null data pointer in passthrough (unused inputs)");
      }

      const auto in_shape = to_4d_shape(in_tensor.sizes);
      auto padding = prop->paddings.empty() ?
                         std::vector<int>(in_shape.size() * 2, 0) :
                         (i < prop->paddings.size() ?
                                 prop->paddings[i] :
                                 std::vector<int>(in_shape.size() * 2, 0));

      // Check if we need dequantization for this tensor
      if (!prop->dequant_scale.empty() && i < prop->dequant_scale.size()) {
        // Need dequantization
        if (in_tensor.bytes != 1) {
          logger(AX_ERROR)
              << "Unused tensor " << i << " is marked for dequantization but has "
              << in_tensor.bytes << " bytes per element (expected 1 for int8)";
          throw std::runtime_error("Invalid tensor type for dequantization");
        }

        std::vector<int> output_shape = process_input_tensor(in_tensor, in_shape,
            padding, prop, i, static_cast<float *>(mutable_out_tensor.data), logger);
        mutable_out_tensor.bytes = sizeof(float);
        mutable_out_tensor.sizes = output_shape;
      } else {
        // Just copy data (no dequantization)
        size_t copy_bytes = in_tensor.total() * in_tensor.bytes;
        std::memcpy(mutable_out_tensor.data, in_tensor.data, copy_bytes);
        mutable_out_tensor.bytes = in_tensor.bytes;
        mutable_out_tensor.sizes = in_tensor.sizes;
      }

      logger(AX_DEBUG) << "Processed unused input tensor " << i << " ("
                       << shape_to_string(in_tensor.sizes)
                       << ") directly to output tensor " << output_idx;

      // Increment output index
      output_idx++;
    } else {
      // This indicates a potential mismatch between calculation in set_output_interface and here
      logger(AX_WARN) << "Not enough output tensors allocated to store all unused input tensors. "
                      << "Stopped after output index " << (output_idx - 1);
      break;
    }
  }
}
