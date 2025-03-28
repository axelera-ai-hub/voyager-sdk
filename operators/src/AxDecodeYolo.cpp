// Copyright Axelera AI, 2024
// General YOLO decoder, the tensor is decoded by ONNXRuntime, filtered by
// parameters, and then passed into the ObjDetectionMeta

#include "AxDataInterface.h"
#include "AxLog.hpp"
#include "AxMetaObjectDetection.hpp"
#include "AxOnnxRuntimeHelper.hpp"
#include "AxOpUtils.hpp"
#include "AxUtils.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <utility>


using OnnxRuntimeInference = ax_onnxruntime::OnnxRuntimeInference;

namespace yolo
{
using lookups = ax_utils::lookups;
using inferences = ax_utils::inferences;

struct properties {
  std::string meta_name{};
  std::vector<lookups> dequantize_tables{};
  std::vector<std::vector<int>> paddings{};
  std::vector<std::string> class_labels{};
  std::vector<int> filter{};
  float confidence{ 0.25F };
  int num_classes{ 80 };
  int topk{ 30000 };
  bool multiclass{ false };
  bool transpose{ true };
  int model_width{ 640 };
  int model_height{ 640 };
  bool normalized_coord{ false };
  std::unique_ptr<OnnxRuntimeInference> feature_decoder;
  mutable std::vector<std::vector<float>> dequantized_data{};
  // TODO: suport xyxy for YOLO-NAS
  // std::string box_format {"xywh"};
};

float
dequantize(int8_t value, const float *the_table)
{
  int index = value + 128;
  return the_table[index];
}

std::vector<Ort::Value>
dequantize_and_prepare_tensors(const AxTensorsInterface &tensors,
    const properties &props, Ax::Logger &logger)
{
  std::vector<Ort::Value> ort_values;
  ort_values.reserve(props.feature_decoder->get_output_node_dims()[0].size());

  // Check if dequantize_tables are provided and match the size of quantized data
  if (props.dequantize_tables.empty()
      || props.dequantize_tables.size() != tensors.size()) {
    throw std::runtime_error("Dequantize tables size mismatch. Expected size: "
                             + std::to_string(tensors.size()) + ", got: "
                             + std::to_string(props.dequantize_tables.size()) + ".");
  }

  for (size_t i = 0; i < tensors.size(); ++i) {
    const auto &tensor = tensors[i];
    const auto *quantized_data = static_cast<const int8_t *>(tensor.data);

    const int dim0 = tensor.sizes[0];
    const int dim1 = tensor.sizes[1] - props.paddings[i][3];
    const int dim2 = tensor.sizes[2] - props.paddings[i][5];
    const int dim3 = tensor.sizes[3] - props.paddings[i][7];

    const auto &table = props.dequantize_tables[i].data();
    props.dequantized_data[i].clear();
    if (props.transpose) {
      // Transpose from NHWC to NCHW format and dequantize
      for (int n = 0; n < dim0; ++n) {
        int n_base = n * (tensor.sizes[1] * tensor.sizes[2] * tensor.sizes[3]);
        for (int c = 0; c < dim3; ++c) {
          for (int h = 0; h < dim1; ++h) {
            int nh_base = n_base + h * (tensor.sizes[2] * tensor.sizes[3]);
            for (int w = 0; w < dim2; ++w) {
              int nhwc_index = nh_base + w * tensor.sizes[3] + c;
              props.dequantized_data[i].push_back(
                  dequantize(quantized_data[nhwc_index], table));
            }
          }
        }
      }
    }

    // expect to be NCHW for ONNX
    std::vector<int64_t> nchw_size = std::vector<int64_t>{ dim0, dim3, dim1, dim2 };
    Ort::MemoryInfo memory_info
        = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value ort_value = Ort::Value::CreateTensor<float>(memory_info,
        props.dequantized_data[i].data(), props.dequantized_data[i].size(),
        nchw_size.data(), 4);
    ort_values.emplace_back(std::move(ort_value));
  }

  return ort_values;
}

template <typename T>
std::pair<ax_utils::stride_iterator<T>, ax_utils::stride_iterator<T>>
make_confidence_iterators(T *start, int num_classes, size_t num_samples, bool is_contiguous)
{
  size_t step = is_contiguous ? 1 : num_samples;
  return { ax_utils::make_stride_iterator(start, 0, step),
    ax_utils::make_stride_iterator(start, num_classes, step) };
}

///
///
inferences
decode_tensors(const AxTensorsInterface &tensors, const properties &props, Ax::Logger &logger)
{
  const int outputs_guess = 1000; //  Guess at the number of outputs to avoid allocs
  inferences output{ outputs_guess };

  // TODO: dequantize and depadding only if feature_decoder_onnx is an empty string
  auto ort_values = dequantize_and_prepare_tensors(tensors, props, logger);
  auto predicts = (*props.feature_decoder)(ort_values);

  if (predicts.size() != 1) {
    throw std::runtime_error("Expected to have 1 output tensor only, , got: "
                             + std::to_string(predicts.size()));
  } // TODO: support predicts.size==2 for YOLO-NAS
  auto &predict = predicts[0];
  float *tensor_data = predict.GetTensorMutableData<float>();
  auto tensor_info = predict.GetTensorTypeAndShapeInfo();
  auto dims = tensor_info.GetShape();

  if (dims.size() != 3) {
    std::stringstream ss;
    ss << "Unexpected tensor dimensions: ";
    for (auto dim : dims) {
      ss << dim << ", ";
    }
    throw std::runtime_error(ss.str());
  }

  // figure out dims[1] and dims[2] which is the number of samples and the number of data per sample
  // here we find the index of the larger one to be the number of samples
  // and the smaller one to be the number of data per sample
  int num_samples = dims[1] > dims[2] ? dims[1] : dims[2];
  int num_data_per_sample = dims[1] > dims[2] ? dims[2] : dims[1];
  bool is_sample_data_contiguous = dims[1] > dims[2];

  auto output_channels_excluding_classes = num_data_per_sample - props.num_classes;
  bool has_object_confidence = true; // for anchor-based models like YOLOv5 and YOLOv7

  if (output_channels_excluding_classes == 4) {
    // for anchor-free models like YOLOv8
    has_object_confidence = false;
  } else if (output_channels_excluding_classes != 5) {
    throw std::runtime_error("Unexpected number of output channels excluding classes: "
                             + std::to_string(output_channels_excluding_classes));
  }


  for (size_t i = 0; i < num_samples; ++i) {
    size_t base_offset = i * (is_sample_data_contiguous ? num_data_per_sample : 1);
    float *sample_ptr = tensor_data + base_offset;
    float obj_conf = has_object_confidence ? sample_ptr[4] : 1.0F;
    if (obj_conf < props.confidence)
      continue;
    float *class_confidences_ptr
        = sample_ptr
          + (has_object_confidence ? 5 : 4) * (is_sample_data_contiguous ? 1 : num_samples);

    auto [begin, end] = make_confidence_iterators(class_confidences_ptr,
        props.num_classes, num_samples, is_sample_data_contiguous);

    auto add_detection_to_output = [&](int class_idx, float score) {
      float cx = sample_ptr[0];
      float cy = sample_ptr[1 * (is_sample_data_contiguous ? 1 : num_samples)];
      float w = sample_ptr[2 * (is_sample_data_contiguous ? 1 : num_samples)];
      float h = sample_ptr[3 * (is_sample_data_contiguous ? 1 : num_samples)];
      ax_utils::fbox box = { cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2 };

      output.class_ids.emplace_back(class_idx);
      output.boxes.emplace_back(box);
      output.scores.emplace_back(score);
    };

    if (props.multiclass) {
      for (auto class_idx : props.filter) {
        auto it = std::next(begin, class_idx);
        float class_confidence = *it;
        float score = obj_conf * class_confidence;

        if (score > props.confidence) {
          add_detection_to_output(class_idx, score);
        }
      }
    } else {
      // Single class case: only consider the class with the highest confidence
      auto max_it = std::max_element(begin, end);
      float max_class_confidence = *max_it;
      int max_class_confidence_id = std::distance(begin, max_it);

      if (max_class_confidence < props.confidence)
        continue;
      float max_score = obj_conf * max_class_confidence;
      if (max_score > props.confidence
          && std::find(props.filter.begin(), props.filter.end(), max_class_confidence_id)
                 != props.filter.end()) {
        add_detection_to_output(max_class_confidence_id, max_score);
      }
    }
  }
  return output;
} // decode_tensors
} // namespace yolo

extern "C" std::shared_ptr<void>
init_and_set_static_properties(
    const std::unordered_map<std::string, std::string> &input, Ax::Logger &logger)
{
  auto props = std::make_shared<yolo::properties>();
  props->meta_name = Ax::get_property(
      input, "meta_key", "yolo_decode_static_properties", props->meta_name);
  auto zero_points = Ax::get_property(input, "zero_points",
      "yolo_decode_static_properties", std::vector<float>{});
  auto scales = Ax::get_property(
      input, "scales", "yolo_decode_static_properties", std::vector<float>{});

  props->num_classes = Ax::get_property(
      input, "classes", "yolo_decode_static_properties", props->num_classes);
  auto topk
      = Ax::get_property(input, "topk", "yolo_decode_static_properties", props->topk);
  if (topk > 0) {
    props->topk = topk;
  }

  auto filename = Ax::get_property(
      input, "classlabels_file", "yolo_decode_static_properties", std::string{});
  if (!filename.empty()) {
    props->class_labels = ax_utils::read_class_labels(
        filename, "yolo_decode_static_properties", logger);
  }
  if (props->num_classes == 0) {
    props->num_classes = props->class_labels.size();
  }
  props->filter = Ax::get_property(
      input, "label_filter", "detection_static_properties", props->filter);
  if (props->filter.empty()) {
    auto size = props->num_classes;
    props->filter.resize(size);
    std::iota(props->filter.begin(), props->filter.end(), 0);
  }
  std::sort(props->filter.begin(), props->filter.end());
  props->filter.erase(std::unique(props->filter.begin(), props->filter.end()),
      props->filter.end());

  props->multiclass = Ax::get_property(
      input, "multiclass", "yolo_decode_static_properties", props->multiclass);
  props->transpose = Ax::get_property(
      input, "transpose", "yolo_decode_static_properties", props->transpose);
  props->normalized_coord = Ax::get_property(input, "normalized_coord",
      "yolo_decode_static_properties", props->normalized_coord);

  //  Build the lookup tables
  if (zero_points.size() != scales.size()) {
    logger(AX_ERROR) << "yolo_decode_static_properties : zero_points and scales must be the same "
                        "size"
                     << std::endl;
    throw std::runtime_error(
        "yolo_decode_static_properties : zero_points and scales must be the same size");
  }

  props->dequantize_tables = ax_utils::build_dequantization_tables(zero_points, scales);

  ax_utils::validate_classes(props->class_labels, props->num_classes,
      "yolo_decode_static_properties", logger);

  props->paddings = Ax::get_property(
      input, "paddings", "yolo_decode_static_properties", props->paddings);
  for (const auto &padding : props->paddings) {
    if (padding.size() != 8) {
      logger(AX_ERROR) << "yolo_decode_static_properties : Padding values must be 8."
                       << std::endl;
      throw std::runtime_error("yolo_decode_static_properties : Padding values must be 8.");
    }
    if (padding[2] != 0 || padding[4] != 0 || padding[6] != 0) {
      logger(AX_ERROR) << "yolo_decode_static_properties : Padding values at lower positions must be zero."
                       << std::endl;
      throw std::runtime_error(
          "yolo_decode_static_properties : Non-zero padding values detected at lower positions.");
    }
  }

  auto feature_decoder_onnx_path = Ax::get_property(input,
      "feature_decoder_onnx", "yolo_decode_static_properties", std::string{});
  if (feature_decoder_onnx_path.empty()) {
    throw std::runtime_error("We don't expect using decode_yolo without an onnx for now.");
  } else {
    // check file exists
    if (!std::filesystem::exists(feature_decoder_onnx_path)) {
      logger(AX_ERROR) << "feature_decoder_onnx: " << feature_decoder_onnx_path
                       << " does not exist" << std::endl;
      throw std::runtime_error(
          "feature_decoder_onnx: " + feature_decoder_onnx_path + " does not exist");
    }
    try {
      props->feature_decoder
          = std::make_unique<OnnxRuntimeInference>(feature_decoder_onnx_path);
      auto input_node_dims = props->feature_decoder->get_input_node_dims();
      props->dequantized_data.resize(input_node_dims.size());
      for (size_t i = 0; i < input_node_dims.size(); ++i) {
        const auto &dims = input_node_dims[i];
        int size = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());
        props->dequantized_data[i].reserve(size);
      }
    } catch (const std::exception &e) {
      logger(AX_ERROR)
          << "Failed to initialize OnnxRuntimeInference with error: " << e.what()
          << std::endl;
      throw;
    }
  }
  return props;
}

extern "C" const std::unordered_set<std::string> &
allowed_properties()
{
  // clang-format off
  static const std::unordered_set<std::string> allowed_properties{
    "meta_key",
    "zero_points",
    "scales",
    "paddings",
    "classes",
    "topk",
    "multiclass",
    "model_width",
    "model_height",
    "normalized_coord",
    "classlabels_file",
    "confidence_threshold",
    "label_filter",
    "feature_decoder_onnx",
    "transpose"
  };
  // clang-format on
  return allowed_properties;
}

extern "C" void
set_dynamic_properties(const std::unordered_map<std::string, std::string> &input,
    yolo::properties *prop, Ax::Logger &logger)
{
  prop->confidence = Ax::get_property(input, "confidence_threshold",
      "detection_dynamic_properties", prop->confidence);
  logger(AX_DEBUG) << "prop->confidence_threshold is " << prop->confidence << std::endl;
}

extern "C" void
decode_to_meta(const AxTensorsInterface &in_tensors, const yolo::properties *prop,
    unsigned int subframe_index, unsigned int subframe_number,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &map,
    const AxDataInterface &video_interface, Ax::Logger &logger)
{
  auto start_time = std::chrono::high_resolution_clock::now();
  auto tensors = in_tensors;
  if (tensors.size() != prop->dequantize_tables.size() && tensors[0].bytes == 1) {
    std::stringstream ss;
    ss << "yolo_decode_to_meta : Number of input tensors (" << tensors.size()
       << ") does not match the number of dequantize parameters ("
       << prop->dequantize_tables.size() << ")";

    throw std::runtime_error(ss.str());
  }
  if (tensors.size() != prop->paddings.size()) {
    std::stringstream ss;
    ss << "yolo_decode_to_meta : Number of input tensors (" << tensors.size()
       << ") does not match the number of paddings (" << prop->paddings.size() << ")";

    throw std::runtime_error(ss.str());
  }

  auto predictions = yolo::decode_tensors(tensors, *prop, logger);
  predictions = ax_utils::topk(predictions, prop->topk);

  auto video_info = std::get<AxVideoInterface>(video_interface).info;

  float scale_width = static_cast<float>(prop->model_width) / video_info.width;
  float scale_height = static_cast<float>(prop->model_height) / video_info.height;
  float scale_ratio = std::min(scale_width, scale_height);
  float padding_width = (prop->model_width - video_info.width * scale_ratio) / 2.0f;
  float padding_height = (prop->model_height - video_info.height * scale_ratio) / 2.0f;

  // Converting coordinates to original image coordinates
  auto to_orig_x = [scale_ratio, padding_width,
                       width = prop->normalized_coord ? prop->model_width : 1,
                       max_width = video_info.width - 1](float x) {
    const int adjusted = (x * width - padding_width) / scale_ratio + 0.5F;
    return std::clamp(adjusted, 0, max_width);
  };

  auto to_orig_y = [scale_ratio, padding_height,
                       height = prop->normalized_coord ? prop->model_height : 1,
                       video_info](float y) {
    const int adjusted = (y * height - padding_height) / scale_ratio + 0.5F;
    return std::clamp(adjusted, 0, video_info.height - 1);
  };

  auto &source_boxes = predictions.boxes;
  std::vector<BboxXyxy> boxes;
  for (auto &b : source_boxes) {
    BboxXyxy box{
      to_orig_x(b.x1),
      to_orig_y(b.y1),
      to_orig_x(b.x2),
      to_orig_y(b.y2),
    };
    boxes.emplace_back(box);
  }

  map[prop->meta_name] = std::make_unique<AxMetaObjDetection>(std::move(boxes),
      std::move(predictions.scores), std::move(predictions.class_ids));

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
  logger(AX_INFO) << "decode_yolo: " << duration.count() << " microseconds" << std::endl;
}
