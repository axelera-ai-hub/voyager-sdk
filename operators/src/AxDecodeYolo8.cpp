// Copyright Axelera AI, 2024
// Optimized anchor-free YOLO(v8) decoder

#include "AxDataInterface.h"
#include "AxLog.hpp"
#include "AxMetaKptsDetection.hpp"
#include "AxMetaObjectDetection.hpp"
#include "AxMetaSegmentsDetection.hpp"
#include "AxOpUtils.hpp"
#include "AxThreadPool.hpp"
#include "AxUtils.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <numeric>
#include <unordered_set>
#include <vector>

namespace yolov8_decode
{
struct properties;
using lookups = std::array<float, 256>;
using inferences = ax_utils::inferences;
constexpr auto max_dfl_bins = 32;


struct decode_args {
  AxTensorsInterface tensors;
  int conf_idx;
  int box_idx;
  int kpt_idx;
  int angle_idx;
  int mask_idx;
  const properties *props;
  int level;
  int start_row;
  int end_row;
  inferences *outputs;
  bool focal_loss;
  Ax::Logger *logger;
  int num_predictions = 0;
};
using pool = Ax::threaded_runner<decode_args>;

struct properties {
  std::vector<lookups> sigmoid_tables{};
  std::vector<lookups> softmax_tables{};
  std::vector<lookups> dequantize_tables{};
  std::vector<ax_utils::sin_cos_lookups> sin_cos_tables{};
  std::vector<lookups> angle_tables{};
  std::vector<std::vector<int>> padding{};
  std::vector<float> zero_points{};
  std::vector<float> scales{};
  std::vector<std::string> class_labels{};
  std::vector<uint8_t> filter{};
  std::vector<float> weights{};

  float confidence{ 0.25F };
  int num_classes{ 0 };
  int topk{ 2000 };
  bool multiclass{ true };
  std::vector<int> kpts_shape{ 0, 0 };
  float kpt_multiplier{ 2.0F }; // YOLO8 uses 2.0, YOLO26 uses 1.0
  float kpt_offset{ 0.0F }; // YOLO8 uses 0.0, YOLO26 uses 0.5
  bool has_angle{ false }; // OBB: true if model outputs rotation angle (1 channel)
  bool raw_radians{ false }; // YOLO26-OBB uses raw radians (true), YOLO8-OBB uses sigmoid transformation (false)
  int num_seg_masks{ 0 }; // Segmentation: number of mask coefficients per detection (32 for YOLOv8-seg)
  std::string meta_name{};
  std::string master_meta{};
  std::string association_meta{};
  std::string decoder_name{};
  bool scale_up{ true };
  bool letterbox{ true };
  bool materialize_masks{ false }; // If true, generate masks immediately in decoder
  bool use_multithreading{ true }; // Enable/disable multithreading
  int model_width{};
  int model_height{};

  mutable std::unique_ptr<Ax::threaded_runner<decode_args>> pool{ nullptr };
};


/// @brief Sort the tensors into the order that they are expected to be in and
///        pairs the box prediction with the correspong confidence predictions.
///        We determine which is which from the channel size.
/// @param tensors - The tensors to sort
/// @param num_classes - The number of classes
/// @param kpts_per_box - The number of kpts per box
/// @param logger - The logger to use for logging
/// @return The sorted tensors

struct tensor_pair {
  int conf_idx;
  int box_idx;
  int kpt_idx;
  int angle_idx; // For OBB models
  int mask_idx; // For segmentation models
  int prototype_idx; // For segmentation models (prototype tensor index)
};

float
dequantize(int8_t value, const float *the_table)
{
  int index = value + 128;
  return the_table[index];
}

ax_utils::prototype_details
build_prototype_vector(const AxTensorsInterface &depadded, int prototype_stride,
    const std::vector<tensor_pair> &tensor_order, const properties &props)
{
  const auto prototype_idx = tensor_order[0].prototype_idx;
  if (prototype_idx == -1) {
    return {}; // No prototype tensor for non-segmentation models
  }

  auto [prototype_width, prototype_height, prototype_depth]
      = ax_utils::get_dims(depadded, prototype_idx, true);
  const auto prototype_tensor = depadded[prototype_idx];
  const auto *prototype_data = static_cast<const int8_t *>(prototype_tensor.data);

  // Validate that prototype depth matches num_seg_masks (e.g., 32 for YOLOv8-seg)
  if (prototype_depth != props.num_seg_masks) {
    throw std::runtime_error("build_prototype_vector: prototype tensor depth ("
                             + std::to_string(prototype_depth) + ") does not match num_seg_masks ("
                             + std::to_string(props.num_seg_masks) + ")");
  }

  auto scale = props.scales[prototype_idx];
  auto zero = props.zero_points[prototype_idx];
  // Dequantize and remove padding of prototype maps
  const size_t prototype_size = prototype_width * prototype_height * prototype_depth;
  const auto prototype_tensor_size = prototype_width * prototype_height * prototype_stride;
  ax_utils::prototype_details proto_details = {
    prototype_width,
    prototype_height,
    prototype_depth,
    scale,
    zero,
    std::make_unique_for_overwrite<uint8_t[]>(prototype_size),
    prototype_size,
  };
  auto *p = proto_details.coefs.get();

  for (int i = 0; i < prototype_tensor_size; i += prototype_stride) {
    std::memcpy(p, &prototype_data[i], prototype_depth);
    p += prototype_depth;
  }
  return proto_details;
}

std::vector<tensor_pair>
sort_tensors(const AxTensorsInterface &tensors, int num_classes,
    const std::vector<int> &kpts_shape, bool has_angle, int num_seg_masks,
    int dfl_size, Ax::Logger &logger)
{
  int kpts_per_box = kpts_shape.empty() ? 0 : kpts_shape[0];
  int kpts_channels = (kpts_shape.size() >= 2 && kpts_shape[0] > 0) ?
                          kpts_shape[0] * kpts_shape[1] :
                          0;

  bool has_segmentation = num_seg_masks > 0;
  bool has_extra_tensor = (kpts_per_box > 0 || has_angle);

  // Check for ambiguities before sorting
  // Box tensor has dfl_size*4 channels (dfl_size=1 for no focal loss -> 4 channels, dfl_size=16 -> 64 channels)
  const int box_channels = dfl_size * 4;

  if (num_classes == box_channels) {
    logger.throw_error("libdecode_yolov8: Cannot disambiguate tensors: num_classes ("
                       + std::to_string(num_classes) + ") == box_channels ("
                       + std::to_string(box_channels)
                       + " = dfl_size*4 = " + std::to_string(dfl_size) + "*4)");
  }
  if (has_segmentation && num_classes == num_seg_masks) {
    logger.throw_error("libdecode_yolov8: Cannot disambiguate tensors: num_classes ("
                       + std::to_string(num_classes) + ") == num_seg_masks ("
                       + std::to_string(num_seg_masks) + ")");
  }
  if (has_segmentation && num_seg_masks == box_channels) {
    logger.throw_error("libdecode_yolov8: Cannot disambiguate tensors: num_seg_masks ("
                       + std::to_string(num_seg_masks) + ") == box_channels ("
                       + std::to_string(box_channels)
                       + " = dfl_size*4 = " + std::to_string(dfl_size) + "*4)");
  }
  if (has_extra_tensor && kpts_channels > 0 && num_classes == kpts_channels) {
    logger.throw_error("libdecode_yolov8: Cannot disambiguate tensors: num_classes ("
                       + std::to_string(num_classes) + ") == kpts_channels ("
                       + std::to_string(kpts_channels) + ")");
  }
  if (has_angle && num_classes == 1) {
    logger.throw_error("libdecode_yolov8: Cannot disambiguate tensors: num_classes ("
                       + std::to_string(num_classes) + ") == angle_channels (1)");
  }

  // For segmentation, we have 1 prototype tensor + 3 tensors per level (conf, box, mask)
  if (has_segmentation) {
    if ((tensors.size() - 1) % 3 != 0) {
      logger.throw_error("libdecode_yolov8: For segmentation, number of tensors - 1 must be multiple of 3, but got "
                         + std::to_string(tensors.size()));
    }
  } else if (tensors.size() % 2 != 0 && !has_extra_tensor) {
    logger.throw_error("libdecode_yolov8: The number of tensors must be even, but got "
                       + Ax::to_string(tensors) + " when num_kpts=0 and has_angle=false");
  } else if (tensors.size() % 3 != 0 && has_extra_tensor) {
    logger.throw_error("libdecode_yolov8: The number of tensors must be multiple of 3, but got "
                       + Ax::to_string(tensors) + " when num_kpts>0 ("
                       + std::to_string(kpts_per_box) + ") or has_angle=true");
  }

  std::vector<int> indices(tensors.size());
  std::iota(std::begin(indices), std::end(indices), 0);
  std::sort(std::begin(indices), std::end(indices), [&tensors](auto a, auto b) {
    const int width_or_height_idx = 2;
    const int channels_idx = 3;
    return std::tie(tensors[a].sizes[width_or_height_idx], tensors[a].sizes[channels_idx])
           > std::tie(tensors[b].sizes[width_or_height_idx], tensors[b].sizes[channels_idx]);
  });

  std::vector<tensor_pair> tensor_pairs;

  if (has_segmentation) {
    // For segmentation: first tensor is prototype, rest are grouped in threes
    int prototype_idx = indices[0]; // Largest tensor is the prototype

    for (auto i = size_t{ 1 }; i < indices.size(); i += 3) {
      // Identify tensors by their channel depth (similar to pose/OBB logic)
      const int channels_idx = 3;
      int conf_idx = -1, box_idx = -1, mask_idx = -1;

      for (auto j = i; j < i + 3; ++j) {
        int channels = tensors[indices[j]].sizes[channels_idx];
        if (channels == num_seg_masks) {
          mask_idx = indices[j];
        } else if (channels == box_channels) {
          box_idx = indices[j];
        } else if (channels == num_classes) {
          conf_idx = indices[j];
        } else {
          logger.throw_error("libdecode_yolov8: Unable to identify segmentation tensor with "
                             + std::to_string(channels) + " channels. Expected mask ("
                             + std::to_string(num_seg_masks) + "), box ("
                             + std::to_string(box_channels) + "), or conf ("
                             + std::to_string(num_classes) + ")");
        }
      }

      // Validate we found all required tensors
      if (conf_idx == -1 || box_idx == -1 || mask_idx == -1) {
        logger.throw_error("libdecode_yolov8: Failed to identify required segmentation tensors (conf, box, mask) at level");
      }

      tensor_pairs.push_back(
          tensor_pair{ conf_idx, box_idx, -1, -1, mask_idx, prototype_idx });
    }
  } else {
    auto swap = num_classes < 64;
    int tensors_per_level = has_extra_tensor ? 3 : 2;

    for (auto i = size_t{}; i != indices.size(); i += tensors_per_level) {
      if (!has_extra_tensor) {
        tensor_pairs.push_back(
            swap ? tensor_pair{ indices[i + 1], indices[i], -1, -1, -1, -1 } :
                   tensor_pair{ indices[i], indices[i + 1], -1, -1, -1, -1 });
      } else {
        // For pose or OBB models, identify tensors by their channel depth
        const int channels_idx = 3;
        int conf_idx = -1, box_idx = -1, kpt_idx = -1, angle_idx = -1;

        for (auto j = i; j < i + 3; ++j) {
          int channels = tensors[indices[j]].sizes[channels_idx];
          if (channels == kpts_channels && kpts_channels > 0) {
            kpt_idx = indices[j];
          } else if (channels == 1 && has_angle) {
            angle_idx = indices[j];
          } else if (channels == box_channels) {
            box_idx = indices[j];
          } else if (channels == num_classes) {
            conf_idx = indices[j];
          } else {
            logger.throw_error("libdecode_yolov8: Unable to identify tensor with "
                               + std::to_string(channels) + " channels. Expected kpts ("
                               + std::to_string(kpts_channels) + "), angle (1), box ("
                               + std::to_string(box_channels) + "), or conf ("
                               + std::to_string(num_classes) + ")");
          }
        }

        // Validate we found required tensors
        if (conf_idx == -1 || box_idx == -1) {
          logger.throw_error("libdecode_yolov8: Failed to identify required tensors (conf, box) at level");
        }
        if (kpts_per_box > 0 && kpt_idx == -1) {
          logger.throw_error("libdecode_yolov8: Failed to identify keypoint tensor at level");
        }
        if (has_angle && angle_idx == -1) {
          logger.throw_error("libdecode_yolov8: Failed to identify angle tensor at level");
        }

        tensor_pairs.push_back(
            tensor_pair{ conf_idx, box_idx, kpt_idx, angle_idx, -1, -1 });
      }
    }
  }

  return tensor_pairs;
}

/// @brief Decode a single cell of the tensor
/// @param box_data - pointer to the raw tensor box data
/// @param score_data - pointer to the raw tensor score
/// @param props - properties of the model
/// @param score_level - index of score tensor
/// @param box_level - index of box tensor
/// @param recip_width - scale facror
/// @param xpos - x position of the cell
/// @param ypos - y position of the cell
/// @param outputs - output inferences
/// @return - The number of predictions added
int
decode_cell(const int8_t *box_data, const int8_t *score_data, const int8_t *kpts_data,
    const int8_t *angle_data, const int8_t *mask_data, const properties &props,
    int score_level, int box_level, int kpt_level, int angle_level, int mask_level,
    float recip_width, int xpos, int ypos, bool focal_loss, inferences &outputs)
{
  // OPTIMIZATION: Check scores first before decoding boxes
  // This avoids expensive box decoding (especially softmax for focal loss) for low-confidence cells
  const auto num_predictions = ax_utils::decode_scores(score_data,
      props.sigmoid_tables[score_level].data(), props.filter, props.confidence,
      props.multiclass, outputs);
  if (!num_predictions) {
    return 0;
  }

  // Now decode boxes only if we have valid scores
  std::array<float, 4> box;
  if (!focal_loss) {
    const auto &box_lookups = props.dequantize_tables[box_level].data();
    for (auto &b : box) {
      b = yolov8_decode::dequantize(*box_data++, box_lookups);
    }
  } else {
    const auto &softmax_lookups = props.softmax_tables[box_level].data();
    std::array<float, max_dfl_bins> softmaxed;
    for (auto &b : box) {
      ax_utils::softmax(
          box_data, props.weights.size(), softmax_lookups, softmaxed.data());
      b = std::transform_reduce(
          props.weights.begin(), props.weights.end(), softmaxed.begin(), 0.0F);
      box_data = std::next(box_data, props.weights.size());
    }
  }
  const float w = (box[0] + box[2]);
  const float h = (box[1] + box[3]);
  // Skip invalid boxes with negative total width or height
  // If box is invalid, we need to remove the scores we already added
  if (w < 0.0F || h < 0.0F) {
    // Remove the scores we just added
    outputs.scores.resize(outputs.scores.size() - num_predictions);
    outputs.class_ids.resize(outputs.class_ids.size() - num_predictions);
    return 0;
  }

  if (angle_level != -1) {
    // For OBB, decode angle and output xywhr format
    const auto &angle_lookups = props.angle_tables[angle_level].data();
    const auto &sin_cos_lookups = props.sin_cos_tables[angle_level].data();
    float angle = yolov8_decode::dequantize(*angle_data, angle_lookups);
    float sin_a = yolov8_decode::dequantize(*angle_data, sin_cos_lookups);
    float cos_a = yolov8_decode::dequantize(*angle_data, sin_cos_lookups + 256);

    // Get half-width and half-height from ltrb offsets
    const auto xf = (box[2] - box[0]) * 0.5F;
    const auto yf = (box[3] - box[1]) * 0.5F;

    // Apply rotation using sin/cos
    const float cx = xf * cos_a - yf * sin_a;
    const float cy = xf * sin_a + yf * cos_a;

    outputs.obb.insert(outputs.obb.end(), num_predictions,
        {
            std::clamp((cx + xpos + 0.5F) * recip_width, 0.0F, 1.0F),
            std::clamp((cy + ypos + 0.5F) * recip_width, 0.0F, 1.0F),
            std::clamp(w * recip_width, 0.0F, 1.0F),
            std::clamp(h * recip_width, 0.0F, 1.0F),
            angle,
        });
  } else {
    // Regular detection: output xyxy format
    const auto x1 = (xpos + 0.5F - box[0]) * recip_width;
    const auto y1 = (ypos + 0.5F - box[1]) * recip_width;
    const auto x2 = (xpos + 0.5F + box[2]) * recip_width;
    const auto y2 = (ypos + 0.5F + box[3]) * recip_width;

    outputs.boxes.insert(outputs.boxes.end(), num_predictions,
        {
            std::clamp(x1, 0.0F, 1.0F),
            std::clamp(y1, 0.0F, 1.0F),
            std::clamp(x2, 0.0F, 1.0F),
            std::clamp(y2, 0.0F, 1.0F),
        });

    // For segmentation, also store mask coefficients
    if (mask_level != -1) {
      outputs.seg_info.insert(outputs.seg_info.end(), num_predictions,
          {
              x1,
              y1,
              x2,
              y2,
              props.scales[mask_level],
              props.zero_points[mask_level],
              std::vector<int8_t>(mask_data, mask_data + props.num_seg_masks),
          });
    }
  }

  if (kpt_level != -1) {
    const auto &dequantize_lookups = props.dequantize_tables[kpt_level].data();
    const auto &sigmoid_lookups = props.sigmoid_tables[kpt_level].data();
    auto *kpts_ptr = kpts_data;

    std::vector<ax_utils::fkpt> kpts;
    kpts.reserve(props.kpts_shape[0]);
    for (auto i = 0; i < props.kpts_shape[0]; ++i) {
      const auto x
          = (xpos + props.kpt_offset
                + props.kpt_multiplier * yolov8_decode::dequantize(kpts_ptr[0], dequantize_lookups))
            * recip_width;
      const auto y
          = (ypos + props.kpt_offset
                + props.kpt_multiplier * yolov8_decode::dequantize(kpts_ptr[1], dequantize_lookups))
            * recip_width;
      const auto v = props.kpts_shape[1] == 3 ?
                         ax_utils::sigmoid(kpts_ptr[2], sigmoid_lookups) :
                         1.0F;

      kpts.push_back({ std::clamp(x, 0.0F, 1.0F), std::clamp(y, 0.0F, 1.0F), v });
      kpts_ptr = std::next(kpts_ptr, props.kpts_shape[1]);
    }

    for (auto i = 0; i < num_predictions; ++i) {
      outputs.kpts.insert(outputs.kpts.end(), kpts.begin(),
          std::next(kpts.begin(), props.kpts_shape[0]));
    }
  }

  return num_predictions;
}

///
/// @brief Decode a single feature map tensor
/// @param tensors - The tensor data
/// @param score_idx - The index of the score tensor
/// @param box_idx - The index of the box tensor
/// @param props - The properties of the model
/// @param level - which of features maps this tensor is
/// @param outputs - output inferences
/// @param start_row - first row to decode (for threading)
/// @param end_row - last row to decode (for threading)
/// @param logger - The logger to use for logging
/// @return - The number of predictions added
int
decode_tensor(const AxTensorsInterface &tensors, int score_idx, int box_idx,
    int kpt_idx, int angle_idx, int mask_idx, const properties &props, int level,
    inferences &outputs, bool focal_loss, int start_row, int end_row, Ax::Logger &logger)
{
  auto [box_width, box_height, box_depth] = ax_utils::get_dims(tensors, box_idx, true);
  auto [score_width, score_height, score_depth]
      = ax_utils::get_dims(tensors, score_idx, true);
  auto [kpts_width, kpts_height, kpts_depth] = ax_utils::get_dims(tensors, kpt_idx, true);
  auto [angle_width, angle_height, angle_depth]
      = ax_utils::get_dims(tensors, angle_idx, true);
  auto [mask_width, mask_height, mask_depth] = ax_utils::get_dims(tensors, mask_idx, true);

  if (box_width != score_width || box_height != score_height) {
    logger(AX_ERROR) << "decode_tensor : box and score tensors must be the same size"
                     << std::endl;
    return 0;
  }
  const auto box_x_stride = box_depth;
  const auto box_y_stride = box_x_stride * box_width;

  const auto score_x_stride = score_depth;
  const auto score_y_stride = score_x_stride * score_width;

  const auto kpts_x_stride = kpts_depth;
  const auto kpts_y_stride = kpts_x_stride * kpts_width;

  const auto angle_x_stride = angle_depth;
  const auto angle_y_stride = angle_x_stride * angle_width;

  const auto mask_x_stride = mask_depth;
  const auto mask_y_stride = mask_x_stride * mask_width;

  const auto box_tensor = tensors[box_idx];
  const auto score_tensor = tensors[score_idx];
  const auto kpts_tensor = kpt_idx == -1 ? nullptr : tensors[kpt_idx].data;
  const auto angle_tensor = angle_idx == -1 ? nullptr : tensors[angle_idx].data;
  const auto mask_tensor = mask_idx == -1 ? nullptr : tensors[mask_idx].data;
  const auto *box_data = static_cast<const int8_t *>(box_tensor.data);
  const auto *score_data = static_cast<const int8_t *>(score_tensor.data);
  const auto *kpts_data = static_cast<const int8_t *>(kpts_tensor);
  const auto *angle_data = static_cast<const int8_t *>(angle_tensor);
  const auto *mask_data = static_cast<const int8_t *>(mask_tensor);

  auto total = 0;
  const auto recip_width = 1.0F / std::max(box_width, box_height);

  for (auto y = start_row; y != end_row; ++y) {
    auto *box_ptr = std::next(box_data, box_y_stride * y);
    auto *score_ptr = std::next(score_data, score_y_stride * y);
    auto *kpts_ptr = kpts_data != nullptr ? std::next(kpts_data, kpts_y_stride * y) : nullptr;
    auto *angle_ptr
        = angle_data != nullptr ? std::next(angle_data, angle_y_stride * y) : nullptr;
    auto *mask_ptr = mask_data != nullptr ? std::next(mask_data, mask_y_stride * y) : nullptr;

    for (auto x = 0; x != box_width; ++x) {
      total += decode_cell(box_ptr, score_ptr, kpts_ptr, angle_ptr, mask_ptr,
          props, score_idx, box_idx, kpt_idx, angle_idx, mask_idx, recip_width,
          x, y, focal_loss, outputs);
      box_ptr = std::next(box_ptr, box_x_stride);
      score_ptr = std::next(score_ptr, score_x_stride);
      kpts_ptr = kpts_data != nullptr ? std::next(kpts_ptr, kpts_x_stride) : nullptr;
      angle_ptr = angle_data != nullptr ? std::next(angle_ptr, angle_x_stride) : nullptr;
      mask_ptr = mask_data != nullptr ? std::next(mask_ptr, mask_x_stride) : nullptr;
    }
  }
  return total;
}

// Wrapper for thread pool
void
decode_tensor(decode_args &details)
{
  details.num_predictions += decode_tensor(details.tensors, details.conf_idx,
      details.box_idx, details.kpt_idx, details.angle_idx, details.mask_idx,
      *details.props, details.level, *details.outputs, details.focal_loss,
      details.start_row, details.end_row, *details.logger);
}

void
add_tasks(const decode_args &args, int total_rows, int num_tasks,
    std::vector<inferences> &outputs, std::vector<pool::packaged_task> &tasks)
{
  if (total_rows % num_tasks != 0) {
    num_tasks = 1;
  }
  int rows_per_task = total_rows / num_tasks;
  int start_row = 0;
  for (int i = 0; i < num_tasks; ++i) {
    auto end_row = start_row + rows_per_task;
    decode_args task_args = args;
    task_args.start_row = start_row;
    task_args.end_row = end_row;
    task_args.outputs = &outputs[tasks.size()];
    tasks.push_back(pool::packaged_task{ decode_tensor, task_args });
    start_row = end_row;
  }
}

AxTensorsInterface
depad_tensors(const AxTensorsInterface &tensors, const std::vector<std::vector<int>> &padding)
{
  auto depadded = tensors;
  for (auto i = 0; i != padding.size(); ++i) {
    depadded[i].sizes[3] = tensors[i].sizes[3] - padding[i][7] - padding[i][6];
  }
  return depadded;
}

/// @brief Decode the tensors into a set of inferences
/// @param tensors - The input tensors
/// @param prop - The properties of the model
/// @param padding - The padding for each tensor
/// @param logger - The logger to use for logging
/// @return The resulting inferences

inferences
decode_tensors(const AxTensorsInterface &tensors, const properties &prop,
    const std::vector<std::vector<int>> &padding, Ax::Logger &logger)
{
  auto depadded = depad_tensors(tensors, padding);
  auto tensor_order = sort_tensors(depadded, prop.num_classes, prop.kpts_shape,
      prop.has_angle, prop.num_seg_masks, prop.weights.size(), logger);

  // Parallel path: use thread pool
  if (prop.use_multithreading) {
    // We split the largest tensor into 4 smaller tensors and process them in
    // parallel hence need 3 extra tasks
    int total_tasks = tensor_order.size() + 3;
    std::vector<inferences> intermediate_outputs;
    intermediate_outputs.reserve(total_tasks);
    const auto preallocation_size = 1000;
    for (auto i = 0; i != total_tasks; ++i) {
      intermediate_outputs.emplace_back(
          preallocation_size, preallocation_size * prop.kpts_shape[0]);
      intermediate_outputs.back().kpts_shape = prop.kpts_shape;
    }

    if (!prop.pool) {
      prop.pool = std::make_unique<pool>(total_tasks);
    }

    pool &runner = *prop.pool;
    std::vector<pool::packaged_task> tasks;
    for (int level = 0; level != tensor_order.size(); ++level) {
      const auto [conf_tensor, loc_tensor, kpt_tensor, angle_tensor, mask_tensor, prototype_tensor]
          = tensor_order[level];

      auto [box_width, box_height, box_depth]
          = ax_utils::get_dims(depadded, loc_tensor, true);
      const int expected_box_depth = prop.weights.size() * 4;
      bool focal_loss = (prop.weights.size() > 1);
      if (box_depth != expected_box_depth) {
        logger.throw_error("decode_tensors : invalid box tensor shape, expect ["
                           + std::to_string(box_width) + ", " + std::to_string(box_height)
                           + ", " + std::to_string(expected_box_depth)
                           + "] (dfl_size=" + std::to_string(prop.weights.size())
                           + " -> " + std::to_string(prop.weights.size()) + "*4) but got: ["
                           + std::to_string(box_width) + ", " + std::to_string(box_height)
                           + ", " + std::to_string(box_depth) + "] ");
      }

      auto args = decode_args{
        .tensors = tensors,
        .conf_idx = conf_tensor,
        .box_idx = loc_tensor,
        .kpt_idx = kpt_tensor,
        .angle_idx = angle_tensor,
        .mask_idx = mask_tensor,
        .props = &prop,
        .level = level,
        .start_row = 0,
        .end_row = box_height,
        .outputs = nullptr,
        .focal_loss = focal_loss,
        .logger = &logger,
      };
      add_tasks(args, box_height, level == 0 ? 4 : 1, intermediate_outputs, tasks);
    }
    runner.run(tasks);

    auto predictions = std::move(intermediate_outputs[0]);
    for (int level = 1; level < tasks.size(); ++level) {
      predictions.extend(std::move(intermediate_outputs[level]));
    }

    // Build prototype vector for segmentation
    if (prop.num_seg_masks > 0 && !tensor_order.empty()) {
      auto prototype_idx = tensor_order[0].prototype_idx;
      auto prototype_stride = ax_utils::get_dims(tensors, prototype_idx, true).depth;
      predictions.set_prototype(
          build_prototype_vector(depadded, prototype_stride, tensor_order, prop));
    }

    return predictions;
  }

  // Sequential path: no threading
  inferences predictions(1000, 1000 * prop.kpts_shape[0]);
  predictions.kpts_shape = prop.kpts_shape;

  for (int level = 0; level != tensor_order.size(); ++level) {
    const auto [conf_tensor, loc_tensor, kpt_tensor, angle_tensor, mask_tensor, prototype_tensor]
        = tensor_order[level];

    auto [box_width, box_height, box_depth]
        = ax_utils::get_dims(depadded, loc_tensor, true);
    const int expected_box_depth = prop.weights.size() * 4;
    bool focal_loss = (prop.weights.size() > 1);
    if (box_depth != expected_box_depth) {
      logger.throw_error("decode_tensors : invalid box tensor shape, expect ["
                         + std::to_string(box_width) + ", " + std::to_string(box_height)
                         + ", " + std::to_string(expected_box_depth)
                         + "] (dfl_size=" + std::to_string(prop.weights.size())
                         + " -> " + std::to_string(prop.weights.size()) + "*4) but got: ["
                         + std::to_string(box_width) + ", " + std::to_string(box_height)
                         + ", " + std::to_string(box_depth) + "] ");
    }
    decode_tensor(tensors, conf_tensor, loc_tensor, kpt_tensor, angle_tensor,
        mask_tensor, prop, level, predictions, focal_loss, 0, box_height, logger);
  }

  // Build prototype vector for segmentation
  if (prop.num_seg_masks > 0 && !tensor_order.empty()) {
    auto prototype_idx = tensor_order[0].prototype_idx;
    auto prototype_stride = ax_utils::get_dims(tensors, prototype_idx, true).depth;
    predictions.set_prototype(
        build_prototype_vector(depadded, prototype_stride, tensor_order, prop));
  }

  return predictions;
}

} // namespace yolov8_decode

extern "C" void
decode_to_meta(const AxTensorsInterface &in_tensors, const yolov8_decode::properties *prop,
    unsigned int subframe_index, unsigned int number_of_subframes,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &map,
    const AxDataInterface &video_interface, Ax::Logger &logger)
{
  auto start_time = std::chrono::high_resolution_clock::now();
  if (!prop) {
    logger(AX_ERROR) << "decode_to_meta : properties not set" << std::endl;
    throw std::runtime_error("decode_to_meta : properties not set");
  }
  auto tensors = in_tensors;
  auto padding = prop->padding;
  if (tensors.size() != prop->sigmoid_tables.size() && tensors[0].bytes == 1) {
    throw std::runtime_error(
        "ssd_decode_to_meta : Number of input tensors or dequantize parameters is incorrect");
  }

  if (tensors.size() == 1) {
    throw std::runtime_error(
        "depadd_tensors : padding cannot be applied when there is only one tensor, consider removing handle_all: True from pipeline");
  }

  if (tensors.size() != padding.size()) {
    throw std::runtime_error(
        "depadd_tensors : number of tensors: " + std::to_string(tensors.size())
        + " and padding size " + std::to_string(padding.size()) + " do not match");
  }

  auto predictions = yolov8_decode::decode_tensors(tensors, *prop, padding, logger);
  predictions = ax_utils::topk(std::move(predictions), prop->topk);

  auto base_box = ax_utils::get_master_box(prop->master_meta, prop->association_meta,
      video_interface, subframe_index, map, prop->decoder_name);

  std::vector<int> ids;
  if (prop->has_angle) {
    // OBB: use predictions.obb with xywhr format
    auto pixel_boxes = ax_utils::scale_shift_boxes(predictions.obb, base_box,
        prop->model_width, prop->model_height, prop->scale_up, prop->letterbox);
    ax_utils::insert_and_associate_meta<AxMetaObjDetectionOBB>(map,
        prop->meta_name, prop->master_meta, subframe_index, number_of_subframes,
        prop->association_meta, std::move(pixel_boxes),
        std::move(predictions.scores), std::move(predictions.class_ids), ids);
  } else if (prop->num_seg_masks > 0) {
    // Segmentation: use predictions.boxes with segmentation info
    auto pixel_boxes = ax_utils::scale_shift_boxes(predictions.boxes, base_box,
        prop->model_width, prop->model_height, true, prop->letterbox);
    auto sizes = SegmentShape{ static_cast<size_t>(predictions.prototype.width),
      static_cast<size_t>(predictions.prototype.height) };
    // Materialize segments if requested
    if (prop->materialize_masks) {
      SegmentList segment_maps;
      segment_maps.reserve(predictions.seg_info.size());
      for (size_t i = 0; i < predictions.seg_info.size(); ++i) {
        segment_maps.push_back(decode_segment(predictions.seg_info[i],
            predictions.prototype, sizes.width, sizes.height, base_box));
      }
      // Create metadata with materialized segments
      ax_utils::insert_and_associate_meta<AxMetaSegmentsDetection>(map,
          prop->meta_name, prop->master_meta, subframe_index,
          number_of_subframes, prop->association_meta, std::move(pixel_boxes),
          std::move(segment_maps), std::move(predictions.scores),
          std::move(predictions.class_ids), ids, sizes, prop->decoder_name);
    } else {
      // Create metadata with recipes
      ax_utils::insert_and_associate_meta<AxMetaSegmentsDetection>(map,
          prop->meta_name, prop->master_meta, subframe_index, number_of_subframes,
          prop->association_meta, std::move(pixel_boxes), std::move(predictions.seg_info),
          std::move(predictions.scores), std::move(predictions.class_ids), ids, sizes,
          std::move(predictions.prototype), std::move(base_box), prop->decoder_name);
    }
  } else if (prop->kpts_shape[0] > 0) {
    // Pose: use predictions.boxes with keypoints
    auto pixel_boxes = ax_utils::scale_shift_boxes(predictions.boxes, base_box,
        prop->model_width, prop->model_height, prop->scale_up, prop->letterbox);
    auto pixel_kpts = ax_utils::scale_shift_kpts(predictions.kpts, base_box,
        prop->model_width, prop->model_height, prop->scale_up, prop->letterbox);
    ax_utils::insert_and_associate_meta<AxMetaKptsDetection>(map,
        prop->meta_name, prop->master_meta, subframe_index, number_of_subframes,
        prop->association_meta, std::move(pixel_boxes), std::move(pixel_kpts),
        std::move(predictions.scores), ids, prop->kpts_shape, prop->decoder_name);
  } else {
    // Regular detection: use predictions.boxes
    auto pixel_boxes = ax_utils::scale_shift_boxes(predictions.boxes, base_box,
        prop->model_width, prop->model_height, prop->scale_up, prop->letterbox);
    ax_utils::insert_and_associate_meta<AxMetaObjDetection>(map,
        prop->meta_name, prop->master_meta, subframe_index, number_of_subframes,
        prop->association_meta, std::move(pixel_boxes),
        std::move(predictions.scores), std::move(predictions.class_ids), ids);
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
  logger(AX_DEBUG) << "decode_to_meta : Decoding took " << duration.count()
                   << " microseconds" << std::endl;
}

extern "C" const std::unordered_set<std::string> &
allowed_properties()
{
  static const std::unordered_set<std::string> allowed_properties{
    "meta_key",
    "master_meta",
    "association_meta",
    "classlabels_file",
    "confidence_threshold",
    "max_boxes",
    "label_filter",
    "topk",
    "zero_points",
    "scales",
    "multiclass",
    "classes",
    "padding",
    "kpts_shape",
    "kpt_multiplier",
    "kpt_offset",
    "has_angle",
    "raw_radians",
    "num_seg_masks",
    "dfl_size",
    "decoder_name",
    "scale_up",
    "letterbox",
    "materialize_masks",
    "use_multithreading",
    "model_width",
    "model_height",
  };
  return allowed_properties;
}

extern "C" std::shared_ptr<void>
init_and_set_static_properties(
    const std::unordered_map<std::string, std::string> &input, Ax::Logger &logger)
{
  auto props = std::make_shared<yolov8_decode::properties>();
  props->meta_name = Ax::get_property(
      input, "meta_key", "detection_static_properties", props->meta_name);
  props->master_meta = Ax::get_property(
      input, "master_meta", "detection_static_properties", props->master_meta);
  props->association_meta = Ax::get_property(input, "association_meta",
      "detection_static_properties", props->association_meta);
  props->zero_points = Ax::get_property(
      input, "zero_points", "detection_static_properties", props->zero_points);
  props->scales = Ax::get_property<float>(
      input, "scales", "detection_static_properties", props->scales);
  props->num_classes = Ax::get_property(
      input, "classes", "detection_static_properties", props->num_classes);
  props->kpts_shape = Ax::get_property(
      input, "kpts_shape", "detection_static_properties", props->kpts_shape);
  props->kpt_multiplier = Ax::get_property(input, "kpt_multiplier",
      "detection_static_properties", props->kpt_multiplier);
  props->kpt_offset = Ax::get_property(
      input, "kpt_offset", "detection_static_properties", props->kpt_offset);
  props->has_angle = Ax::get_property(
      input, "has_angle", "detection_static_properties", props->has_angle);
  props->raw_radians = Ax::get_property(
      input, "raw_radians", "detection_static_properties", props->raw_radians);
  props->num_seg_masks = Ax::get_property(input, "num_seg_masks",
      "detection_static_properties", props->num_seg_masks);
  props->decoder_name = Ax::get_property(
      input, "decoder_name", "detection_static_properties", props->decoder_name);
  auto topk
      = Ax::get_property(input, "topk", "detection_static_properties", props->topk);
  if (topk > 0) {
    props->topk = topk;
  }
  props->scale_up = Ax::get_property(
      input, "scale_up", "detection_static_properties", props->scale_up);
  props->materialize_masks = Ax::get_property(input, "materialize_masks",
      "detection_static_properties", props->materialize_masks);
  props->use_multithreading = Ax::get_property(input, "use_multithreading",
      "detection_static_properties", props->use_multithreading);

  props->model_width = Ax::get_property(
      input, "model_width", "detection_static_properties", props->model_width);
  props->model_height = Ax::get_property(
      input, "model_height", "detection_static_properties", props->model_height);
  if (props->model_height == 0 || props->model_width == 0) {
    logger(AX_ERROR) << "detection_static_properties : model_width and model_height must be "
                        "provided"
                     << std::endl;
    throw std::runtime_error(
        "detection_static_properties : model_width and model_height must be provided");
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

  props->multiclass = Ax::get_property(
      input, "multiclass", "detection_static_properties", props->multiclass);

  //  Build the lookup tables
  if (props->zero_points.size() != props->scales.size()) {
    logger(AX_ERROR) << "detection_static_properties : zero_points and scales must have the same "
                        "number of elements."
                     << std::endl;
    throw std::runtime_error(
        "detection_static_properties : zero_points and scales must be the same size");
  }

  if (props->num_classes == 0) {
    if (!props->class_labels.empty()) {
      props->num_classes = props->class_labels.size();
    }
  }
  if (props->num_classes != 0) {
    ax_utils::validate_classes(props->class_labels, props->num_classes,
        "yolo_decode_static_properties", logger);
  }

  props->sigmoid_tables
      = ax_utils::build_sigmoid_tables(props->zero_points, props->scales);
  props->softmax_tables
      = ax_utils::build_exponential_tables(props->zero_points, props->scales);
  props->dequantize_tables
      = ax_utils::build_dequantization_tables(props->zero_points, props->scales);

  // Build sin/cos and angle tables based on raw_radians mode
  if (props->raw_radians) {
    // YOLO26-OBB: angle is in radians, compute sin/cos directly on dequantized values
    props->sin_cos_tables
        = ax_utils::build_trigonometric_tables(props->zero_points, props->scales);
    // Angle table: just dequantize
    props->angle_tables
        = ax_utils::build_dequantization_tables(props->zero_points, props->scales);
  } else {
    // YOLO8-OBB: apply sigmoid transformation with offset and multiplier
    props->sin_cos_tables = ax_utils::build_sigmoid_trigonometric_tables(
        props->zero_points, props->scales, -0.25F, M_PI);
    // Angle table: (sigmoid(x) - 0.25) * π
    props->angle_tables = ax_utils::build_general_dequantization_tables(
        props->zero_points, props->scales,
        [](float x) { return (ax_utils::to_sigmoid(x) - 0.25F) * M_PI; });
  }

  auto filter = Ax::get_property(
      input, "label_filter", "detection_static_properties", std::vector<int>{});
  props->filter = ax_utils::build_filter(filter, props->num_classes);
  props->letterbox = Ax::get_property(
      input, "letterbox", "detection_static_properties", props->letterbox);
  props->padding = Ax::get_property(
      input, "padding", "detection_static_properties", props->padding);

  int dfl_size = Ax::get_property(input, "dfl_size", "detection_static_properties", 16);
  if (dfl_size <= 0 || dfl_size > yolov8_decode::max_dfl_bins) {
    throw std::runtime_error("detection_static_properties : dfl_size must be between 1 and "
                             + std::to_string(yolov8_decode::max_dfl_bins)
                             + ", got " + std::to_string(dfl_size));
  }
  props->weights.resize(dfl_size);
  std::iota(props->weights.begin(), props->weights.end(), 0);

  return props;
}

extern "C" void
set_dynamic_properties(const std::unordered_map<std::string, std::string> &input,
    yolov8_decode::properties *prop, Ax::Logger &logger)
{
  prop->confidence = Ax::get_property(input, "confidence_threshold",
      "detection_dynamic_properties", prop->confidence);
  logger(AX_DEBUG) << "prop->confidence_threshold is " << prop->confidence << std::endl;
}
