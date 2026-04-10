// Copyright Axelera AI, 2024
#include <AxOpUtils.hpp>
#include <opencv2/core/ocl.hpp>
#include <unordered_map>
#include <unordered_set>
#include "AxDataInterface.h"
#include "AxFaceAlignCommon.h"
#include "AxLog.hpp"
#include "AxMeta.hpp"
#include "AxMetaKptsDetection.hpp"
#include "AxMetaMargin.hpp"
#include "AxMetaTracker.hpp"
#include "AxUtils.hpp"

struct facealign_properties {
  std::string master_meta{};
  std::string association_meta{};
  int width = 0;
  int height = 0;
  float padding = 0.0;
  std::vector<float> XXn{};
  std::vector<float> YYn{};
  bool use_self_normalizing = false;
  bool save_aligned_images = false;
};

extern "C" const std::unordered_set<std::string> &
allowed_properties()
{
  static const std::unordered_set<std::string> allowed_properties{
    "master_meta",
    "association_meta",
    "width",
    "height",
    "padding",
    "template_keypoints_x",
    "template_keypoints_y",
    "use_self_normalizing",
    "save_aligned_images",
  };
  return allowed_properties;
}

extern "C" std::shared_ptr<void>
init_and_set_static_properties(
    const std::unordered_map<std::string, std::string> &input, Ax::Logger &logger)
{
  std::shared_ptr<facealign_properties> prop = std::make_shared<facealign_properties>();
  prop->master_meta = Ax::get_property(
      input, "master_meta", "facealign_static_properties", prop->master_meta);
  if (prop->master_meta.empty()) {
    throw std::runtime_error("facealign: master meta key not provided");
  }
  prop->association_meta = Ax::get_property(input, "association_meta",
      "facealign_static_properties", prop->association_meta);
  prop->width = Ax::get_property(input, "width", "facealign_static_properties", prop->width);
  prop->height
      = Ax::get_property(input, "height", "facealign_static_properties", prop->height);
  prop->padding = Ax::get_property(
      input, "padding", "facealign_static_properties", prop->padding);
  prop->XXn = Ax::get_property(
      input, "template_keypoints_x", "facealign_static_properties", prop->XXn);
  prop->YYn = Ax::get_property(
      input, "template_keypoints_y", "facealign_static_properties", prop->YYn);
  if (prop->XXn.size() != prop->YYn.size()) {
    throw std::runtime_error(
        "facealign: template_keypoints_x and template_keypoints_y must have the same number of elements");
  }
  prop->use_self_normalizing = Ax::get_property(input, "use_self_normalizing",
      "facealign_static_properties", prop->use_self_normalizing);
  prop->save_aligned_images = Ax::get_property(input, "save_aligned_images",
      "facealign_static_properties", prop->save_aligned_images);
  return prop;
}

extern "C" AxDataInterface
set_output_interface(const AxDataInterface &interface,
    const facealign_properties *prop, Ax::Logger &logger)
{
  if (!std::holds_alternative<AxVideoInterface>(interface)) {
    throw std::runtime_error("facealign works on video input only");
  }
  AxDataInterface output = interface;
  if (prop->width == 0 || prop->height == 0) {
    return output;
  }
  auto &info = std::get<AxVideoInterface>(output).info;
  info.width = prop->width;
  info.height = prop->height;
  return output;
}

// Helper function to perform fallback alignment (simple resize)
static void
perform_fallback_alignment(const AxDataInterface &input,
    const AxDataInterface &output, Ax::Logger &logger)
{
  auto &input_video = std::get<AxVideoInterface>(input);
  auto &output_video = std::get<AxVideoInterface>(output);
  cv::Mat input_mat(cv::Size(input_video.info.width, input_video.info.height),
      Ax::opencv_type_u8(input_video.info.format), input_video.data,
      input_video.info.stride);
  cv::Mat output_mat(cv::Size(output_video.info.width, output_video.info.height),
      Ax::opencv_type_u8(output_video.info.format), output_video.data,
      output_video.info.stride);
  cv::resize(input_mat, output_mat, output_mat.size());
}

AxMetaKpts *
extract_keypoints_from_meta(AxMetaBase *meta, int &kpts_per_box)
{

  auto *kpts_meta = dynamic_cast<AxMetaKptsDetection *>(meta);
  if (kpts_meta) {
    kpts_per_box = kpts_meta->get_kpts_shape()[0];
    return kpts_meta;
  }

  // Fall back to AxMetaKpts
  auto *kpts_meta_base = dynamic_cast<AxMetaKpts *>(meta);
  if (kpts_meta_base) {
    kpts_per_box = kpts_meta_base->num_elements();
    return kpts_meta_base;
  }

  return nullptr;
}

AxMetaKpts *
extract_keypoints_from_tracker(AxMetaTracker *tracker_meta, int track_id,
    int &kpts_per_box, Ax::Logger &logger)
{
  if (tracker_meta->track_id_to_tracking_descriptor.empty()) {
    throw std::runtime_error("facealign: tracker meta has no tracking descriptors");
  }

  auto it = tracker_meta->track_id_to_tracking_descriptor.find(track_id);
  if (it == tracker_meta->track_id_to_tracking_descriptor.end()) {
    throw std::runtime_error("facealign: track_id " + std::to_string(track_id)
                             + " not found in tracker meta");
  }

  auto &descriptor = it->second;

  const TrackingElement *element = descriptor.collection->get_frame(descriptor.frame_id);
  if (!element) {
    throw std::runtime_error("facealign: no frame data for current frame in tracker");
  }

  for (const auto &[key, meta_ptr] : element->frame_data_map) {
    auto *kpts = extract_keypoints_from_meta(meta_ptr.get(), kpts_per_box);
    if (kpts)
      return kpts;
  }

  throw std::runtime_error("facealign: could not find keypoints in tracker meta");
}

AxMetaKpts *
extract_keypoints_from_submeta(
    AxMetaBbox *bbox_meta, int box_id, int &kpts_per_box, Ax::Logger &logger)
{
  // Search through all submeta to find one that contains keypoints for this box_id
  // Submeta are additional metadata attached to each box (like classifications, embeddings, keypoints)
  // Each submeta has its own separate data per box

  auto submeta_keys = bbox_meta->submeta_names();
  for (const char *key : submeta_keys) {
    try {
      // Try to get submeta for this specific box_id
      auto *submeta = bbox_meta->get_submeta<AxMetaBase>(
          key, box_id, bbox_meta->get_number_of_subframes());
      if (submeta) {
        // Check if this submeta implements the keypoints interface
        auto *kpts = extract_keypoints_from_meta(submeta, kpts_per_box);
        if (kpts) {
          return kpts; // Found keypoints in this submeta
        }
      }
    } catch (const std::exception &) {
      // This submeta key doesn't exist for this box_id or isn't the right type
      // Continue searching other submeta keys
      continue;
    }
  }

  // No submeta with keypoints found
  return nullptr;
}

float
get_margin(std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &meta_map)
{
  auto margin_meta = meta_map.find("axelera-margin");
  if (margin_meta != meta_map.end()) {
    if (auto *p = dynamic_cast<AxMetaMargin *>(margin_meta->second.get())) {
      return p->margin;
    }
  }
  return 0.0F;
}

box_xyxy
add_margin(box_xyxy box, std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &map)
{
  auto margin = get_margin(map);
  auto [x1, y1, x2, y2] = box;
  auto width = 1 + x2 - x1;
  auto height = 1 + y2 - y1;
  int x_margin = std::round(width * margin);
  int y_margin = std::round(height * margin);
  auto x_start = x1 - x_margin;
  auto y_start = y1 - y_margin;
  auto x_end = x2 + x_margin;
  auto y_end = y2 + y_margin;
  return { x_start, y_start, x_end, y_end };
}

extern "C" void
transform(const AxDataInterface &input, const AxDataInterface &output,
    const facealign_properties *prop, unsigned int subframe_index, unsigned int number_of_subframes,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &map, Ax::Logger &logger)
{
  // ============================================================================
  // STEP 1: Get master_meta (source of keypoints - can be tracker or bbox)
  // ============================================================================
  AxMetaBase *master_meta = map.at(prop->master_meta).get();

  // ============================================================================
  // STEP 2: Determine which meta provides boxes and calculate box_id
  // ============================================================================
  // We need two things:
  // 1. box_meta: Meta that provides the bounding box for this subframe
  // 2. box_id: Index to use when looking up keypoints in master_meta
  //
  // Two scenarios:
  // A) association_meta is set: Use it for boxes, map subframe_index -> box_id
  // B) No association_meta: master_meta provides boxes, box_id = subframe_index

  AxMetaBbox *box_meta = nullptr;
  int box_id = subframe_index;

  if (!prop->association_meta.empty()) {
    // Scenario A: association_meta is a filtered/transformed view of master_meta
    // Example: tracker outputs all tracks, association_meta filters to person class
    auto *association_meta = map.at(prop->association_meta).get();
    box_meta = dynamic_cast<AxMetaBbox *>(association_meta);
    if (!box_meta) {
      throw std::runtime_error("facealign: association_meta must be AxMetaBbox");
    }

    // Map from association_meta's subframe space to master_meta's box space
    // Example: association_meta[0] might map to master_meta's box_id=5
    box_id = box_meta->get_id(subframe_index);
  } else {
    // Scenario B: No association, master_meta provides boxes directly
    box_meta = dynamic_cast<AxMetaBbox *>(master_meta);
    if (!box_meta) {
      throw std::runtime_error(
          "facealign: master_meta must be AxMetaBbox when no association_meta");
    }
    // box_id already set to subframe_index above
  }

  // Validate number of subframes matches what we expect
  if (box_meta->get_number_of_subframes() != number_of_subframes) {
    throw std::runtime_error("facealign: invalid number of subframes");
  }

  // Get the bounding box for this subframe and check if it's valid
  box_xyxy box = box_meta->get_box_xyxy(subframe_index);
  box = add_margin(box, map);
  if (box.x2 <= box.x1 || box.y2 <= box.y1) {
    perform_fallback_alignment(input, output, logger);
    return;
  }

  // ============================================================================
  // STEP 3: Extract keypoints from master_meta
  // ============================================================================
  // Now we extract keypoints using box_id to index into master_meta.
  // The extraction method depends on whether master_meta is a tracker or bbox.
  //
  // Key insight about kpts_start:
  // - If keypoints come from tracker or submeta: Each box has its OWN keypoint set
  //   → kpts_start = 0 (keypoints[0..N] belong to this box)
  // - If keypoints come from bbox direct interface: All boxes share ONE keypoint array
  //   → kpts_start = box_id (keypoints[box_id*N..(box_id+1)*N] belong to this box)

  AxMetaKpts *kpts_meta_base = nullptr;
  int kpts_start = 0;
  int kpts_per_box = 0;

  if (auto *tracker = dynamic_cast<AxMetaTracker *>(master_meta)) {
    // -------------------------------------------------------------------------
    // TRACKER CASE: Keypoints stored per-track in tracker's frame_data_map
    // -------------------------------------------------------------------------
    // The tracker maintains a map: track_id -> TrackingDescriptor
    // Each TrackingDescriptor has a frame_data_map containing metadata (including keypoints)
    // We use box_id as the track_id to look up the correct track's data

    kpts_meta_base = extract_keypoints_from_tracker(tracker, box_id, kpts_per_box, logger);
    kpts_start = 0; // Tracker returns keypoints for ONE specific track

  } else if (auto *bbox_master = dynamic_cast<AxMetaBbox *>(master_meta)) {
    // -------------------------------------------------------------------------
    // BBOX CASE: Keypoints either on bbox directly or in its submeta
    // -------------------------------------------------------------------------

    // Try 1: Check if bbox_master directly implements keypoints interface
    // This means all keypoints for all boxes are in one flat array
    kpts_meta_base = extract_keypoints_from_meta(bbox_master, kpts_per_box);
    bool from_submeta = false;

    if (!kpts_meta_base) {
      // Try 2: Search through all submeta to find keypoints
      // If keypoints are in submeta, each box has its own separate keypoint metadata
      kpts_meta_base
          = extract_keypoints_from_submeta(bbox_master, box_id, kpts_per_box, logger);
      if (kpts_meta_base) {
        from_submeta = true;
      }
    }

    if (!kpts_meta_base) {
      throw std::runtime_error("facealign: no keypoints found in bbox meta or its submeta");
    }

    // Set kpts_start based on keypoint storage layout:
    // - from_submeta=true: Keypoints are per-box, so start at index 0
    // - from_submeta=false: All keypoints in one array, so start at box_id * kpts_per_box
    kpts_start = from_submeta ? 0 : box_id;

  } else {
    throw std::runtime_error("facealign: master_meta must be AxMetaTracker or AxMetaBbox");
  }

  // ============================================================================
  // STEP 4: Extract individual keypoint coordinates for this box
  // ============================================================================
  // We now have:
  // - kpts_meta_base: Pointer to keypoint metadata
  // - kpts_start: Starting index for this box's keypoints
  // - kpts_per_box: Number of keypoints per box
  //
  // Extract keypoints and convert to box-relative coordinates

  std::vector<float> X, Y;
  X.reserve(kpts_per_box);
  Y.reserve(kpts_per_box);

  for (int i = 0; i < kpts_per_box; ++i) {
    // Calculate the absolute index in the keypoint array
    // Example: If kpts_start=0 and i=2, we get keypoint[2] (per-box storage)
    // Example: If kpts_start=5 and i=2, we get keypoint[5*N+2] (flat array storage)
    int kpt_index = kpts_start * kpts_per_box + i;
    if (kpt_index >= static_cast<int>(kpts_meta_base->num_elements())) {
      perform_fallback_alignment(input, output, logger);
      return;
    }

    // Get keypoint in absolute image coordinates
    KptXyv kpt = kpts_meta_base->get_kpt_xy(kpt_index);

    // Convert to box-relative coordinates (relative to box top-left corner)
    // This is needed because face alignment works within the box coordinate system
    float rel_x = kpt.x - box.x1;
    float rel_y = kpt.y - box.y1;

    // Validate keypoint coordinates are finite numbers
    if (!std::isfinite(rel_x) || !std::isfinite(rel_y)) {
      perform_fallback_alignment(input, output, logger);
      return;
    }

    X.push_back(rel_x);
    Y.push_back(rel_y);
  }

  auto &input_video = std::get<AxVideoInterface>(input);
  auto &output_video = std::get<AxVideoInterface>(output);
  cv::Mat input_mat(cv::Size(input_video.info.width, input_video.info.height),
      Ax::opencv_type_u8(input_video.info.format), input_video.data,
      input_video.info.stride);
  cv::Mat output_mat(cv::Size(output_video.info.width, output_video.info.height),
      Ax::opencv_type_u8(output_video.info.format), output_video.data,
      output_video.info.stride);

  // Compute transformation matrix based on alignment mode
  cv::Mat M_inv; // Inverse matrix (destination → source)

  if (prop->use_self_normalizing) {
    // Self-normalizing alignment (eye-based for 5-point landmarks)
    if (kpts_per_box == 5) {
      M_inv = face_align::compute_self_normalizing_matrix(
          X, Y, box, output_video.info.width, output_video.info.height);
    } else {
      M_inv = face_align::compute_fallback_matrix(
          box, output_video.info.width, output_video.info.height);
    }
  } else {
    // Template-based Procrustes alignment
    std::vector<float> XX = prop->XXn;
    std::vector<float> YY = prop->YYn;
    int kpts_offset = 0;

    // Select appropriate template based on number of keypoints
    if (XX.empty() || YY.empty()) {
      if (kpts_per_box >= 68) {
        XX.assign(face_align::template_51pt_x.begin(),
            face_align::template_51pt_x.end());
        YY.assign(face_align::template_51pt_y.begin(),
            face_align::template_51pt_y.end());
        kpts_offset = 17;
      } else if (kpts_per_box >= 51) {
        XX.assign(face_align::template_51pt_x.begin(),
            face_align::template_51pt_x.end());
        YY.assign(face_align::template_51pt_y.begin(),
            face_align::template_51pt_y.end());
        kpts_offset = 0;
      } else if (kpts_per_box >= 5) {
        XX.assign(face_align::template_5pt_x.begin(), face_align::template_5pt_x.end());
        YY.assign(face_align::template_5pt_y.begin(), face_align::template_5pt_y.end());
        kpts_offset = 0;
      }
    }

    // Validate template size matches keypoint count and compute matrix
    if (!XX.empty() && !YY.empty()
        && XX.size() == static_cast<size_t>(kpts_per_box - kpts_offset)) {
      std::vector<float> X_template(X.begin() + kpts_offset, X.end());
      std::vector<float> Y_template(Y.begin() + kpts_offset, Y.end());

      auto margin = get_margin(map);
      M_inv = face_align::compute_template_based_matrix(X_template, Y_template,
          XX, YY, box, margin, output_video.info.width, output_video.info.height);
    } else {
      M_inv = face_align::compute_fallback_matrix(
          box, output_video.info.width, output_video.info.height);
    }
  }

  // cv::warpAffine expects forward matrix (source → destination)
  // The utility functions return inverse matrix, so invert it back
  cv::Mat M;
  cv::invertAffineTransform(M_inv, M);

  if (!face_align::is_valid_matrix(M)) {
    perform_fallback_alignment(input, output, logger);
    return;
  }

  // Use (0,0,0,255) for border pixels
  cv::warpAffine(input_mat, output_mat, M, output_mat.size(), cv::INTER_LINEAR,
      cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0, 255));

  if (prop->save_aligned_images) {
    static int frame_counter = 0;
    try {
      [[maybe_unused]] int ret = system("mkdir -p face_align_debug");

      cv::Mat new_output_mat(cv::Size(output_video.info.width, output_video.info.height),
          Ax::opencv_type_u8(output_video.info.format));

      cv::cvtColor(output_mat, new_output_mat, cv::COLOR_BGRA2RGBA);

      // Crop to central 80% (trim 10% from each side)
      int crop_x = new_output_mat.cols / 10;
      int crop_y = new_output_mat.rows / 10;
      cv::Rect central_roi(crop_x, crop_y, new_output_mat.cols - 2 * crop_x,
          new_output_mat.rows - 2 * crop_y);
      cv::imwrite("face_align_debug/aligned_" + std::to_string(frame_counter) + ".png",
          new_output_mat(central_roi));

      // Create a copy for keypoint visualization (don't modify original input)
      cv::Mat debug_img = input_mat.clone();

      for (size_t i = 0; i < X.size(); ++i) {
        cv::Point2f pt(X[i], Y[i]);
        if (pt.x >= 0 && pt.x < debug_img.cols && pt.y >= 0 && pt.y < debug_img.rows) {
          cv::circle(debug_img, pt, 8, cv::Scalar(0, 255, 0), -1);
        }
      }
      cv::imwrite("face_align_debug/original_" + std::to_string(frame_counter) + ".png",
          debug_img);
      frame_counter++;
    } catch (...) {
      logger(AX_WARN) << "facealign: failed to save aligned images, check permissions or disk space"
                      << std::endl;
    }
  }
}
