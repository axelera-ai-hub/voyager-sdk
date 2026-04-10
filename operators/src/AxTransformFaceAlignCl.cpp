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
#include "AxOpUtils.hpp"
#include "AxOpenCl.hpp"
#include "AxUtils.hpp"

class CLWarpAffine;
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
  bool downstream_supports_opencl{ false };
  AxVideoFormat format{ AxVideoFormat::UNDEFINED };
  std::unique_ptr<CLWarpAffine> warpaffine;
};

/**
 * OpenCL kernel for affine transformation.
 *
 * The affine matrix is a 2x3 matrix in row-major order:
 * [ m00 m01 m02 ]
 * [ m10 m11 m12 ]
 *
 * For optimization and alignment, we convert this to a 3x4 matrix:
 * [ m00 m01 m02 0 ]
 * [ m10 m11 m12 0 ]
 * [ 0   0   1   0 ]
 *
 * The matrix maps destination → source (inverse/backward mapping).
 */

const char *warpaffine_kernel = R"##(

uchar4 color_convert(uchar4 pixel, float16 matrix) {
    float4 in_pixel = convert_float4(pixel);
    float4 color = mad(in_pixel.x, matrix.s0123, mad(in_pixel.y, matrix.s4567, mad(in_pixel.z, matrix.s89ab, matrix.scdef)));
    color.w = in_pixel.w;
    return convert_uchar4_sat(color);
}

__kernel void warpaffine(__global const uchar *in, __global uchar *out, int4 image_dims,
                        int4 strides, int4 offsets, int crop_x, int crop_y,
                        float16 affine_matrix, float16 color_matrix, uchar fill) {

    const int col = get_global_id(0);
    const int row = get_global_id(1);

    // Check bounds for output image
    if (row >= image_dims.w || col >= image_dims.z) {
      return;
    }

    // Apply affine transformation to get source coordinates
    const float4 coord_f = (float4)(col + 0.5F, row + 0.5F, 1.0F, 0.0F);
    float2 corrected = (float2)(dot(coord_f, affine_matrix.s0123), dot(coord_f, affine_matrix.s4567));

    // Define image description for sampling
    image_description img = {
        image_dims,
        strides,
        offsets,
        (int4)(0, 0, image_dims.z, image_dims.w),  // Output region
        (int4)(crop_x, crop_y, 0, 0)                // Crop offset
    };

)##";

using ax_utils::buffer_details;
using ax_utils::CLProgram;
using ax_utils::opencl_details;

// Helper to add alpha channel to format if needed
static AxVideoFormat
add_alpha(AxVideoFormat format)
{
  if (format == AxVideoFormat::BGR) {
    return AxVideoFormat::BGRA;
  } else if (format == AxVideoFormat::RGB) {
    return AxVideoFormat::RGBA;
  }
  return format;
}

class CLWarpAffine
{
  using buffer = CLProgram::ax_buffer;
  using kernel = CLProgram::ax_kernel;

  public:
  CLWarpAffine(opencl_details *ocl, Ax::Logger &logger)
      : program("", ocl, logger)
  {
  }

  ax_utils::CLProgram::ax_kernel build_kernel(ax_utils::CLProgram &program,
      AxVideoFormat in_format, AxVideoFormat out_format)
  {
    std::string kernel_code = warpaffine_kernel;

    auto [_, in_type, sampler_code] = ax_utils::get_input_details(in_format);
    auto [__, out_type, output_code] = ax_utils::get_output_details(in_format, out_format);

    auto n = snprintf(nullptr, 0, kernel_code.c_str(), in_type.c_str(), out_type.c_str());
    std::vector<char> buffer(n + 1);
    snprintf(buffer.data(), buffer.size(), kernel_code.c_str(), in_type.c_str(),
        out_type.c_str());
    auto final_kernel = std::string(buffer.data());
    final_kernel += ax_utils::get_rotation(0); // No flip for warpaffine
    final_kernel += sampler_code;
    final_kernel += output_code;
    final_kernel = ax_utils::get_kernel_utils(0) + final_kernel;

    return program.build_kernel_from_source(final_kernel, "warpaffine");
  }

  cl_kernel get_converter(ax_utils::CLProgram &program, AxVideoFormat in_format,
      AxVideoFormat out_format)
  {
    auto hash = (static_cast<int>(in_format) << 16)
                + (static_cast<int>(out_format) << 8) + 0;
    auto it = std::find_if(std::begin(all_kernels), std::end(all_kernels),
        [hash](auto &x) { return x.hash == hash; });
    if (it != all_kernels.end()) {
      return *it->cl_prog;
    }
    auto k = build_kernel(program, in_format, out_format);
    return *all_kernels.emplace_back(hash, std::move(k)).cl_prog;
  }


  int run(const buffer_details &in, const buffer_details &out,
      const std::vector<cl_float> &matrix_3x4, bool downstream_supports_opencl)
  {
    auto converter = get_converter(program, in.format, out.format);

    bool start_flush = !downstream_supports_opencl;
    auto outbuf = program.create_buffer(out, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR);

    std::array<cl_int, 4> image_dims = { in.width, in.height, out.width, out.height };
    auto strides = ax_utils::build_strides(in, out);
    auto offsets = ax_utils::build_offsets(in, out);
    auto inbuf_y = program.create_buffer(in, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR);
    auto color_matrix = ax_utils::get_color_conversion_matrix(in.format, out.format);
    cl_uchar fill = 0; // Border fill value (0, 0, 0, 255) -> 0 for most channels

    program.set_kernel_args(converter, 0, *inbuf_y, *outbuf, image_dims,
        strides, offsets, in.crop_x, in.crop_y, matrix_3x4, color_matrix, fill);
    return run_kernel(program, converter, in, out, inbuf_y, outbuf, start_flush);
  }

  bool can_use_dmabuf() const
  {
    return program.can_use_dmabuf();
  }

  private:
  CLProgram program;
  int error{};
  struct kernels {
    int hash;
    kernel cl_prog{ nullptr };
  };
  std::vector<kernels> all_kernels;
};

// Helper function to convert 2x3 affine matrix to 3x4 format for OpenCL
static std::vector<cl_float>
convert_affine_matrix_to_3x4(const cv::Mat &M)
{
  if (M.rows != 2 || M.cols != 3) {
    throw std::runtime_error("Affine matrix must be 2x3");
  }

  if (M.type() != CV_32F) {
    throw std::runtime_error("Affine matrix must be CV_32F (float type)");
  }

  std::vector<cl_float> matrix_3x4(12, 0.0f);

  // Convert to 3x4 row-major matrix for float4 alignment
  // Input:  [m00 m01 m02]
  //         [m10 m11 m12]
  // Output: [m00 m01 m02 0]
  //         [m10 m11 m12 0]
  //         [0   0   1   0]

  // Extract float values (all utility functions return CV_32F)
  matrix_3x4[0] = M.at<float>(0, 0);
  matrix_3x4[1] = M.at<float>(0, 1);
  matrix_3x4[2] = M.at<float>(0, 2);
  matrix_3x4[3] = 0.0f;
  matrix_3x4[4] = M.at<float>(1, 0);
  matrix_3x4[5] = M.at<float>(1, 1);
  matrix_3x4[6] = M.at<float>(1, 2);
  matrix_3x4[7] = 0.0f;

  // Third row (identity for homogeneous coordinates)
  matrix_3x4[8] = 0.0f;
  matrix_3x4[9] = 0.0f;
  matrix_3x4[10] = 1.0f;
  matrix_3x4[11] = 0.0f;

  return matrix_3x4;
}

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
    "format",
  };
  return allowed_properties;
}

extern "C" std::shared_ptr<void>
init_and_set_static_properties_with_context(
    const std::unordered_map<std::string, std::string> &input, void *context, Ax::Logger &logger)
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

  // Parse optional format parameter
  auto format = Ax::get_property(
      input, "format", "facealign_static_properties", std::string{});
  if (format == "rgba") {
    prop->format = AxVideoFormat::RGBA;
  } else if (format == "bgra") {
    prop->format = AxVideoFormat::BGRA;
  } else if (format == "rgb") {
    prop->format = AxVideoFormat::RGB;
  } else if (format == "bgr") {
    prop->format = AxVideoFormat::BGR;
  } else if (format == "") {
    prop->format = AxVideoFormat::UNDEFINED;
  } else {
    throw std::runtime_error("FaceAlign with color convert only supports RGBA, BGRA, RGB, or BGR, given: "
                             + format);
  }

  // Initialize OpenCL warpaffine
  prop->warpaffine
      = std::make_unique<CLWarpAffine>(static_cast<opencl_details *>(context), logger);

  return prop;
}

extern "C" void
set_dynamic_properties(const std::unordered_map<std::string, std::string> &input,
    facealign_properties *prop, Ax::Logger & /*logger*/)
{
  prop->downstream_supports_opencl = Ax::get_property(input, "downstream_supports_opencl",
      "facealign_dynamic_properties", prop->downstream_supports_opencl);
}

extern "C" bool
query_supports(Ax::PluginFeature feature,
    const facealign_properties *facealign_properties, Ax::Logger &logger)
{
  if (feature == Ax::PluginFeature::opencl_buffers) {
    return true;
  }
  if (feature == Ax::PluginFeature::crop_meta) {
    return true;
  }
  if (feature == Ax::PluginFeature::dmabuf_buffers) {
    return facealign_properties->warpaffine->can_use_dmabuf();
  }
  return Ax::PluginFeatureDefaults(feature);
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

  // Set output format: use specified format or add alpha channel to input format
  auto in_info = std::get<AxVideoInterface>(interface);
  info.format = prop->format == AxVideoFormat::UNDEFINED ?
                    add_alpha(in_info.info.format) :
                    prop->format;
  ax_utils::remove_cropinfo(output);
  return output;
}

// Helper function to perform fallback alignment using OpenCL warpAffine
static void
perform_fallback_alignment(const AxDataInterface &input, const AxDataInterface &output,
    const box_xyxy &box, const facealign_properties *prop, Ax::Logger &logger)
{
  auto &output_video = std::get<AxVideoInterface>(output);

  // Compute fallback matrix (destination → source) using shared utility
  cv::Mat M = face_align::compute_fallback_matrix(
      box, output_video.info.width, output_video.info.height);

  // Convert to 3x4 format for OpenCL
  auto matrix_3x4 = convert_affine_matrix_to_3x4(M);

  // Extract buffer details using proper utility function
  auto input_details = ax_utils::extract_buffer_details(input);
  auto output_details = ax_utils::extract_buffer_details(output);

  // Run OpenCL kernel
  int result = prop->warpaffine->run(input_details[0], output_details[0],
      matrix_3x4, prop->downstream_supports_opencl);
  if (result != 0) {
    logger(AX_ERROR) << "OpenCL warpaffine failed with error: " << result << std::endl;
  }
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
  // Note: Bounding boxes use closed ranges [x1, x2], so x2 == x1 means width of 1
  box_xyxy box = box_meta->get_box_xyxy(subframe_index);
  box = add_margin(box, map);
  if (box.x2 < box.x1 || box.y2 < box.y1) {
    perform_fallback_alignment(input, output, box, prop, logger);
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
      perform_fallback_alignment(input, output, box, prop, logger);
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
      perform_fallback_alignment(input, output, box, prop, logger);
      return;
    }

    X.push_back(rel_x);
    Y.push_back(rel_y);
  }

  auto &output_video = std::get<AxVideoInterface>(output);

  // Compute transformation matrix based on alignment mode
  cv::Mat M;

  if (prop->use_self_normalizing) {
    // Self-normalizing alignment (eye-based for 5-point landmarks)
    if (kpts_per_box == 5) {
      M = face_align::compute_self_normalizing_matrix(
          X, Y, box, output_video.info.width, output_video.info.height);
    } else {
      M = face_align::compute_fallback_matrix(
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
      M = face_align::compute_template_based_matrix(X_template, Y_template, XX,
          YY, box, margin, output_video.info.width, output_video.info.height);
    } else {
      M = face_align::compute_fallback_matrix(
          box, output_video.info.width, output_video.info.height);
    }
  }

  auto matrix_3x4 = convert_affine_matrix_to_3x4(M);
  auto input_details = ax_utils::extract_buffer_details(input);
  auto output_details = ax_utils::extract_buffer_details(output);

  prop->warpaffine->run(input_details[0], output_details[0], matrix_3x4,
      prop->downstream_supports_opencl);
}
