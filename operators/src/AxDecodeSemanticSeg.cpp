// Copyright Axelera AI, 2024
// Semantic segmentation decoder.
//
// Default path (`interpolate_before_argmax=false`): chip-space argmax over
// the letterbox-cropped tile. `calculate_letterbox_crop` computes the
// chip-space crop rectangle that excludes the letterbox padding region;
// argmax runs on int8 inside that rectangle. Output is sized to chip-cropped
// dims; downstream consumers upsample to source resolution if they need it.
//
// Opt-in path (`interpolate_before_argmax=true`): replicates the torch
// postprocess `_rescale()` math entirely inside the decoder. Single-pass
// fused implementation:
//   1. Read per-frame source image dims from the AxVideoInterface.
//   2. `compute_letterbox_chip_rect` derives the chip-space rectangle that
//      maps to the active (non-padded) image region.
//   3. `upsample_argmax_int8` (multi-class) or `upsample_threshold_int8`
//      (binary) walks each output (src_h, src_w) pixel and computes the
//      four-corner bilinear weighted sum directly on int8 logits, taking
//      the argmax / threshold on the fly. The only materialised buffer is
//      the output class map; no intermediate float dequant or interpolated
//      buffer is allocated.
//   4. Bilinear on int8 is exactly order-preserving with respect to
//      "dequantize -> bilinear -> argmax" because all C output channels
//      share the same (scale, zero_point) for the semantic_seg head.
// Recovers thin classes on strided heads (e.g. yolo26-sem chip 128x128 from
// model 1024x1024) at the cost of one extra per-frame upsample.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <span>
#include <unordered_set>
#include <variant>
#include <vector>
#include "AxLog.hpp"
#include "AxMetaSemanticSegmentation.hpp"
#include "AxOpUtils.hpp"

namespace semantic_seg
{

using lookups = ax_utils::lookups;

// `threshold` is interpreted in probability space when `sigmoid` is true
// (and converted to a logit at init time so the per-pixel loop can compare
// raw logits directly), and in raw-logit space when `sigmoid` is false.
// `threshold` is only used on the single-class path; the multi-class path
// always emits the argmax class.
struct properties {
  std::string meta_name{};
  bool class_map_out{ true };
  std::string decoder_name;
  float threshold{ 0.00001f };
  bool sigmoid{ false };
  // threshold_int8: threshold mapped to quantized space for direct int8 comparison.
  // Precomputed at init when scales are available; only used on the binary int8 path.
  int threshold_int8{ 0 };
  // dequantize_tables: only built (and used) when class_map_out=false requires float logits.
  std::vector<lookups> dequantize_tables{};
  std::vector<int> padding{}; // [N_lo, N_hi, H_lo, H_hi, W_lo, W_hi, C_lo, C_hi]
  // Letterbox parameters for cropping
  int model_width{ 0 };
  int model_height{ 0 };
  bool scale_up{ false };
  bool letterbox{ false };
  // Interpolation parameter - interpolate logits to target resolution before argmax
  bool interpolate_before_argmax{ false };
};

float
dequantize(int8_t value, const float *the_table)
{
  int index = value + 128;
  return the_table[index];
}

// Calculate letterbox crop region in output tensor space
// orig_h/orig_w: original image dimensions
// model_h/model_w: model input dimensions (letterbox target size)
// output_h/output_w: output tensor dimensions (after model downsampling)
// Returns [h_start, h_end, w_start, w_end] in output tensor space
std::array<int, 4>
calculate_letterbox_crop(int orig_h, int orig_w, int model_h, int model_w,
    int output_h, int output_w, bool scale_up, bool letterbox)
{
  // If no letterbox, use full output
  if (!letterbox || model_h == 0 || model_w == 0) {
    return { 0, output_h, 0, output_w };
  }

  // Calculate the letterbox ratio (same as preprocessing)
  float ratio = std::min(static_cast<float>(model_h) / orig_h,
      static_cast<float>(model_w) / orig_w);
  if (!scale_up) {
    ratio = std::min(ratio, 1.0f);
  }

  // Calculate scaled dimensions in model input space
  int scaled_h = static_cast<int>(std::round(orig_h * ratio));
  int scaled_w = static_cast<int>(std::round(orig_w * ratio));

  // Calculate padding in model input space (centered)
  int pad_h_model = (model_h - scaled_h) / 2;
  int pad_w_model = (model_w - scaled_w) / 2;

  // Scale padding from model input space to output tensor space
  float h_scale = static_cast<float>(output_h) / model_h;
  float w_scale = static_cast<float>(output_w) / model_w;

  int crop_h_start = static_cast<int>(std::round(pad_h_model * h_scale));
  int crop_h_end = output_h - static_cast<int>(std::round(pad_h_model * h_scale));
  int crop_w_start = static_cast<int>(std::round(pad_w_model * w_scale));
  int crop_w_end = output_w - static_cast<int>(std::round(pad_w_model * w_scale));

  return { crop_h_start, crop_h_end, crop_w_start, crop_w_end };
}

struct letterbox_chip_rect {
  int top;
  int left;
  int height;
  int width;
};

// Compute the chip-space rectangle that holds the actual (non-padded) content
// of one letterboxed frame. Mirrors the torch `Letterbox._letterbox()` and
// `SemanticSegmentation._rescale()` math:
//   - r = min(model_h / src_h, model_w / src_w); if !scaleup: r = min(r, 1).
//   - new_unpad = (round(src_h * r), round(src_w * r)).
//   - dh, dw = model - new_unpad; pad split as round(d/2 - 0.1) on the top
//     / left side (Ultralytics rounding). The two halves can differ by one
//     pixel when d is odd.
//   - Image-space coords are converted to chip-space by the chip stride
//     (model_dim / chip_dim), rounded to nearest.
// Result is clamped to chip bounds so a degenerate input can never read out
// of range.
letterbox_chip_rect
compute_letterbox_chip_rect(int src_h, int src_w, int model_h, int model_w,
    int chip_h, int chip_w, bool scaleup)
{
  const double r_raw = std::min(static_cast<double>(model_h) / src_h,
      static_cast<double>(model_w) / src_w);
  const double r = scaleup ? r_raw : std::min(r_raw, 1.0);
  const int new_unpad_h = static_cast<int>(std::round(src_h * r));
  const int new_unpad_w = static_cast<int>(std::round(src_w * r));
  const int dh = model_h - new_unpad_h;
  const int dw = model_w - new_unpad_w;
  const int top_img = static_cast<int>(std::round(dh / 2.0 - 0.1));
  const int left_img = static_cast<int>(std::round(dw / 2.0 - 0.1));
  const double stride_h = static_cast<double>(model_h) / chip_h;
  const double stride_w = static_cast<double>(model_w) / chip_w;
  letterbox_chip_rect rect{};
  rect.top = static_cast<int>(std::round(top_img / stride_h));
  rect.left = static_cast<int>(std::round(left_img / stride_w));
  rect.height = static_cast<int>(std::round(new_unpad_h / stride_h));
  rect.width = static_cast<int>(std::round(new_unpad_w / stride_w));
  rect.top = std::clamp(rect.top, 0, chip_h);
  rect.left = std::clamp(rect.left, 0, chip_w);
  rect.height = std::clamp(rect.height, 0, chip_h - rect.top);
  rect.width = std::clamp(rect.width, 0, chip_w - rect.left);
  return rect;
}

// Single-pass bilinear-upsample-and-argmax over a chip-space slice
// [top:top+rect_h, left:left+rect_w, :C] -> (dst_h, dst_w) class map.
//
// The slice lives inside an NHWC tensor strided as
// (chip_h_padded * chip_w_padded * chip_c_padded); the caller passes the
// precomputed row / column strides so this helper stays oblivious to channel
// padding. Channel stride is always 1 (NHWC contiguity) so we hardcode it,
// which lets the auto-vectoriser reason about the inner-loop channel walk.
// Origin pad offsets are folded into `base`.
//
// Bilinear is computed directly on int8: the four-corner weighted sum stays
// in int32 (max magnitude 127 * 256 * 256 ~= 8.3e6). Per-channel values are
// staged in a small stack array, then argmax is one `std::max_element` pass
// per output pixel -- the inner channel loop is branch-free and SIMD-friendly.
// This is exactly order-preserving with the dequantize -> bilinear -> argmax
// reference when all C output channels share the same dequantize affine
// (scale, zero_point), which is the current semantic_seg head convention.
// Single materialised buffer = the output class_map only.
void
upsample_argmax_int8(const int8_t *base, int row_stride, int col_stride,
    const letterbox_chip_rect &rect, int C, int dst_h, int dst_w, int *out)
{
  if (rect.height <= 0 || rect.width <= 0 || C <= 0) {
    std::fill_n(out, dst_h * dst_w, 0);
    return;
  }
  const int rect_h = rect.height;
  const int rect_w = rect.width;
  // Standard align_corners=False (the torch default for bilinear segmentation):
  //   src = (dst + 0.5) * (rect / dst) - 0.5
  // Output pixels outside [0, rect-1] clamp to the nearest valid row/column.
  const double scale_y = static_cast<double>(rect_h) / dst_h;
  const double scale_x = static_cast<double>(rect_w) / dst_w;
  const int8_t *origin = base + rect.top * row_stride + rect.left * col_stride;
  // Precompute per-output-column byte offsets from the row pointer. Baking
  // col_stride in here turns the inner loop into a pair of pointer adds.
  std::vector<int> col_off0(dst_w), col_off1(dst_w);
  std::vector<int> wx_lo(dst_w), wx_hi(dst_w);
  constexpr int FRAC = 1 << 8;
  for (int x = 0; x < dst_w; ++x) {
    const double sx = std::clamp(
        (x + 0.5) * scale_x - 0.5, 0.0, static_cast<double>(rect_w - 1));
    const int xi = static_cast<int>(std::floor(sx));
    const int x0 = xi;
    const int x1 = std::min(xi + 1, rect_w - 1);
    col_off0[x] = x0 * col_stride;
    col_off1[x] = x1 * col_stride;
    const int frac = static_cast<int>(std::round((sx - xi) * FRAC));
    wx_hi[x] = std::clamp(frac, 0, FRAC);
    wx_lo[x] = FRAC - wx_hi[x];
  }
  std::vector<int> vals(C);
  for (int y = 0; y < dst_h; ++y) {
    const double sy = std::clamp(
        (y + 0.5) * scale_y - 0.5, 0.0, static_cast<double>(rect_h - 1));
    const int yi = static_cast<int>(std::floor(sy));
    const int y0 = yi;
    const int y1 = std::min(yi + 1, rect_h - 1);
    const int wy_hi = std::clamp(static_cast<int>(std::round((sy - yi) * FRAC)), 0, FRAC);
    const int wy_lo = FRAC - wy_hi;
    const int8_t *row0 = origin + y0 * row_stride;
    const int8_t *row1 = origin + y1 * row_stride;
    int *out_row = out + y * dst_w;
    for (int x = 0; x < dst_w; ++x) {
      const int8_t *p00 = row0 + col_off0[x];
      const int8_t *p01 = row0 + col_off1[x];
      const int8_t *p10 = row1 + col_off0[x];
      const int8_t *p11 = row1 + col_off1[x];
      const int w00 = wy_lo * wx_lo[x];
      const int w01 = wy_lo * wx_hi[x];
      const int w10 = wy_hi * wx_lo[x];
      const int w11 = wy_hi * wx_hi[x];
      for (int c = 0; c < C; ++c) {
        vals[c] = w00 * static_cast<int>(p00[c]) + w01 * static_cast<int>(p01[c])
                  + w10 * static_cast<int>(p10[c]) + w11 * static_cast<int>(p11[c]);
      }
      out_row[x] = static_cast<int>(
          std::max_element(vals.begin(), vals.begin() + C) - vals.begin());
    }
  }
}

// Binary single-channel variant of upsample_argmax_int8: bilinear-upsample
// the one channel, threshold against threshold_int8 to emit {0, 1}.
void
upsample_threshold_int8(const int8_t *base, int row_stride, int col_stride,
    const letterbox_chip_rect &rect, int dst_h, int dst_w, int threshold_int8, int *out)
{
  if (rect.height <= 0 || rect.width <= 0) {
    std::fill_n(out, dst_h * dst_w, 0);
    return;
  }
  const int rect_h = rect.height;
  const int rect_w = rect.width;
  const double scale_y = static_cast<double>(rect_h) / dst_h;
  const double scale_x = static_cast<double>(rect_w) / dst_w;
  const int8_t *origin = base + rect.top * row_stride + rect.left * col_stride;
  std::vector<int> col_off0(dst_w), col_off1(dst_w);
  std::vector<int> wx_lo(dst_w), wx_hi(dst_w);
  constexpr int FRAC = 1 << 8;
  const int thresh_scaled = threshold_int8 * FRAC * FRAC;
  for (int x = 0; x < dst_w; ++x) {
    const double sx = std::clamp(
        (x + 0.5) * scale_x - 0.5, 0.0, static_cast<double>(rect_w - 1));
    const int xi = static_cast<int>(std::floor(sx));
    const int x0 = xi;
    const int x1 = std::min(xi + 1, rect_w - 1);
    col_off0[x] = x0 * col_stride;
    col_off1[x] = x1 * col_stride;
    const int frac = static_cast<int>(std::round((sx - xi) * FRAC));
    wx_hi[x] = std::clamp(frac, 0, FRAC);
    wx_lo[x] = FRAC - wx_hi[x];
  }
  for (int y = 0; y < dst_h; ++y) {
    const double sy = std::clamp(
        (y + 0.5) * scale_y - 0.5, 0.0, static_cast<double>(rect_h - 1));
    const int yi = static_cast<int>(std::floor(sy));
    const int y0 = yi;
    const int y1 = std::min(yi + 1, rect_h - 1);
    const int wy_hi = std::clamp(static_cast<int>(std::round((sy - yi) * FRAC)), 0, FRAC);
    const int wy_lo = FRAC - wy_hi;
    const int8_t *row0 = origin + y0 * row_stride;
    const int8_t *row1 = origin + y1 * row_stride;
    int *out_row = out + y * dst_w;
    for (int x = 0; x < dst_w; ++x) {
      const int v = wy_lo * wx_lo[x] * static_cast<int>(row0[col_off0[x]])
                    + wy_lo * wx_hi[x] * static_cast<int>(row0[col_off1[x]])
                    + wy_hi * wx_lo[x] * static_cast<int>(row1[col_off0[x]])
                    + wy_hi * wx_hi[x] * static_cast<int>(row1[col_off1[x]]);
      out_row[x] = v > thresh_scaled ? 1 : 0;
    }
  }
}

} // namespace semantic_seg

extern "C" void
decode_to_meta(const AxTensorsInterface &in_tensors, const semantic_seg::properties *prop,
    unsigned int current_frame, unsigned int total_frames,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &map,
    const AxDataInterface &video_interface, Ax::Logger &logger)
{
  auto start_time = std::chrono::high_resolution_clock::now();

  auto &tensor = in_tensors[0];

  if (tensor.bytes == 1) {
    const int H_padded = tensor.sizes[1];
    const int W_padded = tensor.sizes[2];
    const int C_padded = tensor.sizes[3];

    const int pad_H_lo = prop->padding.size() > 2 ? prop->padding[2] : 0;
    const int pad_H_hi = prop->padding.size() > 3 ? prop->padding[3] : 0;
    const int pad_W_lo = prop->padding.size() > 4 ? prop->padding[4] : 0;
    const int pad_W_hi = prop->padding.size() > 5 ? prop->padding[5] : 0;
    const int pad_C_lo = prop->padding.size() > 6 ? prop->padding[6] : 0;
    const int pad_C_hi = prop->padding.size() > 7 ? prop->padding[7] : 0;

    const int H = H_padded - pad_H_lo - pad_H_hi;
    const int W = W_padded - pad_W_lo - pad_W_hi;
    const int C = C_padded - pad_C_lo - pad_C_hi;

    // Get original video dimensions for letterbox crop calculation
    int video_h = 0, video_w = 0;
    if (std::holds_alternative<AxVideoInterface>(video_interface)) {
      const auto &video = std::get<AxVideoInterface>(video_interface);
      video_h = video.info.height;
      video_w = video.info.width;
    }

    // Calculate letterbox crop region in output tensor space
    auto crop = semantic_seg::calculate_letterbox_crop(video_h > 0 ? video_h : H, // Original image height
        video_w > 0 ? video_w : W, // Original image width
        prop->model_height, // Model input height
        prop->model_width, // Model input width
        H, // Output tensor height
        W, // Output tensor width
        prop->scale_up, prop->letterbox);

    // Extract crop coordinates (already in output tensor space)
    int h_start = crop[0], h_end = crop[1], w_start = crop[2], w_end = crop[3];

    int H_cropped = h_end - h_start;
    int W_cropped = w_end - w_start;

    const auto *idata = static_cast<const int8_t *>(tensor.data);
    std::vector<int> size{ H_cropped, W_cropped, C };

    if (prop->class_map_out) {
      const bool interpolate
          = prop->interpolate_before_argmax && video_h > 0 && video_w > 0;
      // When interpolating, target = min(source_dims, model_active_dims).
      // Picks the cheaper bilinear target without losing accuracy: the
      // mIoU gain hinges on bilinear running at >=2x chip pitch, which both
      // candidates satisfy. For large source (>= model) the model-active
      // region is smaller -- caps decoder cost at a constant ~1024*<active>
      // per cityscapes letterbox, independent of 1080p / 4K source. For
      // small source (< model) we skip the upscale and emit directly at
      // source res, which is cheaper and skips the evaluator NN-resize.
      // Implementation: clamp r to <= 1 so target never exceeds source.
      int target_h = H_cropped;
      int target_w = W_cropped;
      if (interpolate) {
        const double r_raw = std::min(static_cast<double>(prop->model_height) / video_h,
            static_cast<double>(prop->model_width) / video_w);
        const double r = std::min(r_raw, 1.0);
        target_h = static_cast<int>(std::round(video_h * r));
        target_w = static_cast<int>(std::round(video_w * r));
      }

      std::vector<int> max_indices(target_h * target_w);

      if (interpolate) {
        // Fused int8 path: a single bilinear pass over the chip-space crop
        // rectangle, with argmax (or threshold for binary) computed on the
        // fly. No intermediate float buffer is materialised -- the only
        // allocation is `max_indices` (the output class map). Bilinear in
        // int8 fixed-point is order-preserving with respect to
        // "dequantize -> bilinear -> argmax" because all C output channels
        // share the same (scale, zero_point) for the semantic_seg head.
        //
        // Strides inside the padded tensor: NHWC layout has row stride
        // W_padded * C_padded, column stride C_padded, channel stride 1.
        // Origin offsets pad_H_lo / pad_W_lo / pad_C_lo are folded into the
        // base pointer so the chip rect coords stay relative to the depadded
        // tile.
        const auto rect = semantic_seg::compute_letterbox_chip_rect(video_h,
            video_w, prop->model_height, prop->model_width, H, W, prop->scale_up);
        const int row_stride = W_padded * C_padded;
        const int col_stride = C_padded;
        const int8_t *base = idata + pad_H_lo * row_stride + pad_W_lo * col_stride + pad_C_lo;
        if (C == 1) {
          semantic_seg::upsample_threshold_int8(base, row_stride, col_stride,
              rect, target_h, target_w, prop->threshold_int8, max_indices.data());
        } else {
          semantic_seg::upsample_argmax_int8(base, row_stride, col_stride, rect,
              C, target_h, target_w, max_indices.data());
        }
      } else if (C == 1) {
        // Binary chip-space path: argmax + dequantization are both
        // unnecessary -- compare directly against threshold_int8.
        auto *out_it = max_indices.data();
        for (int h = h_start; h < h_end; ++h) {
          auto h_off = (h + pad_H_lo) * W_padded * C_padded;
          for (int w = w_start; w < w_end; ++w) {
            auto w_off = (w + pad_W_lo) * C_padded + pad_C_lo;
            *out_it++ = idata[h_off + w_off] > prop->threshold_int8 ? 1 : 0;
          }
        }
      } else {
        // Multi-class chip-space path: argmax is order-preserving on int8,
        // no dequantization needed.
        auto *out_it = max_indices.data();
        for (int h = h_start; h < h_end; ++h) {
          auto h_off = (h + pad_H_lo) * W_padded * C_padded;
          for (int w = w_start; w < w_end; ++w) {
            auto w_off = (w + pad_W_lo) * C_padded + pad_C_lo;
            std::span<const int8_t> row(idata + h_off + w_off, C);
            *out_it++ = std::max_element(row.begin(), row.end()) - row.begin();
          }
        }
      }

      size = { target_h, target_w, C };
      map[prop->meta_name] = std::make_unique<AxMetaSemanticSegmentation>(
          std::move(max_indices), size, prop->decoder_name);
    } else {
      // Raw logits output: dequantize and depad to float.
      const auto *table = prop->dequantize_tables[0].data();
      std::vector<float> data(H_cropped * W_cropped * C);
      int out_idx = 0;
      for (int h = h_start; h < h_end; ++h) {
        auto h_off = (h + pad_H_lo) * W_padded * C_padded;
        for (int w = w_start; w < w_end; ++w) {
          auto w_off = (w + pad_W_lo) * C_padded + pad_C_lo;
          for (int c = 0; c < C; ++c) {
            int src = h_off + w_off + c;
            data[out_idx++] = semantic_seg::dequantize(idata[src], table);
          }
        }
      }
      map[prop->meta_name] = std::make_unique<AxMetaSemanticSegmentation>(
          std::move(data), size, prop->decoder_name);
    }
  } else {
    // Float input (non-compiled / backward-compat path).
    //  Already depadded
    const int H = tensor.sizes[1];
    const int W = tensor.sizes[2];
    const int C = tensor.sizes[3];
    const auto *fdata = static_cast<const float *>(tensor.data);
    std::vector<int> size{ H, W, C };
    const auto total_size = H * W * C;

    if (prop->class_map_out) {
      std::vector<int> max_indices(H * W);
      auto out_it = max_indices.data();
      if (C == 1) {
        for (int offset = 0; offset != total_size; ++offset) {
          *out_it++ = fdata[offset] > prop->threshold ? 1 : 0;
        }
      } else {
        for (int offset = 0; offset != total_size; offset += C) {
          std::span<const float> vec(fdata + offset, C);
          auto max_it = std::max_element(vec.begin(), vec.end());
          *out_it++ = std::distance(vec.begin(), max_it);
        }
      }
      map[prop->meta_name] = std::make_unique<AxMetaSemanticSegmentation>(
          std::move(max_indices), size, prop->decoder_name);
    } else {
      std::vector<float> data(fdata, fdata + total_size);
      map[prop->meta_name] = std::make_unique<AxMetaSemanticSegmentation>(
          std::move(data), size, prop->decoder_name);
    }
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
  logger(AX_DEBUG) << "decode_to_meta : Decoding semantic_seg"
                   << duration.count() << " microseconds" << std::endl;
}

extern "C" const std::unordered_set<std::string> &
allowed_properties()
{
  static const std::unordered_set<std::string> allowed_properties{
    "meta_key",
    "class_map_out",
    "decoder_name",
    "threshold",
    "sigmoid",
    "scales",
    "zero_points",
    "padding",
    "model_width",
    "model_height",
    "scale_up",
    "letterbox",
    "interpolate_before_argmax",
  };
  return allowed_properties;
}

extern "C" std::shared_ptr<void>
init_and_set_static_properties(
    const std::unordered_map<std::string, std::string> &input, Ax::Logger &logger)
{
  auto props = std::make_shared<semantic_seg::properties>();
  props->meta_name = Ax::get_property(
      input, "meta_key", "decode_static_properties", props->meta_name);

  props->decoder_name = Ax::get_property(
      input, "decoder_name", "decode_static_properties", props->decoder_name);

  props->threshold = Ax::get_property(
      input, "threshold", "decode_static_properties", props->threshold);

  props->class_map_out = Ax::get_property(
      input, "class_map_out", "decode_static_properties", props->class_map_out);

  props->sigmoid
      = Ax::get_property(input, "sigmoid", "decode_static_properties", props->sigmoid);

  if (props->sigmoid) {
    // Avoid log(0) and log(1) by clamping threshold to a reasonable range, then convert to logit
    props->threshold = std::clamp(props->threshold, 0.00001F, 0.99999F);
    props->threshold = std::log(props->threshold / (1.0f - props->threshold));
  }

  props->interpolate_before_argmax = Ax::get_property(input, "interpolate_before_argmax",
      "decode_static_properties", props->interpolate_before_argmax);

  auto zero_points = Ax::get_property(
      input, "zero_points", "decode_static_properties", std::vector<float>{});
  auto scales = Ax::get_property<float>(
      input, "scales", "decode_static_properties", std::vector<float>{});

  if (!scales.empty() || !zero_points.empty()) {
    if (zero_points.empty()) {
      zero_points.assign(scales.size(), 0.0f);
    }
    if (scales.empty()) {
      scales.assign(zero_points.size(), 1.0f);
    }

    // Precompute threshold in quantized space for the binary int8 path:
    //   dequantized = (int8 - zero_point) * scale  =>  int8 = threshold / scale + zero_point
    props->threshold_int8 = static_cast<int>(
        std::round(props->threshold / scales[0] + zero_points[0]));

    // Dequantization tables are only needed when class_map_out=false (raw
    // logit output). The class_map_out=true paths -- both chip-space argmax
    // and the int8 fused upsample_argmax -- operate on int8 directly because
    // the head shares one (scale, zero_point) across all output channels, so
    // the per-channel ordering is preserved.
    if (!props->class_map_out) {
      props->dequantize_tables
          = ax_utils::build_dequantization_tables(zero_points, scales);
    }
  }

  props->padding
      = Ax::get_property(input, "padding", "decode_static_properties", props->padding);

  props->model_width = Ax::get_property(
      input, "model_width", "decode_static_properties", props->model_width);
  props->model_height = Ax::get_property(
      input, "model_height", "decode_static_properties", props->model_height);
  props->scale_up = Ax::get_property(
      input, "scale_up", "decode_static_properties", props->scale_up);
  props->letterbox = Ax::get_property(
      input, "letterbox", "decode_static_properties", props->letterbox);

  return props;
}

extern "C" void
set_dynamic_properties(const std::unordered_map<std::string, std::string> &input,
    semantic_seg::properties *prop, Ax::Logger &logger)
{
}
