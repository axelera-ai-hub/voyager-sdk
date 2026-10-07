// Copyright Axelera AI, 2026
#include "gtest/gtest.h"
#include <gmodule.h>
#include "unittest_ax_common.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "AxDataInterface.h"
#include "AxLog.hpp"
#include "AxMeta.hpp"
#include "AxMetaSegmentsDetection.hpp"

namespace
{

// One pyramid level of one cell. The anchor side is scale / max(model size), so
// scale 20 on a 1000 pixel model gives a prior of 0.02 centred on the cell,
// which lands on whole pixels and keeps the expectations exact.
constexpr int model_size = 1000;
constexpr int num_classes = 2;
constexpr int num_seg_masks = 8;
constexpr int proto_size = 2;

struct level {
  std::vector<int8_t> loc;
  std::vector<int8_t> conf;
  std::vector<int8_t> coefficients;
  std::vector<int8_t> prototypes;
  int grid_width = 1;
  int classes = num_classes;
  int proto = proto_size;
};

Ax::StringMap
base_properties(int num_anchors)
{
  const auto tensors = 4;
  auto ones = std::string("1");
  auto zeros = std::string("0");
  auto pads = std::string("0,0,0,0,0,0,0,0");
  for (auto i = 1; i != tensors; ++i) {
    ones += ",1";
    zeros += ",0";
    pads += "|0,0,0,0,0,0,0,0";
  }
  return {
    { "meta_key", "yolact" },
    { "scales", ones },
    { "zero_points", zeros },
    { "padding", pads },
    { "classes", std::to_string(num_classes) },
    { "num_seg_masks", std::to_string(num_seg_masks) },
    { "anchor_scales", "20" },
    { "aspect_ratios", num_anchors == 1 ? "1.0" : "1.0,1.21" },
    { "confidence_threshold", "0.5" },
    { "nms_iou_threshold", "0.5" },
    { "model_width", std::to_string(model_size) },
    { "model_height", std::to_string(model_size) },
  };
}

// The prototype has to stay the widest tensor, because that is how the decoder
// picks it out, so a multi-cell grid needs a bigger prototype than the default.
AxTensorsInterface
tensors_for(level &data, int num_anchors)
{
  const auto width = data.grid_width;
  return {
    { { 1, data.proto, data.proto, num_seg_masks }, 1, data.prototypes.data() },
    { { 1, 1, width, 4 * num_anchors }, 1, data.loc.data() },
    { { 1, 1, width, (data.classes + 1) * num_anchors }, 1, data.conf.data() },
    { { 1, 1, width, num_seg_masks * num_anchors }, 1, data.coefficients.data() },
  };
}

// A prototype whose first channel alternates sign across the 2x2 grid, so a
// coefficient vector of (1, 0, ...) makes the mask a known checkerboard.
std::vector<int8_t>
checkerboard_prototypes()
{
  std::vector<int8_t> prototypes(proto_size * proto_size * num_seg_masks, 0);
  const int8_t signs[] = { 1, -1, -1, 1 };
  for (auto cell = 0; cell != proto_size * proto_size; ++cell) {
    prototypes[cell * num_seg_masks] = signs[cell];
  }
  return prototypes;
}

// Two channels of opposite sign and unequal magnitude on every cell, so the
// channel sum is decided by how the two coefficients are weighted against each
// other rather than by either one alone.
std::vector<int8_t>
opposed_prototypes()
{
  std::vector<int8_t> prototypes(proto_size * proto_size * num_seg_masks, 0);
  for (auto cell = 0; cell != proto_size * proto_size; ++cell) {
    prototypes[cell * num_seg_masks] = 1;
    prototypes[cell * num_seg_masks + 1] = -3;
  }
  return prototypes;
}

constexpr int row_cells = 12;
constexpr int row_proto = 16;
constexpr int row_classes = 4;

// A single row of cells, hot on the first few and background-dominated on the
// rest. With a 300 wide prior on a 1000 pixel model the priors span 0.3 while
// the cells sit 1/12 apart, so cells one apart overlap at IoU 0.565 and cells
// two apart at 0.286, the spread fast NMS needs to be told from greedy NMS.
level
row_of_cells(const std::vector<std::pair<int, int8_t>> &hot)
{
  level data;
  data.grid_width = row_cells;
  data.classes = row_classes;
  data.proto = row_proto;
  data.loc.assign(row_cells * 4, 0);
  data.coefficients.assign(row_cells * num_seg_masks, 0);
  data.prototypes.assign(row_proto * row_proto * num_seg_masks, 0);
  const auto stride = row_classes + 1;
  data.conf.assign(row_cells * stride, 0);
  for (auto cell = 0; cell != row_cells; ++cell) {
    data.conf[cell * stride] = 12;
  }
  for (auto cell = size_t{}; cell != hot.size(); ++cell) {
    data.conf[cell * stride] = 0;
    data.conf[cell * stride + 1 + hot[cell].first] = hot[cell].second;
  }
  return data;
}

Ax::StringMap
row_properties()
{
  auto properties = base_properties(1);
  properties["classes"] = std::to_string(row_classes);
  properties["anchor_scales"] = "300";
  return properties;
}

AxMetaSegmentsDetection *
decode(Ax::Decode &decoder, level &data, int num_anchors,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &map)
{
  AxVideoInterface video{
    { model_size, model_size, model_size * 3, 0, AxVideoFormat::RGB }, nullptr
  };
  auto tensors = tensors_for(data, num_anchors);
  decoder.decode_to_meta(tensors, 0, 1, map, video);
  auto found = map.find("yolact");
  return found == map.end() ?
             nullptr :
             dynamic_cast<AxMetaSegmentsDetection *>(found->second.get());
}

} // namespace

TEST(yolact_properties, missing_model_size_throws)
{
  auto properties = base_properties(1);
  properties.erase("model_width");
  EXPECT_THROW(Ax::LoadDecode("yolact", properties), std::runtime_error);
}

TEST(yolact_properties, mismatched_scales_and_zero_points_throws)
{
  auto properties = base_properties(1);
  properties["scales"] = "1,1,1";
  EXPECT_THROW(Ax::LoadDecode("yolact", properties), std::runtime_error);
}

TEST(yolact_properties, no_classes_and_no_labels_throws)
{
  auto properties = base_properties(1);
  properties.erase("classes");
  EXPECT_THROW(Ax::LoadDecode("yolact", properties), std::runtime_error);
}

TEST(yolact_properties, indistinguishable_head_depths_throws)
{
  //  num_seg_masks 4 collides with the four box channels
  auto properties = base_properties(1);
  properties["num_seg_masks"] = "4";
  EXPECT_THROW(Ax::LoadDecode("yolact", properties), std::runtime_error);
}

TEST(yolact_decode, tensor_count_must_cover_every_level)
{
  auto properties = base_properties(1);
  properties["anchor_scales"] = "20,40";
  auto decoder = Ax::LoadDecode("yolact", properties);
  level data{ std::vector<int8_t>(4), std::vector<int8_t>{ 0, 4, 0 },
    std::vector<int8_t>(num_seg_masks), checkerboard_prototypes() };
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map;
  EXPECT_THROW(decode(*decoder, data, 1, map), std::runtime_error);
}

TEST(yolact_decode, background_class_is_dropped)
{
  auto decoder = Ax::LoadDecode("yolact", base_properties(1));
  //  Softmax over (background, class 0, class 1): the background wins, so the
  //  best foreground score is 0.018 and nothing clears the 0.5 threshold.
  level data{ std::vector<int8_t>(4), std::vector<int8_t>{ 4, 0, 0 },
    std::vector<int8_t>(num_seg_masks), checkerboard_prototypes() };
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map;
  auto *meta = decode(*decoder, data, 1, map);
  ASSERT_NE(meta, nullptr);
  EXPECT_EQ(meta->num_elements(), 0);
}

TEST(yolact_decode, prior_decode_and_mask_threshold)
{
  auto decoder = Ax::LoadDecode("yolact", base_properties(1));
  auto coefficients = std::vector<int8_t>(num_seg_masks, 0);
  coefficients[0] = 1;
  level data{ std::vector<int8_t>(4), std::vector<int8_t>{ 0, 4, 0 },
    coefficients, checkerboard_prototypes() };
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map;
  auto *meta = decode(*decoder, data, 1, map);
  ASSERT_NE(meta, nullptr);
  ASSERT_EQ(meta->num_elements(), 1);
  EXPECT_EQ(meta->class_id(0), 0);
  EXPECT_NEAR(meta->score(0), 0.96466F, 1e-4);

  //  A zero location head leaves the box on the prior: centred on the only
  //  cell with a side of 20/1000, i.e. 0.49 to 0.51 of a 1000 pixel frame.
  const auto box = meta->get_box_xyxy(0);
  EXPECT_EQ(box.x1, 490);
  EXPECT_EQ(box.y1, 490);
  EXPECT_EQ(box.x2, 510);
  EXPECT_EQ(box.y2, 510);

  //  The box padded by one prototype pixel covers the whole 2x2 grid, and the
  //  mask is the sign of the prototype combination.
  const auto segment = meta->get_segment(0);
  EXPECT_EQ(segment.x1, 0);
  EXPECT_EQ(segment.y1, 0);
  EXPECT_EQ(segment.x2, proto_size);
  EXPECT_EQ(segment.y2, proto_size);
  EXPECT_EQ(segment.map, std::vector<uint8_t>({ 255, 0, 0, 255 }));
}

TEST(yolact_decode, the_mask_coefficients_carry_upstreams_tanh)
{
  //  Upstream's mask head ends in tanh and the compiler leaves that activation
  //  out of the AIPU graph, so the decoder owes it. It is not a no-op: tanh
  //  preserves the sign of each coefficient but compresses the larger one much
  //  further, which changes the sign of the sum they contribute to.
  //
  //  Coefficients (4, 1) against prototype channels (1, -3) give 4 - 3 = +1 raw,
  //  so every cell would be set. Through tanh the same pair gives
  //  0.999 - 3 * 0.762 = -1.285 and every cell is clear.
  auto decoder = Ax::LoadDecode("yolact", base_properties(1));
  auto coefficients = std::vector<int8_t>(num_seg_masks, 0);
  coefficients[0] = 4;
  coefficients[1] = 1;
  level data{ std::vector<int8_t>(4), std::vector<int8_t>{ 0, 4, 0 },
    coefficients, opposed_prototypes() };
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map;
  auto *meta = decode(*decoder, data, 1, map);
  ASSERT_NE(meta, nullptr);
  ASSERT_EQ(meta->num_elements(), 1);
  EXPECT_EQ(meta->get_segment(0).map, std::vector<uint8_t>({ 0, 0, 0, 0 }));

  //  Dropping the second coefficient leaves one positive term, which tanh
  //  cannot flip. That pins the result above to the reweighting of the sum
  //  rather than to a sign change in the coefficients themselves.
  coefficients[1] = 0;
  level single{ std::vector<int8_t>(4), std::vector<int8_t>{ 0, 4, 0 },
    coefficients, opposed_prototypes() };
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> single_map;
  auto *kept = decode(*decoder, single, 1, single_map);
  ASSERT_NE(kept, nullptr);
  ASSERT_EQ(kept->num_elements(), 1);
  EXPECT_EQ(kept->get_segment(0).map, std::vector<uint8_t>({ 255, 255, 255, 255 }));
}

TEST(yolact_decode, variance_decode_places_the_box)
{
  auto decoder = Ax::LoadDecode("yolact", base_properties(1));
  //  Offsets of (10, -10) move the centre by offset * 0.1 * side from the prior
  //  centre of 0.5, and (5, 1) scale the side by exp(offset * 0.2), giving
  //  0.52 +/- 0.0272 across and 0.48 +/- 0.0122 down on a 1000 pixel frame.
  level data{ std::vector<int8_t>{ 10, -10, 5, 1 }, std::vector<int8_t>{ 0, 4, 0 },
    std::vector<int8_t>(num_seg_masks), checkerboard_prototypes() };
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map;
  auto *meta = decode(*decoder, data, 1, map);
  ASSERT_NE(meta, nullptr);
  ASSERT_EQ(meta->num_elements(), 1);
  const auto box = meta->get_box_xyxy(0);
  EXPECT_EQ(box.x1, 493);
  EXPECT_EQ(box.y1, 468);
  EXPECT_EQ(box.x2, 547);
  EXPECT_EQ(box.y2, 492);
}

TEST(yolact_decode, fast_nms_suppresses_the_lower_score_of_a_class)
{
  //  Two priors on the same cell with sides 0.020 and 0.022, so their IoU is
  //  (20/22)^2 = 0.826. Both win class 0, the second by more.
  level data{ std::vector<int8_t>(8), std::vector<int8_t>{ 0, 4, 0, 0, 8, 0 },
    std::vector<int8_t>(2 * num_seg_masks), checkerboard_prototypes() };

  auto suppressing = Ax::LoadDecode("yolact", base_properties(2));
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> suppressed_map;
  auto *suppressed = decode(*suppressing, data, 2, suppressed_map);
  ASSERT_NE(suppressed, nullptr);
  EXPECT_EQ(suppressed->num_elements(), 1);

  //  Above that IoU nothing is suppressed, which pins the threshold to the
  //  overlap rather than to some other rejection.
  auto permissive_properties = base_properties(2);
  permissive_properties["nms_iou_threshold"] = "0.9";
  auto permissive = Ax::LoadDecode("yolact", permissive_properties);
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> permissive_map;
  auto *kept = decode(*permissive, data, 2, permissive_map);
  ASSERT_NE(kept, nullptr);
  EXPECT_EQ(kept->num_elements(), 2);
}

TEST(yolact_decode, fast_nms_is_per_class)
{
  //  The same overlapping pair as above, but each prior wins a different
  //  class, so neither can suppress the other.
  level data{ std::vector<int8_t>(8), std::vector<int8_t>{ 0, 4, 0, 0, 0, 8 },
    std::vector<int8_t>(2 * num_seg_masks), checkerboard_prototypes() };
  auto decoder = Ax::LoadDecode("yolact", base_properties(2));
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map;
  auto *meta = decode(*decoder, data, 2, map);
  ASSERT_NE(meta, nullptr);
  ASSERT_EQ(meta->num_elements(), 2);
  EXPECT_EQ(meta->class_id(0), 1);
  EXPECT_EQ(meta->class_id(1), 0);
}

TEST(yolact_decode, a_candidate_is_kept_for_every_class_it_wins)
{
  //  Both foreground classes clear the threshold on the one prior, so the
  //  detection appears twice with different class ids.
  auto properties = base_properties(1);
  properties["confidence_threshold"] = "0.3";
  auto decoder = Ax::LoadDecode("yolact", properties);
  level data{ std::vector<int8_t>(4), std::vector<int8_t>{ -8, 0, 0 },
    std::vector<int8_t>(num_seg_masks), checkerboard_prototypes() };
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map;
  auto *meta = decode(*decoder, data, 1, map);
  ASSERT_NE(meta, nullptr);
  ASSERT_EQ(meta->num_elements(), 2);
  EXPECT_EQ(meta->class_id(0), 0);
  EXPECT_EQ(meta->class_id(1), 1);
}

TEST(yolact_decode, fast_nms_is_not_transitive)
{
  //  Three cells of one class in descending score order. The first two overlap
  //  at 0.565 and so do the last two, but the first and last only at 0.286.
  //  Fast NMS judges every candidate against all higher scoring ones, including
  //  ones that were themselves suppressed, so the third dies by the second and
  //  only one detection survives. Greedy NMS would compare the third against
  //  the surviving first alone and keep it, so this is the assertion that tells
  //  the two apart.
  auto data = row_of_cells({ { 0, 12 }, { 0, 8 }, { 0, 4 } });
  auto decoder = Ax::LoadDecode("yolact", row_properties());
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map;
  auto *meta = decode(*decoder, data, 1, map);
  ASSERT_NE(meta, nullptr);
  EXPECT_EQ(meta->num_elements(), 1);

  //  Above the pair overlap nothing is suppressed, which pins the outcome to
  //  the IoU threshold rather than to some other rejection.
  auto permissive = row_properties();
  permissive["nms_iou_threshold"] = "0.6";
  auto lenient = Ax::LoadDecode("yolact", permissive);
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> lenient_map;
  auto *all = decode(*lenient, data, 1, lenient_map);
  ASSERT_NE(all, nullptr);
  EXPECT_EQ(all->num_elements(), 3);
}

TEST(yolact_decode, a_binding_cap_keeps_the_highest_scores)
{
  //  Three overlapping cells of three different classes, so no candidate can
  //  suppress another and the cap alone decides. It must keep the top two by
  //  score, in order.
  auto data = row_of_cells({ { 0, 12 }, { 1, 8 }, { 2, 4 } });
  auto properties = row_properties();
  properties["topk"] = "2";
  auto decoder = Ax::LoadDecode("yolact", properties);
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map;
  auto *meta = decode(*decoder, data, 1, map);
  ASSERT_NE(meta, nullptr);
  ASSERT_EQ(meta->num_elements(), 2);
  EXPECT_EQ(meta->class_id(0), 0);
  EXPECT_EQ(meta->class_id(1), 1);
  EXPECT_GT(meta->score(0), meta->score(1));
}
