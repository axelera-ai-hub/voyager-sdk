// Copyright Axelera AI, 2023
#include <gtest/gtest.h>

#include "AxNms.hpp"
#include "AxNmsCore.hpp"

namespace
{
///
/// @brief Three boxes in a row where suppression chains.
///
/// At a threshold of 0.5 the best box reaches the middle one but not the last,
/// and the middle one reaches the last. Greedy therefore keeps the last box,
/// because the middle box is gone before it can suppress anything.
///
struct chain {
  std::vector<ax_nms::box_xyxy> boxes{ { 0, 0, 10, 10 }, { 3, 0, 13, 10 },
    { 6, 0, 16, 10 } };
  std::vector<float> scores{ 0.9F, 0.8F, 0.7F };
  std::vector<int> class_ids{ 1, 1, 1 };
};

TEST(nms_core, iou_of_a_box_with_itself_is_one)
{
  const ax_nms::box_xyxy box{ 4, 8, 24, 48 };
  EXPECT_FLOAT_EQ(ax_nms::intersection_over_union(box, box, ax_nms::exclusive_pixels), 1.0F);
  EXPECT_FLOAT_EQ(ax_nms::intersection_over_union(box, box, ax_nms::inclusive_pixels), 1.0F);
}

TEST(nms_core, touching_boxes_overlap_only_when_pixels_are_inclusive)
{
  const ax_nms::box_xyxy lhs{ 0, 0, 10, 10 };
  const ax_nms::box_xyxy rhs{ 10, 0, 20, 10 };
  EXPECT_FLOAT_EQ(ax_nms::intersection_over_union(lhs, rhs, ax_nms::exclusive_pixels), 0.0F);
  EXPECT_GT(ax_nms::intersection_over_union(lhs, rhs, ax_nms::inclusive_pixels), 0.0F);
}

TEST(nms_core, greedy_on_no_boxes_keeps_nothing)
{
  auto keep = ax_nms::greedy_indices(std::span<const ax_nms::box_xyxy>{}, {},
      {}, 0.5F, false, 300, ax_nms::exclusive_pixels);
  EXPECT_TRUE(keep.empty());
}

TEST(nms_core, greedy_returns_survivors_in_descending_score_order)
{
  std::vector<ax_nms::box_xyxy> boxes{ { 0, 0, 10, 10 }, { 40, 40, 50, 50 },
    { 80, 80, 90, 90 } };
  std::vector<float> scores{ 0.2F, 0.9F, 0.5F };
  std::vector<int> class_ids{ 1, 1, 1 };
  auto keep = ax_nms::greedy_indices(
      boxes, scores, class_ids, 0.5F, false, 300, ax_nms::exclusive_pixels);
  EXPECT_EQ(keep, (std::vector<int>{ 1, 2, 0 }));
}

TEST(nms_core, greedy_drops_a_worse_overlapping_box_of_the_same_class)
{
  chain fixture;
  auto keep = ax_nms::greedy_indices(fixture.boxes, fixture.scores,
      fixture.class_ids, 0.5F, false, 300, ax_nms::exclusive_pixels);
  //  The middle box goes, and with it its power to suppress the last box.
  EXPECT_EQ(keep, (std::vector<int>{ 0, 2 }));
}

TEST(nms_core, greedy_keeps_overlapping_boxes_of_different_classes)
{
  chain fixture;
  fixture.class_ids = { 1, 2, 3 };
  auto keep = ax_nms::greedy_indices(fixture.boxes, fixture.scores,
      fixture.class_ids, 0.5F, false, 300, ax_nms::exclusive_pixels);
  EXPECT_EQ(keep, (std::vector<int>{ 0, 1, 2 }));
}

TEST(nms_core, greedy_class_agnostic_ignores_the_class)
{
  chain fixture;
  fixture.class_ids = { 1, 2, 3 };
  auto keep = ax_nms::greedy_indices(fixture.boxes, fixture.scores,
      fixture.class_ids, 0.5F, true, 300, ax_nms::exclusive_pixels);
  EXPECT_EQ(keep, (std::vector<int>{ 0, 2 }));
}

TEST(nms_core, greedy_stops_at_max_boxes)
{
  chain fixture;
  auto keep = ax_nms::greedy_indices(fixture.boxes, fixture.scores,
      fixture.class_ids, 0.5F, false, 1, ax_nms::exclusive_pixels);
  EXPECT_EQ(keep, (std::vector<int>{ 0 }));
}

TEST(nms_core, greedy_at_max_boxes_zero_keeps_nothing)
{
  chain fixture;
  auto keep = ax_nms::greedy_indices(fixture.boxes, fixture.scores,
      fixture.class_ids, 0.5F, false, 0, ax_nms::exclusive_pixels);
  EXPECT_TRUE(keep.empty());
}

TEST(nms_core, fast_on_no_boxes_keeps_nothing)
{
  auto keep = ax_nms::fast_indices(std::span<const ax_nms::box_xyxy>{}, {}, {},
      0.5F, 0, 300, ax_nms::exclusive_pixels);
  EXPECT_TRUE(keep.empty());
}

TEST(nms_core, fast_drops_a_box_whose_suppressor_was_itself_dropped)
{
  chain fixture;
  auto keep = ax_nms::fast_indices(fixture.boxes, fixture.scores,
      fixture.class_ids, 0.5F, 0, 300, ax_nms::exclusive_pixels);
  //  Greedy keeps the last box here. Every overlap is judged against the
  //  original ranking, so the middle box still removes it.
  EXPECT_EQ(keep, (std::vector<int>{ 0 }));
}

TEST(nms_core, fast_keeps_overlapping_boxes_of_different_classes)
{
  chain fixture;
  fixture.class_ids = { 1, 2, 3 };
  auto keep = ax_nms::fast_indices(fixture.boxes, fixture.scores,
      fixture.class_ids, 0.5F, 0, 300, ax_nms::exclusive_pixels);
  EXPECT_EQ(keep, (std::vector<int>{ 0, 1, 2 }));
}

TEST(nms_core, fast_returns_survivors_in_descending_score_order_across_classes)
{
  std::vector<ax_nms::box_xyxy> boxes{ { 0, 0, 10, 10 }, { 40, 40, 50, 50 },
    { 80, 80, 90, 90 } };
  std::vector<float> scores{ 0.2F, 0.9F, 0.5F };
  std::vector<int> class_ids{ 7, 3, 5 };
  auto keep = ax_nms::fast_indices(
      boxes, scores, class_ids, 0.5F, 0, 300, ax_nms::exclusive_pixels);
  EXPECT_EQ(keep, (std::vector<int>{ 1, 2, 0 }));
}

TEST(nms_core, fast_top_k_bounds_the_candidates_of_each_class)
{
  //  Two classes of two well separated boxes each. top_k=1 leaves only the
  //  better box of each class, so no overlap decides anything.
  std::vector<ax_nms::box_xyxy> boxes{ { 0, 0, 10, 10 }, { 100, 100, 110, 110 },
    { 200, 200, 210, 210 }, { 300, 300, 310, 310 } };
  std::vector<float> scores{ 0.9F, 0.8F, 0.7F, 0.6F };
  std::vector<int> class_ids{ 1, 1, 2, 2 };
  auto keep = ax_nms::fast_indices(
      boxes, scores, class_ids, 0.5F, 1, 300, ax_nms::exclusive_pixels);
  EXPECT_EQ(keep, (std::vector<int>{ 0, 2 }));
}

TEST(nms_core, fast_caps_the_total_after_suppressing)
{
  std::vector<ax_nms::box_xyxy> boxes{ { 0, 0, 10, 10 }, { 100, 100, 110, 110 },
    { 200, 200, 210, 210 } };
  std::vector<float> scores{ 0.5F, 0.9F, 0.7F };
  std::vector<int> class_ids{ 1, 2, 3 };
  auto keep = ax_nms::fast_indices(boxes, scores, class_ids, 0.5F, 0, 2, ax_nms::exclusive_pixels);
  EXPECT_EQ(keep, (std::vector<int>{ 1, 2 }));
}

TEST(nms_core, fast_at_max_boxes_zero_caps_nothing)
{
  chain fixture;
  fixture.class_ids = { 1, 2, 3 };
  auto keep = ax_nms::fast_indices(fixture.boxes, fixture.scores,
      fixture.class_ids, 0.5F, 0, 0, ax_nms::exclusive_pixels);
  //  Not positive is no cap here, where greedy reads 0 as keep nothing.
  EXPECT_EQ(keep, (std::vector<int>{ 0, 1, 2 }));
}

TEST(nms, empty_list_returns_empty_list)
{
  auto boxes = std::vector<box_xyxy>{};
  auto scores = std::vector<float>{};
  auto classes = std::vector<int>{};
  auto meta = AxMetaObjDetection(boxes, scores, classes);
  auto actual = non_max_suppression(meta, 0.5f, true, 300, false);
  EXPECT_EQ(actual.num_elements(), 0);
}

TEST(nms, overlapping_boxes_different_classes_do_not_suppress_if_class_aware)
{
  auto boxes = std::vector<box_xyxy>{ { 0, 0, 10, 10 }, { 1, 1, 11, 11 } };
  auto scores = std::vector<float>{ 0.5f, 0.6f };
  auto classes = std::vector<int>{ 1, 2 };
  auto meta = AxMetaObjDetection(boxes, scores, classes);
  auto actual = non_max_suppression(meta, 0.5f, false, 300, false);
  EXPECT_EQ(actual.num_elements(), 2);
}

TEST(nms, overlapping_boxes_different_classes_suppress_if_class_agnostic)
{
  auto boxes = std::vector<box_xyxy>{ { 0, 0, 10, 10 }, { 1, 1, 11, 11 } };
  auto scores = std::vector<float>{ 0.5f, 0.6f };
  auto classes = std::vector<int>{ 1, 2 };
  auto meta = AxMetaObjDetection(boxes, scores, classes);
  auto actual = non_max_suppression(meta, 0.5f, true, 300, false);
  EXPECT_EQ(actual.num_elements(), 1);
}

TEST(nms, non_overlapping_boxes_do_not_suppress)
{
  auto boxes = std::vector<box_xyxy>{ { 0, 0, 10, 10 }, { 10, 10, 21, 21 } };
  auto scores = std::vector<float>{ 0.5f, 0.6f };
  auto classes = std::vector<int>{ 1, 2 };
  auto meta = AxMetaObjDetection(boxes, scores, classes);
  auto actual = non_max_suppression(meta, 0.5f, true, 300, false);
  EXPECT_EQ(actual.num_elements(), 2);
}

TEST(nms, only_topk_is_returned)
{
  auto boxes = std::vector<box_xyxy>{ { 0, 0, 10, 10 }, { 10, 10, 21, 21 } };
  auto scores = std::vector<float>{ 0.5f, 0.6f };
  auto classes = std::vector<int>{ 1, 2 };
  auto meta = AxMetaObjDetection(boxes, scores, classes);
  auto actual = non_max_suppression(meta, 0.5f, true, 1, false);
  EXPECT_EQ(actual.num_elements(), 1);
  auto box_xyxy = actual.get_box_xyxy(0);
  EXPECT_EQ(box_xyxy.x1, 10);
  EXPECT_EQ(box_xyxy.y1, 10);
  EXPECT_EQ(box_xyxy.x2, 21);
  EXPECT_EQ(box_xyxy.y2, 21);
}

TEST(nms, kpts_input)
{
  auto boxes = std::vector<box_xyxy>{ { 0, 0, 10, 10 }, { 10, 10, 21, 21 } };
  auto kpts = std::vector<kpt_xyv>();
  kpts.insert(kpts.end(), 17, { 0, 0, 0.0 });
  kpts.insert(kpts.end(), 17, { 10, 10, 1.0 });

  auto scores = std::vector<float>{ 0.5f, 0.6f };
  auto ids = std::vector<int>{};
  auto meta = AxMetaKptsDetection(boxes, kpts, scores, ids, { 17, 3 });
  auto actual = non_max_suppression(meta, 0.5f, true, 1, false);

  EXPECT_EQ(actual.num_elements(), 1);
  auto box_xyxy = actual.get_box_xyxy(0);
  EXPECT_EQ(box_xyxy.x1, 10);
  EXPECT_EQ(box_xyxy.y1, 10);
  EXPECT_EQ(box_xyxy.x2, 21);
  EXPECT_EQ(box_xyxy.y2, 21);
  auto actual_kpts = actual.get_kpts();
  EXPECT_EQ(actual_kpts.size(), 17);
  EXPECT_TRUE(std::all_of(actual_kpts.begin(), actual_kpts.end(),
      [](const auto &kpt) { return kpt.x == 10; }));
}

TEST(nms, highest_scoring_overlapping_boxes_remains)
{
  auto boxes
      = std::vector<box_xyxy>{ { 0, 0, 10, 10 }, { 1, 1, 11, 11 }, { 1, 1, 10, 10 } };
  auto scores = std::vector<float>{ 0.5F, 0.6F, 0.4F };
  auto classes = std::vector<int>{ 1, 1, 1 };
  auto meta = AxMetaObjDetection(boxes, scores, classes);
  auto actual = non_max_suppression(meta, 0.5F, false, 300, false);
  EXPECT_EQ(actual.num_elements(), 1);
  EXPECT_FLOAT_EQ(actual.score(0), 0.6F);
  auto box_xyxy = actual.get_box_xyxy(0);
  EXPECT_EQ(box_xyxy.x1, 1);
  EXPECT_EQ(box_xyxy.y1, 1);
  EXPECT_EQ(box_xyxy.x2, 11);
  EXPECT_EQ(box_xyxy.y2, 11);
}
} // namespace
