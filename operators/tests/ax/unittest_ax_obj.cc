// Copyright Axelera AI, 2023
#include <tuple>

#include <gtest/gtest.h>

#include "AxMetaObjectDetection.hpp"

namespace
{

TEST(AxMetaObjDetection, empty_meta_has_no_subframes)
{
  auto boxes = std::vector<box_xyxy>{};
  auto scores = std::vector<float>{};
  auto classes = std::vector<int>{};
  auto meta = AxMetaObjDetection(boxes, scores, classes);
  EXPECT_EQ(meta.num_elements(), 0);
}

TEST(AxMetaObjDetection, meta_with_a_single_box_has_one_subframe)
{
  auto boxes = std::vector<box_xyxy>{ { 0, 0, 10, 10 } };
  auto scores = std::vector<float>{ 0.6 };
  auto classes = std::vector<int>{};
  auto meta = AxMetaObjDetection(boxes, scores, classes);
  EXPECT_EQ(meta.num_elements(), 1);
  EXPECT_EQ(meta.is_multi_class(), false);
  EXPECT_FLOAT_EQ(meta.score(0), 0.6);
  EXPECT_EQ(meta.class_id(0), -1);
  auto actual_box = meta.get_box_xyxy(0);
  auto expected_box = box_xyxy{ 0, 0, 10, 10 };
  EXPECT_EQ(std::tie(actual_box.x1, actual_box.y1, actual_box.x2, actual_box.y2),
      std::tie(expected_box.x1, expected_box.y1, expected_box.x2, expected_box.y2));
}

TEST(AxMetaObjDetection, view_as_xywh)
{
  auto boxes = std::vector<box_xyxy>{ { 5, 5, 15, 15 } };
  auto scores = std::vector<float>{ 0.6 };
  auto classes = std::vector<int>{};
  auto meta = AxMetaObjDetection(boxes, scores, classes);
  auto actual_box = meta.get_box_xywh(0);
  auto expected_box = box_xywh{ 10, 10, 10, 10 };

  EXPECT_EQ(std::tie(actual_box.x, actual_box.y, actual_box.w, actual_box.h),
      std::tie(expected_box.x, expected_box.y, expected_box.w, expected_box.h));
}

TEST(AxMetaObjDetection, view_as_xywh_checked)
{
  auto boxes = std::vector<box_xyxy>{ { 5, 5, 15, 15 } };
  auto scores = std::vector<float>{ 0.6 };
  auto classes = std::vector<int>{};
  auto meta = AxMetaObjDetection(boxes, scores, classes);
  auto actual_box = meta.get_box_xywh_at(0);
  auto expected_box = box_xywh{ 10, 10, 10, 10 };

  EXPECT_EQ(std::tie(actual_box.x, actual_box.y, actual_box.w, actual_box.h),
      std::tie(expected_box.x, expected_box.y, expected_box.w, expected_box.h));
}

TEST(AxMetaObjDetection, view_as_ltxywh)
{
  auto boxes = std::vector<box_xyxy>{ { 5, 5, 15, 15 } };
  auto scores = std::vector<float>{ 0.6 };
  auto classes = std::vector<int>{};
  auto meta = AxMetaObjDetection(boxes, scores, classes);
  auto actual_box = meta.get_box_ltxywh(0);
  auto expected_box = box_ltxywh{ 5, 5, 10, 10 };

  EXPECT_EQ(std::tie(actual_box.x, actual_box.y, actual_box.w, actual_box.h),
      std::tie(expected_box.x, expected_box.y, expected_box.w, expected_box.h));
}

TEST(AxMetaObjDetection, view_as_ltxywh_checked)
{
  auto boxes = std::vector<box_xyxy>{ { 5, 5, 15, 15 } };
  auto scores = std::vector<float>{ 0.6 };
  auto classes = std::vector<int>{};
  auto meta = AxMetaObjDetection(boxes, scores, classes);
  auto actual_box = meta.get_box_ltxywh_at(0);
  auto expected_box = box_ltxywh{ 5, 5, 10, 10 };

  EXPECT_EQ(std::tie(actual_box.x, actual_box.y, actual_box.w, actual_box.h),
      std::tie(expected_box.x, expected_box.y, expected_box.w, expected_box.h));
}

TEST(AxMetaObjDetection, meta_with_a_multiple_box_requisite_subframes)
{
  auto boxes = std::vector<box_xyxy>{ { 0, 0, 10, 10 }, { 15, 15, 25, 25 },
    { 30, 30, 50, 50 } };
  auto scores = std::vector<float>{ 0.6, 0.7, 0.3 };
  auto classes = std::vector<int>{ 7, 9, 3 };
  auto meta = AxMetaObjDetection(boxes, scores, classes);
  EXPECT_EQ(meta.num_elements(), 3);
  EXPECT_EQ(meta.is_multi_class(), true);
  EXPECT_FLOAT_EQ(meta.score(1), 0.7);
  EXPECT_EQ(meta.class_id(1), 9);
  auto actual_box = meta.get_box_xyxy(2);
  auto expected_box = box_xyxy{ 30, 30, 50, 50 };
  EXPECT_EQ(std::tie(actual_box.x1, actual_box.y1, actual_box.x2, actual_box.y2),
      std::tie(expected_box.x1, expected_box.y1, expected_box.x2, expected_box.y2));
}

TEST(AxMetaObjDetection, meta_with_a_multiple_box_requisite_subframes_checked)
{
  auto boxes = std::vector<box_xyxy>{ { 0, 0, 10, 10 }, { 15, 15, 25, 25 },
    { 30, 30, 50, 50 } };
  auto scores = std::vector<float>{ 0.6, 0.7, 0.3 };
  auto classes = std::vector<int>{ 7, 9, 3 };
  auto meta = AxMetaObjDetection(boxes, scores, classes);
  EXPECT_EQ(meta.num_elements(), 3);
  EXPECT_EQ(meta.is_multi_class(), true);
  EXPECT_FLOAT_EQ(meta.score_at(1), 0.7);
  EXPECT_EQ(meta.class_id_at(1), 9);
  auto actual_box = meta.get_box_xyxy_at(2);
  auto expected_box = box_xyxy{ 30, 30, 50, 50 };
  EXPECT_EQ(std::tie(actual_box.x1, actual_box.y1, actual_box.x2, actual_box.y2),
      std::tie(expected_box.x1, expected_box.y1, expected_box.x2, expected_box.y2));
}

TEST(AxMetaObjDetection, meta_with_a_multiple_box_requisite_subframes_out_of_range)
{
  auto boxes = std::vector<box_xyxy>{ { 0, 0, 10, 10 }, { 15, 15, 25, 25 },
    { 30, 30, 50, 50 } };
  auto scores = std::vector<float>{ 0.6, 0.7, 0.3 };
  auto classes = std::vector<int>{ 7, 9, 3 };
  auto meta = AxMetaObjDetection(boxes, scores, classes);
  EXPECT_THROW(meta.score_at(5), std::out_of_range);
  EXPECT_THROW(meta.class_id_at(5), std::out_of_range);
  EXPECT_THROW(meta.get_box_xyxy_at(5), std::out_of_range);
  EXPECT_THROW(meta.get_box_ltxywh_at(5), std::out_of_range);
  EXPECT_THROW(meta.get_box_xywh_at(5), std::out_of_range);
}


} // namespace
