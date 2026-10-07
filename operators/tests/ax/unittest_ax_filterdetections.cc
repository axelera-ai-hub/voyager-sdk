// Copyright Axelera AI, 2025
#include "gtest/gtest.h"
#include <memory>
#include "AxFilterDetections.hpp"
#include "AxMetaObjectDetection.hpp"

using namespace Ax;

namespace
{

// Helper to create a dummy AxMetaObjDetection
std::unique_ptr<AxMetaObjDetection>
make_obj_meta(const std::vector<box_xyxy> &boxes,
    const std::vector<float> &scores, const std::vector<int> &classes)
{
  return std::make_unique<AxMetaObjDetection>(boxes, scores, classes);
}

TEST(FilterDetections, FiltersByMinWidthHeight)
{
  MetaMap meta_map;
  std::vector<box_xyxy> boxes = { { 0, 0, 4, 4 }, { 0, 0, 10, 10 } };
  std::vector<float> scores = { 0.9f, 0.8f };
  std::vector<int> classes = { 1, 2 };
  meta_map["input"] = make_obj_meta(boxes, scores, classes);
  FilterDetectionsProperties prop;
  prop.input_meta_key = "input";
  prop.output_meta_key = "output";
  prop.min_width = 6;
  prop.min_height = 6;
  AxDataInterface dummy;
  filter_detections(dummy, prop, meta_map);
  auto *out = dynamic_cast<AxMetaObjDetection *>(meta_map["output"].get());
  ASSERT_NE(out, nullptr);
  EXPECT_EQ(out->num_elements(), 1);
  EXPECT_EQ(out->get_box_xyxy(0).x2, 10);
}

TEST(FilterDetections, FiltersByScore)
{
  MetaMap meta_map;
  std::vector<box_xyxy> boxes = { { 0, 0, 10, 10 }, { 0, 0, 20, 20 } };
  std::vector<float> scores = { 0.4f, 0.9f };
  std::vector<int> classes = { 1, 2 };
  meta_map["input"] = make_obj_meta(boxes, scores, classes);
  FilterDetectionsProperties prop;
  prop.input_meta_key = "input";
  prop.output_meta_key = "output";
  prop.score = 0.5f;
  AxDataInterface dummy;
  filter_detections(dummy, prop, meta_map);
  auto *out = dynamic_cast<AxMetaObjDetection *>(meta_map["output"].get());
  ASSERT_NE(out, nullptr);
  EXPECT_EQ(out->num_elements(), 1);
  EXPECT_FLOAT_EQ(out->score(0), 0.9f);
}

TEST(FilterDetections, FiltersByClassesToKeep)
{
  MetaMap meta_map;
  std::vector<box_xyxy> boxes = { { 0, 0, 10, 10 }, { 0, 0, 20, 20 } };
  std::vector<float> scores = { 0.7f, 0.8f };
  std::vector<int> classes = { 1, 2 };
  meta_map["input"] = make_obj_meta(boxes, scores, classes);
  FilterDetectionsProperties prop;
  prop.input_meta_key = "input";
  prop.output_meta_key = "output";
  prop.classes_to_keep = { 2 };
  AxDataInterface dummy;
  filter_detections(dummy, prop, meta_map);
  auto *out = dynamic_cast<AxMetaObjDetection *>(meta_map["output"].get());
  ASSERT_NE(out, nullptr);
  EXPECT_EQ(out->num_elements(), 1);
  EXPECT_EQ(out->class_id(0), 2);
}

TEST(FilterDetections, TopKByScore)
{
  MetaMap meta_map;
  std::vector<box_xyxy> boxes = { { 0, 0, 10, 10 }, { 0, 0, 20, 20 }, { 0, 0, 30, 30 } };
  std::vector<float> scores = { 0.5f, 0.9f, 0.7f };
  std::vector<int> classes = { 1, 2, 3 };
  meta_map["input"] = make_obj_meta(boxes, scores, classes);
  FilterDetectionsProperties prop;
  prop.input_meta_key = "input";
  prop.output_meta_key = "output";
  prop.which = Which::Score;
  prop.top_k = 2;
  AxDataInterface dummy;
  filter_detections(dummy, prop, meta_map);
  auto *out = dynamic_cast<AxMetaObjDetection *>(meta_map["output"].get());
  ASSERT_NE(out, nullptr);
  EXPECT_EQ(out->num_elements(), 2);
  EXPECT_FLOAT_EQ(out->score(0), 0.9f);
  EXPECT_FLOAT_EQ(out->score(1), 0.7f);
}

TEST(FilterDetections, ThrowsOnMissingMeta)
{
  MetaMap meta_map;
  FilterDetectionsProperties prop;
  prop.input_meta_key = "notfound";
  prop.output_meta_key = "output";
  AxDataInterface dummy;
  EXPECT_THROW(filter_detections(dummy, prop, meta_map), std::runtime_error);
}

TEST(FilterDetections, ThrowsOnWrongType)
{
  // Insert a dummy base meta that is not AxMetaBbox
  struct DummyMeta : public AxMetaBase {
  };
  MetaMap meta_map;
  meta_map["input"] = std::make_unique<DummyMeta>();
  FilterDetectionsProperties prop;
  prop.input_meta_key = "input";
  AxDataInterface dummy;
  EXPECT_THROW(filter_detections(dummy, prop, meta_map), std::runtime_error);
}

TEST(FilterDetectionsTest, FiltersByMinSize)
{
  // Create meta with 3 boxes of different sizes
  std::vector<box_xyxy> boxes = {
    { 0, 0, 9, 9 }, // 10x10
    { 0, 0, 19, 19 }, // 20x20
    { 0, 0, 29, 29 } // 30x30
  };
  std::vector<float> scores = { 0.5, 0.6, 0.7 };
  std::vector<int> classes = { 1, 2, 3 };
  auto meta = std::make_unique<AxMetaObjDetection>(boxes, scores, classes);
  MetaMap meta_map;
  meta_map["input"] = std::move(meta);
  FilterDetectionsProperties prop;
  prop.input_meta_key = "input";
  prop.output_meta_key = "output";
  prop.min_width = 20;
  prop.min_height = 20;
  AxDataInterface dummy_interface;
  filter_detections(dummy_interface, prop, meta_map);
  auto *out = dynamic_cast<AxMetaObjDetection *>(meta_map["output"].get());
  ASSERT_NE(out, nullptr);
  EXPECT_EQ(out->num_elements(), 2);
  EXPECT_EQ(out->get_box_xyxy(0).x2, 19);
  EXPECT_EQ(out->get_box_xyxy(1).x2, 29);
}

TEST(FilterDetectionsTest, FiltersByScore)
{
  std::vector<box_xyxy> boxes = { { 0, 0, 10, 10 }, { 0, 0, 20, 20 }, { 0, 0, 30, 30 } };
  std::vector<float> scores = { 0.1, 0.5, 0.9 };
  std::vector<int> classes = { 1, 1, 1 };
  auto meta = std::make_unique<AxMetaObjDetection>(boxes, scores, classes);
  MetaMap meta_map;
  meta_map["input"] = std::move(meta);
  FilterDetectionsProperties prop;
  prop.input_meta_key = "input";
  prop.output_meta_key = "output";
  prop.score = 0.5;
  AxDataInterface dummy_interface;
  filter_detections(dummy_interface, prop, meta_map);
  auto *out = dynamic_cast<AxMetaObjDetection *>(meta_map["output"].get());
  ASSERT_NE(out, nullptr);
  EXPECT_EQ(out->num_elements(), 2);
  EXPECT_FLOAT_EQ(out->score(0), 0.5);
  EXPECT_FLOAT_EQ(out->score(1), 0.9);
}

TEST(FilterDetectionsTest, FiltersByClasses)
{
  std::vector<box_xyxy> boxes = { { 0, 0, 10, 10 }, { 0, 0, 20, 20 }, { 0, 0, 30, 30 } };
  std::vector<float> scores = { 0.5, 0.5, 0.5 };
  std::vector<int> classes = { 1, 2, 3 };
  auto meta = std::make_unique<AxMetaObjDetection>(boxes, scores, classes);
  MetaMap meta_map;
  meta_map["input"] = std::move(meta);
  FilterDetectionsProperties prop;
  prop.input_meta_key = "input";
  prop.output_meta_key = "output";
  prop.classes_to_keep = { 2, 3 };
  AxDataInterface dummy_interface;
  filter_detections(dummy_interface, prop, meta_map);
  auto *out = dynamic_cast<AxMetaObjDetection *>(meta_map["output"].get());
  ASSERT_NE(out, nullptr);
  EXPECT_EQ(out->num_elements(), 2);
  EXPECT_EQ(out->class_id(0), 2);
  EXPECT_EQ(out->class_id(1), 3);
}

TEST(FilterDetectionsTest, TopKByScore)
{
  std::vector<box_xyxy> boxes = { { 0, 0, 10, 10 }, { 0, 0, 20, 20 }, { 0, 0, 30, 30 } };
  std::vector<float> scores = { 0.2, 0.8, 0.5 };
  std::vector<int> classes = { 1, 1, 1 };
  auto meta = std::make_unique<AxMetaObjDetection>(boxes, scores, classes);
  MetaMap meta_map;
  meta_map["input"] = std::move(meta);
  FilterDetectionsProperties prop;
  prop.input_meta_key = "input";
  prop.output_meta_key = "output";
  prop.which = Which::Score;
  prop.top_k = 2;
  AxDataInterface dummy_interface;
  filter_detections(dummy_interface, prop, meta_map);
  auto *out = dynamic_cast<AxMetaObjDetection *>(meta_map["output"].get());
  ASSERT_NE(out, nullptr);
  EXPECT_EQ(out->num_elements(), 2);
  EXPECT_FLOAT_EQ(out->score(0), 0.8);
  EXPECT_FLOAT_EQ(out->score(1), 0.5);
}

TEST(FilterDetectionsTest, ThrowsOnMissingMetaKey)
{
  MetaMap meta_map;
  FilterDetectionsProperties prop;
  prop.input_meta_key = "notfound";
  AxDataInterface dummy_interface;
  EXPECT_THROW(filter_detections(dummy_interface, prop, meta_map), std::runtime_error);
}


TEST(FilterDetectionsTest, ThrowsOnMissingScore)
{
  MetaMap meta_map;
  FilterDetectionsProperties prop;
  prop.input_meta_key = "notfound";
  AxDataInterface dummy_interface;
  EXPECT_THROW(filter_detections(dummy_interface, prop, meta_map), std::runtime_error);
}

} // namespace
