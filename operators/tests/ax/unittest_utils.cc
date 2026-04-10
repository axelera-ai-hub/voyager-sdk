// Copyright Axelera AI, 2025
#include <gtest/gtest.h>
#include <algorithm>
#include "AxMetaObjectDetection.hpp"
#include "AxMetaTracker.hpp"
#include "AxOpUtils.hpp"
#include "AxUtils.hpp"

const std::unordered_map<std::string, std::string> input = {
  { "empty", "" },
  { "str", "s" },
  { "+ve", "1" },
  { "-ve", "-1" },
  { "zero", "0" },
  { "maxi8", "127" },
  { "mini8", "-128" },
  { "maxu8", "255" },
  { "maxint", "2147483647" },
  { "ints", "1,2,3,4" },
};

TEST(ax_utils_tests, test_get_property_uint8)
{
  const auto d = std::uint8_t{ 42 };
  EXPECT_EQ(Ax::get_property(input, "+ve", "test", d), 1);
  EXPECT_EQ(Ax::get_property(input, "maxi8", "test", d), 127);
  EXPECT_EQ(Ax::get_property(input, "maxu8", "test", d), 255);
  EXPECT_EQ(Ax::get_property(input, "zero", "test", d), 0);
  EXPECT_EQ(Ax::get_property(input, "missing", "test", d), d);
  EXPECT_THROW(Ax::get_property(input, "-ve", "test", d), std::runtime_error);
  EXPECT_THROW(Ax::get_property(input, "str", "test", d), std::runtime_error);
  EXPECT_THROW(Ax::get_property(input, "empty", "test", d), std::runtime_error);
}

TEST(ax_utils_tests, test_get_property_int8)
{
  const auto d = std::int8_t{ -42 };
  EXPECT_EQ(Ax::get_property(input, "+ve", "test", d), 1);
  EXPECT_EQ(Ax::get_property(input, "maxi8", "test", d), 127);
  EXPECT_EQ(Ax::get_property(input, "mini8", "test", d), -128);
  EXPECT_EQ(Ax::get_property(input, "-ve", "test", d), -1);
  EXPECT_EQ(Ax::get_property(input, "zero", "test", d), 0);
  EXPECT_EQ(Ax::get_property(input, "missing", "test", d), d);
  EXPECT_THROW(Ax::get_property(input, "maxu8", "test", d), std::runtime_error);
  EXPECT_THROW(Ax::get_property(input, "str", "test", d), std::runtime_error);
  EXPECT_THROW(Ax::get_property(input, "empty", "test", d), std::runtime_error);
}


TEST(ax_utils_tests, test_get_property_int)
{
  const auto d = int{ -42 };
  EXPECT_EQ(Ax::get_property(input, "+ve", "test", d), 1);
  EXPECT_EQ(Ax::get_property(input, "maxi8", "test", d), 127);
  EXPECT_EQ(Ax::get_property(input, "mini8", "test", d), -128);
  EXPECT_EQ(Ax::get_property(input, "-ve", "test", d), -1);
  EXPECT_EQ(Ax::get_property(input, "zero", "test", d), 0);
  EXPECT_EQ(Ax::get_property(input, "missing", "test", d), d);
  EXPECT_EQ(Ax::get_property(input, "maxu8", "test", d), 255);
  EXPECT_EQ(Ax::get_property(input, "maxint", "test", d), INT_MAX);
  EXPECT_THROW(Ax::get_property(input, "str", "test", d), std::runtime_error);
  EXPECT_THROW(Ax::get_property(input, "empty", "test", d), std::runtime_error);
}

TEST(ax_utils_tests, test_get_property_with_vector_ints)
{
  using v = std::vector<int>;
  EXPECT_EQ(Ax::get_property(input, "+ve", "test", v{}), v{ 1 });
  EXPECT_EQ(Ax::get_property(input, "maxi8", "test", v{}), v{ 127 });
  EXPECT_EQ(Ax::get_property(input, "mini8", "test", v{}), v{ -128 });
  EXPECT_EQ(Ax::get_property(input, "-ve", "test", v{}), v{ -1 });
  EXPECT_EQ(Ax::get_property(input, "zero", "test", v{}), v{ 0 });
  EXPECT_EQ(Ax::get_property(input, "missing", "test", v{}), v{});
  EXPECT_EQ(Ax::get_property(input, "empty", "test", v{}), v{});
  EXPECT_EQ(Ax::get_property(input, "ints", "test", v{}), v({ 1, 2, 3, 4 }));
  EXPECT_THROW(Ax::get_property(input, "str", "test", v{}), std::runtime_error);
}

TEST(ax_utils_tests, test_get_property_with_vector_vector_ints)
{
  using v = std::vector<std::vector<int>>;
  EXPECT_EQ(Ax::get_property(input, "+ve", "test", v{}), v{ { 1 } });
  EXPECT_EQ(Ax::get_property(input, "maxi8", "test", v{}), v{ { 127 } });
  EXPECT_EQ(Ax::get_property(input, "mini8", "test", v{}), v{ { -128 } });
  EXPECT_EQ(Ax::get_property(input, "-ve", "test", v{}), v{ { -1 } });
  EXPECT_EQ(Ax::get_property(input, "zero", "test", v{}), v{ { 0 } });
  EXPECT_EQ(Ax::get_property(input, "missing", "test", v{}), v{});
  EXPECT_EQ(Ax::get_property(input, "empty", "test", v{}), v{});
  EXPECT_EQ(Ax::get_property(input, "ints", "test", v{}), (v{ { 1, 2, 3, 4 } }));
  EXPECT_THROW(Ax::get_property(input, "str", "test", v{}), std::runtime_error);

  const std::unordered_map<std::string, std::string> input1 = {
    { "ints", "1,2,3|4,5,6,7|8" },
  };
  EXPECT_EQ(Ax::get_property(input1, "ints", "test", v{}),
      (v{ { 1, 2, 3 }, { 4, 5, 6, 7 }, { 8 } }));
}

TEST(ax_utils_tests, test_get_optional_propery)
{
  const auto d = std::optional<std::int8_t>{ -42 };
  EXPECT_EQ(*Ax::get_property(input, "+ve", "test", d), 1);
  EXPECT_EQ(*Ax::get_property(input, "maxi8", "test", d), 127);
  EXPECT_EQ(*Ax::get_property(input, "mini8", "test", d), -128);
  EXPECT_EQ(*Ax::get_property(input, "-ve", "test", d), -1);
  EXPECT_EQ(*Ax::get_property(input, "zero", "test", d), 0);
  EXPECT_THROW(*Ax::get_property(input, "str", "test", d), std::runtime_error);
  EXPECT_THROW(*Ax::get_property(input, "empty", "test", d), std::runtime_error);

  EXPECT_EQ(*Ax::get_property(input, "missing", "test", d), d);
  const auto disengaged = std::optional<std::int8_t>{ std::nullopt };
  EXPECT_EQ(Ax::get_property(input, "missing", "test", disengaged), disengaged);
}

TEST(ax_streamer_utils, trim)
{
  using ax_utils::trim;
  EXPECT_EQ("", trim(""));
  EXPECT_EQ("", trim(" "));
  EXPECT_EQ("", trim("  "));
  EXPECT_EQ("", trim("\t"));
  EXPECT_EQ("", trim("\t "));
  EXPECT_EQ("", trim(" \t"));
  EXPECT_EQ("", trim(" \t "));
  EXPECT_EQ("a", trim("a"));
  EXPECT_EQ("a", trim(" a"));
  EXPECT_EQ("a", trim("a "));
  EXPECT_EQ("a", trim(" a "));
  EXPECT_EQ("a", trim("\ta"));
  EXPECT_EQ("a", trim("a\t"));
  EXPECT_EQ("a", trim("\ta\t"));
  EXPECT_EQ("a", trim(" a\t"));
  EXPECT_EQ("a", trim("\ta "));
  EXPECT_EQ("a", trim(" a "));
  EXPECT_EQ("a", trim("\t a"));
  EXPECT_EQ("a", trim(" \ta"));
  EXPECT_EQ("a", trim("\t a\t"));
  EXPECT_EQ("a", trim(" \ta "));
  EXPECT_EQ("a", trim(" \t a"));
  EXPECT_EQ("a", trim(" \t a \t"));
}

TEST(ax_oputils_tests, test_indices_for_topk)
{
  auto image_width = 40;
  auto image_height = 20;

  std::vector<box_xyxy> bboxes = {
    { image_width / 2 + 10, image_height / 2, image_width / 2 + 11, image_height / 2 + 1 },
    { image_width / 2, image_height / 2 + 6, image_width / 2 + 1, image_height / 2 + 7 },
    { image_width / 2, image_height / 2, image_width / 2 + 20, image_height / 2 + 12 },
  };

  auto result_c = ax_utils::indices_for_topk_center(bboxes, 1, image_width, image_height);
  auto result_a = ax_utils::indices_for_topk_area(bboxes, 1);
  ASSERT_EQ(result_c, std::vector<int>{ 1 });
  ASSERT_EQ(result_a, std::vector<int>{ 2 });

  std::vector<float> scores = { 0.1, 0.5, 0.3 };
  auto result_s_topk = ax_utils::indices_for_topk(scores, 5);
  auto result_c_topk
      = ax_utils::indices_for_topk_center(bboxes, 5, image_width, image_height);
  auto result_a_topk = ax_utils::indices_for_topk_area(bboxes, 5);
  std::vector<int> expected_s = { 1, 2, 0 }; // indices sorted by scores
  std::vector<int> expected_c = { 1, 0, 2 }; // indices sorted by distance to center
  std::vector<int> expected_a = { 2, 0, 1 }; // indices sorted by area
  ASSERT_EQ(result_s_topk, expected_s);
  ASSERT_EQ(result_c_topk, expected_c);
  ASSERT_EQ(result_a_topk, expected_a);
}

const std::unordered_map<std::string, std::string> properties = {
  { "test", "test" },
  { "int8_t", "-1" },
  { "int", "128" },
  { "-ve_int", "-129" },
  { "txt", "text" },
  { "uint", "txt" },
  { "vec", "1,2,3" },
  { "vec_vec", "1,2,3|4,5,6" },
};

template <typename T>
std::string
get_error_msg(std::string property, T default_value)
{
  try {
    Ax::get_property(properties, property, "test", default_value);
    return "Did not throw";
  } catch (const std::runtime_error &e) {
    return e.what();
  }
}

TEST(axutils, property_throw_message)
{
  ASSERT_EQ(get_error_msg("test", std::string{}), "Did not throw");
  ASSERT_EQ(get_error_msg("int8_t", uint8_t{}),
      "test : int8_t cannot be converted from '-1' to a type of uint8");
  ASSERT_EQ(get_error_msg("int", int8_t{}),
      "test : int cannot be converted from '128' to a type of int8");
  ASSERT_EQ(get_error_msg("-ve_int", int8_t{}),
      "test : -ve_int cannot be converted from '-129' to a type of int8");
  ASSERT_EQ(get_error_msg("txt", int8_t{}),
      "test : txt cannot be converted from 'text' to a type of int8");
  ASSERT_EQ(get_error_msg("uint", uint32_t{}),
      "test : uint cannot be converted from 'txt' to a type of unsigned int");
  ASSERT_EQ(get_error_msg("uint", int32_t{}),
      "test : uint cannot be converted from 'txt' to a type of int");
  ASSERT_EQ(get_error_msg("uint", std::vector<int32_t>{}),
      "test : uint cannot be converted from 'txt' to a type of std::vector<int>");
  ASSERT_EQ(get_error_msg("uint", std::vector<std::vector<int32_t>>{}),
      "test : uint cannot be converted from 'txt' to a type of std::vector<std::vector<int>>");
}

TEST(scale_boxes, test_scale_boxes)
{
  const auto boxes = std::vector<ax_utils::fbox>{
    { 0.0F, 0.0F, 1.0F, 1.0F },
  };

  AxVideoInterface video_info{ { 1280, 960, 1280 * 4, 0, AxVideoFormat::RGBA }, nullptr };

  const auto model_width = 640;
  const auto model_height = 480;
  const auto scale_up = true;
  auto v_info = BboxXyxy{ 0, 0, video_info.info.width - 1, video_info.info.height - 1 };
  const auto result = ax_utils::scale_shift_boxes(
      boxes, v_info, model_width, model_height, scale_up, true);
  const auto expected = std::vector<box_xyxy>{
    { 0, 0, 1279, 959 },
  };
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0].x1, expected[0].x1);
  ASSERT_EQ(result[0].y1, expected[0].y1);
  ASSERT_EQ(result[0].x2, expected[0].x2);
  ASSERT_EQ(result[0].y2, expected[0].y2);
}


TEST(scale_boxes, test_scale_portrait_boxes)
{
  const auto boxes = std::vector<ax_utils::fbox>{
    { 80.0F / 640.0F, 0.0F / 640.0F, 560.0F / 640.0F, 480.0F / 640.0F },
  };

  AxVideoInterface video_info{ { 960, 1280, 1280 * 4, 0, AxVideoFormat::RGBA }, nullptr };

  const auto model_width = 640;
  const auto model_height = 480;
  const auto scale_up = true;
  auto v_info = BboxXyxy{ 0, 0, video_info.info.width - 1, video_info.info.height - 1 };
  const auto result = ax_utils::scale_shift_boxes(
      boxes, v_info, model_width, model_height, scale_up, true);
  const auto expected = std::vector<box_xyxy>{
    { 0, 0, 959, 1279 },
  };
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0].x1, expected[0].x1);
  ASSERT_EQ(result[0].y1, expected[0].y1);
  ASSERT_EQ(result[0].x2, expected[0].x2);
  ASSERT_EQ(result[0].y2, expected[0].y2);
}

TEST(scale_boxes, test_scale_portrait_boxes_central)
{
  const auto boxes = std::vector<ax_utils::fbox>{
    { 80.0F / 640.0F, 80.0F / 640.0F, 560.0F / 640.0F, 400.0F / 640.0F },
  };

  AxVideoInterface video_info{ { 960, 1280, 1280 * 4, 0, AxVideoFormat::RGBA }, nullptr };

  const auto model_width = 640;
  const auto model_height = 480;
  const auto scale_up = true;
  auto v_info = BboxXyxy{ 0, 0, video_info.info.width - 1, video_info.info.height - 1 };
  const auto result = ax_utils::scale_shift_boxes(
      boxes, v_info, model_width, model_height, scale_up, true);
  const auto expected = std::vector<box_xyxy>{
    { 0, 213, 959, 1067 },
  };
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0].x1, expected[0].x1);
  ASSERT_EQ(result[0].y1, expected[0].y1);
  ASSERT_EQ(result[0].x2, expected[0].x2);
  ASSERT_EQ(result[0].y2, expected[0].y2);
}

TEST(ax_oputils_tests, test_get_meta)
{
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> meta_map{};
  auto meta_map_value = std::make_unique<AxMetaTracker>();
  auto *meta_map_value_ptr = meta_map_value.get();
  meta_map["test"] = std::move(meta_map_value);
  meta_map["unassigned"];

  EXPECT_THROW(ax_utils::get_meta<AxMetaObjDetection>("test", meta_map), std::runtime_error);
  EXPECT_THROW(ax_utils::get_meta<AxMetaTracker>("non_existing", meta_map), std::runtime_error);
  EXPECT_THROW(ax_utils::get_meta<AxMetaTracker>("unassigned", meta_map), std::runtime_error);
  EXPECT_EQ(ax_utils::get_meta<AxMetaTracker>("test", meta_map), meta_map_value_ptr);
}

TEST(ax_oputils_tests, test_insert_and_associate_meta)
{
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> meta_map{};

  auto meta_map_value_master_detections = std::make_unique<AxMetaObjDetection>(
      std::vector<box_xyxy>{ { 10, 10, 20, 20 }, { 30, 30, 40, 40 },
          { 50, 50, 60, 60 }, { 70, 70, 80, 80 } },
      std::vector<float>{ 1.0, 1.0, 1.0, 1.0 }, std::vector<int>{ 0, 0, 0, 0 });

  auto meta_map_value_associate_detections = std::make_unique<AxMetaObjDetection>(
      std::vector<box_xyxy>{ { 70, 70, 80, 80 }, { 30, 30, 40, 40 } },
      std::vector<float>{ 1.0, 1.0 }, std::vector<int>{ 0, 0 });
  int index_of_0th_associate_box_in_master = 3;
  int index_of_1st_associate_box_in_master = 1;
  meta_map_value_associate_detections->set_id(0, index_of_0th_associate_box_in_master);
  meta_map_value_associate_detections->set_id(1, index_of_1st_associate_box_in_master);
  meta_map["master"] = std::move(meta_map_value_master_detections);
  meta_map["associate"] = std::move(meta_map_value_associate_detections);

  ax_utils::insert_and_associate_meta<AxMetaObjDetection>(meta_map, "subkey",
      "master", 0, 2, "associate", std::vector<box_xyxy>{ { 77, 77, 79, 79 } },
      std::vector<float>{ 1.0 }, std::vector<int>{ 0 });
  ax_utils::insert_and_associate_meta<AxMetaObjDetection>(meta_map, "subkey",
      "master", 1, 2, "associate", std::vector<box_xyxy>{ { 33, 33, 36, 36 } },
      std::vector<float>{ 1.0 }, std::vector<int>{ 0 });

  auto submetas = meta_map["master"]->get_submetas<AxMetaObjDetection>("subkey");

  EXPECT_EQ(submetas.size(), 4);
  EXPECT_EQ(submetas[0], nullptr);
  EXPECT_EQ(submetas[1]->get_box_xyxy(0).x1, 33);
  EXPECT_EQ(submetas[1]->get_box_xyxy(0).y1, 33);
  EXPECT_EQ(submetas[1]->get_box_xyxy(0).x2, 36);
  EXPECT_EQ(submetas[1]->get_box_xyxy(0).y2, 36);
  EXPECT_EQ(submetas[2], nullptr);
  EXPECT_EQ(submetas[3]->get_box_xyxy(0).x1, 77);
  EXPECT_EQ(submetas[3]->get_box_xyxy(0).y1, 77);
  EXPECT_EQ(submetas[3]->get_box_xyxy(0).x2, 79);
  EXPECT_EQ(submetas[3]->get_box_xyxy(0).y2, 79);
}
