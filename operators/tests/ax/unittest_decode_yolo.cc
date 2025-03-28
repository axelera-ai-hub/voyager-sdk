// Copyright Axelera AI, 2024
#include "gtest/gtest.h"
#include <gmodule.h>
#include "gmock/gmock.h"
#include "unittest_decode_common.h"

#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <filesystem>
#include "AxDataInterface.h"
#include "AxLog.hpp"
#include "AxMeta.hpp"

namespace fs = std::filesystem;

namespace
{
std::string
get_file_dir()
{
  return fs::path(__FILE__).parent_path().string();
}

struct object_meta {
  std::vector<int32_t> boxes;
  std::vector<float> scores;
  std::vector<int32_t> classes;
};
object_meta
get_meta(const std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &map,
    std::string meta_identifier)
{
  auto position = map.find(meta_identifier);
  if (position == map.end()) {
    return { {}, {} };
  }
  auto *meta = position->second.get();
  EXPECT_NE(meta, nullptr);

  auto actual_metadata = meta->get_extern_meta();
  EXPECT_EQ(actual_metadata.size(), 3);

  auto p_boxes = reinterpret_cast<const int32_t *>(actual_metadata[0].meta);
  auto p_scores = reinterpret_cast<const float *>(actual_metadata[1].meta);
  auto p_classes = reinterpret_cast<const int32_t *>(actual_metadata[2].meta);
  auto actual_boxes = std::vector<int32_t>{ p_boxes,
    p_boxes + actual_metadata[0].meta_size / sizeof(int32_t) };
  auto actual_scores = std::vector<float>{ p_scores,
    p_scores + actual_metadata[1].meta_size / sizeof(float) };
  auto actual_classes = std::vector<int32_t>{ p_classes,
    p_classes + actual_metadata[2].meta_size / sizeof(int32_t) };

  return { actual_boxes, actual_scores, actual_classes };
}

template <typename T>
AxTensorsInterface
tensors_from_vector(std::vector<T> &tensors, std::vector<int> sizes)
{
  return {
    { sizes, sizeof tensors[0], tensors.data() },
  };
}

TEST(yolo_errors, no_zero_points_throws)
{
  std::unordered_map<std::string, std::string> properties = {
    { "scales", "1" },
  };
  EXPECT_THROW(Decoder("libdecode_yolo.so", properties), std::runtime_error);
}

TEST(yolo_errors, no_scales_throws)
{
  std::unordered_map<std::string, std::string> properties = {
    { "zero_points", "0" },
  };
  EXPECT_THROW(Decoder("libdecode_yolo.so", properties), std::runtime_error);
}

TEST(yolo_errors, different_scale_and_zero_point_sizes_throws)
{
  std::unordered_map<std::string, std::string> properties = {
    { "zero_points", "0, 0" },
    { "scales", "1" },
  };
  EXPECT_THROW(Decoder("libdecode_yolo.so", properties), std::runtime_error);
}


TEST(yolo_errors, must_provide_onnx)
{
  std::unordered_map<std::string, std::string> properties = {
    { "zero_points", "0, 0" },
    { "scales", "1, 2" },
  };

  try {
    Decoder("libdecode_yolo.so", properties);
    FAIL() << "Expected std::runtime_error";
  } catch (std::runtime_error const &err) {
    EXPECT_EQ(err.what(),
        std::string("We don't expect using decode_yolo without an onnx for now."));
  } catch (...) {
    FAIL() << "Expected std::runtime_error";
  }
}

TEST(yolo_errors, must_provide_classes)
{
  ;

  std::unordered_map<std::string, std::string> properties = {
    { "zero_points", "0, 0" },
    { "scales", "1, 2" },
  };
  EXPECT_THROW(Decoder("libdecode_yolo.so", properties), std::runtime_error);
}


TEST(yolo_errors, onnx_must_exist)
{
  std::unordered_map<std::string, std::string> properties = {
    { "zero_points", "0, 0" },
    { "scales", "1, 2" },
    { "feature_decoder_onnx", "doesnt_exist.onnx" },
  };
  try {
    Decoder("libdecode_yolo.so", properties);
    FAIL() << "Expected std::runtime_error";
  } catch (std::runtime_error const &err) {
    EXPECT_EQ(err.what(),
        std::string("feature_decoder_onnx: doesnt_exist.onnx does not exist"));
  } catch (...) {
    FAIL() << "Expected std::runtime_error";
  }
}

TEST(yolo_errors, padding_length_must_all_be_8)
{
  std::unordered_map<std::string, std::string> properties = {
    { "zero_points", "0, 0" },
    { "scales", "1, 2" },
    { "feature_decoder_onnx", "doesnt_exist.onnx" },
    { "paddings", "0,0,0,0,0,0,0,1|0,0,0,0,0,0,0,2|0,0,0,2,0,1" },
  };
  try {
    Decoder("libdecode_yolo.so", properties);
    FAIL() << "Expected std::runtime_error";
  } catch (std::runtime_error const &err) {
    EXPECT_EQ(err.what(),
        std::string("yolo_decode_static_properties : Padding values must be 8."));
  } catch (...) {
    FAIL() << "Expected std::runtime_error";
  }
}

TEST(yolo_error, different_tensor_sizes_and_dequantize_tables_size)
{
  // when anchor_free, there is no class_confidence, so the scores
  // are class confidences
  std::vector<int8_t> coordinates = { 0, 0, 0, 0, 0, 0, 0, 0 };
  // according to zero_points and scales, the dequantized values:
  // 0.4284, 0.784, 0.1386, 0.2382; 0.8, 0.5, 0.2, 0.8
  std::vector<int8_t> confidences = { 80, 127, 32, 48, 28, 22, 16, 28 };
  std::vector<int> coordinates_shape = { 1, 4, 2, 1 };
  std::vector<int> confidences_shape = { 1, 3, 2, 1 };
  std::string meta_identifier = "yolov8";

  std::unordered_map<std::string, std::string> properties = {
    { "meta_key", meta_identifier },
    { "zero_points", "10" },
    { "scales", "0.0063" },
    { "paddings", "0,0,0,0,0,0,0,0" },
    { "confidence_threshold", "0.20" },
    { "classes", "4" }, // anchor free no object confidence
    { "multiclass", "0" },
    { "feature_decoder_onnx", get_file_dir() + "/assets/test_concat-1-8-10.onnx" },
  };
  Decoder decoder("libdecode_yolo.so", properties);

  AxVideoInterface video_info{ { 64, 48, 64, 0, AxVideoFormat::RGB }, nullptr };
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map{};
  auto tensors = tensors_from_vector(coordinates, coordinates_shape);
  auto tensors_conf = tensors_from_vector(confidences, confidences_shape);
  tensors.push_back(std::move(tensors_conf[0]));
  EXPECT_THROW(decoder.decode_to_meta(tensors, 0, 1, map, video_info), std::runtime_error);
}

std::vector<int8_t>
nchw_to_nhwc_with_padding(const std::vector<int8_t> &nchwData, int N, int C,
    int H, int W, int newC = -1)
{
  // If newC is not provided or less than C, no padding is needed. Use C as the channel dimension.
  if (newC < C || newC == -1) {
    newC = C;
  }

  std::vector<int8_t> nhwcData(N * H * W * newC,
      0); // Initialize with zeros, considering padding if newC > C

  for (int n = 0; n < N; ++n) {
    for (int h = 0; h < H; ++h) {
      for (int w = 0; w < W; ++w) {
        for (int c = 0; c < C; ++c) { // Only iterate over original C for copying data
          int nchwIndex = n * C * H * W + c * H * W + h * W + w;
          int nhwcIndex = n * H * W * newC + h * W * newC + w * newC
                          + c; // Use newC for calculating index
          nhwcData[nhwcIndex] = nchwData[nchwIndex];
        }
        // No need to explicitly pad with zeros since the vector is initialized with zeros
      }
    }
  }
  return nhwcData;
}

// a test fixture for buiding input tensors for yolo decoder,
// coordinates are all 0
class YoloDecodeScoresTest : public ::testing::Test
{
  protected:
  std::vector<int8_t> m_coordinates;
  std::vector<int8_t> m_confidences;
  std::vector<int> m_coordinates_shape;
  std::vector<int> m_confidences_shape;
  std::string m_zero_points = "16,10";
  std::string m_scales = "2.1,0.005";
  std::string m_paddings = "0,0,0,0,0,0,0,0|0,0,0,0,0,0,0,0";
  // YOLOv5's output is [N, num_of_samples, num_data_per_sample], we use 1-10-8
  // YOLOv8's output is [N, num_data_per_sample, num_of_samples], we use 1-8-10
  std::string m_test_onnx_v5 = get_file_dir() + "/assets/test_concat-1-10-8.onnx";
  std::string m_test_onnx_v8 = get_file_dir() + "/assets/test_concat-1-8-10.onnx";
  std::string m_classes_v5 = "3"; // anchor-based with object confidence
  std::string m_classes_v8 = "4"; // anchor-free without object confidence

  AxTensorsInterface m_tensors;
  AxVideoInterface m_video_info{ { 640, 480, 640, 0, AxVideoFormat::RGB }, nullptr };

  void CustomSetUp(bool padding = false)
  {
    // assume 10 samples, but use the first 2 samples only
    // the reason to have 10 samples it to have number of samples > number data
    // clang-format off
    // per sample 20 30 10 8; 30 20 8 10
    m_coordinates = std::vector<int8_t>{
      88, 110, 0, 0, 0, 0, 0, 0, 0, 0,
      127, 86, 0, 0, 0, 0, 0, 0, 0, 0,
      100, 26, 0, 0, 0, 0, 0, 0, 0, 0,
      80,  31, 0, 0, 0, 0, 0, 0, 0, 0,
    };

    // 0.585,0.55,0.115,0.21; 0.55,0.40,0.165,0.56
    m_confidences = std::vector<int8_t>{
      127, 120, 0, 0, 0, 0, 0, 0, 0, 0,
      120,  90, 0, 0, 0, 0, 0, 0, 0, 0,
      33,   43, 0, 0, 0, 0, 0, 0, 0, 0,
      52,  122, 0, 0, 0, 0, 0, 0, 0, 0,
    };
    // clang-format on
    if (padding) {
      std::vector<int8_t> transposed_confidences
          = nchw_to_nhwc_with_padding(m_confidences, 1, 1, 4, 10, 4);
      m_confidences = transposed_confidences;
      transposed_confidences = nchw_to_nhwc_with_padding(m_coordinates, 1, 1, 4, 10, 4);
      m_coordinates = transposed_confidences;
      m_coordinates_shape = { 1, 4, 10, 4 };
      m_confidences_shape = { 1, 4, 10, 4 };
      m_paddings = "0,0,0,0,0,0,0,3|0,0,0,0,0,0,0,3";
    } else {
      std::vector<int8_t> transposed_confidences
          = nchw_to_nhwc_with_padding(m_confidences, 1, 1, 4, 10);
      m_confidences = transposed_confidences;
      transposed_confidences = nchw_to_nhwc_with_padding(m_coordinates, 1, 1, 4, 10);
      m_coordinates = transposed_confidences;
      m_coordinates_shape = { 1, 4, 10, 1 };
      m_confidences_shape = { 1, 4, 10, 1 };
    }

    m_tensors = tensors_from_vector(m_coordinates, m_coordinates_shape);
    auto tensors_conf = tensors_from_vector(m_confidences, m_confidences_shape);
    m_tensors.push_back(std::move(tensors_conf[0]));
  }

  void SetUp() override
  {
    CustomSetUp();
  }
};

TEST_F(YoloDecodeScoresTest, unsupported_output_channels_excluding_classes)
{
  std::string meta_identifier = "yolov8";

  std::unordered_map<std::string, std::string> properties = {
    { "meta_key", meta_identifier },
    { "zero_points", m_zero_points },
    { "scales", m_scales },
    { "paddings", m_paddings },
    { "confidence_threshold", "0.20" },
    { "classes", "2" }, // this is not possible
    { "multiclass", "0" },
    { "feature_decoder_onnx", m_test_onnx_v8 },
  };
  Decoder decoder("libdecode_yolo.so", properties);

  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map{};
  try {
    decoder.decode_to_meta(m_tensors, 0, 1, map, m_video_info);
    FAIL() << "Expected std::runtime_error, but got nothing";
  } catch (std::runtime_error const &err) {
    EXPECT_EQ(err.what(),
        std::string("Unexpected number of output channels excluding classes: 6"));
  } catch (...) {
    FAIL() << "Expected std::runtime_error";
  }
}

TEST_F(YoloDecodeScoresTest, depadding)
{
  CustomSetUp(true);
  std::string meta_identifier = "yolov8";


  std::unordered_map<std::string, std::string> properties = {
    { "meta_key", meta_identifier },
    { "zero_points", m_zero_points },
    { "scales", m_scales },
    { "paddings", m_paddings },
    { "confidence_threshold", "0.20" },
    { "classes", "4" },
    { "multiclass", "0" },
    { "feature_decoder_onnx", m_test_onnx_v8 },
  };
  Decoder decoder("libdecode_yolo.so", properties);

  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map{};
  decoder.decode_to_meta(m_tensors, 0, 1, map, m_video_info);

  auto [actual_boxes, actual_scores, actual_classes] = get_meta(map, meta_identifier);

  // print out actual_scores, actual_classes with gtest
  auto expected_classes = std::vector<int32_t>{ 0, 3 };
  auto expected_scores = std::vector<float>{ 0.585F, 0.56F };
  ASSERT_EQ(actual_classes, expected_classes);
  EXPECT_FLOAT_EQ(actual_scores[0], expected_scores[0]);
  EXPECT_FLOAT_EQ(actual_scores[1], expected_scores[1]);
}

TEST_F(YoloDecodeScoresTest, multiclass)
{
  std::string meta_identifier = "yolov8";

  std::unordered_map<std::string, std::string> properties = {
    { "meta_key", meta_identifier },
    { "zero_points", m_zero_points },
    { "scales", m_scales },
    { "paddings", m_paddings },
    { "confidence_threshold", "0.45" },
    { "classes", "4" },
    { "multiclass", "1" },
    { "feature_decoder_onnx", m_test_onnx_v8 },
  };
  Decoder decoder("libdecode_yolo.so", properties);

  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map{};
  decoder.decode_to_meta(m_tensors, 0, 1, map, m_video_info);

  auto [actual_boxes, actual_scores, actual_classes] = get_meta(map, meta_identifier);
  auto expected_classes = std::vector<int32_t>{ 0, 1, 0, 3 };
  auto expected_scores = std::vector<float>{ 0.585F, 0.55F, 0.55F, 0.56F };
  ASSERT_EQ(actual_classes, expected_classes);
  EXPECT_FLOAT_EQ(actual_scores[0], expected_scores[0]);
  EXPECT_FLOAT_EQ(actual_scores[1], expected_scores[1]);
}

TEST_F(YoloDecodeScoresTest, yolov8)
{
  std::string meta_identifier = "yolov8";

  std::unordered_map<std::string, std::string> properties = {
    { "meta_key", meta_identifier },
    { "zero_points", m_zero_points },
    { "scales", m_scales },
    { "paddings", m_paddings },
    { "confidence_threshold", "0.20" },
    { "classes", "4" },
    { "multiclass", "0" },
    { "model_width", "640" },
    { "model_height", "640" },
    { "feature_decoder_onnx", m_test_onnx_v8 },
  };
  Decoder decoder("libdecode_yolo.so", properties);

  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map{};
  decoder.decode_to_meta(m_tensors, 0, 1, map, m_video_info);

  auto [actual_boxes, actual_scores, actual_classes] = get_meta(map, meta_identifier);

  auto expected_boxes = std::vector<int32_t>{ 63, 86, 239, 220, 187, 51, 208, 83 };

  auto expected_classes = std::vector<int32_t>{ 0, 3 };
  auto expected_scores = std::vector<float>{ 0.585F, 0.56F };
  ASSERT_EQ(actual_classes, expected_classes);
  EXPECT_EQ(actual_boxes, expected_boxes);
}

TEST_F(YoloDecodeScoresTest, yolov5)
{
  std::string meta_identifier = "yolov5";

  std::unordered_map<std::string, std::string> properties = {
    { "meta_key", meta_identifier },
    { "zero_points", m_zero_points },
    { "scales", m_scales },
    { "paddings", m_paddings },
    { "confidence_threshold", "0.20" },
    { "classes", "3" },
    { "multiclass", "0" },
    { "model_width", "640" },
    { "model_height", "640" },
    { "feature_decoder_onnx", m_test_onnx_v5 },
  };
  Decoder decoder("libdecode_yolo.so", properties);

  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map{};
  decoder.decode_to_meta(m_tensors, 0, 1, map, m_video_info);

  auto [actual_boxes, actual_scores, actual_classes] = get_meta(map, meta_identifier);
  auto expected_boxes = std::vector<int32_t>{ 63, 86, 239, 220, 187, 51, 208, 83 };

  auto expected_classes = std::vector<int32_t>{ 0, 2 };
  auto expected_scores = std::vector<float>{ 0.32175F, 0.308F };
  ASSERT_EQ(actual_classes, expected_classes);
  EXPECT_EQ(actual_boxes, expected_boxes);
}

} // namespace
