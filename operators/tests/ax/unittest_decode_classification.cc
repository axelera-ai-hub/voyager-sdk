// Copyright Axelera AI, 2023
#include "gmock/gmock.h"
#include "unittest_decode_common.h"

std::pair<std::vector<std::vector<int32_t>>, std::vector<std::vector<float>>>
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

  std::vector<std::vector<float>> actual_scores;
  std::vector<std::vector<int32_t>> actual_idx;

  for (const auto &meta : actual_metadata) {
    std::string type = reinterpret_cast<const char *>(meta.subtype);
    if (type == "scores") {
      auto p_scores = reinterpret_cast<const float *>(meta.meta);
      actual_scores.emplace_back(p_scores, p_scores + meta.meta_size / sizeof(float));
    } else if (type == "classes") {
      auto p_idx = reinterpret_cast<const int32_t *>(meta.meta);
      actual_idx.emplace_back(p_idx, p_idx + meta.meta_size / sizeof(int32_t));
    }
  }

  return { actual_idx, actual_scores };
}


TEST(no_softmax, topk_1_should_return_highest_scoring_index)
{
  std::vector<float> scores = { 0.1F, 0.2F, 0.3F, 0.4F, 0.8F };
  std::string meta_identifier = "top_k";

  std::unordered_map<std::string, std::string> input
      = { { "meta_key", meta_identifier }, { "top_k", "1" }, { "softmax", "0" } };
  Decoder decoder("libdecode_classification.so", input);
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> metadata;
  auto tensors = tensors_from_vector(scores);
  AxVideoInterface video_info{ { 640, 480, 640, 0, AxVideoFormat::RGB }, nullptr };
  decoder.decode_to_meta(tensors, 0, 1, metadata, video_info);
  auto [actual_idx, actual_score] = get_meta(metadata, meta_identifier);

  std::vector<std::vector<int>> expected_idx = { { 4 } };
  std::vector<std::vector<float>> expected_score = { { 0.8f } };

  EXPECT_EQ(expected_idx, actual_idx);
  EXPECT_EQ(expected_score, actual_score);
}

TEST(no_softmax, topk_3_should_return_3_highest_scoring_indices)
{
  std::vector<float> scores = { 0.1F, 0.2F, 0.4F, 0.3F, 0.8F };
  std::string meta_identifier = "top_k";

  std::unordered_map<std::string, std::string> input = { { "meta_key", meta_identifier },
    { "top_k", "3" }, { "softmax", "0" }, { "sorted", "0" } };
  Decoder decoder("libdecode_classification.so", input);

  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> metadata;
  auto tensors = tensors_from_vector(scores);

  AxVideoInterface video_info{ { 640, 480, 640, 0, AxVideoFormat::RGB }, nullptr };

  decoder.decode_to_meta(tensors, 0, 1, metadata, video_info);

  auto [actual_idx, actual_score] = get_meta(metadata, meta_identifier);

  std::vector<std::vector<int>> expected_idx = { { 4, 2, 3 } };
  std::vector<std::vector<float>> expected_score = { { 0.8F, 0.4F, 0.3F } };

  EXPECT_TRUE(std::is_permutation(
      expected_idx.begin(), expected_idx.end(), actual_idx.begin()));
  EXPECT_TRUE(std::is_permutation(
      expected_score.begin(), expected_score.end(), actual_score.begin()));
}

TEST(no_softmax, sorted_topk_3_should_return_3_highest_scoring_indices_in_order)
{
  std::vector<float> scores = { 0.1F, 0.2F, 0.4F, 0.3F, 0.8F };
  std::string meta_identifier = "top_k";

  std::unordered_map<std::string, std::string> input = { { "meta_key", meta_identifier },
    { "top_k", "3" }, { "softmax", "0" }, { "sorted", "1" } };
  Decoder decoder("libdecode_classification.so", input);
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> metadata;
  auto tensors = tensors_from_vector(scores);
  AxVideoInterface video_info{ { 640, 480, 640, 0, AxVideoFormat::RGB }, nullptr };
  decoder.decode_to_meta(tensors, 0, 1, metadata, video_info);

  auto [actual_idx, actual_score] = get_meta(metadata, meta_identifier);

  std::vector<std::vector<int>> expected_idx = { { 4, 2, 3 } };
  std::vector<std::vector<float>> expected_score = { { 0.8F, 0.4F, 0.3F } };

  EXPECT_EQ(expected_idx, actual_idx);
  EXPECT_EQ(expected_score, actual_score);
}

TEST(no_softmax, bottom_k_1_should_return_lowest_scoring_index)
{
  std::vector<float> scores = { 0.1F, 0.2F, 0.01F, 0.4F, 0.8F };
  std::string meta_identifier = "top_k";


  std::unordered_map<std::string, std::string> input = { { "meta_key", meta_identifier },
    { "top_k", "1" }, { "softmax", "0" }, { "largest", "0" } };
  Decoder decoder("libdecode_classification.so", input);
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> metadata;
  auto tensors = tensors_from_vector(scores);

  AxVideoInterface video_info{ { 640, 480, 640, 0, AxVideoFormat::RGB }, nullptr };
  decoder.decode_to_meta(tensors, 0, 1, metadata, video_info);

  auto [actual_idx, actual_score] = get_meta(metadata, meta_identifier);

  std::vector<std::vector<int>> expected_idx = { { 2 } };
  std::vector<std::vector<float>> expected_score = { { 0.01f } };

  EXPECT_EQ(expected_idx, actual_idx);
  EXPECT_EQ(expected_score, actual_score);
}

TEST(no_softmax, bottom_k_3_should_return_3_lowest_scoring_indices)
{
  std::vector<float> scores = { 0.1F, 0.2F, 0.4F, 0.3F, 0.8F };
  std::string meta_identifier = "top_k";

  std::unordered_map<std::string, std::string> input = { { "meta_key", meta_identifier },
    { "top_k", "3" }, { "softmax", "0" }, { "largest", "0" }, { "sorted", "0" } };
  Decoder decoder("libdecode_classification.so", input);
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> metadata;
  auto tensors = tensors_from_vector(scores);

  AxVideoInterface video_info{ { 640, 480, 640, 0, AxVideoFormat::RGB }, nullptr };
  decoder.decode_to_meta(tensors, 0, 1, metadata, video_info);

  auto [actual_idx, actual_score] = get_meta(metadata, meta_identifier);

  std::vector<std::vector<int>> expected_idx = { { 0, 1, 3 } };
  std::vector<std::vector<float>> expected_score = { { 0.3F, 0.2F, 0.1F } };

  EXPECT_TRUE(std::is_permutation(
      expected_idx[0].begin(), expected_idx[0].end(), actual_idx[0].begin()));
  EXPECT_TRUE(std::is_permutation(expected_score[0].begin(),
      expected_score[0].end(), actual_score[0].begin()));
}

TEST(no_softmax, sorted_bottom_k_3_should_return_3_lowest_scoring_indices_in_order)
{
  std::vector<float> scores = { 0.1F, 0.2F, 0.4F, 0.3F, 0.8F };
  std::string meta_identifier = "top_k";

  std::unordered_map<std::string, std::string> input = { { "meta_key", meta_identifier },
    { "top_k", "3" }, { "softmax", "0" }, { "largest", "0" }, { "sorted", "0" } };
  Decoder decoder("libdecode_classification.so", input);
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> metadata;
  auto tensors = tensors_from_vector(scores);

  AxVideoInterface video_info{ { 640, 480, 640, 0, AxVideoFormat::RGB }, nullptr };
  decoder.decode_to_meta(tensors, 0, 1, metadata, video_info);

  auto [actual_idx, actual_score] = get_meta(metadata, meta_identifier);

  std::vector<std::vector<int>> expected_idx = { { 0, 1, 3 } };
  std::vector<std::vector<float>> expected_score = { { 0.1F, 0.2F, 0.3F } };

  EXPECT_EQ(expected_idx, actual_idx);
  EXPECT_EQ(expected_score, actual_score);
}

TEST(softmax, topk_3_should_return_highest_softmaxed_scores)
{
  std::vector<float> scores = { 0.5F, 0.4F, 0.0F, 1.0F, 0.1F };
  std::string meta_identifier = "top_k";

  std::unordered_map<std::string, std::string> input
      = { { "meta_key", meta_identifier }, { "top_k", "3" }, { "softmax", "1" } };
  Decoder decoder("libdecode_classification.so", input);
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> metadata;
  auto tensors = tensors_from_vector(scores);

  AxVideoInterface video_info{ { 640, 480, 640, 0, AxVideoFormat::RGB }, nullptr };
  decoder.decode_to_meta(tensors, 0, 1, metadata, video_info);

  auto [actual_idx, actual_score] = get_meta(metadata, meta_identifier);

  std::vector<std::vector<int>> expected_idx = { { 3, 0, 1 } };
  std::vector<float> expected_scores = { 0.34132123F, 0.20702179F, 0.187321F };
  EXPECT_EQ(expected_idx, actual_idx);
  EXPECT_FLOAT_EQ(expected_scores[0], actual_score[0][0]);
  EXPECT_FLOAT_EQ(expected_scores[1], actual_score[0][1]);
  EXPECT_FLOAT_EQ(expected_scores[2], actual_score[0][2]);
}

TEST(no_softmax, test_empty_labels)
{
  std::vector<float> scores = { 0.5F, 0.4F, 0.0F, 1.0F, 0.1F };
  std::string meta_identifier = "top_k";

  auto labels_file = tempfile("");

  std::unordered_map<std::string, std::string> input
      = { { "meta_key", meta_identifier }, { "top_k", "3" }, { "softmax", "1" },
          { "classlabels_file", labels_file.filename() } };
  Decoder decoder("libdecode_classification.so", input);
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> metadata;
  auto tensors = tensors_from_vector(scores);

  AxVideoInterface video_info{ { 640, 480, 640, 0, AxVideoFormat::RGB }, nullptr };
  decoder.decode_to_meta(tensors, 0, 1, metadata, video_info);

  auto position = metadata.find(meta_identifier);
  ASSERT_NE(position, metadata.end());
  auto *meta = dynamic_cast<AxMetaClassification *>(position->second.get());
  ASSERT_NE(meta, nullptr);
  auto labels = meta->get_labels();

  EXPECT_EQ(labels.size(), 1);
  std::vector<std::string> expected_labels = { "Class: 3", "Class: 0", "Class: 1" };
  EXPECT_EQ(expected_labels, labels[0]);
}

TEST(no_softmax, test_labels)
{
  std::vector<float> scores = { 0.5F, 0.4F, 0.0F, 1.0F, 0.1F };
  std::string meta_identifier = "top_k";

  auto labels_file = tempfile("Beaver\nDog\nPony\nLion\nKettle\n");
  std::unordered_map<std::string, std::string> input
      = { { "meta_key", meta_identifier }, { "top_k", "3" }, { "softmax", "1" },
          { "classlabels_file", labels_file.filename() } };
  Decoder decoder("libdecode_classification.so", input);
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> metadata;
  auto tensors = tensors_from_vector(scores);

  AxVideoInterface video_info{ { 640, 480, 640, 0, AxVideoFormat::RGB }, nullptr };
  decoder.decode_to_meta(tensors, 0, 1, metadata, video_info);

  auto position = metadata.find(meta_identifier);
  ASSERT_NE(position, metadata.end());
  auto *meta = dynamic_cast<AxMetaClassification *>(position->second.get());
  ASSERT_NE(meta, nullptr);
  auto labels = meta->get_labels();

  EXPECT_EQ(labels.size(), 1);
  std::vector<std::string> expected_labels = { "Lion", "Beaver", "Dog" };
  EXPECT_EQ(expected_labels, labels[0]);
}

TEST(no_softmax, test_labels_size_too_small)
{
  std::vector<float> scores = { 0.5F, 0.4F, 0.0F, 1.0F, 0.1F };
  std::string meta_identifier = "top_k";

  auto labels_file = tempfile("Beaver\nDog\nPony\nLion\n");
  std::unordered_map<std::string, std::string> input
      = { { "meta_key", meta_identifier }, { "top_k", "3" }, { "softmax", "1" },
          { "classlabels_file", labels_file.filename() } };
  Decoder decoder("libdecode_classification.so", input);
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> metadata;
  auto tensors = tensors_from_vector(scores);

  AxVideoInterface video_info{ { 640, 480, 640, 0, AxVideoFormat::RGB }, nullptr };
  //  Should not throw
  decoder.decode_to_meta(tensors, 0, 1, metadata, video_info);
}

TEST(no_softmax, test_labels_size_too_large)
{
  std::vector<float> scores = { 0.5F, 0.4F, 0.0F };
  std::string meta_identifier = "top_k";

  auto labels_file = tempfile("Beaver\nDog\nPony\nLion\n");
  std::unordered_map<std::string, std::string> input
      = { { "meta_key", meta_identifier }, { "top_k", "1" }, { "softmax", "1" },
          { "classlabels_file", labels_file.filename() } };
  Decoder decoder("libdecode_classification.so", input);
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> metadata;
  auto tensors = tensors_from_vector(scores);

  AxVideoInterface video_info{ { 640, 480, 640, 0, AxVideoFormat::RGB }, nullptr };
  //  Should not throw
  decoder.decode_to_meta(tensors, 0, 1, metadata, video_info);
}

TEST(classification, throws_on_invalid_subframe)
{
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> metadata;
  std::vector<float> scores = { 0.5F, 0.4F, 0.0F, 1.0F, 0.1F };
  auto tensors = tensors_from_vector(scores);
  std::string meta_identifier = "top_k";
  std::unordered_map<std::string, std::string> input
      = { { "meta_key", meta_identifier }, { "top_k", "3" }, { "softmax", "1" } };
  Decoder decoder("libdecode_classification.so", input);
  AxVideoInterface video_info{ { 640, 480, 640, 0, AxVideoFormat::RGB }, nullptr };
  ASSERT_THROW(decoder.decode_to_meta(tensors, 2, 2, metadata, video_info), std::runtime_error);
}

TEST(classification, throws_on_invalid_subframe_stream)
{
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> metadata;
  std::vector<float> scores = { 0.5F, 0.4F, 0.0F, 1.0F, 0.1F };
  auto tensors = tensors_from_vector(scores);
  std::string meta_identifier = "top_k";
  std::unordered_map<std::string, std::string> input
      = { { "meta_key", meta_identifier }, { "top_k", "3" }, { "softmax", "1" } };
  Decoder decoder("libdecode_classification.so", input);
  AxVideoInterface video_info{ { 640, 480, 640, 0, AxVideoFormat::RGB }, nullptr };
  decoder.decode_to_meta(tensors, 0, 1, metadata, video_info);
  ASSERT_THROW(decoder.decode_to_meta(tensors, 1, 2, metadata, video_info), std::runtime_error);
}

TEST(softmax, out_of_order_frames)
{
  std::vector<float> scores0 = { 0.5F, 0.4F, 0.0F, 1.0F, 0.1F };
  std::vector<float> scores1 = { 0.1F, 0.5F, 0.4F, 0.0F, 0.2F };
  std::string meta_identifier = "top_k";

  std::unordered_map<std::string, std::string> input
      = { { "meta_key", meta_identifier }, { "top_k", "3" }, { "softmax", "1" } };
  Decoder decoder("libdecode_classification.so", input);
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> metadata;
  auto tensors0 = tensors_from_vector(scores0);
  auto tensors1 = tensors_from_vector(scores1);

  AxVideoInterface video_info{ { 640, 480, 640, 0, AxVideoFormat::RGB }, nullptr };
  decoder.decode_to_meta(tensors1, 1, 2, metadata, video_info);
  decoder.decode_to_meta(tensors0, 0, 2, metadata, video_info);

  auto [actual_idx, actual_score] = get_meta(metadata, meta_identifier);

  std::vector<std::vector<int>> expected_idx = {
    { 3, 0, 1 },
    { 1, 2, 4 },
  };
  EXPECT_EQ(expected_idx, actual_idx);
  EXPECT_THAT(actual_score, testing::ElementsAreArray({
                                testing::ElementsAreArray({
                                    testing::FloatEq(0.34132123F),
                                    testing::FloatEq(0.20702179F),
                                    testing::FloatEq(0.187321F),
                                }),
                                testing::ElementsAreArray({
                                    testing::FloatEq(0.25493905F),
                                    testing::FloatEq(0.23067838F),
                                    testing::FloatEq(0.18886349F),
                                }),
                            }));
}

TEST(softmax, in_order_frames)
{
  std::vector<float> scores0 = { 0.5F, 0.4F, 0.0F, 1.0F, 0.1F };
  std::vector<float> scores1 = { 0.1F, 0.5F, 0.4F, 0.0F, 0.2F };
  std::string meta_identifier = "top_k";

  std::unordered_map<std::string, std::string> input
      = { { "meta_key", meta_identifier }, { "top_k", "3" }, { "softmax", "1" } };
  Decoder decoder("libdecode_classification.so", input);
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> metadata;
  auto tensors0 = tensors_from_vector(scores0);
  auto tensors1 = tensors_from_vector(scores1);

  AxVideoInterface video_info{ { 640, 480, 640, 0, AxVideoFormat::RGB }, nullptr };
  decoder.decode_to_meta(tensors0, 0, 2, metadata, video_info);
  decoder.decode_to_meta(tensors1, 1, 2, metadata, video_info);

  auto [actual_idx, actual_score] = get_meta(metadata, meta_identifier);

  std::vector<std::vector<int>> expected_idx = {
    { 3, 0, 1 },
    { 1, 2, 4 },
  };
  EXPECT_EQ(expected_idx, actual_idx);
  EXPECT_THAT(actual_score, testing::ElementsAreArray({
                                testing::ElementsAreArray({
                                    testing::FloatEq(0.34132123F),
                                    testing::FloatEq(0.20702179F),
                                    testing::FloatEq(0.187321F),
                                }),
                                testing::ElementsAreArray({
                                    testing::FloatEq(0.25493905F),
                                    testing::FloatEq(0.23067838F),
                                    testing::FloatEq(0.18886349F),
                                }),
                            }));
}
