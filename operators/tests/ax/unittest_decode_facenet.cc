// Copyright Axelera AI, 2025
#include <Eigen/Core>
#include <algorithm>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <fstream>
#include <nlohmann/json.hpp>
#include <numeric>
#include "AxMeta.hpp"
#include "AxMetaClassification.hpp"
#include "AxOpUtils.hpp"
#include "AxUtils.hpp"
#include "gmock/gmock.h"
#include "unittest_ax_common.h"

std::vector<std::vector<float>>
get_embeddings_meta(const std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &map,
    std::string meta_identifier)
{
  auto position = map.find(meta_identifier);
  if (position == map.end()) {
    return {};
  }
  auto *meta = position->second.get();
  EXPECT_NE(meta, nullptr);

  auto *embeddings_meta = dynamic_cast<AxMetaEmbeddings *>(meta);
  EXPECT_NE(embeddings_meta, nullptr);

  return embeddings_meta->get_embeddings();
}

std::pair<std::vector<std::string>, std::vector<float>>
get_classification_meta(
    const std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &map,
    std::string meta_identifier)
{
  auto position = map.find(meta_identifier);
  if (position == map.end()) {
    return { {}, {} };
  }
  auto *meta = position->second.get();
  EXPECT_NE(meta, nullptr);

  auto *classification_meta = dynamic_cast<AxMetaClassification *>(meta);
  EXPECT_NE(classification_meta, nullptr);

  auto labels = classification_meta->get_labels();
  auto scores = classification_meta->get_scores();

  return { labels.empty() ? std::vector<std::string>{} : labels[0],
    scores.empty() ? std::vector<float>{} : scores[0] };
}

class MockEigenMatrixProvider
{
  public:
  Eigen::MatrixXf create_test_embeddings(int rows, int cols)
  {
    Eigen::MatrixXf matrix(rows, cols);
    // Fill with some predictable values
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        matrix(i, j) = 0.1f * i + 0.01f * j;
      }
    }
    return matrix;
  }
};

TEST(AxOpUtils, EmbeddingsCosineSimilarityNormalise)
{
  Eigen::MatrixXf embeddings(2, 2);
  embeddings << 3.0f, 0.0f, 0.0f, 4.0f;

  std::vector<float> desc = { 6.0f, 0.0f };
  auto sims = ax_utils::embeddings_cosine_similarity(desc, embeddings, true);
  ASSERT_EQ(sims.size(), 2u);
  EXPECT_NEAR(sims[0], 1.0f, 1e-6f);
  EXPECT_NEAR(sims[1], 0.0f, 1e-6f);

  std::vector<float> zero_desc = { 0.0f, 0.0f };
  auto zero_sims = ax_utils::embeddings_cosine_similarity(zero_desc, embeddings, true);
  ASSERT_EQ(zero_sims.size(), 2u);
  EXPECT_NEAR(zero_sims[0], 0.0f, 1e-6f);
  EXPECT_NEAR(zero_sims[1], 0.0f, 1e-6f);
}

TEST(AxOpUtils, EmbeddingsEuclideanDistanceNoNormalise)
{
  Eigen::MatrixXf embeddings(3, 2);
  embeddings << 1.0f, 2.0f, 2.0f, 2.0f, 4.0f, 6.0f;

  std::vector<float> desc = { 1.0f, 2.0f };
  auto distances = ax_utils::embeddings_euclidean_distance(desc, embeddings, false);

  ASSERT_EQ(distances.size(), 3u);
  EXPECT_NEAR(distances[0], 0.0f, 1e-6f);
  EXPECT_NEAR(distances[1], 1.0f, 1e-6f);
  EXPECT_NEAR(distances[2], 5.0f, 1e-6f);
}

TEST(AxOpUtils, EmbeddingsEuclideanDistanceNormalise)
{
  Eigen::MatrixXf embeddings(3, 2);
  embeddings << 3.0f, 0.0f, 0.0f, 4.0f, 0.0f, 0.0f;

  std::vector<float> desc = { 6.0f, 0.0f };
  auto distances = ax_utils::embeddings_euclidean_distance(desc, embeddings, true);

  ASSERT_EQ(distances.size(), 3u);
  EXPECT_NEAR(distances[0], 0.0f, 1e-6f);
  EXPECT_NEAR(distances[1], std::sqrt(2.0f), 1e-6f);
  EXPECT_NEAR(distances[2], 1.0f, 1e-6f);

  std::vector<float> zero_desc = { 0.0f, 0.0f };
  auto zero_distances
      = ax_utils::embeddings_euclidean_distance(zero_desc, embeddings, true);
  ASSERT_EQ(zero_distances.size(), 3u);
  EXPECT_NEAR(zero_distances[0], 1.0f, 1e-6f);
  EXPECT_NEAR(zero_distances[1], 1.0f, 1e-6f);
  EXPECT_NEAR(zero_distances[2], 0.0f, 1e-6f);
}

TEST(AxOpUtils, EmbeddingsCosineDistance)
{
  Eigen::MatrixXf embeddings(4, 2);
  embeddings.row(0) << 2.0f, 1.0f;
  embeddings.row(1) << 2.0f, -1.0f;
  embeddings.row(2) << 1.0f, 0.0f;
  embeddings.row(3) << 0.0f, 2.0f;

  std::vector<float> desc = { 1.0f, 2.0f };
  auto distances = ax_utils::embeddings_cosine_distance(desc, embeddings);

  ASSERT_EQ(distances.size(), 4u);
  const float inv_sqrt5 = 1.0f / std::sqrt(5.0f);
  const float kPi = static_cast<float>(M_PI);
  EXPECT_NEAR(distances[0], std::acos(0.8f) / kPi, 1e-5f);
  EXPECT_NEAR(distances[1], 0.5f, 1e-5f);
  EXPECT_NEAR(distances[2], std::acos(inv_sqrt5) / kPi, 1e-5f);
  EXPECT_NEAR(distances[3], std::acos(2.0f * inv_sqrt5) / kPi, 1e-5f);
}

TEST(AxOpUtils, EmbeddingsCosineDistanceOriginalEdge)
{
  Eigen::MatrixXf embeddings(4, 2);
  embeddings << 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f;

  std::vector<float> desc = { 1.0f, 0.0f };
  auto distances = ax_utils::embeddings_cosine_distance(desc, embeddings);

  ASSERT_EQ(distances.size(), 4u);
  EXPECT_NEAR(distances[0], 0.0f, 1e-6f);
  EXPECT_NEAR(distances[1], 0.5f, 1e-6f);
  EXPECT_NEAR(distances[2], 0.25f, 1e-6f);
  EXPECT_NEAR(distances[3], 0.5f, 1e-6f);
}

TEST(AxOpUtils, EmbeddingsSquaredEuclideanDistanceNoNormalise)
{
  Eigen::MatrixXf embeddings(3, 2);
  embeddings << 1.0f, 2.0f, 2.0f, 2.0f, 4.0f, 6.0f;

  std::vector<float> desc = { 1.0f, 2.0f };
  auto distances
      = ax_utils::embeddings_squared_euclidean_distance(desc, embeddings, false);

  ASSERT_EQ(distances.size(), 3u);
  EXPECT_NEAR(distances[0], 0.0f, 1e-6f);
  EXPECT_NEAR(distances[1], 1.0f, 1e-6f);
  EXPECT_NEAR(distances[2], 25.0f, 1e-6f);
}

TEST(AxOpUtils, EmbeddingsSquaredEuclideanDistanceNormalise)
{
  Eigen::MatrixXf embeddings(3, 2);
  embeddings << 3.0f, 0.0f, 0.0f, 4.0f, 0.0f, 0.0f;

  std::vector<float> desc = { 6.0f, 0.0f };
  auto distances = ax_utils::embeddings_squared_euclidean_distance(desc, embeddings, true);

  ASSERT_EQ(distances.size(), 3u);
  EXPECT_NEAR(distances[0], 0.0f, 1e-6f);
  EXPECT_NEAR(distances[1], 2.0f, 1e-6f);
  EXPECT_NEAR(distances[2], 1.0f, 1e-6f);

  std::vector<float> zero_desc = { 0.0f, 0.0f };
  auto zero_distances
      = ax_utils::embeddings_squared_euclidean_distance(zero_desc, embeddings, true);
  ASSERT_EQ(zero_distances.size(), 3u);
  EXPECT_NEAR(zero_distances[0], 1.0f, 1e-6f);
  EXPECT_NEAR(zero_distances[1], 1.0f, 1e-6f);
  EXPECT_NEAR(zero_distances[2], 0.0f, 1e-6f);
}

TEST(FacenetDecoder, PairValidationMode)
{
  std::vector<float> embeddings = { 0.134798, 0.269696, 0.404494, 0.539393, 0.674191 };
  std::string meta_identifier = "embeddings";

  std::unordered_map<std::string, std::string> input = { { "meta_key", meta_identifier },
    { "pair_validation", "1" }, { "decoder_name", "facenet" } };

  auto tmp_file = tempfile("[]"); // Empty embeddings file
  input["embeddings_file"] = tmp_file.filename();

  auto decoder = Ax::LoadDecode("facenet", input);
  Ax::MetaMap metadata;
  auto tensors = tensors_from_vector(embeddings);
  AxVideoInterface video_info{ { 640, 480, 640, 0, AxVideoFormat::RGB }, nullptr };

  decoder->decode_to_meta(tensors, 0, 1, metadata, video_info);

  auto actual_embeddings = get_embeddings_meta(metadata, meta_identifier);

  ASSERT_EQ(actual_embeddings.size(), 1);

  // The implementation normalizes the embeddings, so we need to compare with normalized values
  float norm = std::sqrt(std::inner_product(
      embeddings.begin(), embeddings.end(), embeddings.begin(), 0.0f));
  std::vector<float> normalized_embeddings = embeddings;
  if (norm > 0) {
    std::transform(normalized_embeddings.begin(), normalized_embeddings.end(),
        normalized_embeddings.begin(), [norm](float val) { return val / norm; });
  }

  // Compare with normalized embeddings
  for (size_t i = 0; i < normalized_embeddings.size(); ++i) {
    EXPECT_NEAR(actual_embeddings[0][i], normalized_embeddings[i], 1e-5);
  }
}
TEST(FacenetDecoder, ClassificationModeEuclidean)
{
  std::vector<float> input_embedding = { 0.1348, 0.2697, 0.4045, 0.5394, 0.6742 };
  std::string meta_identifier = "classification";

  // Create JSON with proper format expected by ax_utils::read_embedding_json
  nlohmann::json embeddings_json
      = { { "person1", { 0.1543, 0.2571, 0.3599, 0.4628, 0.5656 } },
          { "person2", { 0.1348, 0.2697, 0.4045, 0.5394, 0.6742 } } };

  auto tmp_file = tempfile(embeddings_json.dump());

  std::unordered_map<std::string, std::string> input = { { "meta_key", meta_identifier },
    { "pair_validation", "0" }, { "metric_type", "euclidean_distance" }, { "top_k", "1" },
    { "distance_threshold", "0.5" }, { "embeddings_file", tmp_file.filename() } };

  auto decoder = Ax::LoadDecode("facenet", input);
  Ax::MetaMap metadata;
  auto tensors = tensors_from_vector(input_embedding);
  AxVideoInterface video_info{ { 640, 480, 640, 0, AxVideoFormat::RGB }, nullptr };

  decoder->decode_to_meta(tensors, 0, 1, metadata, video_info);

  auto [labels, scores] = get_classification_meta(metadata, meta_identifier);

  ASSERT_EQ(labels.size(), 1);
  EXPECT_EQ(labels[0], "person2"); // Should match the closest embedding
  ASSERT_EQ(scores.size(), 1);
  // For Euclidean distance, we may not get exactly 0.0 due to normalization
  // Allow a small tolerance for the distance
  EXPECT_LT(scores[0], 0.1);
}
TEST(FacenetDecoder, ClassificationModeCosine)
{
  std::vector<float> input_embedding = { 0.1348, 0.2697, 0.4045, 0.5394, 0.6742 };
  std::string meta_identifier = "classification";

  // Create JSON with proper format expected by ax_utils::read_embedding_json
  nlohmann::json embeddings_json
      = { { "person1", { 0.1543, 0.2571, 0.3599, 0.4628, 0.5656 } },
          { "person2", { 0.9f, 0.8f, 0.7f, 0.6f, 0.5f } } };

  auto tmp_file = tempfile(embeddings_json.dump());

  std::unordered_map<std::string, std::string> input = { { "meta_key", meta_identifier },
    { "pair_validation", "0" }, { "metric_type", "cosine_similarity" }, { "top_k", "2" },
    { "distance_threshold", "0.7" }, { "embeddings_file", tmp_file.filename() } };

  auto decoder = Ax::LoadDecode("facenet", input);
  Ax::MetaMap metadata;
  auto tensors = tensors_from_vector(input_embedding);
  AxVideoInterface video_info{ { 640, 480, 640, 0, AxVideoFormat::RGB }, nullptr };

  decoder->decode_to_meta(tensors, 0, 1, metadata, video_info);

  auto [labels, scores] = get_classification_meta(metadata, meta_identifier);

  ASSERT_EQ(labels.size(), 2);
  // With cosine similarity, higher score is better (more similar)
  EXPECT_GT(scores[0], scores[1]);

  // The normalized input embeddings should have the highest similarity with person1
  // given the test data. However, we're testing the order from the implementation here.
  EXPECT_EQ(labels[0], "person1");
  EXPECT_EQ(labels[1], "person2");
}

TEST(FacenetDecoder, InvalidInput)
{
  std::vector<float> embeddings = { 0.1348, 0.2697, 0.4045, 0.5394, 0.6742 };
  std::string meta_identifier = "embeddings";

  // Create JSON with proper format expected by ax_utils::read_embedding_json
  nlohmann::json embeddings_json
      = { { "person1", { 0.1543, 0.2571, 0.3599, 0.4628, 0.5656 } },
          { "person2", { 0.9f, 0.8f, 0.7f, 0.6f, 0.5f } } };

  auto tmp_file = tempfile(embeddings_json.dump());

  std::unordered_map<std::string, std::string> input
      = { { "meta_key", meta_identifier }, { "pair_validation", "1" },
          { "decoder_name", "facenet" }, { "embeddings_file", tmp_file.filename() } };

  auto decoder = Ax::LoadDecode("facenet", input);
  Ax::MetaMap metadata;

  // Create a tensor with multiple tensors (which is invalid)
  AxTensorsInterface invalid_tensors = { { { 5 }, sizeof(float), embeddings.data() },
    { { 5 }, sizeof(float), embeddings.data() } };

  AxVideoInterface video_info{ { 640, 480, 640, 0, AxVideoFormat::RGB }, nullptr };

  EXPECT_THROW(decoder->decode_to_meta(invalid_tensors, 0, 1, metadata, video_info),
      std::runtime_error);
}

TEST(FacenetDecoder, ThresholdMatching)
{
  std::vector<float> input_embedding = { 0.1348, 0.2697, 0.4045, 0.5394, 0.6742 };
  std::string meta_identifier = "classification";

  // Create JSON with embeddings that are far from the input
  nlohmann::json embeddings_json = { { "person1", { 0.9f, 0.9f, 0.9f, 0.9f, 0.9f } },
    { "person2", { 0.8f, 0.8f, 0.8f, 0.8f, 0.8f } } };

  auto tmp_file = tempfile(embeddings_json.dump());

  // Use a very low threshold so matches will be invalid
  std::unordered_map<std::string, std::string> input = { { "meta_key", meta_identifier },
    { "pair_validation", "0" }, { "metric_type", "euclidean_distance" },
    { "top_k", "2" }, { "distance_threshold", "0.1" }, // Very strict threshold
    { "embeddings_file", tmp_file.filename() } };

  auto decoder = Ax::LoadDecode("facenet", input);
  Ax::MetaMap metadata;
  auto tensors = tensors_from_vector(input_embedding);
  AxVideoInterface video_info{ { 640, 480, 640, 0, AxVideoFormat::RGB }, nullptr };

  decoder->decode_to_meta(tensors, 0, 1, metadata, video_info);

  auto [labels, scores] = get_classification_meta(metadata, meta_identifier);

  ASSERT_EQ(labels.size(), 2);
  // With strict threshold, all matches should be invalid
  EXPECT_EQ(labels[0], "");
  EXPECT_EQ(labels[1], "");
  EXPECT_EQ(scores[0], -1.0f); // Invalid match marker
  EXPECT_EQ(scores[1], -1.0f);
}

TEST(FacenetDecoder, UpdateEmbeddings)
{
  std::vector<float> new_embedding = { 0.1348, 0.2697, 0.4045, 0.5394, 0.6742 };
  std::string meta_identifier = "embeddings";
  std::string person_name = "new_person";

  // Start with empty embeddings
  nlohmann::json embeddings_json = nlohmann::json::object();
  auto tmp_file = tempfile(embeddings_json.dump());

  // The implementation expects labels_for_update to be a comma-separated string
  // Not a JSON array
  std::string labels_str = person_name; // For a single name, just use the string directly

  std::unordered_map<std::string, std::string> input
      = { { "meta_key", meta_identifier }, { "pair_validation", "0" },
          { "update_embeddings", "1" }, { "labels_for_update", labels_str },
          { "embeddings_file", tmp_file.filename() } };

  auto decoder = Ax::LoadDecode("facenet", input);
  Ax::MetaMap metadata;
  auto tensors = tensors_from_vector(new_embedding);
  AxVideoInterface video_info{ { 640, 480, 640, 0, AxVideoFormat::RGB }, nullptr };

  // This should add the new embedding
  decoder->decode_to_meta(tensors, 0, 1, metadata, video_info);

  // Read the updated embeddings file to verify
  nlohmann::json updated_json;
  try {
    std::ifstream f(tmp_file.filename());
    if (!f.is_open()) {
      FAIL() << "Could not open the embeddings file: " << tmp_file.filename();
    }
    std::string content(
        (std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (content.empty()) {
      FAIL() << "Embeddings file is empty";
    }
    try {
      updated_json = nlohmann::json::parse(content);
    } catch (const nlohmann::json::parse_error &e) {
      FAIL() << "Failed to parse JSON content: " << e.what() << "\nContent: " << content;
    }
  } catch (const std::exception &e) {
    FAIL() << "Failed to read updated JSON file: " << e.what();
  }

  // Debug output to help diagnose the issue
  std::cout << "JSON content: " << updated_json.dump(2) << std::endl;

  ASSERT_TRUE(updated_json.contains(person_name))
      << "JSON does not contain expected person: " << person_name;

  // Get the stored embedding
  ASSERT_TRUE(updated_json[person_name].is_array()) << "Stored embedding is not an array";
  auto stored_embedding = updated_json[person_name].get<std::vector<float>>();

  // The implementation normalizes the embeddings, so we need to compare with normalized values
  float norm = std::sqrt(std::inner_product(
      new_embedding.begin(), new_embedding.end(), new_embedding.begin(), 0.0f));
  std::vector<float> normalized_embedding = new_embedding;
  if (norm > 0) {
    std::transform(normalized_embedding.begin(), normalized_embedding.end(),
        normalized_embedding.begin(), [norm](float val) { return val / norm; });
  }

  ASSERT_EQ(stored_embedding.size(), normalized_embedding.size());
  for (size_t i = 0; i < stored_embedding.size(); ++i) {
    EXPECT_NEAR(stored_embedding[i], normalized_embedding[i], 1e-5);
  }
}

TEST(FacenetDecoder, DeduplicateExactOnLoad)
{
  // Two labels with identical vectors
  nlohmann::json embeddings_json
      = { { "A", { 0.1f, 0.2f, 0.3f } }, { "B", { 0.1f, 0.2f, 0.3f } } };
  auto tmp_file = tempfile(embeddings_json.dump());

  std::unordered_map<std::string, std::string> input = { { "meta_key", "classification" },
    { "pair_validation", "0" }, { "metric_type", "cosine_similarity" }, { "top_k", "1" },
    { "distance_threshold", "0.5" }, { "embeddings_file", tmp_file.filename() } };

  // Recognition mode: should deduplicate and proceed
  auto decoder = Ax::LoadDecode("facenet", input);
  Ax::MetaMap metadata;
  std::vector<float> v = { 1.f, 0.f, 0.f };
  auto tensors = tensors_from_vector(v);
  AxVideoInterface video_info{ { 640, 480, 640, 0, AxVideoFormat::RGB }, nullptr };
  EXPECT_NO_THROW(decoder->decode_to_meta(tensors, 0, 1, metadata, video_info));
}

TEST(FacenetDecoder, UpdateModeAlertsNearDuplicates)
{
  // Two very similar embeddings but different labels
  // We just build a file and enable update_embeddings; decoder should start fresh
  // if exact dups exist; otherwise it should log alerts (not asserted here).
  nlohmann::json embeddings_json = { { "C", { 0.1f, 0.2f, 0.3f } },
    { "D", { 0.1000001f, 0.2000001f, 0.3000001f } } };
  auto tmp_file = tempfile(embeddings_json.dump());

  std::unordered_map<std::string, std::string> input = { { "meta_key", "classification" },
    { "pair_validation", "0" }, { "metric_type", "cosine_similarity" },
    { "top_k", "1" }, { "distance_threshold", "0.5" },
    { "embeddings_file", tmp_file.filename() }, { "update_embeddings", "1" } };

  auto decoder = Ax::LoadDecode("facenet", input);
  Ax::MetaMap metadata;
  std::vector<float> v2 = { 0.1f, 0.2f, 0.3f };
  auto tensors = tensors_from_vector(v2);
  AxVideoInterface video_info{ { 640, 480, 640, 0, AxVideoFormat::RGB }, nullptr };
  // Should not throw; alerts go to logger
  EXPECT_NO_THROW(decoder->decode_to_meta(tensors, 0, 1, metadata, video_info));
}
