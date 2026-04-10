// Copyright Axelera AI, 2024
// Optimized Facenet decoder

#include "AxDataInterface.h"
#include "AxLog.hpp"
#include "AxMetaClassification.hpp"
#include "AxOpUtils.hpp"
#include "AxUtils.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <iterator>
#include <limits>
#include <mutex>
#include <ranges>
#include <unordered_set>
#include <vector>

#include <Eigen/Core>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>

namespace facenet_recog
{

struct properties {
  std::string embeddings_file{};
  float distance_threshold{ 0.5 };
  int metric_type{ ax_utils::EUCLIDEAN_DISTANCE };
  bool pair_validation{ true };
  int top_k{ 1 };
  bool update_embeddings{ false };

  std::string meta_name{};
  std::string master_meta{};
  std::string association_meta{};
  std::string decoder_name{};

  // For update mode: labels to be processed in sequence
  std::vector<std::string> labels_for_update{};
  mutable size_t current_update_index{ 0 };

  mutable Eigen::MatrixXf embeddings{};
  mutable std::vector<std::string> labels{};

  // Guards concurrent calls to decode_to_meta when update mode is enabled
  mutable std::mutex update_mutex;
};

} // namespace facenet_recog

// Deduplicate exact duplicate embeddings by keeping the first occurrence
static bool
deduplicate_exact_embeddings(Eigen::MatrixXf &embeddings,
    std::vector<std::string> &labels, Ax::Logger &logger)
{
  if (embeddings.rows() <= 1 || static_cast<size_t>(embeddings.rows()) != labels.size()) {
    return false;
  }

  std::unordered_map<std::string, std::vector<size_t>> key_to_rows;
  for (int r = 0; r < embeddings.rows(); ++r) {
    std::string key;
    key.reserve(static_cast<size_t>(embeddings.cols()) * 8);
    for (int c = 0; c < embeddings.cols(); ++c) {
      float v = embeddings(r, c);
      int iv = static_cast<int>(std::llround(static_cast<long double>(v * 1'000'000.0L)));
      key.append(reinterpret_cast<const char *>(&iv), sizeof(iv));
    }
    key_to_rows[key].push_back(static_cast<size_t>(r));
  }

  bool has_dups = false;
  for (const auto &kv : key_to_rows) {
    if (kv.second.size() > 1) {
      has_dups = true;
      break;
    }
  }
  if (!has_dups)
    return false;

  std::vector<size_t> keep_indices;
  keep_indices.reserve(static_cast<size_t>(embeddings.rows()));
  for (const auto &kv : key_to_rows) {
    keep_indices.push_back(kv.second.front());
  }
  std::sort(keep_indices.begin(), keep_indices.end());

  Eigen::MatrixXf new_embeddings;
  new_embeddings.resize(static_cast<int>(keep_indices.size()), embeddings.cols());
  std::vector<std::string> new_labels;
  new_labels.reserve(keep_indices.size());
  for (int i = 0; i < static_cast<int>(keep_indices.size()); ++i) {
    size_t r = keep_indices[i];
    for (int c = 0; c < embeddings.cols(); ++c) {
      new_embeddings(i, c) = embeddings(static_cast<int>(r), c);
    }
    new_labels.push_back(labels[r]);
  }
  const auto removed = labels.size() - new_labels.size();
  embeddings = std::move(new_embeddings);
  labels = std::move(new_labels);
  logger(AX_WARN)
      << "facenet_recog: Deduplication complete. Removed " << removed
      << " duplicate entries; remaining unique embeddings: " << labels.size()
      << std::endl;
  return true;
}

// Alert if there are near-duplicate pairs (cosine similarity >= threshold) across different labels
static void
alert_near_duplicate_pairs(const Eigen::MatrixXf &embeddings,
    const std::vector<std::string> &labels, float sim_threshold, Ax::Logger &logger)
{
  const int rows = embeddings.rows();
  if (rows <= 1 || static_cast<size_t>(rows) != labels.size())
    return;
  // avoid O(n^2) blow-up on large galleries
  constexpr int kMaxPairsScan = 1000;
  if (rows > kMaxPairsScan) {
    logger(AX_WARN) << "facenet_recog: Skipping near-duplicate scan (" << rows
                    << ") > " << kMaxPairsScan << std::endl;
    return;
  }
  // normalize rows
  Eigen::MatrixXf norm = embeddings;
  for (int r = 0; r < rows; ++r) {
    float len2 = 0.f;
    for (int c = 0; c < norm.cols(); ++c)
      len2 += norm(r, c) * norm(r, c);
    float inv = (len2 > 0.f) ? (1.0f / std::sqrt(len2)) : 0.f;
    for (int c = 0; c < norm.cols(); ++c)
      norm(r, c) *= inv;
  }
  for (int i = 0; i < rows; ++i) {
    for (int j = i + 1; j < rows; ++j) {
      float dot = 0.f;
      for (int c = 0; c < norm.cols(); ++c)
        dot += norm(i, c) * norm(j, c);
      if (dot >= sim_threshold && labels[i] != labels[j]) {
        logger(AX_WARN) << "facenet_recog: ALERT: near-duplicate embeddings across labels '"
                        << labels[i] << "' and '" << labels[j]
                        << "' (cosine sim=" << dot << ")" << std::endl;
      }
    }
  }
}

extern "C" void
decode_to_meta(const AxTensorsInterface &in_tensors, const facenet_recog::properties *prop,
    unsigned int current_frame, unsigned int total_frames,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &map,
    const AxDataInterface &video_interface, Ax::Logger &logger)
{
  auto start_time = std::chrono::high_resolution_clock::now();
  if (!prop) {
    logger(AX_ERROR) << "decode_to_meta : properties not set" << std::endl;
    throw std::runtime_error("decode_to_meta : properties not set");
  }
  auto tensors = in_tensors;
  if (tensors.size() != 1) {
    throw std::runtime_error("facenet_recog_to_meta : Number of input tensors must be 1");
  }
  auto *data = static_cast<float *>(tensors[0].data);
  const auto total_size = tensors[0].total();

  // Normalize the embedding (guard against degenerate all-zero)
  std::vector<std::vector<float>> embeddings_vec;
  std::vector<float> normalized_data(data, data + total_size);
  // Lightweight path: skip hashing in production
  float norm = std::sqrt(std::inner_product(normalized_data.begin(),
      normalized_data.end(), normalized_data.begin(), 0.0f));
  if (norm > 0) {
    std::transform(normalized_data.begin(), normalized_data.end(),
        normalized_data.begin(), [norm](float val) { return val / norm; });
  } else {
    // If degenerate, skip update/classification for this frame
    logger(AX_WARN) << "facenet_recog: Received degenerate embedding (norm=0); skipping frame"
                    << std::endl;
    return;
  }
  embeddings_vec.emplace_back(std::move(normalized_data));

  if (prop->pair_validation) {
    ax_utils::insert_meta<AxMetaEmbeddings>(map, prop->meta_name, prop->master_meta,
        current_frame, total_frames, std::move(embeddings_vec), prop->decoder_name);
  } else if (prop->update_embeddings) {
    // Ensure only one thread updates embeddings and the index at a time
    std::lock_guard<std::mutex> guard(prop->update_mutex);
    // Two modes
    // 1) labels_for_update provided: add/update specific identities in order
    // 2) no labels provided: refine best-matching existing identity if threshold allows

    if (!prop->labels_for_update.empty()) {
      if (prop->current_update_index >= prop->labels_for_update.size()) {
        logger(AX_ERROR) << "facenet_recog: Processed more frames than labels_for_update provided"
                         << std::endl;
        throw std::runtime_error(
            "facenet_recog: Processed more frames than labels_for_update provided");
      }

      auto &current_embedding = embeddings_vec[0];
      const std::string &person_name = prop->labels_for_update[prop->current_update_index];
      logger(AX_INFO) << "facenet_recog: updating for label '" << person_name
                      << "' at index " << prop->current_update_index << std::endl;

      auto label_it = std::find(prop->labels.begin(), prop->labels.end(), person_name);
      // Always overwrite on update mode to avoid drift
      if (label_it != prop->labels.end()) {
        // Update existing entry
        size_t person_idx
            = static_cast<size_t>(std::distance(prop->labels.begin(), label_it));
        for (size_t i = 0; i < current_embedding.size(); ++i) {
          prop->embeddings(person_idx, i) = current_embedding[i];
        }
      } else {
        // Guard against exact/near-duplicate embeddings corrupting the file.
        bool is_duplicate = false;
        if (prop->embeddings.rows() > 0) {
          auto sims = ax_utils::embeddings_cosine_similarity(
              current_embedding, prop->embeddings);
          float max_sim = -1.0f;
          for (float s : sims)
            max_sim = std::max(max_sim, s);
          if (max_sim > 0.999f) { // identical or nearly identical
            is_duplicate = true;
            logger(AX_WARN)
                << "facenet_recog: Skipping add for '" << person_name
                << "' due to near-duplicate embedding (max cosine sim=" << max_sim
                << ")" << std::endl;
          }
        }
        if (!is_duplicate) {
          ax_utils::add_vec_to_matrix(current_embedding, prop->embeddings);
          prop->labels.push_back(person_name);
          logger(AX_INFO) << "facenet_recog: Added new person: '" << person_name
                          << "'" << std::endl;
        }
      }

      prop->current_update_index++;

      try {
        ax_utils::write_embedding_json(
            prop->embeddings, prop->labels, prop->embeddings_file, logger);
        logger(AX_DEBUG) << "facenet_recog: Successfully saved embeddings to: "
                         << prop->embeddings_file << std::endl;
      } catch (const std::exception &e) {
        logger(AX_ERROR)
            << "facenet_recog: Failed to save embeddings: " << e.what() << std::endl;
        throw;
      }
    } else {
      // Label-free refine of existing entries (never add unnamed entries)
      if (prop->labels.empty()) {
        logger(AX_WARN) << "facenet_recog: update_embeddings is true but no embeddings to update. Skipping."
                        << std::endl;
        return;
      }

      const auto &current_embedding = embeddings_vec[0];
      std::vector<float> distance;
      switch (prop->metric_type) {
        case ax_utils::EUCLIDEAN_DISTANCE:
          distance = ax_utils::embeddings_euclidean_distance(
              current_embedding, prop->embeddings);
          break;
        case ax_utils::SQUARED_EUCLIDEAN_DISTANCE:
          distance = ax_utils::embeddings_squared_euclidean_distance(
              current_embedding, prop->embeddings);
          break;
        case ax_utils::COSINE_DISTANCE:
          distance = ax_utils::embeddings_cosine_distance(current_embedding, prop->embeddings);
          break;
        case ax_utils::COSINE_SIMILARITY:
          distance = ax_utils::embeddings_cosine_similarity(
              current_embedding, prop->embeddings);
          break;
        default:
          throw std::runtime_error("facenet_recog: Unsupported metric in update mode");
      }

      size_t best_idx = 0;
      for (size_t i = 1; i < distance.size(); ++i) {
        bool better = (prop->metric_type == ax_utils::COSINE_SIMILARITY) ?
                          (distance[i] > distance[best_idx]) :
                          (distance[i] < distance[best_idx]);
        if (better)
          best_idx = i;
      }

      const bool use_similarity = (prop->metric_type == ax_utils::COSINE_SIMILARITY);
      const bool valid_match
          = use_similarity ? (distance[best_idx] >= prop->distance_threshold) :
                             (distance[best_idx] <= prop->distance_threshold);
      if (!valid_match) {
        logger(AX_DEBUG) << "facenet_recog: update_embeddings: current frame did not meet threshold; skipping update"
                         << std::endl;
        return;
      }

      for (size_t i = 0; i < current_embedding.size(); ++i) {
        prop->embeddings(best_idx, i) = current_embedding[i];
      }
      logger(AX_INFO) << "facenet_recog: Updated embedding for person: '"
                      << prop->labels[best_idx] << "'" << std::endl;

      try {
        ax_utils::write_embedding_json(
            prop->embeddings, prop->labels, prop->embeddings_file, logger);
        logger(AX_DEBUG) << "facenet_recog: Successfully saved embeddings to: "
                         << prop->embeddings_file << std::endl;
      } catch (const std::exception &e) {
        logger(AX_ERROR)
            << "facenet_recog: Failed to save embeddings: " << e.what() << std::endl;
        throw;
      }
    }

  } else {
    // Recognition mode - never modify embeddings or file
    const auto &current_embedding = embeddings_vec[0];
    const auto &const_embeddings = prop->embeddings;
    const auto &const_labels = prop->labels;

    if (const_labels.empty()) {
      logger(AX_ERROR)
          << "facenet_recog: No embeddings loaded for recognition" << std::endl;
      throw std::runtime_error("facenet_recog: No embeddings loaded for recognition");
    }

    std::vector<float> distance;
    // Ensure current embedding normalization is consistent with the chosen
    // metric We already normalized the current embedding to unit length above.
    switch (prop->metric_type) {
      case ax_utils::EUCLIDEAN_DISTANCE:
        distance = ax_utils::embeddings_euclidean_distance(current_embedding, const_embeddings);
        break;
      case ax_utils::SQUARED_EUCLIDEAN_DISTANCE:
        distance = ax_utils::embeddings_squared_euclidean_distance(
            current_embedding, const_embeddings);
        break;
      case ax_utils::COSINE_DISTANCE:
        distance = ax_utils::embeddings_cosine_distance(current_embedding, const_embeddings);
        break;
      case ax_utils::COSINE_SIMILARITY:
        distance = ax_utils::embeddings_cosine_similarity(current_embedding, const_embeddings);
        break;
      default:
        {
          std::stringstream ss;
          ss << "facenet_recog_to_meta : Unsupported metric id: " << prop->metric_type;
          logger(AX_ERROR) << ss.str() << std::endl;
          throw std::runtime_error(ss.str());
        }
    }

    std::vector<size_t> indices(distance.size());
    std::iota(indices.begin(), indices.end(), 0);
    auto top_k = std::min(indices.size(), static_cast<size_t>(prop->top_k));

    // Sort the indices based on the distance metric - match Python logic
    std::partial_sort(indices.begin(), std::next(indices.begin(), top_k),
        indices.end(), [&](size_t a, size_t b) {
          return (prop->metric_type == ax_utils::COSINE_SIMILARITY) ?
                     distance[a] > distance[b] :
                     distance[a] < distance[b];
        });

    std::vector<float> top_scores(top_k);
    std::vector<std::string> top_labels(top_k);
    std::vector<int> top_ids;
    top_ids.reserve(top_k);

    for (size_t i = 0; i < top_k; i++) {
      const auto idx = indices[i];
      const bool use_similarity = (prop->metric_type == ax_utils::COSINE_SIMILARITY);
      const bool invalid_match = use_similarity ?
                                     (distance[idx] < prop->distance_threshold) :
                                     (distance[idx] > prop->distance_threshold);

      if (invalid_match) {
        top_scores[i] = -1.0f; // Mark as invalid
        top_labels[i] = "";
        top_ids.push_back(-1);
        continue;
      }
      top_ids.push_back(static_cast<int>(idx));

      top_scores[i] = distance[idx];
      top_labels[i] = const_labels[idx];
    }

    logger(AX_INFO) << "facenet_recognition: Recognized person: '" << top_labels[0]
                    << "(" << top_ids[0] << ")' with score: " << top_scores[0]
                    << " (threshold: " << prop->distance_threshold << ")" << std::endl;

    ax_utils::insert_and_associate_meta<AxMetaClassification>(map,
        prop->meta_name, prop->master_meta, current_frame, total_frames,
        prop->association_meta, AxMetaClassification::scores_vec{ top_scores },
        AxMetaClassification::classes_vec{ top_ids },
        AxMetaClassification::labels_vec{ top_labels });
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
  logger(AX_DEBUG) << "decode_to_meta : Decoding facenet_recog: " << duration.count()
                   << " microseconds" << std::endl;
}

extern "C" const std::unordered_set<std::string> &
allowed_properties()
{
  static const std::unordered_set<std::string> allowed_properties{ "embeddings_file",
    "distance_threshold", "metric_type", "meta_key", "master_meta",
    "association_meta", "pair_validation", "top_k", "decoder_name",
    "update_embeddings", "labels_for_update" };
  return allowed_properties;
}

extern "C" std::shared_ptr<void>
init_and_set_static_properties(
    const std::unordered_map<std::string, std::string> &input, Ax::Logger &logger)
{
  auto props = std::make_shared<facenet_recog::properties>();
  props->meta_name = Ax::get_property(
      input, "meta_key", "recog_static_properties", props->meta_name);
  props->master_meta = Ax::get_property(
      input, "master_meta", "recog_static_properties", props->master_meta);
  props->association_meta = Ax::get_property(input, "association_meta",
      "recog_static_properties", props->association_meta);
  props->decoder_name = Ax::get_property(
      input, "decoder_name", "recog_static_properties", props->decoder_name);
  props->distance_threshold = Ax::get_property(input, "distance_threshold",
      "recog_static_properties", props->distance_threshold);
  props->metric_type = ax_utils::parse_metric_type(
      input, props->metric_type, "recog_static_properties");

  props->pair_validation = Ax::get_property(input, "pair_validation",
      "recog_static_properties", props->pair_validation);
  props->update_embeddings = Ax::get_property(input, "update_embeddings",
      "recog_static_properties", props->update_embeddings);
  props->embeddings_file = Ax::get_property(input, "embeddings_file",
      "recog_static_properties", props->embeddings_file);

  if (!props->embeddings_file.empty() && std::ifstream(props->embeddings_file).good()) {
    // Preserve legacy behavior: do not normalize reference embeddings here.
    // Current embedding is normalized below; reference magnitudes remain as stored.
    auto [embeddings, labels] = ax_utils::read_embedding_json(
        props->embeddings_file, /*normalise=*/false, logger);
    props->embeddings = std::move(embeddings);
    props->labels = std::move(labels);
    logger(AX_INFO)
        << "facenet_recog: Loaded " << props->labels.size()
        << " existing embeddings from: " << props->embeddings_file << std::endl;

    // Detect exact duplicates and optionally alert near-duplicates
    if (props->embeddings.rows() > 1
        && props->labels.size() == static_cast<size_t>(props->embeddings.rows())) {
      if (props->update_embeddings) {
        Eigen::MatrixXf tmp = props->embeddings;
        auto lcopy = props->labels;
        if (deduplicate_exact_embeddings(tmp, lcopy, logger)) {
          logger(AX_WARN) << "facenet_recog: update_embeddings=true and existing file contains exact duplicates;"
                          << " starting fresh in-memory." << std::endl;
          props->embeddings = Eigen::MatrixXf();
          props->labels.clear();
        } else {
          // No exact duplicates; still alert for near-duplicate pairs across labels
          alert_near_duplicate_pairs(props->embeddings, props->labels, 0.999f, logger);
        }
      } else {
        (void) deduplicate_exact_embeddings(props->embeddings, props->labels, logger);
      }
    }
  }

  bool embeddings_required = !props->pair_validation && !props->update_embeddings;
  if (embeddings_required && props->labels.empty()) {
    throw std::runtime_error(
        "facenet_recog: Embeddings must be provided when update_embeddings is false and pair_validation is false");
  }

  if (props->update_embeddings) {
    // Optional: accept labels_for_update when provided for deterministic add/update
    auto opt_labels = Ax::get_property(input, "labels_for_update",
        "recog_static_properties", std::vector<std::string>());
    if (!opt_labels.empty()) {
      props->labels_for_update = std::move(opt_labels);
      props->current_update_index = 0;
      logger(AX_INFO) << "facenet_recog: update_embeddings enabled with labels_for_update ("
                      << props->labels_for_update.size() << ")" << std::endl;
    } else {
      logger(AX_INFO) << "facenet_recog: update_embeddings enabled (label-free refine mode)"
                      << std::endl;
    }
  }

  props->top_k
      = Ax::get_property(input, "top_k", "recog_static_properties", props->top_k);

  return props;
}

extern "C" void
set_dynamic_properties(const std::unordered_map<std::string, std::string> &input,
    facenet_recog::properties *prop, Ax::Logger &logger)
{
}
