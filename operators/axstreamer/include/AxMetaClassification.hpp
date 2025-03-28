// Copyright Axelera AI, 2025
#pragma once

#include "AxMetaBBox.hpp"


class AxMetaClassification : public AxMetaBase
{
  public:
  static inline const auto red = cv::Scalar(255, 0, 0);

  using scores_vec = std::vector<std::vector<float>>;
  using classes_vec = std::vector<std::vector<int32_t>>;
  using labels_vec = std::vector<std::vector<std::string>>;

  AxMetaClassification(scores_vec scores, classes_vec classes,
      labels_vec labels, std::string box_meta = "")
      : scores_(std::move(scores)), classes_(std::move(classes)),
        labels_(std::move(labels)), box_meta_(std::move(box_meta))
  {
    if (scores_.size() != classes_.size()) {
      throw std::logic_error("AxMetaClassification: scores and classes must have the same size");
    }
  }

  void append(std::vector<float> scores, std::vector<int> classes,
      std::vector<std::string> labels)
  {
    scores_.push_back(std::move(scores));
    classes_.push_back(std::move(classes));
    labels_.push_back(std::move(labels));
  }

  void replace(int idx, std::vector<float> scores, std::vector<int> classes,
      std::vector<std::string> labels)
  {
    scores_[idx] = std::move(scores);
    classes_[idx] = std::move(classes);
    labels_[idx] = std::move(labels);
  }

  void draw(const AxVideoInterface &video,
      const std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &meta_map) override
  {
    if (video.info.format != AxVideoFormat::RGB && video.info.format != AxVideoFormat::RGBA) {
      throw std::runtime_error("Labels can only be drawn on RGB or RGBA");
    }
    cv::Mat mat(cv::Size(video.info.width, video.info.height),
        Ax::opencv_type_u8(video.info.format), video.data, video.info.stride);
    if (box_meta_.empty()) {
      cv::putText(mat, labels_[0][0],
          cv::Point(video.info.width / 2, video.info.height / 2),
          cv::FONT_HERSHEY_SIMPLEX, 2.0, red);
    } else {
      const auto &ref = meta_map.at(box_meta_);
      AxMetaBbox *bboxes = dynamic_cast<AxMetaBbox *>(ref.get());
      if (bboxes == nullptr) {
        throw std::runtime_error("AxMetaLabels not provided with AxMetaBbox");
      }
      if (bboxes->num_elements() != labels_.size()) {
        throw std::runtime_error("AxMetaLabels number of labels does not match number of boxes");
      }
      for (int i = 0; i < labels_.size(); ++i) {
        const auto [x, y, x1, y1] = bboxes->get_box_xyxy(i);
        cv::putText(mat, labels_[i][0], cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, 1.0, red);
      }
    }
  }

  std::vector<extern_meta> get_extern_meta() const override
  {
    const char *class_meta = "ClassificationMeta";
    auto results = std::vector<extern_meta>();
    for (const auto &score : scores_) {
      results.push_back({ class_meta, "scores", int(score.size() * sizeof(float)),
          reinterpret_cast<const char *>(score.data()) });
    }
    for (const auto &cl : classes_) {
      results.push_back({ class_meta, "classes", int(cl.size() * sizeof(int)),
          reinterpret_cast<const char *>(cl.data()) });
    }
    return results;
  }

  scores_vec get_scores() const
  {
    return scores_;
  }

  classes_vec get_classes() const
  {
    return classes_;
  }

  labels_vec get_labels() const
  {
    return labels_;
  }

  size_t get_number_of_subframes() const override
  {
    return scores_.size();
  }

  private:
  scores_vec scores_;
  classes_vec classes_;
  labels_vec labels_;
  std::string box_meta_;
};


// AxMetaEmbeddings is a class that represents embeddings for each frame or ROI.
// It is a specialization of ClassificationMeta that decodes the output tensor
// to embeddings. It can be used for pair validation, streaming embedding
// features from C++ to Python, or any other applications where users want to
// leverage the embeddings for different recognition tasks or business logic,
// providing flexibility in various use cases.
class AxMetaEmbeddings : public AxMetaBase
{
  public:
  using embeddings_vec = std::vector<std::vector<float>>;

  AxMetaEmbeddings(embeddings_vec embeddings, std::string box_meta = "")
      : embeddings_(std::move(embeddings))
  {
  }

  void append(std::vector<float> embedding)
  {
    embeddings_.push_back(std::move(embedding));
  }

  void replace(int idx, std::vector<float> embedding)
  {
    embeddings_[idx] = std::move(embedding);
  }

  void draw(const AxVideoInterface &video,
      const std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &meta_map) override
  {
    throw std::runtime_error("Embeddings are not supported to be drawn");
  }

  std::vector<extern_meta> get_extern_meta() const override
  {
    std::vector<extern_meta> result;
    int num_embeddings = embeddings_.size();
    size_t total_size = sizeof(int); // For num_embeddings

    for (const auto &embedding : embeddings_) {
      total_size += embedding.size() * sizeof(float);
    }

    std::vector<std::byte> buffer(total_size);
    auto ptr = buffer.data();

    std::memcpy(ptr, &num_embeddings, sizeof(int));
    ptr += sizeof(int);

    for (const auto &embedding : embeddings_) {
      size_t embedding_size = embedding.size() * sizeof(float);
      std::memcpy(ptr, embedding.data(), embedding_size);
      ptr += embedding_size;
    }

    result.push_back({ "embeddings", "data", static_cast<int>(total_size),
        reinterpret_cast<const char *>(buffer.data()) });
    return result;
  }

  embeddings_vec get_embeddings() const
  {
    return embeddings_;
  }

  size_t get_number_of_subframes() const override
  {
    return embeddings_.size();
  }

  private:
  embeddings_vec embeddings_;
};
