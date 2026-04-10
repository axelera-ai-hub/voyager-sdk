// Copyright Axelera AI, 2025
#pragma once

#include <opencv2/opencv.hpp>
#include <span>
#include <vector>

#include "AxDataInterface.h"
#include "AxMeta.hpp"
#include "AxUtils.hpp"


struct kpt_xyv {
  int x;
  int y;
  float visibility;
};

using KptXyv = kpt_xyv;
using KptXyvVector = std::vector<KptXyv>;

class AxMetaKpts : public virtual AxMetaBase
{
  public:
  AxMetaKpts(KptXyvVector kpts)
      : kptsvec(std::move(kpts))
  {
  }

  /// @brief Get the keypoints for the metadata at the given index
  /// @param idx - The starting index
  /// @param num_keypoints - The number of keypoints per element of metadata
  /// @return The actual keypoints
  std::span<const kpt_xyv> get_kpts_xyv(size_t idx, size_t num_keypoints) const
  {
    auto start = idx * num_keypoints;
    if (start + num_keypoints > kptsvec.size()) {
      throw std::out_of_range("Index out of range in get_kpts_xyv");
    }
    auto begin = std::next(kptsvec.begin(), start);
    return std::span<const kpt_xyv>(begin, num_keypoints);
  }

  /// @brief Set the keypoints for the metadata at the given index
  /// @param idx - The starting index
  /// @param kpts - The keypoints to set
  void set_kpts_xyv(size_t idx, const std::span<kpt_xyv> &kpts)
  {
    auto start = idx * kpts.size();
    if (start + kpts.size() > kptsvec.size()) {
      throw std::out_of_range("Index out of range in set_kpts_xyv");
    }
    auto begin = std::next(kptsvec.begin(), start);
    std::copy(kpts.begin(), kpts.end(), begin);
  }

  std::vector<extern_meta> get_extern_meta() const override
  {
    return { { "kpts", "kpts", int(kptsvec.size() * sizeof(KptXyv)),
        reinterpret_cast<const char *>(kptsvec.data()) } };
  }

  size_t num_elements() const
  {
    return kptsvec.size();
  }

  KptXyv get_kpt_xy(size_t idx) const
  {
    return kptsvec[idx];
  }

  std::vector<KptXyv> get_kpts() const
  {
    return kptsvec;
  }

  const KptXyv *get_kpts_data() const
  {
    return kptsvec.data();
  }

  void extend(const AxMetaKpts &other)
  {
    kptsvec.insert(kptsvec.end(), other.kptsvec.begin(), other.kptsvec.end());
  }

  void extend(AxMetaKpts &&other)
  {
    kptsvec.insert(kptsvec.end(), std::make_move_iterator(other.kptsvec.begin()),
        std::make_move_iterator(other.kptsvec.end()));
  }


  private:
  KptXyvVector kptsvec;
};
