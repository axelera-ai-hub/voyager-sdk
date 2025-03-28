// Copyright Axelera AI, 2025
#pragma once

#include <opencv2/opencv.hpp>

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
  AxMetaKpts(KptXyvVector kpts) : kptsvec(std::move(kpts))
  {
  }

  std::vector<kpt_xyv> get_kpts_xyv(size_t start, size_t count) const
  {
    auto begin = std::next(kptsvec.begin(), start);
    return { begin, std::next(begin, count) };
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


  private:
  KptXyvVector kptsvec;
};
