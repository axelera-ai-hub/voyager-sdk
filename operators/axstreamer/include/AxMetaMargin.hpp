// Copyright Axelera AI, 2026
#pragma once

#include <time.h>

#include <atomic>
#include <chrono>
#include <vector>

#include "AxDataInterface.h"
#include "AxMeta.hpp"
#include "AxUtils.hpp"

class AxMetaMargin : public AxMetaBase
{
  public:
  float margin;

  explicit AxMetaMargin(float margin)
      : margin{ margin }
  {
    enable_extern = false;
  }

  std::vector<extern_meta> get_extern_meta() const override
  {
    return {};
  }
};
