// Copyright Axelera AI, 2023
#include "AxDataInterface.h"
#include "AxLog.hpp"
#include "AxMeta.hpp"

extern "C" void
inplace(const AxDataInterface &data, const void *, unsigned int, unsigned int,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &map, Ax::Logger &logger)
{
  if (!std::holds_alternative<AxVideoInterface>(data)) {
    throw std::runtime_error("inplace_draw works with video only");
  }

  for (const auto &ele : map) {
    if (ele.second->enable_draw) {
      ele.second->draw(std::get<AxVideoInterface>(data), map);
    }
  }
}
