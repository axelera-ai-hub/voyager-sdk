// Copyright Axelera AI, 2025
#pragma once
#include <AxMeta.hpp>
#include <vector>

class AxLicensePlateMeta : public AxMetaBase
{
  public:
  AxLicensePlateMeta(std::string label)
      : label_(std::move(label))
  {
  }

  std::vector<extern_meta> get_extern_meta() const override
  {
    const char *class_meta = "LicensePlateMeta";
    auto results = std::vector<extern_meta>();
    results.push_back({ class_meta, "label", int(label_.size()),
        reinterpret_cast<const char *>(label_.c_str()) });

    return results;
  }

  private:
  std::string label_;
};
