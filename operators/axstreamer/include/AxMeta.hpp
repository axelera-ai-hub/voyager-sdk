// Copyright Axelera AI, 2025
#pragma once

#include <cstddef>
#include <cstring>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "AxDataInterface.h"

struct extern_meta {
  const char *type{}; // Names the type of metadata e.g object_meta, classification_meta
  const char *subtype{}; //  Names the subtype of metadata e.g. Object Detection contains BBox, scores, class labels
  int meta_size{}; //  Size of this chunk in bytes
  const char *meta{}; //   Pointer to the raw data.
};

struct extern_meta_container {
  std::string type{};
  std::string subtype{};
  std::vector<char> meta{};
  extern_meta_container(
      const char *_type, const char *_subtype, int _meta_size, const char *_meta)
      : type(_type), subtype(_subtype), meta(_meta, _meta + _meta_size)
  {
  }
};

class SubmetaMap;

class AxMetaBase
{
  public:
  bool enable_draw = true;
  std::shared_ptr<SubmetaMap> submeta_map;
  virtual void draw(const AxVideoInterface &video,
      const std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &meta_map)
  {
    throw std::runtime_error("Disable drawing of metadata without overloaded draw function");
  }
  virtual size_t get_number_of_subframes() const
  {
    return 1;
  }

  virtual std::vector<extern_meta> get_extern_meta() const = 0;

  inline AxMetaBase();
  virtual ~AxMetaBase() = default;
};


class SubmetaMap
{
  public:
  void insert(const std::string &name, int subframe_index, int subframe_number,
      std::shared_ptr<AxMetaBase> meta)
  {
    std::unique_lock lock(mutex);
    auto [map_itr, created] = map.try_emplace(
        name, std::vector<std::shared_ptr<AxMetaBase>>(subframe_number));
    if (!created && map_itr->second.size() != subframe_number) {
      throw std::runtime_error("Subframe number mismatch in insert of SubmetaMap");
    }
    map_itr->second[subframe_index] = std::move(meta);
  }

  void erase(const std::string &name)
  {
    std::unique_lock lock(mutex);
    map.erase(name);
  }

  template <typename T = AxMetaBase>
  T *get(const std::string &name, int subframe_index, int subframe_number) const
  {
    std::shared_lock lock(mutex);
    auto map_itr = map.find(name);
    if (map_itr == map.end()) {
      throw std::runtime_error("Submodel name " + name + " not found in get of SubmetaMap");
    }
    if (map_itr->second.size() != subframe_number) {
      throw std::runtime_error("Subframe number in SubmetaMap is "
                               + std::to_string(map_itr->second.size()) + " but queried is "
                               + std::to_string(subframe_number));
    }
    auto ptr = map_itr->second[subframe_index].get();
    if (ptr == nullptr) {
      throw std::runtime_error("Submodel name " + name + " with index "
                               + std::to_string(subframe_index) + " is nullptr");
    }
    auto result = dynamic_cast<T *>(ptr);
    if (result == nullptr) {
      throw std::runtime_error("Submodel name " + name + " with index "
                               + std::to_string(subframe_index) + " is not of type "
                               + typeid(T).name() + " but " + typeid(*ptr).name());
    }
    return result;
  }

  template <typename T = AxMetaBase>
  std::vector<T *> get(const std::string &name) const
  {
    std::shared_lock lock(mutex);
    std::vector<T *> result;
    auto map_itr = map.find(name);
    if (map_itr == map.end()) {
      return result;
    }
    for (const auto &meta : map_itr->second) {
      auto &rmeta = *meta;
      auto result_meta = dynamic_cast<T *>(&rmeta);
      if (result_meta == nullptr) {
        throw std::runtime_error("Submodel name " + name + " is not of type "
                                 + typeid(T).name() + " but " + typeid(rmeta).name());
      }
      result.push_back(result_meta);
    }
    return result;
  }

  const std::pair<std::vector<const char *>, std::vector<int>> contents() const
  {
    std::shared_lock lock(mutex);
    std::vector<const char *> submodel_names;
    std::vector<int> subframe_numbers;
    for (const auto &pair : map) {
      submodel_names.push_back(pair.first.c_str());
      subframe_numbers.push_back(pair.second.size());
    }
    return std::make_pair(std::move(submodel_names), std::move(subframe_numbers));
  }

  private:
  std::unordered_map<std::string, std::vector<std::shared_ptr<AxMetaBase>>> map;
  mutable std::shared_mutex mutex;
};

inline AxMetaBase::AxMetaBase() : submeta_map(std::make_shared<SubmetaMap>())
{
}
