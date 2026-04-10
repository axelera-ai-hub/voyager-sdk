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
      : type(_type),
        subtype(_subtype),
        meta(_meta, _meta + _meta_size)
  {
  }
};

class AxMetaBase
{
  public:
  bool enable_extern = true;

  virtual void draw(const AxVideoInterface &video,
      const std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &meta_map)
  {
  }

  virtual size_t get_number_of_subframes() const
  {
    return 1;
  }

  virtual std::vector<extern_meta> get_extern_meta() const
  {
    throw std::runtime_error("get_extern_meta not implemented");
  }

  void insert_submeta(const std::string &name, int subframe_index,
      int subframe_number, std::shared_ptr<AxMetaBase> meta)
  {
    if (!submeta_map) {
      submeta_map = std::make_shared<SubmetaMap>();
    }
    submeta_map->insert(name, subframe_index, subframe_number, std::move(meta));
  }

  template <typename T = AxMetaBase>
  T *get_submeta(const std::string &name, int subframe_index, int subframe_number) const
  {
    if (!submeta_map) {
      throw std::runtime_error("Submeta map is not initialized");
    }
    return submeta_map->get<T>(name, subframe_index, subframe_number);
  }

  template <typename T = AxMetaBase>
  std::vector<T *> get_submetas(const std::string &name) const
  {
    if (!submeta_map) {
      return {};
    }
    return submeta_map->get<T>(name);
  }

  const std::vector<const char *> submeta_names() const
  {
    if (!submeta_map) {
      return {};
    }
    return submeta_map->keys();
  }

  virtual ~AxMetaBase() = default;

  private:
  class SubmetaMap
  {
    public:
    void insert(const std::string &name, int subframe_index,
        int number_of_subframes, std::shared_ptr<AxMetaBase> meta)
    {
      if (subframe_index < 0) {
        throw std::runtime_error("Subframe index cannot be negative in insert of SubmetaMap");
      }
      if (number_of_subframes <= 0) {
        throw std::runtime_error("Number of subframes must be positive in insert of SubmetaMap");
      }
      if (subframe_index >= number_of_subframes) {
        throw std::runtime_error("Subframe index out of bounds in insert of SubmetaMap");
      }
      std::unique_lock lock(mutex);
      auto &subframes = map[name];
      subframes.resize(
          std::max(static_cast<size_t>(number_of_subframes), subframes.size()));
      subframes[subframe_index] = std::move(meta);
    }

    template <typename T = AxMetaBase>
    T *get(const std::string &name, int subframe_index, int number_of_subframes) const
    {
      std::shared_lock lock(mutex);
      auto map_itr = map.find(name);
      if (map_itr == map.end()) {
        throw std::runtime_error("Submodel name " + name + " not found in get of SubmetaMap");
      }
      if (map_itr->second.size() != number_of_subframes) {
        throw std::runtime_error("Submodel name " + name + " has number of subframes "
                                 + std::to_string(map_itr->second.size()) + " but queried is "
                                 + std::to_string(number_of_subframes));
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
        auto result_meta = dynamic_cast<T *>(meta.get());
        if (meta.get() != nullptr && result_meta == nullptr) {
          auto &rmeta = *meta;
          throw std::runtime_error("Submodel name " + name + " is not of type "
                                   + typeid(T).name() + " but " + typeid(rmeta).name());
        }
        result.push_back(result_meta);
      }
      return result;
    }

    const std::vector<const char *> keys() const
    {
      std::shared_lock lock(mutex);
      std::vector<const char *> submodel_names;
      for (const auto &[name, submodel_vector] : map) {
        submodel_names.push_back(name.c_str());
      }
      return submodel_names;
    }

    private:
    std::unordered_map<std::string, std::vector<std::shared_ptr<AxMetaBase>>> map;
    mutable std::shared_mutex mutex;
  };

  std::shared_ptr<SubmetaMap> submeta_map;
};
