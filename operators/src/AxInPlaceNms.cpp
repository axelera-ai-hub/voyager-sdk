// Copyright Axelera AI, 2025
#include "AxDataInterface.h"
#include "AxLog.hpp"
#include "AxMetaKptsDetection.hpp"
#include "AxMetaObjectDetection.hpp"
#include "AxMetaSegmentsDetection.hpp"
#include "AxUtils.hpp"

#include "AxNms.hpp"

#include <iostream>
#include <unordered_set>

struct nms_properties {
  std::string meta_key;
  std::string master_meta{};
  int max_boxes{ 10000 };
  float nms_threshold{ 0.5F };
  int class_agnostic{ true };
  std::string location{ "CPU" };
  bool flatten{ false };
  bool merge{ false };
};

extern "C" const std::unordered_set<std::string> &
allowed_properties()
{
  static const std::unordered_set<std::string> allowed_properties{
    "meta_key",
    "max_boxes",
    "class_agnostic",
    "nms_threshold",
    "location",
    "master_meta",
    "flatten_meta",
    "merge",
  };
  return allowed_properties;
}

extern "C" std::shared_ptr<void>
init_and_set_static_properties(
    const std::unordered_map<std::string, std::string> &input, Ax::Logger &logger)
{
  std::shared_ptr<nms_properties> prop = std::make_shared<nms_properties>();
  prop->meta_key
      = Ax::get_property(input, "meta_key", "nms_static_properties", prop->meta_key);
  prop->master_meta = Ax::get_property(
      input, "master_meta", "nms_static_properties", prop->master_meta);
  prop->max_boxes
      = Ax::get_property(input, "max_boxes", "nms_static_properties", prop->max_boxes);
  prop->class_agnostic = Ax::get_property(
      input, "class_agnostic", "nms_static_properties", prop->class_agnostic);
  prop->location
      = Ax::get_property(input, "location", "nms_static_properties", prop->location);
  prop->flatten = Ax::get_property(
      input, "flatten_meta", "nms_static_properties", prop->flatten);
  prop->merge = Ax::get_property(input, "merge", "nms_static_properties", prop->merge);

  if (prop->location == "GPU") {
    logger(AX_WARN) << "OpenCL implementation of NMS is not available on this platform, running on CPU."
                    << std::endl;
  } else if (prop->location != "CPU") {
    logger(AX_WARN)
        << prop->location << " is not a valid location. Using CPU." << std::endl;
  }
  prop->location = "CPU";
  return prop;
}

extern "C" void
set_dynamic_properties(const std::unordered_map<std::string, std::string> &input,
    nms_properties *prop, Ax::Logger &logger)
{
  prop->nms_threshold = Ax::get_property(
      input, "nms_threshold", "nms_dynamic_properties", prop->nms_threshold);
}

template <typename T>
void
run_on_cpu_or_gpu(T &meta, const nms_properties *details, Ax::Logger &logger)
{
  auto merge = details->flatten;
  if constexpr (std::is_same_v<T, AxMetaSegmentsDetection>) {
    merge = false;
  }

  meta = non_max_suppression(meta, details->nms_threshold,
      details->class_agnostic, details->max_boxes, merge);
}

std::unique_ptr<AxMetaObjDetection>
make_new_meta(const AxMetaObjDetection &meta)
{
  return std::make_unique<AxMetaObjDetection>();
}

std::unique_ptr<AxMetaKptsDetection>
make_new_meta(const AxMetaKptsDetection &meta)
{
  return std::make_unique<AxMetaKptsDetection>(std::vector<box_xyxy>{},
      KptXyvVector{}, std::vector<float>{}, std::vector<int>{},
      meta.get_kpts_shape(), meta.get_decoder_name());
}

template <typename T>
std::unique_ptr<T>
flatten_metadata_t(std::span<AxMetaBase *const> meta)
{
  auto result = make_new_meta(*dynamic_cast<T *>(meta[0]));
  bool found = false;
  for (auto m : meta) {
    if (auto obj = dynamic_cast<T *>(m)) {
      result->extend(*obj);
      found = true;
    }
  }
  if (found) {
    return result;
  }
  return {};
}

std::unique_ptr<AxMetaBase>
flatten_metadata(const std::vector<AxMetaBase *> &meta)
{
  if (meta.empty()) {
    return nullptr;
  }
  if (dynamic_cast<AxMetaObjDetection *>(meta[0])) {
    return flatten_metadata_t<AxMetaObjDetection>(meta);
  } else if (dynamic_cast<AxMetaKptsDetection *>(meta[0])) {
    return flatten_metadata_t<AxMetaKptsDetection>(meta);
  } else if (dynamic_cast<AxMetaSegmentsDetection *>(meta[0])) {
    //  We do not flatten segments metadata yet
    return nullptr;
  } else {
    throw std::runtime_error("flatten_metadata : Metadata type not supported yet: "
                             + std::string(typeid(*meta[0]).name()));
  }
  return nullptr;
}

extern "C" void
inplace(const AxDataInterface &, const nms_properties *details, unsigned int,
    unsigned int, std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &map,
    Ax::Logger &logger)
{
  std::vector<AxMetaBase *> metas;
  if (details->master_meta.empty()) {
    auto it = map.find(details->meta_key);
    if (it == map.end() || !it->second) {
      logger(AX_INFO) << "No metadata key in inplace_nms" << std::endl;
      return;
    }
    metas = { it->second.get() };
  } else {
    auto master_itr = map.find(details->master_meta);
    if (master_itr == map.end()) {
      logger(AX_ERROR) << "inplace_nms : master_meta not found" << std::endl;
      return;
    }
    metas = master_itr->second->get_submetas(details->meta_key);
  }

  auto flattened = details->flatten ? flatten_metadata(metas) : nullptr;
  if (flattened) {
    for (auto m : metas) {
      if (m) {
        m->enable_extern = false;
      }
    }
    auto ret = map.try_emplace(details->meta_key, std::move(flattened));
    if (!ret.second) {
      logger(AX_ERROR) << "inplace_nms : Failed to insert flattened metadata" << std::endl;
      return;
    }
    metas = { ret.first->second.get() };
  } else if (details->flatten) {
    //  If we get here we have a request to flatten on a type of metadata that does not support it
    //  Currently this is only AxMetaSegmentsDetection
    //  We need to do some smart flattening so that we can perform NMS on all of the submetas together
    //  and then move the submetas to meta_key
    auto flattened = non_max_suppression(metas, details->nms_threshold,
        details->class_agnostic, details->max_boxes);
    auto ret = map.try_emplace(details->meta_key, std::move(flattened));
    if (!ret.second) {
      logger(AX_ERROR) << "inplace_nms : Failed to insert flattened metadata" << std::endl;
    }

    return;
  }
  for (auto m : metas) {
    if (!m) {
      continue;
    }
    if (auto meta = dynamic_cast<AxMetaObjDetection *>(m)) {
      run_on_cpu_or_gpu(*meta, details, logger);
    } else if (auto meta = dynamic_cast<AxMetaKptsDetection *>(m)) {
      run_on_cpu_or_gpu(*meta, details, logger);
    } else if (auto meta = dynamic_cast<AxMetaSegmentsDetection *>(m)) {
      run_on_cpu_or_gpu(*meta, details, logger);
    } else if (auto meta = dynamic_cast<AxMetaPoseSegmentsDetection *>(m)) {
      run_on_cpu_or_gpu(*meta, details, logger);
    } else if (auto meta = dynamic_cast<AxMetaObjDetectionOBB *>(m)) {
      run_on_cpu_or_gpu(*meta, details, logger);
    } else {
      throw std::runtime_error("inplace_nms : Metadata type not supported");
    }
  }
}
