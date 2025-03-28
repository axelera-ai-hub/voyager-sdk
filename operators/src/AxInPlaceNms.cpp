// Copyright Axelera AI, 2023
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
#ifdef OPENCL
  std::unique_ptr<CLNms> nms;
#endif
  std::string meta_key;
  std::string master_meta{};
  int max_boxes{ 10000 };
  float nms_threshold{ 0.5F };
  int class_agnostic{ true };
  std::string location{ "CPU" };
};

bool
is_valid_location(std::string_view location)
{
  return location == "GPU" || location == "CPU";
}

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
  if (!is_valid_location(prop->location)) {
    logger(AX_WARN)
        << prop->location << " is not a valid location. Using CPU." << std::endl;
  }
#ifdef OPENCL
  prop->nms = std::make_unique<CLNms>();
#else
  if (prop->location == "GPU") {
    logger(AX_WARN)
        << "OpenCL not available on this platform, running on CPU." << std::endl;
  }
#endif
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
  if (details->location == "GPU") {
#ifdef OPENCL
    meta = details->nms->run(meta, details->nms_threshold, details->class_agnostic);
    return;
#endif
  }
  meta = non_max_suppression(
      meta, details->nms_threshold, details->class_agnostic, details->max_boxes);
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
    metas.push_back(it->second.get());
  } else {
    auto master_itr = map.find(details->master_meta);
    if (master_itr == map.end()) {
      logger(AX_ERROR) << "inplace_nms : master_meta not found" << std::endl;
      return;
    }
    if (!master_itr->second->submeta_map) {
      logger(AX_ERROR) << "inplace_nms : master_meta has no submeta_map" << std::endl;
      return;
    }
    metas = master_itr->second->submeta_map->get(details->meta_key);
  }

  for (auto m : metas) {
    if (auto meta = dynamic_cast<AxMetaObjDetection *>(m)) {
      run_on_cpu_or_gpu(*meta, details, logger);
    } else if (auto meta = dynamic_cast<AxMetaKptsDetection *>(m)) {
      run_on_cpu_or_gpu(*meta, details, logger);
    } else if (auto meta = dynamic_cast<AxMetaSegmentsDetection *>(m)) {
      run_on_cpu_or_gpu(*meta, details, logger);
    } else {
      throw std::runtime_error("inplace_nms : Metadata type not supported");
    }
  }
}
