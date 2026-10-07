// Copyright Axelera AI, 2024
#include "AxDataInterface.h"
#include "AxFilterDetections.hpp"
#include "AxLog.hpp"
#include "AxOpUtils.hpp"
#include "AxUtils.hpp"

#include <unordered_set>


extern "C" const std::unordered_set<std::string> &
allowed_properties()
{
  static const std::unordered_set<std::string> allowed_properties{ "input_meta_key",
    "output_meta_key", "hide_output_meta", "which", "top_k", "classes_to_keep",
    "min_width", "min_height", "score" };
  return allowed_properties;
}

extern "C" std::shared_ptr<void>
init_and_set_static_properties(
    const std::unordered_map<std::string, std::string> &input, Ax::Logger &logger)
{
  auto prop = std::make_shared<Ax::FilterDetectionsProperties>();
  prop->input_meta_key = Ax::get_property(input, "input_meta_key",
      "FilterDetectionsProperties", prop->input_meta_key);
  if (prop->input_meta_key.empty()) {
    throw std::runtime_error("filterdetections : input_meta_key is empty");
  }
  prop->output_meta_key = Ax::get_property(input, "output_meta_key",
      "FilterDetectionsProperties", prop->output_meta_key);
  if (prop->output_meta_key.empty()) {
    throw std::runtime_error("filterdetections : output_meta_key is empty");
  }
  prop->hide_output_meta = Ax::get_property(input, "hide_output_meta",
      "FilterDetectionsProperties", prop->hide_output_meta);
  prop->top_k = Ax::get_property(input, "top_k", "FilterDetectionsProperties", prop->top_k);
  std::string which = Ax::get_property(
      input, "which", "FilterDetectionsProperties", std::string{ "NONE" });
  if (which == "NONE") {
    prop->which = Ax::Which::None;
  } else if (which == "SCORE") {
    prop->which = Ax::Which::Score;
  } else if (which == "CENTER") {
    prop->which = Ax::Which::Center;
  } else if (which == "AREA") {
    prop->which = Ax::Which::Area;
  } else {
    throw std::runtime_error("filterdetections : 'which' must be one of NONE, SCORE, CENTER, AREA, instead received "
                             + which);
  }
  if (prop->top_k && prop->which == Ax::Which::None) {
    throw std::runtime_error("filterdetections : top_k is set but which is NONE");
  }
  if (!prop->top_k && prop->which != Ax::Which::None) {
    throw std::runtime_error(
        "filterdetections : which is set to one of SCORE, CENTER, or AREA, but top_k is not set");
  }
  prop->classes_to_keep = Ax::get_property(input, "classes_to_keep",
      "FilterDetectionsProperties", prop->classes_to_keep);
  prop->min_width = Ax::get_property(
      input, "min_width", "FilterDetectionsProperties", prop->min_width);
  prop->min_height = Ax::get_property(
      input, "min_height", "FilterDetectionsProperties", prop->min_height);
  prop->score = Ax::get_property(input, "score", "FilterDetectionsProperties", prop->score);
  return prop;
}

extern "C" void
inplace(const AxDataInterface &interface,
    const Ax::FilterDetectionsProperties *prop, unsigned int, unsigned int,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &meta_map,
    Ax::Logger &logger)
{
  Ax::filter_detections(interface, *prop, meta_map, logger);
}
