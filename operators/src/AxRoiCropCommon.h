// Copyright Axelera AI, 2026
#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>
#include "AxDataInterface.h"
#include "AxLog.hpp"
#include "AxMetaBBox.hpp"

// Common property fields shared by the CPU and OpenCL roicrop operators.
struct RoiCropParams {
  std::string meta_key{};
  int left{ -1 };
  int top{ -1 };
  int width{ -1 };
  int height{ -1 };
  bool downstream_supports_opencl{ false };
};

const std::unordered_set<std::string> &roicrop_allowed_properties();

// Parse and validate the static roicrop properties from the input map.
// `name` is used in error messages to identify which operator is reporting.
void parse_roicrop_params(const std::unordered_map<std::string, std::string> &input,
    RoiCropParams &prop, const std::string &name, Ax::Logger &logger);

float roicrop_get_margin(
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &meta_map);

BboxXyxy roicrop_get_roi_with_margin(const RoiCropParams &prop,
    unsigned int subframe_index, unsigned int number_of_subframes,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &meta_map,
    const std::string &name, Ax::Logger &logger);

AxDataInterface roicrop_set_output_interface(const AxDataInterface &interface,
    const RoiCropParams &prop, unsigned int subframe_index, unsigned int number_of_subframes,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &meta_map,
    const std::string &name, Ax::Logger &logger);
