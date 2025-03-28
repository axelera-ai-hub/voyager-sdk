// Copyright Axelera AI, 2025
#pragma once
#include "AxDataInterface.h"
#include "AxLog.hpp"
#include "AxMeta.hpp"

#include <unordered_map>
#include <unordered_set>


namespace Ax
{
using AsyncCompleter = std::function<void()>;
using MetaMap = std::unordered_map<std::string, std::unique_ptr<AxMetaBase>>;
using StringMap = std::unordered_map<std::string, std::string>;
using StringSet = std::unordered_set<std::string>;

namespace V1Plugin
{

struct Base {
  std::shared_ptr<void> (*init_and_set_static_properties)(
      const StringMap &options, Logger &logger)
      = nullptr;
  const StringSet &(*allowed_properties)() = nullptr;
  void (*set_dynamic_properties)(
      const StringMap &options, void *subplugin_properties, Logger &logger)
      = nullptr;
};

struct InPlace : Base {
  void (*inplace)(const AxDataInterface &interface,
      const void *subplugin_properties, unsigned int subframe_index,
      unsigned int number_of_subframes, MetaMap &meta_map, Logger &logger)
      = nullptr;
};

struct Transform : Base {
  void (*transform)(const AxDataInterface &input_interface,
      const AxDataInterface &output_interface, const void *subplugin_properties,
      unsigned int subframe_index, unsigned int number_of_subframes,
      MetaMap &meta_map, Logger &logger)
      = nullptr;

  AsyncCompleter (*transform_async)(const AxDataInterface &input_interface,
      const AxDataInterface &output_interface, const void *subplugin_properties,
      unsigned int subframe_index, unsigned int number_of_subframes,
      MetaMap &meta_map, Logger &logger)
      = nullptr;

  AxDataInterface (*set_output_interface)(const AxDataInterface &interface,
      const void *subplugin_properties, Logger &logger)
      = nullptr;

  bool (*can_passthrough)(const AxDataInterface &input,
      const AxDataInterface &output, const void *subplugin_properties, Logger &logger)
      = nullptr;

  bool (*handles_crop_meta)() = nullptr;

  AxDataInterface (*set_output_interface_from_meta)(const AxDataInterface &interface,
      const void *subplugin_properties, unsigned int subframe_index,
      unsigned int number_of_subframes, MetaMap &meta_map, Logger &logger)
      = nullptr;

  bool (*can_use_dmabuf)(const void *subplugin_properties, Ax::Logger &logger) = nullptr;
};

struct Decoder : Base {

  void (*decode_to_meta)(const AxTensorsInterface &tensors_interface,
      const void *subplugin_properties, unsigned int subframe_index,
      unsigned int number_of_subframes, MetaMap &meta_map,
      const AxDataInterface &srcpad_info, Logger &logger)
      = nullptr;
};
} // namespace V1Plugin

} // namespace Ax
