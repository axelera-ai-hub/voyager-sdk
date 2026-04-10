// Copyright Axelera AI, 2025
#pragma once
#include "AxDataInterface.h"
#include "AxLog.hpp"
#include "AxMeta.hpp"

#include <memory>
#include <unordered_map>
#include <unordered_set>

struct TrackingElement;
struct TrackingDescriptor;

namespace Ax
{
using MetaMap = std::unordered_map<std::string, std::unique_ptr<AxMetaBase>>;
using StringMap = std::unordered_map<std::string, std::string>;
using StringSet = std::unordered_set<std::string>;

enum class PluginFeature {
  no_feature = 0,
  vaapi_buffers,
  dmabuf_buffers,
  opencl_buffers,
  crop_meta,
  video_meta,
};

inline bool
PluginFeatureDefaults(PluginFeature feature)
{
  switch (feature) {
    case PluginFeature::video_meta:
      return true;
    default:
      return false;
  }
}

namespace V1Plugin
{

struct Base {
  std::shared_ptr<void> (*init_and_set_static_properties)(
      const StringMap &options, Logger &logger)
      = nullptr;
  const StringSet &(*allowed_properties)() = nullptr;
  void (*set_dynamic_properties)(const StringMap &options, void *properties, Logger &logger)
      = nullptr;

  std::shared_ptr<void> (*init_and_set_static_properties_with_context)(
      const StringMap &options, AxAllocationContext *context, Logger &logger)
      = nullptr;
};

struct InPlace : Base {
  void (*inplace)(const AxDataInterface &interface, const void *properties,
      unsigned int subframe_index, unsigned int subframe_count,
      MetaMap &meta_map, Logger &logger)
      = nullptr;
};

struct Transform : Base {
  void (*transform)(const AxDataInterface &input, const AxDataInterface &output,
      const void *properties, unsigned int subframe_index,
      unsigned int subframe_count, MetaMap &meta_map, Logger &logger)
      = nullptr;

  AxDataInterface (*set_output_interface)(
      const AxDataInterface &input, const void *properties, Logger &logger)
      = nullptr;

  bool (*can_passthrough)(const AxDataInterface &input,
      const AxDataInterface &output, const void *properties, Logger &logger)
      = nullptr;

  AxDataInterface (*set_output_interface_from_meta)(const AxDataInterface &interface,
      const void *properties, unsigned int subframe_index,
      unsigned int subframe_count, MetaMap &meta_map, Logger &logger)
      = nullptr;

  // always valid (we provide a default implementation if not set)
  bool (*query_supports)(PluginFeature feature, const void *properties, Logger &logger)
      = nullptr;
};

struct Decode : Base {

  void (*decode_to_meta)(const AxTensorsInterface &tensors_interface,
      const void *properties, unsigned int subframe_index, unsigned int subframe_count,
      MetaMap &meta_map, const AxDataInterface &video, Logger &logger)
      = nullptr;
};

struct DetermineObjectAttribute : Base {
  std::unique_ptr<AxMetaBase> (*determine_object_attribute)(const void *properties,
      int first_id, int frame_id, uint8_t key,
      const std::unordered_map<int, TrackingElement> &frame_id_to_element, Logger &logger)
      = nullptr;
};

struct TrackerFilter : Base {
  bool (*filter)(const void *properties,
      const TrackingDescriptor &tracking_descriptor, Logger &logger)
      = nullptr;
};

} // namespace V1Plugin

// pure interfaces to the
class Plugin
{
  public:
  virtual std::string name() const = 0;
  virtual std::string mode() const = 0; // TODO needs a better name
  virtual const Ax::StringSet &allowed_properties() const = 0;
  virtual void set_dynamic_properties(const Ax::StringMap &options) = 0;

  virtual ~Plugin() = default;
};


class InPlace : public Plugin
{
  public:
  virtual bool has_inplace() const = 0;
  virtual void inplace(const AxDataInterface &interface,
      unsigned int subframe_index, unsigned int number_of_subframes, MetaMap &map)
      = 0;
};

class Transform : public Plugin
{
  public:
  virtual bool has_transform() const = 0;
  virtual bool has_set_output_interface() const = 0;
  virtual bool has_set_output_interface_from_meta() const = 0;

  virtual void transform(const AxDataInterface &input, const AxDataInterface &output,
      unsigned int subframe_index, unsigned int subframe_count, MetaMap &meta_map)
      = 0;

  virtual AxDataInterface set_output_interface(const AxDataInterface &input) = 0;

  virtual AxDataInterface set_output_interface_from_meta(const AxDataInterface &input,
      unsigned int subframe_index, unsigned int number_of_subframes,
      std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &meta_map)
      = 0;

  virtual bool can_passthrough(const AxDataInterface &input, const AxDataInterface &output) const
      = 0;

  virtual bool query_supports(PluginFeature feature) const = 0;
};

class Decode : public Plugin
{
  public:
  virtual void decode_to_meta(const AxTensorsInterface &tensors, unsigned int subframe_index,
      unsigned int number_of_subframes, MetaMap &map, const AxDataInterface &video)
      = 0;
};


/// @brief  Load a plugin from a shared library name
/// @returns Returns a Plugin pointer (InPlace, Transform, or Decode)
/// @throws std::runtime_error if the library cannot be loaded
/// @param logger Logger to use for logging
/// @param name Name of the plugin to load.
/// This can be a full path, a library name, or just the plugin name. For example
/// "/path/to/libdecode_yolo.so", "libdecode_yolo.so", or "decode_yolo" are all valid.
/// @param options Options string to pass to the plugin
/// @param context Allocation context to use for the plugin
/// @param mode Mode string to pass to the plugin, this is only relevant to InPlace plugins.
std::unique_ptr<Plugin> load_plugin(Logger &logger, const std::string &name,
    const std::string &options = "", AxAllocationContext *context = nullptr,
    const std::string &mode = "none");

} // namespace Ax
