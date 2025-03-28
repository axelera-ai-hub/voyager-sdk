// Copyright Axelera AI, 2023
#include "unittest_ax_common.h"

struct FormatParam {
  AxVideoFormat format;
  int bgra_out;
};

class ColorFormatFixture : public ::testing::TestWithParam<FormatParam>
{
};

class Transformer : public Plugin
{

  public:
  Transformer(const std::string &path,
      const std::unordered_map<std::string, std::string> &input)
      : Plugin(path, input)
  {
    plugin.initialise_function("set_output_interface", p_set_output_interface);
    plugin.initialise_function("transform", p_transform);
    plugin.initialise_function("can_passthrough", p_can_passthrough, false);
    plugin.initialise_function("can_use_dmabuf", p_can_use_dmabuf, false);
  }

  void transform(const AxDataInterface &input, const AxDataInterface &output,
      std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &map,
      unsigned int subframe_index = 0, unsigned int number_of_subframes = 1)
  {
    p_transform(input, output, properties.get(), subframe_index,
        number_of_subframes, map, logger);
  }

  AxDataInterface set_output_interface(const AxDataInterface &input)
  {
    return p_set_output_interface(input, properties.get(), logger);
  }

  void transform(const AxDataInterface &inp, const AxDataInterface &out)
  {
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map = {};
    return transform(inp, out, map);
  }

  bool can_passthrough(const AxDataInterface &input, const AxDataInterface &output)
  {
    return p_can_passthrough && p_can_passthrough(input, output, logger);
  }

  private:
  void (*p_transform)(const AxDataInterface &input_interface,
      const AxDataInterface &output_interface, const void *subplugin_properties,
      unsigned int subframe_index, unsigned int number_of_subframes,
      std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &meta_map,
      Ax::Logger &logger)
      = nullptr;

  AxDataInterface (*p_set_output_interface)(const AxDataInterface &interface,
      const void *subplugin_properties, Ax::Logger &logger)
      = nullptr;

  bool (*p_can_passthrough)(const AxDataInterface &input,
      const AxDataInterface &output, Ax::Logger &logger)
      = nullptr;

  bool (*p_can_use_dmabuf)(const void *subplugin_properties, Ax::Logger &logger) = nullptr;
};
