// Copyright Axelera AI, 2023
#include "unittest_ax_common.h"


class Inplacer : public Plugin
{
  public:
  Inplacer(const std::string &path, const std::unordered_map<std::string, std::string> &input)
      : Plugin(path, input)
  {
    plugin.initialise_function("inplace", p_inplace);
  }

  void inplace(const AxDataInterface &interface,
      std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &map,
      unsigned int subframe_index = 0, unsigned int number_of_subframes = 1)
  {
    p_inplace(interface, properties.get(), subframe_index, number_of_subframes, map, logger);
  }

  void inplace(const AxDataInterface &interface)
  {
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map = {};
    inplace(interface, map);
  }

  private:
  void (*p_inplace)(const AxDataInterface &interface, const void *subplugin_properties,
      unsigned int subframe_index, unsigned int number_of_subframes,
      std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &meta_map,
      Ax::Logger &logger)
      = nullptr;
};
