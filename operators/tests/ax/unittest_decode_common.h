// Copyright Axelera AI, 2023
#include "AxStreamerUtils.hpp"
#include "unittest_ax_common.h"


class Decoder : public Plugin
{
  public:
  explicit Decoder(const std::string &path,
      const std::unordered_map<std::string, std::string> &input)
      : Plugin(path, input)
  {
    plugin.initialise_function("decode_to_meta", p_decode_to_meta);
  }

  void decode_to_meta(const AxTensorsInterface &tensors,
      unsigned int subframe_index, unsigned int number_of_subframes,
      std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &map,
      const AxVideoInterface &video_info)
  {
    p_decode_to_meta(tensors, properties.get(), subframe_index,
        number_of_subframes, map, video_info, logger);
  }

  private:
  void (*p_decode_to_meta)(const AxTensorsInterface &tensors, const void *data,
      unsigned int subframe_index, unsigned int number_of_subframes,
      std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &map,
      const AxDataInterface &video_info, Ax::Logger &logger)
      = nullptr;
};
