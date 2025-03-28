// Copyright Axelera AI, 2023
#include "AxDataInterface.h"
#include "AxLog.hpp"
#include "AxMetaStreamId.hpp"
#include "AxUtils.hpp"

#include <unordered_set>

struct streamid_properties {
  std::string meta_key{ "stream_id" };
  int stream_id{ 0 };
};

extern "C" const std::unordered_set<std::string> &
allowed_properties()
{
  static const std::unordered_set<std::string> allowed_properties{ "meta_key", "stream_id" };
  return allowed_properties;
}

extern "C" std::shared_ptr<void>
init_and_set_static_properties(
    const std::unordered_map<std::string, std::string> &input, Ax::Logger &logger)
{
  auto prop = std::make_shared<streamid_properties>();
  prop->meta_key = Ax::get_property(
      input, "meta_key", "addstreamid_static_properties", prop->meta_key);
  prop->stream_id = Ax::get_property(
      input, "stream_id", "addstreamid_static_properties", prop->stream_id);
  return prop;
}

extern "C" void
inplace(const AxDataInterface &, const streamid_properties *details, unsigned int,
    unsigned int, std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &map,
    Ax::Logger &logger)
{
  if (map.count(details->meta_key)) {
    logger(AX_ERROR) << "inplace_addstreamid: meta_key (" << details->meta_key
                     << ") already exists" << std::endl;
    throw std::runtime_error("inplace_addstreamid: meta_key already exists");
  }
  map[details->meta_key] = std::make_unique<AxMetaStreamId>(details->stream_id);
}
