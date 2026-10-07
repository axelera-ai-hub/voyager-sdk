// Copyright Axelera AI, 2023
#include "AxDataInterface.h"
#include "AxLog.hpp"
#include "AxMetaStreamId.hpp"
#include "AxUtils.hpp"

#include <chrono>
#include <thread>
#include <unordered_set>

using std::chrono::steady_clock;
template <typename Clock> using time_point = std::chrono::time_point<Clock>;

struct streamid_properties {
  std::string meta_key{ "stream_id" };
  int stream_id{ 0 };
  int fps_limit{ 0 };
  time_point<steady_clock> last_time;
};

extern "C" const std::unordered_set<std::string> &
allowed_properties()
{
  static const std::unordered_set<std::string> allowed_properties{ "meta_key",
    "stream_id", "fps_limit" };
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
  prop->fps_limit = Ax::get_property(
      input, "fps_limit", "addstreamid_static_properties", prop->fps_limit);
  return prop;
}

static void
wait_for_duration(time_point<steady_clock> start_time, std::chrono::microseconds duration)
{
  auto now = steady_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - start_time);
  if (elapsed < duration) {
    std::this_thread::sleep_for(duration - elapsed);
  }
}

extern "C" void
inplace(const AxDataInterface &, streamid_properties *details, unsigned int, unsigned int,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &map, Ax::Logger &logger)
{
  if (map.count(details->meta_key)) {
    logger(AX_ERROR) << "inplace_addstreamid: meta_key (" << details->meta_key
                     << ") already exists" << std::endl;
    throw std::runtime_error("inplace_addstreamid: meta_key already exists");
  }
  if (details->fps_limit > 0) {
    const auto us = std::chrono::microseconds(1000000 / details->fps_limit);
    wait_for_duration(details->last_time, us);
    details->last_time = steady_clock::now();
  }
  map[details->meta_key] = std::make_unique<AxMetaStreamId>(details->stream_id);
}
