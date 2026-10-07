// Copyright Axelera AI, 2023
// Shared helpers for customer decoder unit tests.
#pragma once

#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "AxDataInterface.h"
#include "AxLog.hpp"
#include "AxMeta.hpp"
#include "AxPlugin.hpp"

// Loads a decoder plugin (.so) by name and forwards decode_to_meta to it.
//
// The plugin is resolved via AX_SUBPLUGIN_PATH, which add_customer_tests points
// at the test's own build directory where the customer library is co-located,
// so tests can load their decoder by bare library name.
class Decoder
{
  public:
  Decoder(const std::string &lib, const std::unordered_map<std::string, std::string> &properties)
  {
#ifdef AX_TEST_PLUGIN_DIR
    // When run as a bare binary (outside ctest), point the plugin loader at the
    // test's build directory, where the customer library is co-located. Respect
    // an externally provided value (e.g. ctest's ENVIRONMENT) if already set.
    if (std::getenv("AX_SUBPLUGIN_PATH") == nullptr) {
      setenv("AX_SUBPLUGIN_PATH", AX_TEST_PLUGIN_DIR, 0);
    }
#endif
    std::string options;
    for (const auto &[key, value] : properties) {
      if (!options.empty()) {
        options += ";";
      }
      options += key + ":" + value;
    }
    plugin_ = Ax::load_plugin(logger_, lib, options);
    decode_ = dynamic_cast<Ax::Decode *>(plugin_.get());
    if (!decode_) {
      throw std::runtime_error("Plugin '" + lib + "' is not a decoder");
    }
  }

  void decode_to_meta(const AxTensorsInterface &tensors,
      unsigned int subframe_index, unsigned int number_of_subframes,
      std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &map,
      const AxDataInterface &video)
  {
    decode_->decode_to_meta(tensors, subframe_index, number_of_subframes, map, video);
  }

  private:
  Ax::Logger logger_{ Ax::Severity::error, nullptr, nullptr };
  std::unique_ptr<Ax::Plugin> plugin_;
  Ax::Decode *decode_ = nullptr;
};
