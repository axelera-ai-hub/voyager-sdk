// Copyright Axelera AI, 2023
#include <gtest/gtest.h>

#include <gmodule.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include <filesystem>
#include "AxDataInterface.h"
#include "AxLog.hpp"
#include "AxMeta.hpp"
#include "AxMetaClassification.hpp"
#include "AxStreamerUtils.hpp"

namespace fs = std::filesystem;


class tempfile
{
  public:
  explicit tempfile(const std::string &content)
  {
    int fd = ::mkstemp(name.data());
    if (fd == -1) {
      throw std::runtime_error("Failed to create temporary file");
    }
    auto total_written = ssize_t{ 0 };
    auto to_write = content.size();
    const char *p = content.c_str();
    while (total_written != content.size()) {
      auto num_written = ::write(fd, p, to_write - total_written);
      if (num_written == -1) {
        ::close(fd);
        throw std::runtime_error("Failed to write temporary file");
      }
      total_written += num_written;
      p += num_written;
    }
    ::close(fd);
  }

  tempfile(const tempfile &) = delete;
  tempfile &operator=(const tempfile &) = delete;

  ~tempfile()
  {
    ::unlink(name.c_str());
  }

  std::string filename() const
  {
    return name;
  }

  private:
  std::string name = "/tmp/ax.XXXXXX";
};

template <typename T>
AxTensorsInterface
tensors_from_vector(std::vector<T> &tensors)
{
  return { { { int(tensors.size()) }, sizeof tensors[0], tensors.data() } };
}

inline bool
has_dma_heap()
{
  return fs::is_directory("/dev/dma_heap");
}

class Plugin
{
  static std::string get_plugin_path(std::string plugin)
  {
    const auto val = std::getenv("AX_SUBPLUGIN_PATH");
    return Ax::libname(val && val[0] ? std::string{ fs::path(val) / plugin } : plugin);
  }

  public:
  explicit Plugin(const std::string &path,
      const std::unordered_map<std::string, std::string> &input)
      : plugin(logger, get_plugin_path(path))
  {

    std::shared_ptr<void> (*p_init_and_set_static_properties)(
        const std::unordered_map<std::string, std::string> &input, Ax::Logger &logger)
        = nullptr;
    plugin.initialise_function("init_and_set_static_properties", p_init_and_set_static_properties);

    properties = p_init_and_set_static_properties(input, logger);

    void (*p_set_dynamic_properties)(const std::unordered_map<std::string, std::string> &input,
        void *data, Ax::Logger &logger)
        = nullptr;
    plugin.initialise_function("set_dynamic_properties", p_set_dynamic_properties, false);
    if (p_set_dynamic_properties) {
      p_set_dynamic_properties(input, properties.get(), logger);
    }
  }

  protected:
  std::shared_ptr<void> properties;
  Ax::Logger logger{ Ax::Severity::error, nullptr, nullptr };
  Ax::SharedLib plugin;
};
