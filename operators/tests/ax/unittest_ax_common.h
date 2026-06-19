// Copyright Axelera AI, 2023
#include <gtest/gtest.h>

#include <gmodule.h>

#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif
#include "AxDataInterface.h"
#include "AxLog.hpp"
#include "AxMeta.hpp"
#include "AxMetaBBox.hpp"
#include "AxMetaClassification.hpp"
#include "AxOpenClExtensions.hpp"
#include "AxStreamerUtils.hpp"


namespace fs = std::filesystem;


class tempfile
{
  public:
  explicit tempfile(const std::string &content)
  {
#ifdef _WIN32
    char buf[MAX_PATH];
    if (!GetTempFileNameA(fs::temp_directory_path().string().c_str(), "ax_", 0, buf)) {
      throw std::runtime_error("Failed to create temporary file");
    }
    path_ = buf;
#else
    auto tmpl = (fs::temp_directory_path() / "ax.XXXXXX").string();
    int fd = ::mkstemp(tmpl.data());
    if (fd == -1) {
      throw std::runtime_error("Failed to create temporary file");
    }
    ::close(fd);
    path_ = tmpl;
#endif
    std::ofstream f(path_, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!f.write(content.data(), content.size())) {
      fs::remove(path_);
      throw std::runtime_error("Failed to write temporary file");
    }
  }

  tempfile(const tempfile &) = delete;
  tempfile &operator=(const tempfile &) = delete;

  ~tempfile()
  {
    fs::remove(path_);
  }

  std::string filename() const
  {
    return path_.string();
  }

  private:
  fs::path path_;
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
#ifdef _WIN32
  return false;
#else
  return fs::is_directory("/dev/dma_heap");
#endif
}

struct FormatParam {
  AxVideoFormat format;
  int out_format;
};

namespace Ax
{
inline std::string
StringMapAsOptions(const StringMap &m)
{
  std::vector<std::string> pairs;
  pairs.reserve(m.size());
  for (const auto &p : m) {
    pairs.push_back(p.first + ":" + p.second);
  }
  return Ax::Internal::join(pairs, ";");
}

template <typename Plugin>
std::unique_ptr<Plugin>
LoadPlugin(std::string name, const StringMap &input)
{
  static Ax::Logger logger{ Ax::Severity::error, nullptr, nullptr };
  auto opts = StringMapAsOptions(input);
  auto plugin = Ax::load_plugin(logger, name, opts);
  return std::unique_ptr<Plugin>(static_cast<Plugin *>(plugin.release()));
}

inline auto
LoadInPlace(const std::string &name, const StringMap &input)
{
  return LoadPlugin<Ax::InPlace>("inplace_" + name, input);
}

inline auto
LoadTransform(const std::string &name, const StringMap &input)
{
  return LoadPlugin<Ax::Transform>("transform_" + name, input);
}

inline auto
LoadDecode(const std::string &name, const StringMap &input)
{
  return LoadPlugin<Ax::Decode>("decode_" + name, input);
}

class buffer_planes
{
  public:
  buffer_planes(std::vector<std::vector<uint8_t>> inmem_planes)
      : planes(inmem_planes)
  {
    for (auto &p : planes) {
      opencl_buffer cl_plane = { {}, ax_utils::cl_object<cl_event>{ nullptr },
        { p.data(), p.size() }, p.data(), {} };
      cl_planes.push_back(std::move(cl_plane));
    }
  }

  buffer_planes(buffer_planes &&rhs)
      : planes(std::move(rhs.planes)),
        cl_planes(std::move(rhs.cl_planes))
  {
  }

  opencl_planes get_planes()
  {
    opencl_planes result{};
    for (auto &p : cl_planes) {
      result.planes.push_back(&p);
    }
    return result;
  }

  ~buffer_planes()
  {
    for (auto &p : cl_planes) {
      if (p.buffer) {
        clReleaseMemObject(p.buffer);
        p.buffer = nullptr;
      }
      p.event.reset();
    }
  }

  private:
  std::vector<std::vector<uint8_t>> planes;
  std::vector<opencl_buffer> cl_planes;
};


} // namespace Ax
