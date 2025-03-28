// Copyright Axelera AI, 2025
#pragma once

#include <iostream>
#include <optional>
#include <ostream>
#include <streambuf>
#include <string>

namespace Ax
{
enum class Severity { trace, log, debug, info, fixme, warning, error };

struct SeverityTag {
  Severity severity;
  std::string file;
  int line;
  std::string function;
};
std::ostream &operator<<(std::ostream &s, Severity logSeverity);

struct Tag {
  void *ptr{};
  void *category{};
};

class Logger
{
  class null_buffer : public std::streambuf
  {
    public:
    null_buffer();

    int overflow(int c);
  };

  public:
  class null_stream : public std::ostream
  {
    public:
    null_stream();

    private:
    null_buffer m_sb;
  };


  public:
  /// @brief Constructor
  /// @param severity - The severity level to filter for this logger
  /// @param source - For a gstreamer element this should be the gst element
  /// @param debug - For a gstreamer element this should be a debug category
  Logger(Severity severity, void *source, void *debug);

  /// @brief Overloaded operator() to return the logger or the null sink
  /// @param severity - Return real logger if severity_ is less than or equal to this
  /// @return - The logger or the null sink
  ///
  /// This enables usage like this:
  ///   logger (Ax::Severity::info) << "This is an info message";
  ///   logger (Ax::Severity::debug) << "This is a debug message";
  std::ostream &operator()(Ax::SeverityTag severity);


  using Sink = void (*)(SeverityTag, Tag, const std::string &);
  void init_log_sink(Ax::Severity severity, Sink sink);

  private:
  Ax::Severity severity_;
  null_stream null_sink_;
  void *source_;
  void *debug_;
  inline static bool init = false;
};

///
/// @brief Validate the severity level
/// @param severity - string representing the desired log level
/// @return - If valid an engaged optional holding the actual severity
///           else a disenegaged optional
///
inline std::optional<Severity> extract_severity(std::string severity);

std::string to_string(Ax::Severity logSeverity);

} // namespace Ax

#define AX_TRACE                                          \
  Ax::SeverityTag                                         \
  {                                                       \
    Ax::Severity::trace, __FILE__, __LINE__, __FUNCTION__ \
  }
#define AX_LOG                                          \
  Ax::SeverityTag                                       \
  {                                                     \
    Ax::Severity::log, __FILE__, __LINE__, __FUNCTION__ \
  }
#define AX_DEBUG                                          \
  Ax::SeverityTag                                         \
  {                                                       \
    Ax::Severity::debug, __FILE__, __LINE__, __FUNCTION__ \
  }
#define AX_INFO                                          \
  Ax::SeverityTag                                        \
  {                                                      \
    Ax::Severity::info, __FILE__, __LINE__, __FUNCTION__ \
  }
#define AX_FIXME                                          \
  Ax::SeverityTag                                         \
  {                                                       \
    Ax::Severity::fixme, __FILE__, __LINE__, __FUNCTION__ \
  }
#define AX_WARN                                             \
  Ax::SeverityTag                                           \
  {                                                         \
    Ax::Severity::warning, __FILE__, __LINE__, __FUNCTION__ \
  }
#define AX_ERROR                                          \
  Ax::SeverityTag                                         \
  {                                                       \
    Ax::Severity::error, __FILE__, __LINE__, __FUNCTION__ \
  }
