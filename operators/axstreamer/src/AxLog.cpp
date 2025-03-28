// Copyright Axelera AI, 2025
#include "AxLog.hpp"
#include <algorithm>
#include <mutex>
#include <sstream>
#include <thread>
#include <unordered_map>

namespace Ax
{

/// @brief The AxLogger class is an internal class and should not be
/// instantiated directly. Instead, use the Ax::Logger class.
class AxLogger : public std::streambuf
{
  //  Each thread has its own stream which means it potentially has its own
  //  severity and tag. This struct stores the details of the stream for each
  //  thread. This, alomg with some thread detection and a mutex helps prevent
  //  interleaving of messages from different threads.
  struct stream_details {
    std::stringstream strm{};
    Tag tag{};
    SeverityTag severity{};
  };

  public:
  AxLogger() noexcept;

  ~AxLogger() noexcept;

  static AxLogger &instance();

  /// Without "init" every LOG(X) will simply go to clog
  static void init(Severity level, Logger::Sink log_sink);

  void add_logsink(Severity level, Logger::Sink sink);

  private:
  int sync(stream_details &strm);

  int sync() override;

  int overflow(int c) override;

  std::streamsize xsputn(const char *s, std::streamsize n) override;

  friend std::ostream &operator<<(std::ostream &os, Ax::SeverityTag log_severity);
  friend std::ostream &operator<<(std::ostream &os, const Tag &tag);

  stream_details &get_stream();

  Severity severity_filter;
  // one buffer per thread to avoid mixed log lines
  std::mutex mutex_;
  std::unordered_map<std::thread::id, stream_details> buffer_;
  /// the last thread id
  std::thread::id last_id_;
  /// the last buffer
  stream_details *last_buffer_ = nullptr;
  Logger::Sink log_sink;
};

AxLogger::AxLogger() noexcept : last_buffer_(nullptr)
{
  std::clog.rdbuf(this);
  //  Initialise the first stream
  std::lock_guard<std::mutex> lock(mutex_);
  auto id = std::this_thread::get_id();
  last_id_ = id;
  last_buffer_ = &(buffer_[id]);
}

AxLogger::~AxLogger() noexcept
{
  sync();
}

AxLogger &
AxLogger::instance()
{
  static AxLogger instance_;
  return instance_;
}

/// Without "init" every LOG(X) will simply go to clog
void
AxLogger::init(Severity level, Logger::Sink log_sink)
{
  AxLogger::instance().add_logsink(level, log_sink);
}

void
AxLogger::add_logsink(Severity level, Logger::Sink sink)
{
  severity_filter = level;
  log_sink = sink;
}

int
AxLogger::sync(stream_details &strm)
{
  if (!strm.strm.str().empty()) {
    if (strm.severity.severity >= severity_filter) {
      log_sink(strm.severity, strm.tag, strm.strm.str());
    }
  }
  strm.strm.str("");
  strm.strm.clear();
  return 0;
}

int
AxLogger::sync()
{
  auto &strm = get_stream();
  return sync(strm);
}

int
AxLogger::overflow(int c)
{
  if (c == '\n' || c == EOF) {
    sync();
  } else {
    auto &strm = get_stream();
    strm.strm << static_cast<char>(c);
  }
  return c;
}

std::streamsize
AxLogger::xsputn(const char *s, std::streamsize n)
{
  auto &strm = get_stream();
  strm.strm.write(s, n);
  return n;
}

AxLogger::stream_details &
AxLogger::get_stream()
{
  std::lock_guard<std::mutex> lock(mutex_);
  auto id = std::this_thread::get_id();
  if (last_id_ != id) {
    last_id_ = id;
    last_buffer_ = &(buffer_[id]);
  }
  return *last_buffer_;
}


std::ostream &
operator<<(std::ostream &os, Ax::SeverityTag log_severity)
{
  if (AxLogger *log = dynamic_cast<AxLogger *>(os.rdbuf())) {
    auto &strm = log->get_stream();
    if (strm.severity.severity != log_severity.severity) {
      log->sync(strm);
      strm.severity = log_severity;
    }
  } else {
    os << std::to_string(
        static_cast<std::underlying_type_t<Severity>>(log_severity.severity));
  }
  return os;
}

std::ostream &
operator<<(std::ostream &os, const Tag &tag)
{
  if (AxLogger *log = dynamic_cast<AxLogger *>(os.rdbuf())) {
    auto &strm = log->get_stream();
    if (strm.tag.ptr != tag.ptr || strm.tag.category != tag.category) {
      log->sync(strm);
      strm.tag = tag;
    }
  }
  return os;
}

Logger::null_buffer::null_buffer() : std::streambuf()
{
  setp(nullptr, nullptr);
}

int
Logger::null_buffer::overflow(int c)
{
  return c;
}
Logger::null_stream::null_stream() : std::ostream(&m_sb)
{
}


Logger::Logger(Severity severity, void *source, void *debug)
    : severity_(severity), source_(source), debug_(debug)
{
}

void
Logger::init_log_sink(Ax::Severity severity, Sink sink)
{
  if (!init) {
    AxLogger::instance().add_logsink(severity, sink);
    init = true;
  }
}


/// @brief Overloaded operator() to return the logger or the null sink
/// @param severity - Return real logger if severity_ is less than or equal to this
/// @return - The logger or the null sink
///
/// This enables usage like this:
///   logger (Ax::Severity::info) << "This is an info message";
///   logger (Ax::Severity::debug) << "This is a debug message";
std::ostream &
Logger::operator()(Ax::SeverityTag severity)
{
  //  If the requested setting is less than or equal to the current setting
  //  then return the logger, else return the null sink

  if (severity_ <= severity.severity) {
    if (source_ == nullptr && debug_ == nullptr) {
      return std::clog << to_string(severity.severity) << " : ";
    }
    return std::clog << severity << Tag{ source_, debug_ };
  }
  return null_sink_;
}


///
/// @brief Validate the severity level
/// @param severity - string representing the desired log level
/// @return - If valid an engaged optional holding the actual severity
///           else a disenegaged optional
///
std::optional<Severity>
extract_severity(std::string severity)
{
  struct log_levels {
    std::string name;
    Ax::Severity severity;
  };

  log_levels valid_levels[] = {
    { "trace", Ax::Severity::trace },
    { "debug", Ax::Severity::debug },
    { "info", Ax::Severity::info },
    { "warning", Ax::Severity::warning },
    { "error", Ax::Severity::error },
  };

  std::transform(severity.begin(), severity.end(), severity.begin(),
      [](unsigned char c) { return std::tolower(c); });
  auto found = std::find_if(std::begin(valid_levels), std::end(valid_levels),
      [&](const auto &valid) { return severity == valid.name; });
  if (found != std::end(valid_levels)) {
    return found->severity;
  }
  return {};
}

std::string
to_string(Ax::Severity logSeverity)
{
  switch (logSeverity) {
    case Ax::Severity::trace:
      return "Trace";
    case Ax::Severity::log:
      return "Log";
    case Ax::Severity::debug:
      return "Debug";
    case Ax::Severity::info:
      return "Info";
    case Ax::Severity::fixme:
      return "Fixme";
    case Ax::Severity::warning:
      return "Warn";
    case Ax::Severity::error:
      return "Error";
    default:
      return std::to_string(static_cast<int>(logSeverity));
  }
}

std::ostream &
operator<<(std::ostream &s, Ax::Severity logSeverity)
{
  return s << to_string(logSeverity);
}

} // namespace Ax
