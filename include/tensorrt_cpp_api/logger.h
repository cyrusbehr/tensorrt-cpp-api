#pragma once

#include <cstdint>
#include <memory>
#include <string_view>

namespace trtcpp {

/// Log severity, ordered ascending by importance (kVerbose=0 .. kInternalError=4). Each
/// value corresponds to an nvinfer1::ILogger::Severity BY NAME, not by integer: nvinfer1's
/// integer order is the reverse (kINTERNAL_ERROR=0 .. kVERBOSE=4), so the TensorRT
/// bridge must translate per enumerator and never via a plain static_cast (that would
/// invert every level).
enum class Severity : std::uint8_t {
    kVerbose,
    kInfo,
    kWarning,
    kError,
    kInternalError,
};

std::string_view toString(Severity s) noexcept;

/// Injectable logger. The library NEVER logs to a global -- every component takes an
/// ILogger (or accepts the default), fixing v6's hard-wired global spdlog. Implementations
/// must be thread-safe and must not throw.
class ILogger {
public:
    virtual ~ILogger() = default;
    virtual void log(Severity severity, std::string_view message) noexcept = 0;
};

/// Process-wide thread-safe stderr logger. Its threshold is read once from the
/// TRTCPP_LOG_LEVEL env var (verbose|info|warning|error|off; default info), with
/// LOG_LEVEL honored as a v6-compatibility fallback.
std::shared_ptr<ILogger> defaultLogger();

/// A stderr logger with an explicit severity threshold (messages below it are dropped).
std::shared_ptr<ILogger> makeStderrLogger(Severity threshold);

#ifdef TRT_CPP_API_WITH_SPDLOG
/// Routes log records through spdlog's default logger. Only declared/available when the
/// library is built with -DTRT_CPP_API_WITH_SPDLOG=ON (which links spdlog::spdlog and
/// defines this macro on the public interface).
std::shared_ptr<ILogger> makeSpdlogLogger();
#endif

} // namespace trtcpp
