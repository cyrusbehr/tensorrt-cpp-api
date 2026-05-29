#include "tensorrt_cpp_api/logger.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>

#ifdef TRT_CPP_API_WITH_SPDLOG
#include <spdlog/spdlog.h>
#endif

namespace trtcpp {

std::string_view toString(Severity s) noexcept {
    switch (s) {
    case Severity::kVerbose:
        return "VERBOSE";
    case Severity::kInfo:
        return "INFO";
    case Severity::kWarning:
        return "WARNING";
    case Severity::kError:
        return "ERROR";
    case Severity::kInternalError:
        return "INTERNAL_ERROR";
    }
    return "UNKNOWN";
}

namespace {

// threshold_ is an int (not a Severity) so the "off" level can sit one past
// kInternalError (value 5) and drop every real message, including internal errors.
class StderrLogger final : public ILogger {
public:
    explicit StderrLogger(int threshold) : threshold_(threshold) {}

    void log(Severity severity, std::string_view message) noexcept override {
        if (static_cast<int>(severity) < threshold_) {
            return;
        }
        const std::string_view tag = toString(severity);
        std::lock_guard<std::mutex> lock(mutex_);
        std::fprintf(stderr, "[trtcpp] [%.*s] %.*s\n", static_cast<int>(tag.size()), tag.data(), static_cast<int>(message.size()),
                     message.data());
        std::fflush(stderr);
    }

private:
    int threshold_;
    std::mutex mutex_;
};

constexpr int kSilentThreshold = static_cast<int>(Severity::kInternalError) + 1;

int thresholdFromEnv() {
    const char *value = std::getenv("TRTCPP_LOG_LEVEL");
    if (value == nullptr) {
        value = std::getenv("LOG_LEVEL"); // v6 compatibility
    }
    std::string level = (value != nullptr) ? value : "info";
    for (char &c : level) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    if (level == "verbose" || level == "trace" || level == "debug") {
        return static_cast<int>(Severity::kVerbose);
    }
    if (level == "warning" || level == "warn") {
        return static_cast<int>(Severity::kWarning);
    }
    if (level == "error" || level == "err") {
        return static_cast<int>(Severity::kError);
    }
    if (level == "off") {
        return kSilentThreshold; // suppress everything, including internal errors
    }
    return static_cast<int>(Severity::kInfo);
}

} // namespace

std::shared_ptr<ILogger> makeStderrLogger(Severity threshold) { return std::make_shared<StderrLogger>(static_cast<int>(threshold)); }

std::shared_ptr<ILogger> defaultLogger() {
    static std::shared_ptr<ILogger> instance = std::make_shared<StderrLogger>(thresholdFromEnv());
    return instance;
}

#ifdef TRT_CPP_API_WITH_SPDLOG
namespace {
class SpdlogLogger final : public ILogger {
public:
    void log(Severity severity, std::string_view message) noexcept override {
        // ILogger::log is noexcept; never let a logging-backend exception escape, even if
        // spdlog/fmt were built to throw a non-std::exception type.
        try {
            switch (severity) {
            case Severity::kVerbose:
                spdlog::debug("{}", message);
                break;
            case Severity::kInfo:
                spdlog::info("{}", message);
                break;
            case Severity::kWarning:
                spdlog::warn("{}", message);
                break;
            case Severity::kError:
                spdlog::error("{}", message);
                break;
            case Severity::kInternalError:
                spdlog::critical("{}", message);
                break;
            }
        } catch (...) {
        }
    }
};
} // namespace

std::shared_ptr<ILogger> makeSpdlogLogger() { return std::make_shared<SpdlogLogger>(); }
#endif

} // namespace trtcpp
