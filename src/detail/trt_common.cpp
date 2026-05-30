#include "detail/trt_common.h"

#include <utility>

namespace trtcpp::detail {

Severity fromNvSeverity(nvinfer1::ILogger::Severity severity) noexcept {
    switch (severity) {
    case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
        return Severity::kInternalError;
    case nvinfer1::ILogger::Severity::kERROR:
        return Severity::kError;
    case nvinfer1::ILogger::Severity::kWARNING:
        return Severity::kWarning;
    case nvinfer1::ILogger::Severity::kINFO:
        return Severity::kInfo;
    case nvinfer1::ILogger::Severity::kVERBOSE:
        return Severity::kVerbose;
    }
    return Severity::kInfo;
}

nvinfer1::ILogger::Severity toNvSeverity(Severity severity) noexcept {
    switch (severity) {
    case Severity::kInternalError:
        return nvinfer1::ILogger::Severity::kINTERNAL_ERROR;
    case Severity::kError:
        return nvinfer1::ILogger::Severity::kERROR;
    case Severity::kWarning:
        return nvinfer1::ILogger::Severity::kWARNING;
    case Severity::kInfo:
        return nvinfer1::ILogger::Severity::kINFO;
    case Severity::kVerbose:
        return nvinfer1::ILogger::Severity::kVERBOSE;
    }
    return nvinfer1::ILogger::Severity::kINFO;
}

TrtLoggerBridge::TrtLoggerBridge(std::shared_ptr<trtcpp::ILogger> logger) : logger_(std::move(logger)) {
    if (!logger_) {
        logger_ = defaultLogger();
    }
}

void TrtLoggerBridge::log(nvinfer1::ILogger::Severity severity, const char *msg) noexcept {
    logger_->log(fromNvSeverity(severity), msg != nullptr ? msg : "");
}

Status loadPluginLibraries(const std::vector<std::string> &paths) {
    if (paths.empty()) {
        return Status{};
    }
    nvinfer1::IPluginRegistry *registry = ::getPluginRegistry(); // global free function, not a member
    if (registry == nullptr) {
        return Status{StatusCode::kTensorRtError, "getPluginRegistry() returned null"};
    }
    for (const std::string &path : paths) {
        // loadLibrary returns null for a missing/unreadable library OR one whose plugins
        // are already registered -- both are failures to propagate.
        if (registry->loadLibrary(path.c_str()) == nullptr) {
            return Status{StatusCode::kTensorRtError,
                          "failed to load plugin library (missing, unreadable, or already registered): " + path};
        }
    }
    return Status{};
}

} // namespace trtcpp::detail
