#pragma once

// Internal TensorRT glue. This header includes <NvInfer.h> and is NOT installed -- it is
// included only by .cpp files in src/, keeping nvinfer1 types out of every public header.

#include <memory>
#include <string>
#include <vector>

#include <NvInfer.h>

#include "tensorrt_cpp_api/logger.h"
#include "tensorrt_cpp_api/status.h"

namespace trtcpp::detail {

// TensorRT 10/11 destroy API objects via `delete` (the old destroy() is gone), so a plain
// unique_ptr with the default deleter is the correct RAII wrapper.
template <class T> using TrtUniquePtr = std::unique_ptr<T>;

// Severity translation BY NAME. nvinfer1::ILogger::Severity is numerically the REVERSE of
// trtcpp::Severity (kINTERNAL_ERROR=0 .. kVERBOSE=4), so these must switch on the
// enumerator and must never static_cast between the two.
Severity fromNvSeverity(nvinfer1::ILogger::Severity severity) noexcept;
nvinfer1::ILogger::Severity toNvSeverity(Severity severity) noexcept;

// Adapts a trtcpp::ILogger to the nvinfer1::ILogger TensorRT requires. Holds the trtcpp
// logger alive; pass nv() (or the bridge itself) to createInferBuilder/createInferRuntime.
class TrtLoggerBridge final : public nvinfer1::ILogger {
public:
    // trtcpp:: is required: inside a class derived from nvinfer1::ILogger, an unqualified
    // ILogger/Severity would resolve to the base class's names.
    explicit TrtLoggerBridge(std::shared_ptr<trtcpp::ILogger> logger);
    void log(nvinfer1::ILogger::Severity severity, const char *msg) noexcept override;
    nvinfer1::ILogger &nv() noexcept { return *this; }

private:
    std::shared_ptr<trtcpp::ILogger> logger_;
};

// Load custom-plugin shared libraries into the global plugin registry (IPluginV3 path,
// via IPluginRegistry::loadLibrary). Returns the first failure, or ok if all loaded.
Status loadPluginLibraries(const std::vector<std::string> &paths);

} // namespace trtcpp::detail
