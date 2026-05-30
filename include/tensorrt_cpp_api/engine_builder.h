#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "tensorrt_cpp_api/build_options.h"
#include "tensorrt_cpp_api/engine.h"
#include "tensorrt_cpp_api/logger.h"
#include "tensorrt_cpp_api/status.h"

namespace trtcpp {

/// Builds a serialized TensorRT engine from an ONNX model. Stateless apart from the
/// logger; one builder can build many engines. The cache-backed buildOrLoad and the
/// build+deserialize buildAndLoad convenience entry points are provided by the cache
/// and runtime modules respectively.
class EngineBuilder {
public:
    explicit EngineBuilder(std::shared_ptr<ILogger> logger = defaultLogger());

    /// Parse the ONNX at `onnxPath` and build a serialized engine for `options`.
    Result<std::vector<std::byte>> buildFromOnnxFile(const std::string &onnxPath, const BuildOptions &options);
    /// Parse ONNX bytes (e.g. an in-memory or decrypted model) and build a serialized engine.
    Result<std::vector<std::byte>> buildFromOnnxBytes(std::span<const std::byte> onnx, const BuildOptions &options);

    /// Build the engine if no fresh cached one exists in options.engineCacheDir, otherwise
    /// reuse the cache. Returns the engine file path. The cache is keyed by ONNX content
    /// hash + TRT version + GPU UUID + build options, with a JSON sidecar and atomic write;
    /// a stale cache (changed ONNX/options/version/GPU) is detected and rebuilt.
    Result<std::string> buildOrLoad(const std::string &onnxPath, const BuildOptions &options);

    /// One-shot: buildOrLoad + deserialize into a ready-to-run Engine. The common entry point.
    Result<Engine> buildAndLoad(const std::string &onnxPath, const BuildOptions &options, const EngineOptions &engineOptions = {});

private:
    std::shared_ptr<ILogger> logger_;
};

} // namespace trtcpp
