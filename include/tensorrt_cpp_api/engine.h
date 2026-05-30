#pragma once

#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorrt_cpp_api/cuda.h"
#include "tensorrt_cpp_api/device_tensor.h"
#include "tensorrt_cpp_api/dtype.h"
#include "tensorrt_cpp_api/logger.h"
#include "tensorrt_cpp_api/shape.h"
#include "tensorrt_cpp_api/status.h"
#include "tensorrt_cpp_api/tensor.h"

namespace trtcpp {

struct EngineOptions {
    int deviceIndex = 0;
    std::shared_ptr<ILogger> logger = defaultLogger();
    std::vector<std::string> pluginLibraries; ///< dlopen + register (IPluginV3) before deserialize
};

/// Metadata for one IO tensor (always name-keyed; never index-keyed).
struct TensorInfo {
    std::string name;
    bool isInput = false;
    DType dtype = DType::kFloat32;
    Shape shape; ///< build-time shape; dynamic axes are -1
};

/// A loaded TensorRT engine. Thread-COMPATIBLE, not thread-safe: for concurrent inference
/// use EnginePool. TensorRT types are hidden behind a PImpl, so this header pulls in no
/// nvinfer1. The owning ICudaEngine/IExecutionContext/IRuntime are released on destruction.
class Engine {
public:
    Engine(Engine &&) noexcept;
    Engine &operator=(Engine &&) noexcept;
    Engine(const Engine &) = delete;
    Engine &operator=(const Engine &) = delete;
    ~Engine();

    static Result<Engine> loadFromFile(const std::string &enginePath, const EngineOptions &options = {});
    static Result<Engine> loadFromMemory(std::span<const std::byte> engineData, const EngineOptions &options = {});

    std::vector<TensorInfo> tensors() const;
    std::vector<std::string> inputNames() const;
    std::vector<std::string> outputNames() const;
    int nbOptimizationProfiles() const;
    Result<Shape> tensorShape(const std::string &name) const; ///< build-time (may be -1 dynamic)
    Result<DType> tensorDType(const std::string &name) const;

    /// Caller-allocated, zero-copy, no implicit sync. Selects `profileIndex`
    /// (setOptimizationProfileAsync on `stream`) BEFORE setInputShape from the input views,
    /// then setTensorAddress for every IO tensor + enqueueV3 -- all on `stream`. The caller
    /// owns the sync. Out-of-range profileIndex returns kInvalidArgument.
    Status enqueue(const std::unordered_map<std::string, TensorView> &inputs, const std::unordered_map<std::string, TensorView> &outputs,
                   const Stream &stream, int profileIndex = 0);

    /// Library-allocated convenience: resolves output shapes from the inputs, allocates
    /// owning device Tensors sized from getTensorShape AFTER setInputShape (never build-time
    /// -1 dims), enqueues, and returns them. No implicit sync -- read back via Tensor::toHost.
    /// For concurrent use, prefer EnginePool. Engine itself is thread-compatible.
    Result<std::unordered_map<std::string, Tensor>> infer(const std::unordered_map<std::string, TensorView> &inputs, const Stream &stream,
                                                          int profileIndex = 0);

    /// Single-output shortcut (the classifier/detector common case): returns the sole
    /// output, or kInvalidArgument if the engine has != 1 output. Avoids hard-coding a name.
    Result<Tensor> inferSingle(const std::unordered_map<std::string, TensorView> &inputs, const Stream &stream, int profileIndex = 0);

    /// Resolve ALL output shapes for a set of input views + profile (sets input shapes on
    /// the execution context, reads getTensorShape -- dynamic-aware). Use to pre-allocate
    /// output buffers for the enqueue path.
    Result<std::unordered_map<std::string, Shape>> outputShapes(const std::unordered_map<std::string, TensorView> &inputs,
                                                                int profileIndex = 0);

private:
    Engine();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace trtcpp
