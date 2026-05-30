#include "tensorrt_cpp_api/engine.h"

#include "detail/engine_cache.h"
#include "detail/execution.h"
#include "detail/trt_common.h"

#include <cuda_runtime.h>

#include <NvInfer.h>

#include <utility>

namespace trtcpp {

struct Engine::Impl {
    EngineOptions options;
    detail::TrtLoggerBridge bridge;
    detail::TrtUniquePtr<nvinfer1::IRuntime> runtime;
    detail::TrtUniquePtr<nvinfer1::ICudaEngine> engine;
    detail::TrtUniquePtr<nvinfer1::IExecutionContext> context;
    detail::EngineMeta meta;

    explicit Impl(EngineOptions opts) : options(std::move(opts)), bridge(options.logger) {}
};

Engine::Engine() = default;
Engine::Engine(Engine &&) noexcept = default;
Engine &Engine::operator=(Engine &&) noexcept = default;
Engine::~Engine() = default;

Result<Engine> Engine::loadFromFile(const std::string &enginePath, const EngineOptions &options) {
    auto data = detail::readFile(enginePath); // hardened: rejects directories/missing without throwing
    if (!data) {
        return data.status();
    }
    return loadFromMemory(data.value(), options);
}

Result<Engine> Engine::loadFromMemory(std::span<const std::byte> engineData, const EngineOptions &options) {
    if (cudaError_t code = cudaSetDevice(options.deviceIndex); code != cudaSuccess) {
        return cudaToStatus(code, "cudaSetDevice");
    }
    if (Status status = detail::loadPluginLibraries(options.pluginLibraries); !status) {
        return status;
    }

    Engine result;
    result.impl_ = std::make_unique<Impl>(options);
    Impl &impl = *result.impl_;

    impl.runtime.reset(nvinfer1::createInferRuntime(impl.bridge.nv()));
    if (!impl.runtime) {
        return Status{StatusCode::kTensorRtError, "createInferRuntime failed"};
    }
    impl.engine.reset(impl.runtime->deserializeCudaEngine(engineData.data(), engineData.size()));
    if (!impl.engine) {
        return Status{StatusCode::kTensorRtError, "deserializeCudaEngine failed (corrupt or incompatible engine)"};
    }
    impl.context.reset(impl.engine->createExecutionContext());
    if (!impl.context) {
        return Status{StatusCode::kTensorRtError, "createExecutionContext failed"};
    }
    impl.meta = detail::introspectEngine(*impl.engine, options.deviceIndex);
    return result;
}

std::vector<TensorInfo> Engine::tensors() const { return impl_->meta.tensors; }
std::vector<std::string> Engine::inputNames() const { return impl_->meta.inputNames; }
std::vector<std::string> Engine::outputNames() const { return impl_->meta.outputNames; }
int Engine::nbOptimizationProfiles() const { return impl_->meta.nbProfiles; }

Result<Shape> Engine::tensorShape(const std::string &name) const {
    for (const TensorInfo &info : impl_->meta.tensors) {
        if (info.name == name) {
            return info.shape;
        }
    }
    return Status{StatusCode::kNotFound, "no such tensor: " + name};
}

Result<DType> Engine::tensorDType(const std::string &name) const {
    for (const TensorInfo &info : impl_->meta.tensors) {
        if (info.name == name) {
            return info.dtype;
        }
    }
    return Status{StatusCode::kNotFound, "no such tensor: " + name};
}

Status Engine::enqueue(const std::unordered_map<std::string, TensorView> &inputs,
                       const std::unordered_map<std::string, TensorView> &outputs, const Stream &stream, int profileIndex) {
    return detail::enqueue(*impl_->context, impl_->meta, inputs, outputs, stream, profileIndex);
}

Result<std::unordered_map<std::string, Shape>> Engine::outputShapes(const std::unordered_map<std::string, TensorView> &inputs,
                                                                    int profileIndex) {
    return detail::outputShapes(*impl_->context, impl_->meta, inputs, profileIndex);
}

Result<std::unordered_map<std::string, Tensor>> Engine::infer(const std::unordered_map<std::string, TensorView> &inputs,
                                                              const Stream &stream, int profileIndex) {
    return detail::infer(*impl_->context, impl_->meta, inputs, stream, profileIndex);
}

Result<Tensor> Engine::inferSingle(const std::unordered_map<std::string, TensorView> &inputs, const Stream &stream, int profileIndex) {
    if (impl_->meta.outputNames.size() != 1) {
        return Status{StatusCode::kInvalidArgument,
                      "inferSingle requires exactly one output; this engine has " + std::to_string(impl_->meta.outputNames.size())};
    }
    auto outputs = infer(inputs, stream, profileIndex);
    if (!outputs) {
        return outputs.status();
    }
    return std::move(outputs.value().begin()->second);
}

} // namespace trtcpp
