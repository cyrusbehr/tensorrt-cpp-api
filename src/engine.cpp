#include "tensorrt_cpp_api/engine.h"

#include "detail/engine_cache.h"
#include "detail/trt_common.h"

#include <cuda_runtime.h>

#include <NvInfer.h>

#include <utility>

namespace trtcpp {
namespace {

DType fromTrtDataType(nvinfer1::DataType type) {
    switch (type) {
    case nvinfer1::DataType::kFLOAT:
        return DType::kFloat32;
    case nvinfer1::DataType::kHALF:
        return DType::kFloat16;
    case nvinfer1::DataType::kINT8:
        return DType::kInt8;
    case nvinfer1::DataType::kINT32:
        return DType::kInt32;
    case nvinfer1::DataType::kBOOL:
        return DType::kBool;
    case nvinfer1::DataType::kUINT8:
        return DType::kUInt8;
    case nvinfer1::DataType::kFP8:
        return DType::kFp8;
    case nvinfer1::DataType::kBF16:
        return DType::kBFloat16;
    case nvinfer1::DataType::kINT64:
        return DType::kInt64;
    case nvinfer1::DataType::kINT4:
        return DType::kInt4;
    }
    return DType::kFloat32;
}

nvinfer1::Dims toDims(const Shape &shape) {
    nvinfer1::Dims dims;
    dims.nbDims = shape.rank();
    for (int i = 0; i < shape.rank(); ++i) {
        dims.d[i] = shape[i];
    }
    return dims;
}

static_assert(Shape::kMaxRank == nvinfer1::Dims::MAX_DIMS, "Shape::kMaxRank must match nvinfer1::Dims::MAX_DIMS");

Shape fromDims(const nvinfer1::Dims &dims) {
    if (dims.nbDims < 0) {
        return Shape{}; // TensorRT signals invalid dims with nbDims == -1
    }
    std::array<std::int64_t, Shape::kMaxRank> values{};
    for (int i = 0; i < dims.nbDims; ++i) {
        values[static_cast<std::size_t>(i)] = dims.d[i];
    }
    return Shape{std::span<const std::int64_t>(values.data(), static_cast<std::size_t>(dims.nbDims))};
}

} // namespace

struct Engine::Impl {
    EngineOptions options;
    detail::TrtLoggerBridge bridge;
    detail::TrtUniquePtr<nvinfer1::IRuntime> runtime;
    detail::TrtUniquePtr<nvinfer1::ICudaEngine> engine;
    detail::TrtUniquePtr<nvinfer1::IExecutionContext> context;
    std::vector<TensorInfo> tensors;
    std::vector<std::string> inputNames;
    std::vector<std::string> outputNames;

    explicit Impl(EngineOptions opts) : options(std::move(opts)), bridge(options.logger) {}

    // Select the profile and set every input's shape (and optionally its device address) on
    // the shared context. The ordering -- profile then input shapes -- is the TRT requirement.
    Status bindInputs(const std::unordered_map<std::string, TensorView> &inputs, int profileIndex, cudaStream_t stream, bool setAddresses) {
        const int nbProfiles = engine->getNbOptimizationProfiles();
        const int profileCeiling = nbProfiles > 0 ? nbProfiles : 1;
        if (profileIndex < 0 || profileIndex >= profileCeiling) {
            return Status{StatusCode::kInvalidArgument, "profileIndex out of range"};
        }
        if (nbProfiles > 0) {
            if (!context->setOptimizationProfileAsync(profileIndex, stream)) {
                return Status{StatusCode::kInvalidArgument,
                              "setOptimizationProfileAsync failed for profile " + std::to_string(profileIndex)};
            }
        }
        for (const std::string &name : inputNames) {
            auto it = inputs.find(name);
            if (it == inputs.end()) {
                return Status{StatusCode::kInvalidArgument, "missing input tensor: " + name};
            }
            if (!context->setInputShape(name.c_str(), toDims(it->second.shape()))) {
                return Status{StatusCode::kShapeMismatch, "setInputShape rejected the shape for input: " + name};
            }
            if (setAddresses && !context->setTensorAddress(name.c_str(), it->second.data())) {
                return Status{StatusCode::kTensorRtError, "setTensorAddress failed for input: " + name};
            }
        }
        if (!context->allInputDimensionsSpecified()) {
            return Status{StatusCode::kInvalidArgument, "not all input dimensions were specified"};
        }
        return Status{};
    }
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

    const int nbTensors = impl.engine->getNbIOTensors();
    for (int i = 0; i < nbTensors; ++i) {
        const char *name = impl.engine->getIOTensorName(i);
        TensorInfo info;
        info.name = name;
        info.isInput = impl.engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT;
        info.dtype = fromTrtDataType(impl.engine->getTensorDataType(name));
        info.shape = fromDims(impl.engine->getTensorShape(name));
        if (info.isInput) {
            impl.inputNames.push_back(info.name);
        } else {
            impl.outputNames.push_back(info.name);
        }
        impl.tensors.push_back(std::move(info));
    }
    return result;
}

std::vector<TensorInfo> Engine::tensors() const { return impl_->tensors; }
std::vector<std::string> Engine::inputNames() const { return impl_->inputNames; }
std::vector<std::string> Engine::outputNames() const { return impl_->outputNames; }
int Engine::nbOptimizationProfiles() const { return impl_->engine->getNbOptimizationProfiles(); }

Result<Shape> Engine::tensorShape(const std::string &name) const {
    for (const TensorInfo &info : impl_->tensors) {
        if (info.name == name) {
            return info.shape;
        }
    }
    return Status{StatusCode::kNotFound, "no such tensor: " + name};
}

Result<DType> Engine::tensorDType(const std::string &name) const {
    for (const TensorInfo &info : impl_->tensors) {
        if (info.name == name) {
            return info.dtype;
        }
    }
    return Status{StatusCode::kNotFound, "no such tensor: " + name};
}

Status Engine::enqueue(const std::unordered_map<std::string, TensorView> &inputs,
                       const std::unordered_map<std::string, TensorView> &outputs, const Stream &stream, int profileIndex) {
    if (Status status = impl_->bindInputs(inputs, profileIndex, stream.handle(), /*setAddresses=*/true); !status) {
        return status;
    }
    for (const std::string &name : impl_->outputNames) {
        auto it = outputs.find(name);
        if (it == outputs.end()) {
            return Status{StatusCode::kInvalidArgument, "missing output tensor: " + name};
        }
        if (!impl_->context->setTensorAddress(name.c_str(), it->second.data())) {
            return Status{StatusCode::kTensorRtError, "setTensorAddress failed for output: " + name};
        }
    }
    if (!impl_->context->enqueueV3(stream.handle())) {
        return Status{StatusCode::kTensorRtError, "enqueueV3 failed"};
    }
    return Status{};
}

Result<std::unordered_map<std::string, Shape>> Engine::outputShapes(const std::unordered_map<std::string, TensorView> &inputs,
                                                                    int profileIndex) {
    Stream stream;
    if (Status status = impl_->bindInputs(inputs, profileIndex, stream.handle(), /*setAddresses=*/false); !status) {
        return status;
    }
    if (Status status = stream.synchronize(); !status) {
        return status;
    }
    std::unordered_map<std::string, Shape> result;
    for (const std::string &name : impl_->outputNames) {
        result.emplace(name, fromDims(impl_->context->getTensorShape(name.c_str())));
    }
    return result;
}

Result<std::unordered_map<std::string, Tensor>> Engine::infer(const std::unordered_map<std::string, TensorView> &inputs,
                                                              const Stream &stream, int profileIndex) {
    if (Status status = impl_->bindInputs(inputs, profileIndex, stream.handle(), /*setAddresses=*/true); !status) {
        return status;
    }
    std::unordered_map<std::string, Tensor> outputs;
    for (const TensorInfo &info : impl_->tensors) {
        if (info.isInput) {
            continue;
        }
        const Shape shape = fromDims(impl_->context->getTensorShape(info.name.c_str()));
        if (shape.isDynamic()) {
            return Status{StatusCode::kInternal, "output shape unresolved after setInputShape: " + info.name};
        }
        auto tensor = Tensor::allocate(info.dtype, shape, Device::kCuda, impl_->options.deviceIndex);
        if (!tensor) {
            return tensor.status();
        }
        if (!impl_->context->setTensorAddress(info.name.c_str(), tensor.value().data())) {
            return Status{StatusCode::kTensorRtError, "setTensorAddress failed for output: " + info.name};
        }
        outputs.emplace(info.name, std::move(tensor).value());
    }
    if (!impl_->context->enqueueV3(stream.handle())) {
        return Status{StatusCode::kTensorRtError, "enqueueV3 failed"};
    }
    return outputs;
}

Result<Tensor> Engine::inferSingle(const std::unordered_map<std::string, TensorView> &inputs, const Stream &stream, int profileIndex) {
    if (impl_->outputNames.size() != 1) {
        return Status{StatusCode::kInvalidArgument,
                      "inferSingle requires exactly one output; this engine has " + std::to_string(impl_->outputNames.size())};
    }
    auto outputs = infer(inputs, stream, profileIndex);
    if (!outputs) {
        return outputs.status();
    }
    return std::move(outputs.value().begin()->second);
}

} // namespace trtcpp
