#include "detail/execution.h"

#include <array>
#include <utility>

namespace trtcpp::detail {
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

static_assert(Shape::kMaxRank == nvinfer1::Dims::MAX_DIMS, "Shape::kMaxRank must match nvinfer1::Dims::MAX_DIMS");

nvinfer1::Dims toDims(const Shape &shape) {
    nvinfer1::Dims dims;
    dims.nbDims = shape.rank();
    for (int i = 0; i < shape.rank(); ++i) {
        dims.d[i] = shape[i];
    }
    return dims;
}

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

// Select the profile and set every input's shape (and optionally its address) on `context`.
// Ordering -- profile then input shapes -- is the TRT requirement; the profile-set stream
// must match the eventual enqueue stream.
Status bindInputs(nvinfer1::IExecutionContext &context, const EngineMeta &meta, const TensorMap &inputs, int profileIndex,
                  cudaStream_t stream, bool setAddresses) {
    const int profileCeiling = meta.nbProfiles > 0 ? meta.nbProfiles : 1;
    if (profileIndex < 0 || profileIndex >= profileCeiling) {
        return Status{StatusCode::kInvalidArgument, "profileIndex out of range"};
    }
    // Only dynamic engines need per-context profile selection; for static engines the
    // default profile 0 is shared safely across concurrent contexts.
    if (meta.hasDynamic) {
        if (!context.setOptimizationProfileAsync(profileIndex, stream)) {
            return Status{StatusCode::kInvalidArgument, "setOptimizationProfileAsync failed for profile " + std::to_string(profileIndex)};
        }
    }
    for (const std::string &name : meta.inputNames) {
        auto it = inputs.find(name);
        if (it == inputs.end()) {
            return Status{StatusCode::kInvalidArgument, "missing input tensor: " + name};
        }
        if (!context.setInputShape(name.c_str(), toDims(it->second.shape()))) {
            return Status{StatusCode::kShapeMismatch, "setInputShape rejected the shape for input: " + name};
        }
        if (setAddresses && !context.setTensorAddress(name.c_str(), it->second.data())) {
            return Status{StatusCode::kTensorRtError, "setTensorAddress failed for input: " + name};
        }
    }
    if (!context.allInputDimensionsSpecified()) {
        return Status{StatusCode::kInvalidArgument, "not all input dimensions were specified"};
    }
    return Status{};
}

} // namespace

EngineMeta introspectEngine(nvinfer1::ICudaEngine &engine, int deviceIndex) {
    EngineMeta meta;
    meta.deviceIndex = deviceIndex;
    meta.nbProfiles = engine.getNbOptimizationProfiles();
    const int nbTensors = engine.getNbIOTensors();
    for (int i = 0; i < nbTensors; ++i) {
        const char *name = engine.getIOTensorName(i);
        TensorInfo info;
        info.name = name;
        info.isInput = engine.getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT;
        info.dtype = fromTrtDataType(engine.getTensorDataType(name));
        info.shape = fromDims(engine.getTensorShape(name));
        if (info.isInput) {
            meta.inputNames.push_back(info.name);
        } else {
            meta.outputNames.push_back(info.name);
        }
        if (info.shape.isDynamic()) {
            meta.hasDynamic = true;
        }
        meta.tensors.push_back(std::move(info));
    }
    return meta;
}

Status enqueue(nvinfer1::IExecutionContext &context, const EngineMeta &meta, const TensorMap &inputs, const TensorMap &outputs,
               const Stream &stream, int profileIndex) {
    if (Status status = bindInputs(context, meta, inputs, profileIndex, stream.handle(), /*setAddresses=*/true); !status) {
        return status;
    }
    for (const std::string &name : meta.outputNames) {
        auto it = outputs.find(name);
        if (it == outputs.end()) {
            return Status{StatusCode::kInvalidArgument, "missing output tensor: " + name};
        }
        if (!context.setTensorAddress(name.c_str(), it->second.data())) {
            return Status{StatusCode::kTensorRtError, "setTensorAddress failed for output: " + name};
        }
    }
    if (!context.enqueueV3(stream.handle())) {
        return Status{StatusCode::kTensorRtError, "enqueueV3 failed"};
    }
    return Status{};
}

Result<std::unordered_map<std::string, Tensor>> infer(nvinfer1::IExecutionContext &context, const EngineMeta &meta, const TensorMap &inputs,
                                                      const Stream &stream, int profileIndex) {
    if (Status status = bindInputs(context, meta, inputs, profileIndex, stream.handle(), /*setAddresses=*/true); !status) {
        return status;
    }
    std::unordered_map<std::string, Tensor> outputs;
    for (const TensorInfo &info : meta.tensors) {
        if (info.isInput) {
            continue;
        }
        const Shape shape = fromDims(context.getTensorShape(info.name.c_str()));
        if (shape.isDynamic()) {
            return Status{StatusCode::kInternal, "output shape unresolved after setInputShape: " + info.name};
        }
        auto tensor = Tensor::allocate(info.dtype, shape, Device::kCuda, meta.deviceIndex);
        if (!tensor) {
            return tensor.status();
        }
        if (!context.setTensorAddress(info.name.c_str(), tensor.value().data())) {
            return Status{StatusCode::kTensorRtError, "setTensorAddress failed for output: " + info.name};
        }
        outputs.emplace(info.name, std::move(tensor).value());
    }
    if (!context.enqueueV3(stream.handle())) {
        return Status{StatusCode::kTensorRtError, "enqueueV3 failed"};
    }
    return outputs;
}

Result<std::unordered_map<std::string, Shape>> outputShapes(nvinfer1::IExecutionContext &context, const EngineMeta &meta,
                                                            const TensorMap &inputs, int profileIndex) {
    Stream stream;
    if (Status status = bindInputs(context, meta, inputs, profileIndex, stream.handle(), /*setAddresses=*/false); !status) {
        return status;
    }
    if (Status status = stream.synchronize(); !status) {
        return status;
    }
    std::unordered_map<std::string, Shape> result;
    for (const std::string &name : meta.outputNames) {
        result.emplace(name, fromDims(context.getTensorShape(name.c_str())));
    }
    return result;
}

} // namespace trtcpp::detail
