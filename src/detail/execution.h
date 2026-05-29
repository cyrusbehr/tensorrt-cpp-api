#pragma once

// Internal: per-execution-context inference logic shared by Engine (one context) and
// EnginePool::Lease (N contexts, one optimization profile each). Includes NvInfer.h, so it
// is never installed.

#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

#include <NvInfer.h>

#include "tensorrt_cpp_api/cuda.h"
#include "tensorrt_cpp_api/device_tensor.h"
#include "tensorrt_cpp_api/engine.h" // TensorInfo
#include "tensorrt_cpp_api/shape.h"
#include "tensorrt_cpp_api/status.h"
#include "tensorrt_cpp_api/tensor.h"

namespace trtcpp::detail {

// Name-keyed engine metadata, introspected once and shared across contexts.
struct EngineMeta {
    std::vector<TensorInfo> tensors;
    std::vector<std::string> inputNames;
    std::vector<std::string> outputNames;
    int nbProfiles = 0;
    int deviceIndex = 0;
    bool hasDynamic = false; ///< any IO tensor has a dynamic (-1) dim; static engines report a profile count of 1
};

EngineMeta introspectEngine(nvinfer1::ICudaEngine &engine, int deviceIndex);

using TensorMap = std::unordered_map<std::string, TensorView>;

// Caller-allocated: bind profile + input shapes/addresses, bind output addresses, enqueueV3.
Status enqueue(nvinfer1::IExecutionContext &context, const EngineMeta &meta, const TensorMap &inputs, const TensorMap &outputs,
               const Stream &stream, int profileIndex);

// Library-allocated: bind inputs, allocate outputs sized from the resolved shapes, enqueueV3.
Result<std::unordered_map<std::string, Tensor>> infer(nvinfer1::IExecutionContext &context, const EngineMeta &meta, const TensorMap &inputs,
                                                      const Stream &stream, int profileIndex);

// Resolve all output shapes for the given inputs + profile (on a private stream).
Result<std::unordered_map<std::string, Shape>> outputShapes(nvinfer1::IExecutionContext &context, const EngineMeta &meta,
                                                            const TensorMap &inputs, int profileIndex);

} // namespace trtcpp::detail
