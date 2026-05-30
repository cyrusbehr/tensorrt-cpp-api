#pragma once

#include "tensorrt_cpp_api/build_config.h"

// Legacy INT8 post-training calibration. TensorRT 11 removed the IInt8Calibrator family, so
// this whole header compiles out when the library is built against TRT >= 11. The forward
// path is Precision::kInt8Qdq with a modelopt-quantized ONNX (no calibrator).
#if TRT_CPP_API_TENSORRT_VERSION_MAJOR < 11

#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorrt_cpp_api/cuda.h"
#include "tensorrt_cpp_api/shape.h"
#include "tensorrt_cpp_api/tensor.h"

namespace trtcpp {

/// Supplies calibration batches for legacy INT8 PTQ. Implement nextBatch to fill the named
/// input device buffers (views over TensorRT's calibration buffers) with PRE-PROCESSED data
/// -- using the SAME preprocessing as inference, to avoid the v6 calibration/inference
/// mismatch.
class ICalibrator {
public:
    virtual ~ICalibrator() = default;
    virtual int batchSize() const = 0;
    /// Fill `inputs` with the next batch on `stream`; return false when exhausted.
    virtual bool nextBatch(const std::unordered_map<std::string, TensorView> &inputs, const Stream &stream) = 0;
    /// Optional calibration-cache persistence (return nullopt to recompute).
    virtual std::optional<std::vector<std::byte>> readCache() = 0;
    virtual void writeCache(std::span<const std::byte>) = 0;
};

/// A single-input calibrator that streams pre-processed raw batches (contiguous NCHW float,
/// one binary file per batch of shape `batchShape`) from disk -- no image decoding, so it
/// needs neither OpenCV nor the preprocessing sublibrary. (The decode+preprocess
/// "image directory" convenience lives in the OpenCV interop module.) `cachePath`, if set,
/// persists the calibration table.
std::shared_ptr<ICalibrator> makeRawBatchCalibrator(std::string inputName, Shape batchShape, std::vector<std::string> batchFiles,
                                                    std::string cachePath = {});

} // namespace trtcpp

#endif // TRT_CPP_API_TENSORRT_VERSION_MAJOR < 11
