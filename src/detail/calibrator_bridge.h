#pragma once

#include "tensorrt_cpp_api/build_config.h"

#if TRT_CPP_API_TENSORRT_VERSION_MAJOR < 11

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include <NvInfer.h>

#include "tensorrt_cpp_api/calibrator.h"
#include "tensorrt_cpp_api/cuda.h"
#include "tensorrt_cpp_api/shape.h"

namespace trtcpp::detail {

// Adapts a trtcpp::ICalibrator to the legacy nvinfer1::IInt8EntropyCalibrator2 the TRT < 11
// builder consumes. Single-input only (the legacy calibrator contract). Owns a stream for
// the per-batch uploads and synchronizes before each getBatch returns.
class Int8CalibratorBridge final : public nvinfer1::IInt8EntropyCalibrator2 {
public:
    Int8CalibratorBridge(std::shared_ptr<ICalibrator> calibrator, std::string inputName, Shape batchShape);

    int32_t getBatchSize() const noexcept override;
    bool getBatch(void *bindings[], const char *names[], int32_t nbBindings) noexcept override;
    const void *readCalibrationCache(std::size_t &length) noexcept override;
    void writeCalibrationCache(const void *ptr, std::size_t length) noexcept override;

private:
    std::shared_ptr<ICalibrator> calibrator_;
    std::string inputName_;
    Shape batchShape_;
    Stream stream_;
    std::vector<std::byte> cache_;
};

} // namespace trtcpp::detail

#endif // TRT_CPP_API_TENSORRT_VERSION_MAJOR < 11
