#include "detail/calibrator_bridge.h"

#if TRT_CPP_API_TENSORRT_VERSION_MAJOR < 11

#include <unordered_map>
#include <utility>

namespace trtcpp::detail {

Int8CalibratorBridge::Int8CalibratorBridge(std::shared_ptr<ICalibrator> calibrator, std::string inputName, Shape batchShape)
    : calibrator_(std::move(calibrator)), inputName_(std::move(inputName)), batchShape_(std::move(batchShape)) {}

int32_t Int8CalibratorBridge::getBatchSize() const noexcept { return calibrator_ ? calibrator_->batchSize() : 0; }

bool Int8CalibratorBridge::getBatch(void *bindings[], const char *names[], int32_t nbBindings) noexcept {
    try {
        std::unordered_map<std::string, TensorView> inputs;
        for (int32_t i = 0; i < nbBindings; ++i) {
            if (inputName_ == names[i]) {
                inputs.emplace(inputName_, TensorView{bindings[i], DType::kFloat32, batchShape_, Device::kCuda, 0, Layout::kLinear});
            }
        }
        if (inputs.empty()) {
            return false; // calibrator's input name not among the bindings
        }
        if (!calibrator_->nextBatch(inputs, stream_)) {
            // No more batches (or the provider errored). nextBatch may have queued async work on
            // stream_ before failing, so drain it before returning -- otherwise a late DMA could
            // write into a binding buffer TRT no longer expects to be touched.
            stream_.synchronize();
            return false;
        }
        return stream_.synchronize().ok(); // the buffer must be filled before TRT reads it
    } catch (...) {
        return false;
    }
}

const void *Int8CalibratorBridge::readCalibrationCache(std::size_t &length) noexcept {
    try {
        auto cache = calibrator_->readCache();
        if (!cache.has_value() || cache->empty()) {
            length = 0;
            return nullptr;
        }
        cache_ = std::move(*cache);
        length = cache_.size();
        return cache_.data();
    } catch (...) {
        length = 0;
        return nullptr;
    }
}

void Int8CalibratorBridge::writeCalibrationCache(const void *ptr, std::size_t length) noexcept {
    try {
        calibrator_->writeCache(std::span<const std::byte>(static_cast<const std::byte *>(ptr), length));
    } catch (...) {
    }
}

} // namespace trtcpp::detail

#endif // TRT_CPP_API_TENSORRT_VERSION_MAJOR < 11
