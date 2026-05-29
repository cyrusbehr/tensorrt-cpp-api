#include "tensorrt_cpp_api/calibrator.h"

#if TRT_CPP_API_TENSORRT_VERSION_MAJOR < 11

#include "detail/engine_cache.h" // detail::readFile / writeAtomic

#include <cuda_runtime.h>

#include <fstream>
#include <utility>

namespace trtcpp {
namespace {

class RawBatchCalibrator final : public ICalibrator {
public:
    RawBatchCalibrator(std::string inputName, Shape batchShape, std::vector<std::string> batchFiles, std::string cachePath)
        : inputName_(std::move(inputName)), batchShape_(std::move(batchShape)), batchFiles_(std::move(batchFiles)),
          cachePath_(std::move(cachePath)) {}

    int batchSize() const override { return batchShape_.rank() > 0 ? static_cast<int>(batchShape_[0]) : 1; }

    bool nextBatch(const std::unordered_map<std::string, TensorView> &inputs, const Stream &stream) override {
        if (index_ >= batchFiles_.size()) {
            return false;
        }
        auto it = inputs.find(inputName_);
        if (it == inputs.end()) {
            return false;
        }
        auto bytes = detail::readFile(batchFiles_[index_]);
        if (!bytes || bytes.value().size() != it->second.nbytes()) {
            return false; // missing file or size mismatch with the calibration buffer
        }
        if (cudaMemcpyAsync(it->second.data(), bytes.value().data(), bytes.value().size(), cudaMemcpyHostToDevice, stream.handle()) !=
            cudaSuccess) {
            return false;
        }
        // Synchronize before the local host buffer is freed -- the async copy reads it, and a
        // pageable-source H2D copy is not contractually host-synchronous.
        if (cudaStreamSynchronize(stream.handle()) != cudaSuccess) {
            return false;
        }
        ++index_;
        return true;
    }

    std::optional<std::vector<std::byte>> readCache() override {
        if (cachePath_.empty()) {
            return std::nullopt;
        }
        auto bytes = detail::readFile(cachePath_);
        if (!bytes) {
            return std::nullopt;
        }
        return std::move(bytes.value());
    }

    void writeCache(std::span<const std::byte> data) override {
        if (!cachePath_.empty()) {
            detail::writeAtomic(cachePath_, data);
        }
    }

private:
    std::string inputName_;
    Shape batchShape_;
    std::vector<std::string> batchFiles_;
    std::string cachePath_;
    std::size_t index_ = 0;
};

} // namespace

std::shared_ptr<ICalibrator> makeRawBatchCalibrator(std::string inputName, Shape batchShape, std::vector<std::string> batchFiles,
                                                    std::string cachePath) {
    return std::make_shared<RawBatchCalibrator>(std::move(inputName), std::move(batchShape), std::move(batchFiles), std::move(cachePath));
}

} // namespace trtcpp

#endif // TRT_CPP_API_TENSORRT_VERSION_MAJOR < 11
