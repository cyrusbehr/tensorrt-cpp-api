#include "tensorrt_cpp_api/calibrator.h"
#include "tensorrt_cpp_api/cuda.h"
#include "tensorrt_cpp_api/device_tensor.h"
#include "tensorrt_cpp_api/engine_builder.h"

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// The legacy calibrator path only exists on TRT < 11 (calibrator.h compiles out otherwise).
#if TRT_CPP_API_TENSORRT_VERSION_MAJOR < 11

using namespace trtcpp;

namespace {

std::string modelPath(const char *name) { return std::string(TRTCPP_TEST_MODEL_DIR) + "/" + name; }

std::vector<std::string> writeRawBatches(const std::filesystem::path &dir, int count, int floatsPerBatch) {
    std::filesystem::create_directories(dir);
    std::vector<std::string> paths;
    for (int b = 0; b < count; ++b) {
        std::vector<float> data(static_cast<std::size_t>(floatsPerBatch));
        for (int i = 0; i < floatsPerBatch; ++i) {
            data[static_cast<std::size_t>(i)] = static_cast<float>((i + b * 7) % 11) - 5.0f;
        }
        const std::string path = (dir / ("batch_" + std::to_string(b) + ".bin")).string();
        std::ofstream out(path, std::ios::binary);
        out.write(reinterpret_cast<const char *>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(float)));
        paths.push_back(path);
    }
    return paths;
}

// A calibrator that records how many batches TensorRT requested and fills each with varying
// data, so we can assert the build actually drove calibration through our bridge.
class CountingCalibrator final : public ICalibrator {
public:
    explicit CountingCalibrator(int maxBatches) : maxBatches_(maxBatches) {}
    int batchSize() const override { return 1; }
    bool nextBatch(const std::unordered_map<std::string, TensorView> &inputs, const Stream &stream) override {
        ++calls;
        if (calls > maxBatches_ || inputs.empty()) {
            return false;
        }
        const TensorView &view = inputs.begin()->second;
        std::vector<float> host(view.nbytes() / sizeof(float));
        for (std::size_t i = 0; i < host.size(); ++i) {
            host[i] = static_cast<float>((i + static_cast<std::size_t>(calls) * 3) % 13) - 6.0f;
        }
        cudaMemcpyAsync(view.data(), host.data(), view.nbytes(), cudaMemcpyHostToDevice, stream.handle());
        cudaStreamSynchronize(stream.handle()); // keep host alive until the copy completes
        return true;
    }
    std::optional<std::vector<std::byte>> readCache() override { return std::nullopt; }
    void writeCache(std::span<const std::byte>) override { ++cacheWrites; }

    int calls = 0;
    int cacheWrites = 0;

private:
    int maxBatches_;
};

} // namespace

TEST(Calibrator, Int8RequiresCalibrator) {
    EngineBuilder builder;
    BuildOptions options;
    options.precision = Precision::kInt8CalibLegacy;
    options.engineCacheDir = (std::filesystem::temp_directory_path() / "trtcpp_e10").string();
    auto engine = builder.buildFromOnnxFile(modelPath("conv_1x3x8x8.onnx"), options);
    EXPECT_FALSE(engine.ok());
    EXPECT_EQ(engine.status().code(), StatusCode::kInvalidArgument); // missing calibrator
}

TEST(Calibrator, CalibratorInvokedDuringInt8Build) {
    const auto dir = std::filesystem::temp_directory_path() / "trtcpp_e10_build";
    std::filesystem::remove_all(dir);
    auto calibrator = std::make_shared<CountingCalibrator>(8);

    EngineBuilder builder;
    BuildOptions options;
    options.precision = Precision::kInt8CalibLegacy;
    options.calibrator = calibrator;
    options.engineCacheDir = dir.string();
    // The build drives calibration through the bridge. (Whether TRT ultimately emits an INT8
    // engine depends on the model; end-to-end INT8 emission is validated later on a real model.)
    builder.buildFromOnnxFile(modelPath("conv_1x3x8x8.onnx"), options);
    EXPECT_GT(calibrator->calls, 0); // TensorRT requested batches through our IInt8 bridge

    std::filesystem::remove_all(dir);
}

TEST(Calibrator, RawBatchCalibratorMechanics) {
    const auto dir = std::filesystem::temp_directory_path() / "trtcpp_e10_raw";
    std::filesystem::remove_all(dir);
    auto batches = writeRawBatches(dir, 3, 1 * 3 * 8 * 8);
    const std::string cachePath = (dir / "calib.cache").string();
    auto calibrator = makeRawBatchCalibrator("input", Shape{1, 3, 8, 8}, batches, cachePath);

    EXPECT_EQ(calibrator->batchSize(), 1);
    auto buffer = Tensor::allocate(DType::kFloat32, Shape{1, 3, 8, 8}, Device::kCuda);
    ASSERT_TRUE(buffer.ok());
    Stream stream;
    std::unordered_map<std::string, TensorView> inputs{{"input", buffer.value().view()}};

    EXPECT_TRUE(calibrator->nextBatch(inputs, stream));
    EXPECT_TRUE(calibrator->nextBatch(inputs, stream));
    EXPECT_TRUE(calibrator->nextBatch(inputs, stream));
    EXPECT_FALSE(calibrator->nextBatch(inputs, stream)); // exhausted after 3 files

    EXPECT_FALSE(calibrator->readCache().has_value()); // nothing written yet
    const std::vector<std::byte> table{std::byte{1}, std::byte{2}, std::byte{3}};
    calibrator->writeCache(table);
    auto read = calibrator->readCache();
    ASSERT_TRUE(read.has_value());
    EXPECT_EQ(read->size(), 3u);

    std::filesystem::remove_all(dir);
}

#endif // TRT_CPP_API_TENSORRT_VERSION_MAJOR < 11
