#include "tensorrt_cpp_api/cuda.h"
#include "tensorrt_cpp_api/device_tensor.h"
#include "tensorrt_cpp_api/engine.h"
#include "tensorrt_cpp_api/engine_builder.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>

using namespace trtcpp;

namespace {

std::string modelPath(const char *name) { return std::string(TRTCPP_TEST_MODEL_DIR) + "/" + name; }

std::string cacheDir() { return (std::filesystem::temp_directory_path() / "trtcpp_e8_cache").string(); }

Result<Engine> buildAndLoad(const char *model, BuildOptions options) {
    options.engineCacheDir = cacheDir();
    EngineBuilder builder;
    auto path = builder.buildOrLoad(modelPath(model), options);
    if (!path) {
        return path.status();
    }
    return Engine::loadFromFile(path.value());
}

} // namespace

TEST(Engine, IntrospectsIO) {
    BuildOptions options;
    options.precision = Precision::kFp32;
    auto engine = buildAndLoad("relu_1x3x8x8.onnx", options);
    ASSERT_TRUE(engine.ok()) << engine.status().message();
    EXPECT_EQ(engine.value().inputNames(), (std::vector<std::string>{"input"}));
    EXPECT_EQ(engine.value().outputNames(), (std::vector<std::string>{"output"}));
    auto shape = engine.value().tensorShape("input");
    ASSERT_TRUE(shape.ok());
    EXPECT_EQ(shape.value(), (Shape{1, 3, 8, 8}));
    auto dtype = engine.value().tensorDType("output");
    ASSERT_TRUE(dtype.ok());
    EXPECT_EQ(dtype.value(), DType::kFloat32);
    EXPECT_FALSE(engine.value().tensorShape("nope").ok());
}

TEST(Engine, ReluInferenceRoundtrip) {
    BuildOptions options;
    options.precision = Precision::kFp32;
    auto engine = buildAndLoad("relu_1x3x8x8.onnx", options);
    ASSERT_TRUE(engine.ok()) << engine.status().message();

    Stream stream;
    std::vector<float> hostInput(1 * 3 * 8 * 8);
    for (std::size_t i = 0; i < hostInput.size(); ++i) {
        hostInput[i] = (i % 2 == 0) ? -static_cast<float>(i) : static_cast<float>(i);
    }
    auto deviceInput = Tensor::allocate(DType::kFloat32, Shape{1, 3, 8, 8}, Device::kCuda);
    ASSERT_TRUE(deviceInput.ok());
    TensorView hostView{hostInput.data(), DType::kFloat32, Shape{1, 3, 8, 8}, Device::kHost};
    ASSERT_TRUE(deviceInput.value().copyFrom(hostView, stream).ok());

    auto output = engine.value().inferSingle({{"input", deviceInput.value().view()}}, stream);
    ASSERT_TRUE(output.ok()) << output.status().message();

    auto host = output.value().toHost(stream);
    ASSERT_TRUE(host.ok());
    auto values = host.value().as<float>();
    ASSERT_TRUE(values.ok());
    ASSERT_EQ(values.value().size(), hostInput.size());
    for (std::size_t i = 0; i < hostInput.size(); ++i) {
        EXPECT_FLOAT_EQ(values.value()[i], std::max(hostInput[i], 0.0f)); // Relu
    }
}

TEST(Engine, EnqueueCallerAllocated) {
    BuildOptions options;
    options.precision = Precision::kFp32;
    auto engine = buildAndLoad("relu_1x3x8x8.onnx", options);
    ASSERT_TRUE(engine.ok());

    Stream stream;
    std::vector<float> hostInput(1 * 3 * 8 * 8, -2.0f); // all negative -> Relu 0
    auto deviceInput = Tensor::allocate(DType::kFloat32, Shape{1, 3, 8, 8}, Device::kCuda);
    ASSERT_TRUE(deviceInput.ok());
    TensorView hostView{hostInput.data(), DType::kFloat32, Shape{1, 3, 8, 8}, Device::kHost};
    ASSERT_TRUE(deviceInput.value().copyFrom(hostView, stream).ok());

    auto shapes = engine.value().outputShapes({{"input", deviceInput.value().view()}});
    ASSERT_TRUE(shapes.ok());
    ASSERT_EQ(shapes.value().at("output"), (Shape{1, 3, 8, 8}));

    auto deviceOutput = Tensor::allocate(DType::kFloat32, shapes.value().at("output"), Device::kCuda);
    ASSERT_TRUE(deviceOutput.ok());
    ASSERT_TRUE(engine.value().enqueue({{"input", deviceInput.value().view()}}, {{"output", deviceOutput.value().view()}}, stream).ok());

    auto host = deviceOutput.value().toHost(stream);
    ASSERT_TRUE(host.ok());
    auto values = host.value().as<float>();
    ASSERT_TRUE(values.ok());
    for (float v : values.value()) {
        EXPECT_FLOAT_EQ(v, 0.0f);
    }
}

TEST(Engine, DynamicBatchInference) {
    BuildOptions options;
    options.precision = Precision::kFp32;
    OptimizationProfile profile;
    profile.inputs.push_back({"input", Shape{1, 3, 8, 8}, Shape{2, 3, 8, 8}, Shape{4, 3, 8, 8}});
    options.profiles.push_back(profile);
    auto engine = buildAndLoad("relu_dynamic_batch.onnx", options);
    ASSERT_TRUE(engine.ok()) << engine.status().message();

    Stream stream;
    const int batch = 2; // the #1 community bug: batch > 1 must work
    std::vector<float> hostInput(static_cast<std::size_t>(batch) * 3 * 8 * 8, -1.0f);
    auto deviceInput = Tensor::allocate(DType::kFloat32, Shape{batch, 3, 8, 8}, Device::kCuda);
    ASSERT_TRUE(deviceInput.ok());
    TensorView hostView{hostInput.data(), DType::kFloat32, Shape{batch, 3, 8, 8}, Device::kHost};
    ASSERT_TRUE(deviceInput.value().copyFrom(hostView, stream).ok());

    auto output = engine.value().inferSingle({{"input", deviceInput.value().view()}}, stream);
    ASSERT_TRUE(output.ok()) << output.status().message();
    EXPECT_EQ(output.value().shape(), (Shape{batch, 3, 8, 8})); // dynamic batch resolved

    auto host = output.value().toHost(stream);
    ASSERT_TRUE(host.ok());
    auto values = host.value().as<float>();
    ASSERT_TRUE(values.ok());
    ASSERT_EQ(values.value().size(), hostInput.size());
    for (float v : values.value()) {
        EXPECT_FLOAT_EQ(v, 0.0f); // Relu(-1) == 0
    }
}

TEST(Engine, RejectsOutOfRangeProfile) {
    BuildOptions options;
    options.precision = Precision::kFp32;
    auto engine = buildAndLoad("relu_1x3x8x8.onnx", options);
    ASSERT_TRUE(engine.ok());
    Stream stream;
    auto input = Tensor::allocate(DType::kFloat32, Shape{1, 3, 8, 8}, Device::kCuda);
    ASSERT_TRUE(input.ok());
    auto output = engine.value().infer({{"input", input.value().view()}}, stream, /*profileIndex=*/5);
    EXPECT_FALSE(output.ok());
    EXPECT_EQ(output.status().code(), StatusCode::kInvalidArgument);
}

TEST(EngineBuilder, BuildAndLoadOneShot) {
    EngineBuilder builder;
    BuildOptions options;
    options.precision = Precision::kFp32;
    options.engineCacheDir = cacheDir();
    auto engine = builder.buildAndLoad(modelPath("relu_1x3x8x8.onnx"), options);
    ASSERT_TRUE(engine.ok()) << engine.status().message();
    EXPECT_EQ(engine.value().inputNames().size(), 1u);
    EXPECT_EQ(engine.value().outputNames().size(), 1u);
}

TEST(Engine, EnqueueRejectsMissingOutput) {
    BuildOptions options;
    options.precision = Precision::kFp32;
    auto engine = buildAndLoad("relu_1x3x8x8.onnx", options);
    ASSERT_TRUE(engine.ok());
    Stream stream;
    auto deviceInput = Tensor::allocate(DType::kFloat32, Shape{1, 3, 8, 8}, Device::kCuda);
    ASSERT_TRUE(deviceInput.ok());
    Status status = engine.value().enqueue({{"input", deviceInput.value().view()}}, {}, stream); // no output buffer
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}
