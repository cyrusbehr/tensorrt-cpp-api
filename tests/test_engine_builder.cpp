#include "detail/engine_cache.h"
#include "tensorrt_cpp_api/cuda.h"
#include "tensorrt_cpp_api/engine_builder.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <span>
#include <string>

using namespace trtcpp;

namespace {
std::string modelPath(const char *name) { return std::string(TRTCPP_TEST_MODEL_DIR) + "/" + name; }
} // namespace

TEST(EngineBuilder, BuildsFixedShapeFp32) {
    EngineBuilder builder;
    BuildOptions options;
    options.precision = Precision::kFp32;
    auto engine = builder.buildFromOnnxFile(modelPath("relu_1x3x8x8.onnx"), options);
    ASSERT_TRUE(engine.ok()) << engine.status().message();
    EXPECT_GT(engine.value().size(), 0u);
}

TEST(EngineBuilder, BuildsFixedShapeFp16) {
    EngineBuilder builder;
    BuildOptions options;
    options.precision = Precision::kFp16; // weak-typed kFP16 path on TRT 10
    auto engine = builder.buildFromOnnxFile(modelPath("relu_1x3x8x8.onnx"), options);
    ASSERT_TRUE(engine.ok()) << engine.status().message();
    EXPECT_GT(engine.value().size(), 0u);
}

TEST(EngineBuilder, BuildsDynamicBatchWithProfile) {
    EngineBuilder builder;
    BuildOptions options;
    options.precision = Precision::kFp32;
    OptimizationProfile profile;
    profile.inputs.push_back({"input", Shape{1, 3, 8, 8}, Shape{2, 3, 8, 8}, Shape{4, 3, 8, 8}});
    options.profiles.push_back(profile);
    auto engine = builder.buildFromOnnxFile(modelPath("relu_dynamic_batch.onnx"), options);
    ASSERT_TRUE(engine.ok()) << engine.status().message();
    EXPECT_GT(engine.value().size(), 0u);
}

TEST(EngineBuilder, MissingFileIsNotFound) {
    EngineBuilder builder;
    auto engine = builder.buildFromOnnxFile(modelPath("does_not_exist.onnx"), BuildOptions{});
    EXPECT_FALSE(engine.ok());
    EXPECT_EQ(engine.status().code(), StatusCode::kNotFound);
}

TEST(EngineBuilder, DirectoryPathDoesNotCrash) {
    EngineBuilder builder;
    auto engine = builder.buildFromOnnxFile(TRTCPP_TEST_MODEL_DIR, BuildOptions{}); // a directory, not a file
    EXPECT_FALSE(engine.ok());
    EXPECT_EQ(engine.status().code(), StatusCode::kNotFound);
}

TEST(EngineBuilder, GarbageOnnxFailsToParse) {
    EngineBuilder builder;
    const char junk[] = "not a real onnx model";
    auto engine = builder.buildFromOnnxBytes(std::as_bytes(std::span<const char>(junk, sizeof(junk))), BuildOptions{});
    EXPECT_FALSE(engine.ok());
    EXPECT_EQ(engine.status().code(), StatusCode::kTensorRtError);
}

TEST(EngineBuilder, Fp8RejectedOnAmpere) {
    auto info = queryDevice(0);
    if (info.ok() && (info.value().computeMajor * 10 + info.value().computeMinor) >= 89) {
        GTEST_SKIP() << "device supports FP8";
    }
    EngineBuilder builder;
    BuildOptions options;
    options.precision = Precision::kFp8;
    auto engine = builder.buildFromOnnxFile(modelPath("relu_1x3x8x8.onnx"), options);
    EXPECT_FALSE(engine.ok());
    EXPECT_EQ(engine.status().code(), StatusCode::kUnsupported);
}

TEST(EngineBuilder, BuildOrLoadCacheHitMissStale) {
    const auto dir = std::filesystem::temp_directory_path() / "trtcpp_engine_cache";
    std::filesystem::remove_all(dir);

    EngineBuilder builder;
    BuildOptions options;
    options.precision = Precision::kFp32;
    options.engineCacheDir = dir.string();

    // Miss -> build, engine + sidecar created.
    auto first = builder.buildOrLoad(modelPath("relu_1x3x8x8.onnx"), options);
    ASSERT_TRUE(first.ok()) << first.status().message();
    ASSERT_TRUE(std::filesystem::exists(first.value()));
    ASSERT_TRUE(std::filesystem::exists(first.value() + ".json"));
    const auto builtTime = std::filesystem::last_write_time(first.value());

    // Hit -> same path, engine file NOT rewritten.
    auto second = builder.buildOrLoad(modelPath("relu_1x3x8x8.onnx"), options);
    ASSERT_TRUE(second.ok());
    EXPECT_EQ(second.value(), first.value());
    EXPECT_EQ(std::filesystem::last_write_time(second.value()), builtTime);

    // Stale -> corrupt the sidecar's ONNX hash; next call detects it and rebuilds.
    {
        std::ofstream sidecar(first.value() + ".json", std::ios::trunc);
        sidecar << R"({ "onnx_sha256": "deadbeef" })";
    }
    auto third = builder.buildOrLoad(modelPath("relu_1x3x8x8.onnx"), options);
    ASSERT_TRUE(third.ok());
    EXPECT_EQ(third.value(), first.value());
    auto meta = trtcpp::detail::readSidecar(third.value() + ".json");
    ASSERT_TRUE(meta.ok());
    EXPECT_NE(meta.value().onnxSha256, "deadbeef"); // sidecar was rewritten with the real hash

    std::filesystem::remove_all(dir);
}
