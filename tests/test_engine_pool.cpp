#include "tensorrt_cpp_api/cuda.h"
#include "tensorrt_cpp_api/device_tensor.h"
#include "tensorrt_cpp_api/engine_builder.h"
#include "tensorrt_cpp_api/engine_pool.h"

#include <gtest/gtest.h>

#include <atomic>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

using namespace trtcpp;

namespace {

std::string modelPath(const char *name) { return std::string(TRTCPP_TEST_MODEL_DIR) + "/" + name; }
std::string cacheDir() { return (std::filesystem::temp_directory_path() / "trtcpp_e9_cache").string(); }

// Build the dynamic-batch engine with `profiles` identical profiles and return its path.
Result<std::string> buildDynamic(int profiles) {
    BuildOptions options;
    options.precision = Precision::kFp32;
    options.engineCacheDir = cacheDir();
    for (int i = 0; i < profiles; ++i) {
        OptimizationProfile profile;
        profile.inputs.push_back({"input", Shape{1, 3, 8, 8}, Shape{2, 3, 8, 8}, Shape{4, 3, 8, 8}});
        options.profiles.push_back(profile);
    }
    return EngineBuilder{}.buildOrLoad(modelPath("relu_dynamic_batch.onnx"), options);
}

Result<std::string> buildStatic() {
    BuildOptions options;
    options.precision = Precision::kFp32;
    options.engineCacheDir = cacheDir();
    return EngineBuilder{}.buildOrLoad(modelPath("relu_1x3x8x8.onnx"), options);
}

} // namespace

TEST(EnginePool, Metadata) {
    auto path = buildStatic();
    ASSERT_TRUE(path.ok()) << path.status().message();
    auto pool = EnginePool::create(path.value(), 2);
    ASSERT_TRUE(pool.ok()) << pool.status().message();
    EXPECT_EQ(pool.value().size(), 2);
    EXPECT_EQ(pool.value().inputNames(), (std::vector<std::string>{"input"}));
    EXPECT_EQ(pool.value().outputNames(), (std::vector<std::string>{"output"}));
}

TEST(EnginePool, RejectsMoreContextsThanProfiles) {
    auto path = buildDynamic(2);
    ASSERT_TRUE(path.ok());
    auto pool = EnginePool::create(path.value(), 4); // engine has only 2 profiles
    EXPECT_FALSE(pool.ok());
    EXPECT_EQ(pool.status().code(), StatusCode::kInvalidArgument);
}

TEST(EnginePool, TryAcquireExhaustionAndRelease) {
    auto path = buildStatic();
    ASSERT_TRUE(path.ok());
    auto pool = EnginePool::create(path.value(), 2);
    ASSERT_TRUE(pool.ok());

    auto a = pool.value().tryAcquire();
    auto b = pool.value().tryAcquire();
    ASSERT_TRUE(a.has_value());
    ASSERT_TRUE(b.has_value());
    EXPECT_FALSE(pool.value().tryAcquire().has_value()); // both contexts in use

    a.reset(); // return one lease
    auto c = pool.value().tryAcquire();
    EXPECT_TRUE(c.has_value());
}

TEST(EnginePool, ConcurrentInferenceFourContexts) {
    auto path = buildDynamic(4);
    ASSERT_TRUE(path.ok()) << path.status().message();
    auto pool = EnginePool::create(path.value(), 4);
    ASSERT_TRUE(pool.ok()) << pool.status().message();
    EXPECT_EQ(pool.value().size(), 4);

    std::atomic<int> successes{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < 8; ++t) {
        threads.emplace_back([&pool, &successes] {
            auto lease = pool.value().acquire(); // blocks until a context is free
            if (!lease.ok()) {
                return;
            }
            Stream stream;
            std::vector<float> hostInput(2 * 3 * 8 * 8, -1.0f); // batch 2, all negative
            auto deviceInput = Tensor::allocate(DType::kFloat32, Shape{2, 3, 8, 8}, Device::kCuda);
            if (!deviceInput.ok()) {
                return;
            }
            TensorView hostView{hostInput.data(), DType::kFloat32, Shape{2, 3, 8, 8}, Device::kHost};
            if (!deviceInput.value().copyFrom(hostView, stream).ok()) {
                return;
            }
            auto output = lease.value().inferSingle({{"input", deviceInput.value().view()}}, stream);
            if (!output.ok()) {
                return;
            }
            auto host = output.value().toHost(stream);
            if (!host.ok()) {
                return;
            }
            auto values = host.value().as<float>();
            if (!values.ok() || values.value().size() != hostInput.size()) {
                return;
            }
            for (float v : values.value()) {
                if (v != 0.0f) {
                    return;
                }
            }
            successes.fetch_add(1);
        });
    }
    for (std::thread &thread : threads) {
        thread.join();
    }
    EXPECT_EQ(successes.load(), 8); // all 8 jobs ran correctly across 4 concurrent contexts
}
