#include "detail/engine_cache.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>

using namespace trtcpp;
using namespace trtcpp::detail;

TEST(EngineCache, FileNameFormat) {
    EXPECT_EQ(cacheFileName("yolov8n", "deadbeefcafef00d0011", "10.0.0", "0123456789abcdef", "fp16"),
              "yolov8n.deadbeef.trt10.0.0.01234567.fp16.engine");
}

TEST(EngineCache, SidecarRoundtrip) {
    CacheMeta meta;
    meta.onnxSha256 = "aabbcc";
    meta.trtVersion = "10.0.0";
    meta.cudaVersion = "12060";
    meta.gpuName = "NVIDIA GeForce RTX 3080 Laptop GPU";
    meta.gpuUuid = "0123456789abcdef0123456789abcdef";
    meta.precision = "fp16";
    meta.buildOptionsDigest = "digest123";
    meta.versionCompatible = true;
    meta.hardwareCompatible = false;
    meta.createdUnix = 1234567890;

    const auto dir = std::filesystem::temp_directory_path() / "trtcpp_cache_roundtrip";
    std::filesystem::create_directories(dir);
    const std::string path = (dir / "engine.json").string();
    ASSERT_TRUE(writeSidecar(path, meta).ok());

    auto read = readSidecar(path);
    ASSERT_TRUE(read.ok()) << read.status().message();
    EXPECT_EQ(read.value().onnxSha256, "aabbcc");
    EXPECT_EQ(read.value().trtVersion, "10.0.0");
    EXPECT_EQ(read.value().gpuName, meta.gpuName);
    EXPECT_EQ(read.value().buildOptionsDigest, "digest123");
    EXPECT_TRUE(read.value().versionCompatible);
    EXPECT_FALSE(read.value().hardwareCompatible);
    std::filesystem::remove_all(dir);
}

TEST(EngineCache, FreshnessChecks) {
    CacheMeta expected;
    expected.onnxSha256 = "hash1";
    expected.trtVersion = "10.0.0";
    expected.gpuUuid = "gpuA";
    expected.buildOptionsDigest = "d1";

    EXPECT_TRUE(isFresh(expected, expected, false, false));

    auto changed = expected;
    changed.onnxSha256 = "hash2";
    EXPECT_FALSE(isFresh(changed, expected, false, false));

    changed = expected;
    changed.buildOptionsDigest = "d2";
    EXPECT_FALSE(isFresh(changed, expected, false, false));

    changed = expected;
    changed.trtVersion = "10.2.0";
    EXPECT_FALSE(isFresh(changed, expected, false, false)); // exact TRT mismatch

    // version-compatible: cached 10.0.0, runtime 10.2.0 (same major, runtime newer) -> fresh
    {
        CacheMeta cached = expected;
        cached.trtVersion = "10.0.0";
        CacheMeta runtime = expected;
        runtime.trtVersion = "10.2.0";
        EXPECT_TRUE(isFresh(cached, runtime, true, false));
        // different major -> stale even when version-compatible
        cached.trtVersion = "11.0.0";
        EXPECT_FALSE(isFresh(cached, runtime, true, false));
    }

    changed = expected;
    changed.gpuUuid = "gpuB";
    EXPECT_FALSE(isFresh(changed, expected, false, false)); // GPU mismatch
    EXPECT_TRUE(isFresh(changed, expected, false, true));   // hardware-compatible ignores UUID
}

TEST(EngineCache, WriteAtomicCreatesParentDirs) {
    const auto dir = std::filesystem::temp_directory_path() / "trtcpp_cache_atomic";
    std::filesystem::remove_all(dir);
    const std::string path = (dir / "nested" / "engine.bin").string();
    const std::string data = "hello-engine";
    auto bytes = std::span<const std::byte>(reinterpret_cast<const std::byte *>(data.data()), data.size());
    ASSERT_TRUE(writeAtomic(path, bytes).ok());
    ASSERT_TRUE(std::filesystem::exists(path));

    std::ifstream in(path, std::ios::binary);
    const std::string back((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    EXPECT_EQ(back, data);
    std::filesystem::remove_all(dir);
}
