#pragma once

#include <cstddef>
#include <span>
#include <string>
#include <vector>

#include "tensorrt_cpp_api/status.h"

namespace trtcpp::detail {

// Read a regular file fully into memory. Rejects missing paths and directories (whose
// tellg() returns a huge sentinel that would otherwise trigger a giant allocation), so it
// never throws out of the no-throw API. Shared by the builder and the engine loader.
Result<std::vector<std::byte>> readFile(const std::string &path);

// Metadata recorded in the JSON sidecar next to a cached engine, used to detect stale
// caches across ONNX changes, TensorRT versions, GPUs, and build options.
struct CacheMeta {
    std::string onnxSha256;
    std::string trtVersion;  // e.g. "10.0.0"
    std::string cudaVersion; // runtime version, e.g. "12060"
    std::string gpuName;
    std::string gpuUuid;
    std::string precision;
    std::string buildOptionsDigest;
    bool versionCompatible = false;
    bool hardwareCompatible = false;
    long long createdUnix = 0;
};

// Engine cache filename: <stem>.<sha8>.trt<ver>.<gpuUuid8>.<precision>.engine
std::string cacheFileName(const std::string &onnxStem, const std::string &onnxSha256, const std::string &trtVersion,
                          const std::string &gpuUuid, const std::string &precision);

// Write `bytes` to `path` atomically (temp file + rename), creating parent dirs.
Status writeAtomic(const std::string &path, std::span<const std::byte> bytes);

// Write/read the JSON sidecar.
Status writeSidecar(const std::string &path, const CacheMeta &meta);
Result<CacheMeta> readSidecar(const std::string &path);

// A cached engine is fresh iff the ONNX hash and build-options digest match, the TRT
// version matches (only the major when versionCompatible, requiring runtime >= cached),
// and the GPU UUID matches (skipped when hardwareCompatible).
bool isFresh(const CacheMeta &sidecar, const CacheMeta &expected, bool versionCompatible, bool hardwareCompatible);

} // namespace trtcpp::detail
