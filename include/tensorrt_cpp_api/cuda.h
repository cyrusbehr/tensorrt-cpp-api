#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

#include <cuda_runtime.h>

#include "tensorrt_cpp_api/status.h"

// CUDA is a hard dependency of the engine core (caller-provided streams cross the public
// API), so this header includes <cuda_runtime.h>. OpenCV/spdlog/TensorRT, by contrast,
// never appear in public headers.

namespace trtcpp {

/// Convert a CUDA error code into a Status (kOk on cudaSuccess). `context` prefixes the
/// message, e.g. cudaToStatus(err, "cudaMalloc").
Status cudaToStatus(cudaError_t code, std::string_view context = {});

/// RAII CUDA stream. Either OWNS a stream it created, or WRAPS a caller-provided one
/// (non-owning) -- the model that callers asked for and the Python bindings
/// require (an integer handle from torch/cupy). Replaces v6's per-call create/destroy.
///
/// A default-constructed Stream owns a new non-blocking stream; if creation fails (only
/// under catastrophic OOM) it degrades to the CUDA default stream (non-owning). Use
/// Stream::create() when you need to observe that failure.
class Stream {
public:
    Stream();
    static Result<Stream> create();                       ///< owned non-blocking stream, error-checked
    static Stream wrap(cudaStream_t existing) noexcept;   ///< non-owning
    static Stream wrap(std::uintptr_t existing) noexcept; ///< non-owning, for language bindings
    ~Stream();

    Stream(Stream &&other) noexcept;
    Stream &operator=(Stream &&other) noexcept;
    Stream(const Stream &) = delete;
    Stream &operator=(const Stream &) = delete;

    cudaStream_t handle() const noexcept { return stream_; } ///< nullptr == CUDA default stream
    std::uintptr_t raw() const noexcept { return reinterpret_cast<std::uintptr_t>(stream_); }
    bool owns() const noexcept { return owns_; }
    Status synchronize() const noexcept;

private:
    Stream(cudaStream_t stream, bool owns) noexcept : stream_(stream), owns_(owns) {}
    cudaStream_t stream_ = nullptr;
    bool owns_ = false;
};

struct DeviceInfo {
    int index = 0;
    std::string name;
    std::string uuid; ///< 32-char hex
    int computeMajor = 0;
    int computeMinor = 0;
    std::size_t totalMemoryBytes = 0;
};

Result<int> deviceCount();
Result<DeviceInfo> queryDevice(int index);

} // namespace trtcpp
