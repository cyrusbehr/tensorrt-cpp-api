#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorrt_cpp_api/device_tensor.h"
#include "tensorrt_cpp_api/tensor.h"

namespace trtcpp::detail {

// Name-keyed reusable device buffers for an engine's IO tensors. Each slot owns a byte
// buffer that is reallocated only when a tensor's required size grows (or its device
// changes), so a fixed-shape inference loop allocates once and reuses thereafter --
// addressing the dynamic-output-size and per-call-allocation issues.
//
// Backed by owning Tensors (synchronous cudaMalloc/cudaFree), so a slot is destruction-safe
// without a stream; reuse amortizes the synchronous allocation cost across inferences.
//
// Device buffers only -- host staging is not managed here; the readback path already pins
// host memory via Tensor::toHost, so a separate pinned-host pool is not provided.
// has()/names() report that a slot exists, not that it holds a live (non-zero) allocation.
class NamedBuffers {
public:
    NamedBuffers() = default;

    // Ensure the buffer named `name` is large enough for (dtype, shape) on `deviceId`,
    // reallocating only if it must grow or the device changed. Returns a TensorView of
    // (dtype, shape) over the buffer. `shape` must be fully resolved (not dynamic).
    Result<TensorView> ensure(const std::string &name, DType dtype, Shape shape, int deviceId = 0);

    void *address(const std::string &name) const noexcept; // device address, or nullptr if absent
    bool has(const std::string &name) const noexcept;
    std::vector<std::string> names() const;
    std::size_t capacity(const std::string &name) const noexcept; // bytes; 0 if absent
    void clear() noexcept;

private:
    struct Slot {
        Tensor buffer; // a kUInt8 device buffer of `capacity` bytes
        std::size_t capacity = 0;
        int deviceId = 0;
    };
    std::unordered_map<std::string, Slot> slots_;
};

} // namespace trtcpp::detail
