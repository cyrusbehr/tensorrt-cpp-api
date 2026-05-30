#pragma once

#include <cstddef>
#include <memory>

#include "tensorrt_cpp_api/cuda.h"

namespace trtcpp {

/// Device-memory allocator abstraction so a host application can hand TensorRT its own
/// arena. The default (defaultDeviceAllocator) is stream-ordered (cudaMallocAsync) from a
/// private pool. Used by the engine's buffer management; the standalone owning Tensor
/// uses synchronous cudaMalloc since it has no stream at construction.
class IDeviceAllocator {
public:
    virtual ~IDeviceAllocator() = default;
    /// Returns device memory ordered on `stream`, or nullptr on failure. `alignment` is a
    /// hint; the default allocator already satisfies 256-byte alignment.
    virtual void *allocate(std::size_t bytes, std::size_t alignment, const Stream &stream) = 0;
    /// Frees memory ordered on `stream`. Must be called on a stream after which the memory
    /// is no longer used.
    virtual void deallocate(void *ptr, const Stream &stream) noexcept = 0;
};

/// Stream-ordered allocator backed by a private CUDA memory pool with the release
/// threshold set to UINT64_MAX (freed memory is retained for reuse across iterations
/// rather than returned to the OS on every sync -- the documented best practice).
std::shared_ptr<IDeviceAllocator> defaultDeviceAllocator(int deviceIndex = 0);

} // namespace trtcpp
