#pragma once

#include <cstddef>
#include <span>

#include "tensorrt_cpp_api/cuda.h"
#include "tensorrt_cpp_api/tensor.h"

namespace trtcpp {

/// An owning tensor: a RAII device buffer (cudaMalloc) or pinned host buffer
/// (cudaMallocHost). Move-only. Returned when the library allocates outputs for the
/// caller, and the type the Python bindings own. The owning counterpart
/// to the non-owning TensorView.
class Tensor {
public:
    Tensor() = default;
    ~Tensor();

    /// Allocate a contiguous buffer for `shape` of `dtype` on the given device. Device
    /// memory is synchronous cudaMalloc; host memory is pinned (cudaMallocHost) for fast
    /// async transfers. Errors (kInvalidArgument) on a dynamic shape.
    static Result<Tensor> allocate(DType dtype, Shape shape, Device device, int deviceId = 0);

    Tensor(Tensor &&other) noexcept;
    Tensor &operator=(Tensor &&other) noexcept;
    Tensor(const Tensor &) = delete;
    Tensor &operator=(const Tensor &) = delete;

    TensorView view() const noexcept;
    void *data() const noexcept { return data_; }
    DType dtype() const noexcept { return dtype_; }
    const Shape &shape() const noexcept { return shape_; }
    Device device() const noexcept { return device_; }
    int deviceId() const noexcept { return deviceId_; }
    bool empty() const noexcept { return data_ == nullptr; }
    std::size_t nbytes() const noexcept { return byteSizeOf(dtype_, static_cast<std::size_t>(shape_.numel())); }

    /// Async copy from `src` into this tensor on `stream`. The byte size and dtype must
    /// match. The transfer direction (H2D/D2H/D2D/H2H) is inferred from the devices.
    /// Cross-device (different deviceId, both CUDA) copies are rejected with kUnsupported
    /// (v7 targets single-GPU inference; stage through host for multi-GPU).
    Status copyFrom(TensorView src, const Stream &stream);

    /// Allocate a new tensor on the target device and async-copy this one into it.
    Result<Tensor> to(Device device, int deviceId, const Stream &stream) const;
    /// Materialize on the host: copy to a pinned host tensor AND synchronize `stream`, so
    /// the result is immediately readable. The common readback path.
    Result<Tensor> toHost(const Stream &stream) const;

    /// Typed, dtype-checked host span (errors on a device tensor); pair with toHost().
    template <class T> Result<std::span<const T>> as() const { return view().as<T>(); }

private:
    Tensor(void *data, DType dtype, Shape shape, Device device, int deviceId) noexcept
        : data_(data), dtype_(dtype), shape_(std::move(shape)), device_(device), deviceId_(deviceId) {}
    void reset() noexcept;

    void *data_ = nullptr;
    DType dtype_ = DType::kFloat32;
    Shape shape_;
    Device device_ = Device::kCuda;
    int deviceId_ = 0;
};

} // namespace trtcpp
