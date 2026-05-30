#pragma once

#include <cstddef>
#include <cstdint> // SIZE_MAX
#include <span>
#include <utility>

#include "tensorrt_cpp_api/dtype.h"
#include "tensorrt_cpp_api/layout.h"
#include "tensorrt_cpp_api/shape.h"
#include "tensorrt_cpp_api/status.h"

namespace trtcpp {

/// A non-owning view over host or device memory. The primary input/output type at the
/// enqueue boundary -- zero-copy by construction. Carries enough to drive
/// setTensorAddress + setInputShape and to build DLPack / __cuda_array_interface__.
///
/// The owning Tensor (allocate/toHost/copyFrom) is CUDA-backed.
class TensorView {
public:
    struct Desc {
        void *data = nullptr;
        DType dtype = DType::kFloat32;
        Shape shape;
        Device device = Device::kCuda;
        int deviceId = 0;
        Layout layout = Layout::kLinear;
    };

    TensorView() = default;
    explicit TensorView(Desc d) : d_(std::move(d)) {}
    TensorView(void *data, DType dtype, Shape shape, Device device, int deviceId = 0, Layout layout = Layout::kLinear)
        : d_{data, dtype, std::move(shape), device, deviceId, layout} {}

    void *data() const noexcept { return d_.data; }
    DType dtype() const noexcept { return d_.dtype; }
    const Shape &shape() const noexcept { return d_.shape; }
    Device device() const noexcept { return d_.device; }
    int deviceId() const noexcept { return d_.deviceId; }
    Layout layout() const noexcept { return d_.layout; }
    bool isCuda() const noexcept { return d_.device == Device::kCuda; }
    /// Byte size of the data. A dynamic (unresolved) shape has numel 0, so this returns
    /// 0 -- resolve the shape (set concrete dims) before using nbytes for allocation.
    std::size_t nbytes() const noexcept { return byteSizeOf(d_.dtype, static_cast<std::size_t>(d_.shape.numel())); }

    /// Typed, dtype-checked host span. Errors (kInvalidArgument) on a device view --
    /// never an implicit D2H copy -- or (kDtypeMismatch) if T does not match dtype().
    /// A dynamic shape yields an ok, length-0 span (numel 0): resolve the shape first.
    template <class T> Result<std::span<const T>> as() const;

private:
    Desc d_;
};

/// Byte size of a resolved (dtype, shape), with overflow checking -- the guard the buffer
/// layer owes before any allocation. Errors (kInvalidArgument) on a dynamic shape or if
/// the element count / byte size would overflow std::size_t. (GCC/Clang builtins; the
/// library is Linux/GCC-Clang only.)
inline Result<std::size_t> checkedByteSize(DType dtype, const Shape &shape) {
    if (shape.isDynamic()) {
        return Status{StatusCode::kInvalidArgument, "checkedByteSize requires a fully resolved (non-dynamic) shape"};
    }
    std::size_t count = 1;
    for (int i = 0; i < shape.rank(); ++i) {
        if (__builtin_mul_overflow(count, static_cast<std::size_t>(shape[i]), &count)) {
            return Status{StatusCode::kInvalidArgument, "shape element count overflows std::size_t"};
        }
    }
    std::size_t totalBits = 0;
    if (__builtin_mul_overflow(count, static_cast<std::size_t>(bitsPerElement(dtype)), &totalBits)) {
        return Status{StatusCode::kInvalidArgument, "tensor byte size overflows std::size_t"};
    }
    if (totalBits > SIZE_MAX - 7) { // guard the round-up below from wrapping past SIZE_MAX
        return Status{StatusCode::kInvalidArgument, "tensor byte size overflows std::size_t"};
    }
    return (totalBits + 7) / 8;
}

template <class T> Result<std::span<const T>> TensorView::as() const {
    if (d_.device != Device::kHost) {
        return Status{StatusCode::kInvalidArgument, "TensorView::as<T>() requires a host tensor; copy to host first (Tensor::toHost)"};
    }
    if (d_.dtype != DTypeOf<T>::value) {
        return Status{StatusCode::kDtypeMismatch, "TensorView::as<T>() dtype does not match the tensor's dtype"};
    }
    return std::span<const T>(static_cast<const T *>(d_.data), static_cast<std::size_t>(d_.shape.numel()));
}

} // namespace trtcpp
