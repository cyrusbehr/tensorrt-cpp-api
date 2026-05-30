#include "tensorrt_cpp_api/device_tensor.h"

#include <utility>

namespace trtcpp {
namespace {

cudaMemcpyKind memcpyKind(Device src, Device dst) {
    if (src == Device::kHost && dst == Device::kCuda) {
        return cudaMemcpyHostToDevice;
    }
    if (src == Device::kCuda && dst == Device::kHost) {
        return cudaMemcpyDeviceToHost;
    }
    if (src == Device::kCuda && dst == Device::kCuda) {
        return cudaMemcpyDeviceToDevice;
    }
    return cudaMemcpyHostToHost;
}

} // namespace

Result<Tensor> Tensor::allocate(DType dtype, Shape shape, Device device, int deviceId) {
    auto byteSize = checkedByteSize(dtype, shape); // rejects dynamic shapes and overflow
    if (!byteSize) {
        return byteSize.status();
    }
    const std::size_t bytes = byteSize.value();
    void *ptr = nullptr;
    if (bytes > 0) {
        if (device == Device::kCuda) {
            // Set the target device only around the allocation and restore the caller's
            // current device, so allocate() leaves no thread-local device-state side effect.
            int previousDevice = 0;
            cudaGetDevice(&previousDevice);
            if (cudaError_t code = cudaSetDevice(deviceId); code != cudaSuccess) {
                cudaSetDevice(previousDevice);
                return cudaToStatus(code, "cudaSetDevice");
            }
            cudaError_t code = cudaMalloc(&ptr, bytes);
            cudaSetDevice(previousDevice);
            if (code != cudaSuccess) {
                return cudaToStatus(code, "cudaMalloc");
            }
        } else {
            if (cudaError_t code = cudaMallocHost(&ptr, bytes); code != cudaSuccess) {
                return cudaToStatus(code, "cudaMallocHost");
            }
        }
    }
    return Tensor{ptr, dtype, std::move(shape), device, deviceId};
}

void Tensor::reset() noexcept {
    if (data_ != nullptr) {
        if (device_ == Device::kCuda) {
            cudaFree(data_);
        } else {
            cudaFreeHost(data_);
        }
        data_ = nullptr;
    }
}

Tensor::~Tensor() { reset(); }

Tensor::Tensor(Tensor &&other) noexcept
    : data_(other.data_), dtype_(other.dtype_), shape_(std::move(other.shape_)), device_(other.device_), deviceId_(other.deviceId_) {
    other.data_ = nullptr;
}

Tensor &Tensor::operator=(Tensor &&other) noexcept {
    if (this != &other) {
        reset();
        data_ = other.data_;
        dtype_ = other.dtype_;
        shape_ = std::move(other.shape_);
        device_ = other.device_;
        deviceId_ = other.deviceId_;
        other.data_ = nullptr;
    }
    return *this;
}

TensorView Tensor::view() const noexcept { return TensorView{data_, dtype_, shape_, device_, deviceId_, Layout::kLinear}; }

Status Tensor::copyFrom(TensorView src, const Stream &stream) {
    if (src.dtype() != dtype_) {
        return Status{StatusCode::kDtypeMismatch, "Tensor::copyFrom dtype mismatch"};
    }
    if (src.nbytes() != nbytes()) {
        return Status{StatusCode::kShapeMismatch, "Tensor::copyFrom byte-size mismatch"};
    }
    if (src.device() == Device::kCuda && device_ == Device::kCuda && src.deviceId() != deviceId_) {
        // Cross-device (peer) copies need peer-access setup; not supported in v7 (CNN
        // inference is single-GPU). Stage through host explicitly if you need this.
        return Status{StatusCode::kUnsupported, "Tensor::copyFrom does not support cross-device copies; stage through host"};
    }
    if (nbytes() == 0) {
        return Status{};
    }
    cudaError_t code = cudaMemcpyAsync(data_, src.data(), nbytes(), memcpyKind(src.device(), device_), stream.handle());
    return cudaToStatus(code, "cudaMemcpyAsync");
}

Result<Tensor> Tensor::to(Device device, int deviceId, const Stream &stream) const {
    auto dst = Tensor::allocate(dtype_, shape_, device, deviceId);
    if (!dst) {
        return dst.status();
    }
    if (Status status = dst.value().copyFrom(view(), stream); !status) {
        return status;
    }
    return std::move(dst).value();
}

Result<Tensor> Tensor::toHost(const Stream &stream) const {
    auto dst = to(Device::kHost, 0, stream);
    if (!dst) {
        return dst;
    }
    if (Status status = stream.synchronize(); !status) {
        return status;
    }
    return dst;
}

} // namespace trtcpp
