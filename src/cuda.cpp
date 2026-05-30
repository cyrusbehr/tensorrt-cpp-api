#include "tensorrt_cpp_api/cuda.h"

#include <array>
#include <cstdio>

namespace trtcpp {

Status cudaToStatus(cudaError_t code, std::string_view context) {
    if (code == cudaSuccess) {
        return Status{};
    }
    std::string message;
    if (!context.empty()) {
        message.append(context);
        message.append(": ");
    }
    message.append(cudaGetErrorName(code));
    message.append(" (");
    message.append(cudaGetErrorString(code));
    message.append(")");
    return Status{StatusCode::kCudaError, std::move(message)};
}

Stream::Stream() {
    cudaStream_t stream = nullptr;
    if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess) {
        stream_ = stream;
        owns_ = true;
    } else {
        stream_ = nullptr; // degrade to the default stream
        owns_ = false;
    }
}

Result<Stream> Stream::create() {
    cudaStream_t stream = nullptr;
    cudaError_t code = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (code != cudaSuccess) {
        return cudaToStatus(code, "cudaStreamCreateWithFlags");
    }
    return Stream{stream, true};
}

Stream Stream::wrap(cudaStream_t existing) noexcept { return Stream{existing, false}; }

Stream Stream::wrap(std::uintptr_t existing) noexcept { return Stream{reinterpret_cast<cudaStream_t>(existing), false}; }

Stream::~Stream() {
    if (owns_ && stream_ != nullptr) {
        cudaStreamDestroy(stream_);
    }
}

Stream::Stream(Stream &&other) noexcept : stream_(other.stream_), owns_(other.owns_) {
    other.stream_ = nullptr;
    other.owns_ = false;
}

Stream &Stream::operator=(Stream &&other) noexcept {
    if (this != &other) {
        if (owns_ && stream_ != nullptr) {
            cudaStreamDestroy(stream_);
        }
        stream_ = other.stream_;
        owns_ = other.owns_;
        other.stream_ = nullptr;
        other.owns_ = false;
    }
    return *this;
}

Status Stream::synchronize() const noexcept { return cudaToStatus(cudaStreamSynchronize(stream_), "cudaStreamSynchronize"); }

Result<int> deviceCount() {
    int count = 0;
    cudaError_t code = cudaGetDeviceCount(&count);
    if (code != cudaSuccess) {
        return cudaToStatus(code, "cudaGetDeviceCount");
    }
    return count;
}

Result<DeviceInfo> queryDevice(int index) {
    cudaDeviceProp prop{};
    cudaError_t code = cudaGetDeviceProperties(&prop, index);
    if (code != cudaSuccess) {
        return cudaToStatus(code, "cudaGetDeviceProperties");
    }
    DeviceInfo info;
    info.index = index;
    info.name = prop.name;
    info.computeMajor = prop.major;
    info.computeMinor = prop.minor;
    info.totalMemoryBytes = prop.totalGlobalMem;
    std::array<char, 33> hex{};
    for (int i = 0; i < 16; ++i) {
        std::snprintf(hex.data() + i * 2, 3, "%02x", static_cast<unsigned char>(prop.uuid.bytes[i]));
    }
    info.uuid.assign(hex.data(), 32);
    return info;
}

} // namespace trtcpp
