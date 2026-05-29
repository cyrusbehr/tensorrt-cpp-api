#include "tensorrt_cpp_api/dtype.h"
#include "tensorrt_cpp_api/layout.h"
#include "tensorrt_cpp_api/shape.h"
#include "tensorrt_cpp_api/status.h"

#include <cassert>
#include <stdexcept>

namespace trtcpp {

std::string_view toString(DType t) noexcept {
    switch (t) {
    case DType::kFloat32:
        return "float32";
    case DType::kFloat16:
        return "float16";
    case DType::kBFloat16:
        return "bfloat16";
    case DType::kInt32:
        return "int32";
    case DType::kInt64:
        return "int64";
    case DType::kInt8:
        return "int8";
    case DType::kUInt8:
        return "uint8";
    case DType::kBool:
        return "bool";
    case DType::kFp8:
        return "fp8";
    case DType::kInt4:
        return "int4";
    }
    return "unknown";
}

std::string_view toString(Layout l) noexcept {
    switch (l) {
    case Layout::kNCHW:
        return "NCHW";
    case Layout::kNHWC:
        return "NHWC";
    case Layout::kLinear:
        return "linear";
    case Layout::kUnknown:
        return "unknown";
    }
    return "unknown";
}

std::string_view toString(Device d) noexcept {
    switch (d) {
    case Device::kHost:
        return "host";
    case Device::kCuda:
        return "cuda";
    }
    return "unknown";
}

std::string_view toString(StatusCode code) noexcept {
    switch (code) {
    case StatusCode::kOk:
        return "ok";
    case StatusCode::kInvalidArgument:
        return "invalid_argument";
    case StatusCode::kNotFound:
        return "not_found";
    case StatusCode::kIoError:
        return "io_error";
    case StatusCode::kCudaError:
        return "cuda_error";
    case StatusCode::kTensorRtError:
        return "tensorrt_error";
    case StatusCode::kShapeMismatch:
        return "shape_mismatch";
    case StatusCode::kDtypeMismatch:
        return "dtype_mismatch";
    case StatusCode::kUnsupported:
        return "unsupported";
    case StatusCode::kStaleCache:
        return "stale_cache";
    case StatusCode::kInternal:
        return "internal";
    }
    return "unknown";
}

Shape::Shape(std::initializer_list<std::int64_t> dims) : Shape(std::span<const std::int64_t>(dims.begin(), dims.size())) {}

Shape::Shape(std::span<const std::int64_t> dims) {
    assert(dims.size() <= static_cast<std::size_t>(kMaxRank) && "Shape rank exceeds kMaxRank");
    rank_ = static_cast<int>(dims.size() < static_cast<std::size_t>(kMaxRank) ? dims.size() : static_cast<std::size_t>(kMaxRank));
    for (int i = 0; i < rank_; ++i) {
        dims_[static_cast<std::size_t>(i)] = dims[static_cast<std::size_t>(i)];
    }
}

std::int64_t Shape::at(int i) const {
    if (i < 0 || i >= rank_) {
        throw std::out_of_range("Shape::at index out of range");
    }
    return dims_[static_cast<std::size_t>(i)];
}

bool Shape::isDynamic() const noexcept {
    for (int i = 0; i < rank_; ++i) {
        if (dims_[static_cast<std::size_t>(i)] < 0) {
            return true;
        }
    }
    return false;
}

std::int64_t Shape::numel() const noexcept {
    if (isDynamic()) {
        return 0;
    }
    std::int64_t n = 1;
    for (int i = 0; i < rank_; ++i) {
        n *= dims_[static_cast<std::size_t>(i)];
    }
    return n;
}

bool Shape::operator==(const Shape &other) const noexcept {
    if (rank_ != other.rank_) {
        return false;
    }
    for (int i = 0; i < rank_; ++i) {
        if (dims_[static_cast<std::size_t>(i)] != other.dims_[static_cast<std::size_t>(i)]) {
            return false;
        }
    }
    return true;
}

std::string Shape::toString() const {
    std::string s = "[";
    for (int i = 0; i < rank_; ++i) {
        if (i != 0) {
            s += ',';
        }
        s += std::to_string(dims_[static_cast<std::size_t>(i)]);
    }
    s += ']';
    return s;
}

} // namespace trtcpp
