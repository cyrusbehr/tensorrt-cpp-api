#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace trtcpp {

/// Element type at the API boundary. Library-owned so consumers never include NvInfer.h.
/// Mirrors nvinfer1::DataType; kFp8/kInt4 are type tags whose acceleration is gated on
/// recent TensorRT and GPU architecture (validated at engine build time, not here).
enum class DType : std::uint8_t {
    kFloat32,
    kFloat16,
    kBFloat16,
    kInt32,
    kInt64,
    kInt8,
    kUInt8,
    kBool,
    kFp8,
    kInt4,
};

/// Bit width of a single element. kInt4 is sub-byte (4 bits).
constexpr int bitsPerElement(DType t) noexcept {
    switch (t) {
    case DType::kFloat32:
    case DType::kInt32:
        return 32;
    case DType::kInt64:
        return 64;
    case DType::kFloat16:
    case DType::kBFloat16:
        return 16;
    case DType::kInt8:
    case DType::kUInt8:
    case DType::kBool:
    case DType::kFp8:
        return 8;
    case DType::kInt4:
        return 4;
    }
    return 0;
}

/// Storage bytes for one element, rounded up. For sub-byte types prefer byteSizeOf,
/// which accounts for packing (kInt4 stores two elements per byte).
constexpr std::size_t byteSize(DType t) noexcept { return static_cast<std::size_t>((bitsPerElement(t) + 7) / 8); }

/// Total bytes to store `count` elements, honoring sub-byte packing. Assumes the result
/// is representable in std::size_t; callers that allocate from untrusted counts must
/// bounds-check first (the engine's buffer allocation does so before cudaMalloc).
constexpr std::size_t byteSizeOf(DType t, std::size_t count) noexcept {
    return (static_cast<std::size_t>(bitsPerElement(t)) * count + 7) / 8;
}

std::string_view toString(DType t) noexcept;

/// Maps a C++ scalar type to its DType so typed accessors can dtype-check. Left
/// undefined for unsupported T (using as<T>() with an unsupported T is a compile error).
/// fp16/bf16/fp8 have no portable dep-free C++ scalar; their specializations are defined by
/// the buffer/interop layer that needs them, in a CUDA-including header.
template <class T> struct DTypeOf;
template <> struct DTypeOf<float> {
    static constexpr DType value = DType::kFloat32;
};
template <> struct DTypeOf<std::int32_t> {
    static constexpr DType value = DType::kInt32;
};
template <> struct DTypeOf<std::int64_t> {
    static constexpr DType value = DType::kInt64;
};
template <> struct DTypeOf<std::int8_t> {
    static constexpr DType value = DType::kInt8;
};
template <> struct DTypeOf<std::uint8_t> {
    static constexpr DType value = DType::kUInt8;
};
template <> struct DTypeOf<bool> {
    static constexpr DType value = DType::kBool;
};

} // namespace trtcpp
