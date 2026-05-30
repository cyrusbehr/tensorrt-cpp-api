#pragma once

#include <cstdint>
#include <string_view>

namespace trtcpp {

/// Memory layout hint for interop and preprocessing. TensorRT IO is layout-agnostic (it
/// sees raw dims); Layout drives the preprocessing kernels and OpenCV adapters only.
enum class Layout : std::uint8_t {
    kNCHW,
    kNHWC,
    kLinear,
    kUnknown,
};

/// Where a tensor's data lives.
enum class Device : std::uint8_t {
    kHost,
    kCuda,
};

std::string_view toString(Layout l) noexcept;
std::string_view toString(Device d) noexcept;

} // namespace trtcpp
