#pragma once

#include "tensorrt_cpp_api/build_config.h"

#include <string>

namespace trtcpp {

struct Version {
    int major = 0;
    int minor = 0;
    int patch = 0;
};

/// The tensorrt_cpp_api library version.
constexpr Version libraryVersion() noexcept {
    return Version{TRT_CPP_API_VERSION_MAJOR, TRT_CPP_API_VERSION_MINOR, TRT_CPP_API_VERSION_PATCH};
}

/// The TensorRT major/minor the library was BUILT against (compile-time, from FindTensorRT).
constexpr int tensorrtBuildMajor() noexcept { return TRT_CPP_API_TENSORRT_VERSION_MAJOR; }
constexpr int tensorrtBuildMinor() noexcept { return TRT_CPP_API_TENSORRT_VERSION_MINOR; }

/// Human-readable, e.g. "7.0.0 (built against TensorRT 10.0)".
inline std::string versionString() {
    return std::to_string(TRT_CPP_API_VERSION_MAJOR) + "." + std::to_string(TRT_CPP_API_VERSION_MINOR) + "." +
           std::to_string(TRT_CPP_API_VERSION_PATCH) + " (built against TensorRT " + std::to_string(TRT_CPP_API_TENSORRT_VERSION_MAJOR) +
           "." + std::to_string(TRT_CPP_API_TENSORRT_VERSION_MINOR) + ")";
}

} // namespace trtcpp
