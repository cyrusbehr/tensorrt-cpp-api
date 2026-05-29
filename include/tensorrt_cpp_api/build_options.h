#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "tensorrt_cpp_api/quant.h"
#include "tensorrt_cpp_api/shape.h"

namespace trtcpp {

class ICalibrator; // forward-declared (defined in the calibration module, E10, gated < TRT 11)

/// One input's min/opt/max extents for an optimization profile. Multi-input,
/// multi-dimension, multi-profile -- the dynamic-shape fix (community #80/#86/#29/#20).
struct ProfileShape {
    std::string inputName;
    Shape min;
    Shape opt;
    Shape max;
};

/// A complete optimization profile: one ProfileShape per dynamic input. For multi-stream
/// concurrent dynamic-shape inference, build one profile per execution context (EnginePool).
struct OptimizationProfile {
    std::vector<ProfileShape> inputs;
};

struct BuildOptions {
    Precision precision = Precision::kFp16;
    /// Empty => static shapes taken from the ONNX. Otherwise one profile per concurrent
    /// context you intend to run.
    std::vector<OptimizationProfile> profiles;
    int deviceIndex = 0;
    int dlaCore = -1;                          ///< -1 = GPU; >= 0 selects a DLA core
    std::optional<std::size_t> workspaceBytes; ///< setMemoryPoolLimit(kWORKSPACE); unset => TRT default
    /// nullopt => auto: true for kInt8Qdq and on TRT >= 11; false for kFp16/kFp32 on
    /// TRT < 11 (so the weak-typed precision flag is honored).
    std::optional<bool> stronglyTyped;
    bool versionCompatible = false;  ///< kVERSION_COMPATIBLE (relaxes cache staleness)
    bool hardwareCompatible = false; ///< Ampere-plus hardware compatibility (relaxes cache staleness)
    std::string engineCacheDir = ".";
    std::string timingCachePath; ///< empty => engineCacheDir/<hash>.timing
    std::vector<std::string> pluginLibraries;
    /// kInt8CalibLegacy only; ignored otherwise. Forward-declared and inert on TRT >= 11
    /// so BuildOptions stays ABI-stable across the calibrator-removal boundary.
    std::shared_ptr<ICalibrator> calibrator;
};

} // namespace trtcpp
