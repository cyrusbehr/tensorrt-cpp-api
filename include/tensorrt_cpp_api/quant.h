#pragma once

#include <cstdint>
#include <string_view>

namespace trtcpp {

/// Precision / quantization mode for engine building. Version-aware and never a silent no-op
/// (an unachievable mode fails the build with a clear error rather than downgrading):
///   kFp32           - full precision.
///   kFp16           - half precision. On TRT < 11 via the weak-typed kFP16 flag; on
///                     TRT >= 11 requires an FP16/QDQ ONNX (else kUnsupported).
///   kInt8Qdq        - explicit Q/DQ ONNX, strongly typed, no precision flags. The
///                     forward path that compiles on both TRT 10 and 11. Default for
///                     quantized models.
///   kInt8CalibLegacy- legacy calibrator PTQ. Only available when built against TRT < 11
///                     (wired up in the calibration module); rejected on TRT >= 11.
///   kFp8 / kNvfp4   - explicit-QDQ FP8 / NVFP4; require Ada (8.9+) / Blackwell (10.0+)
///                     hardware respectively (validated at build).
enum class Precision : std::uint8_t {
    kFp32,
    kFp16,
    kInt8Qdq,
    kInt8CalibLegacy,
    kFp8,
    kNvfp4,
};

std::string_view toString(Precision p) noexcept;

} // namespace trtcpp
