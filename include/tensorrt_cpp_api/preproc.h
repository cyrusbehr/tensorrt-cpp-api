#pragma once

#include <array>
#include <cstdint>

#include "tensorrt_cpp_api/cuda.h"
#include "tensorrt_cpp_api/dtype.h"
#include "tensorrt_cpp_api/status.h"
#include "tensorrt_cpp_api/tensor.h"

// Optional preprocessing sublibrary (tensorrt_cpp_api::preproc). A single fused CUDA kernel
// covers the ~5 ops a CNN TRT input needs, so the engine core links nothing here and there
// is no OpenCV dependency. Build it with -DTRT_CPP_API_BUILD_PREPROC=ON (default ON).

namespace trtcpp::preproc {

/// How to turn a decoded HWC uint8 image into a TRT-ready NCHW tensor. The normalization is
/// out = (pixel - mean[c]) * scale[c] applied per channel (fixes the v6 broken per-channel
/// path): e.g. [0,1] scaling is mean=0, scale=1/255; ImageNet folds 1/(255*std) into scale.
/// The output channel count and dtype are taken from `dst` (not duplicated here). Resize,
/// layout, and channel copy handle any channel count; per-channel mean/scale cover up to 4
/// channels (channels >= 4 use identity normalization).
struct PreprocSpec {
    std::array<float, 4> mean{0.0f, 0.0f, 0.0f, 0.0f};
    std::array<float, 4> scale{1.0f, 1.0f, 1.0f, 1.0f};
    bool swapRB = false;            ///< swap channels 0 and 2 (BGR<->RGB), 3-channel only
    bool keepAspectRatioPad = true; ///< letterbox (resize keeping aspect, pad right/bottom) vs stretch
    std::uint8_t padValue = 0;      ///< fill for the letterbox padding (then normalized like a pixel)
};

/// Fused: letterbox-resize -> optional BGR<->RGB -> (pixel-mean)*scale -> HWC->NCHW -> cast,
/// writing `dst` with no intermediate buffers. Async on `stream` (no implicit sync).
///   src: HWC uint8 device tensor, shape [H,W,C] or [1,H,W,C].
///   dst: pre-allocated NCHW device tensor [1, C, Hout, Wout], dtype kFloat32 or kFloat16.
Status letterboxToTensor(TensorView src, TensorView dst, const PreprocSpec &spec, const Stream &stream);

} // namespace trtcpp::preproc
