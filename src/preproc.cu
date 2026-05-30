#include "tensorrt_cpp_api/preproc.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>

namespace trtcpp::preproc {
namespace {

template <typename T> __device__ T toOut(float x);
template <> __device__ float toOut<float>(float x) { return x; }
template <> __device__ __half toOut<__half>(float x) { return __float2half(x); }

struct Norm {
    float mean[4];
    float scale[4];
};

template <typename OutT>
__global__ void letterboxKernel(const unsigned char *src, int srcH, int srcW, int channels, OutT *dst, int outC, int outH, int outW,
                                int newW, int newH, float invScaleX, float invScaleY, Norm norm, bool swapRB, unsigned char padValue) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = outC * outH * outW;
    if (index >= total) {
        return;
    }
    const int ox = index % outW;
    const int oy = (index / outW) % outH;
    const int oc = index / (outW * outH);
    const float mean = oc < 4 ? norm.mean[oc] : 0.0f;
    const float scale = oc < 4 ? norm.scale[oc] : 1.0f;

    float value;
    if (ox < newW && oy < newH) {
        const float sx = (ox + 0.5f) * invScaleX - 0.5f;
        const float sy = (oy + 0.5f) * invScaleY - 0.5f;
        int x0 = static_cast<int>(floorf(sx));
        int y0 = static_cast<int>(floorf(sy));
        const float ax = sx - x0;
        const float ay = sy - y0;
        const int x1 = min(x0 + 1, srcW - 1);
        const int y1 = min(y0 + 1, srcH - 1);
        x0 = max(0, min(x0, srcW - 1));
        y0 = max(0, min(y0, srcH - 1));
        // BGR<->RGB swap only applies to the first 3 channels; for an output channel >= 3 (or a
        // non-3-channel source) keep identity. Clamp to [0, channels-1] so a swapped index can
        // never go negative or out of bounds (e.g. oc>=3 would make 2-oc negative).
        int ic = (swapRB && channels == 3 && oc < 3) ? (2 - oc) : oc;
        ic = max(0, min(ic, channels - 1));
        const float p00 = src[(y0 * srcW + x0) * channels + ic];
        const float p01 = src[(y0 * srcW + x1) * channels + ic];
        const float p10 = src[(y1 * srcW + x0) * channels + ic];
        const float p11 = src[(y1 * srcW + x1) * channels + ic];
        const float top = p00 * (1.0f - ax) + p01 * ax;
        const float bottom = p10 * (1.0f - ax) + p11 * ax;
        value = top * (1.0f - ay) + bottom * ay;
    } else {
        value = static_cast<float>(padValue);
    }
    dst[oc * outH * outW + oy * outW + ox] = toOut<OutT>((value - mean) * scale);
}

// Read (H, W, C) from a TensorView whose shape is [H,W,C] or [1,H,W,C].
bool readHwc(const TensorView &src, int &h, int &w, int &c) {
    const Shape &shape = src.shape();
    if (shape.rank() == 3) {
        h = static_cast<int>(shape[0]);
        w = static_cast<int>(shape[1]);
        c = static_cast<int>(shape[2]);
        return true;
    }
    if (shape.rank() == 4 && shape[0] == 1) {
        h = static_cast<int>(shape[1]);
        w = static_cast<int>(shape[2]);
        c = static_cast<int>(shape[3]);
        return true;
    }
    return false;
}

} // namespace

Status letterboxToTensor(TensorView src, TensorView dst, const PreprocSpec &spec, const Stream &stream) {
    if (!src.isCuda() || !dst.isCuda()) {
        return Status{StatusCode::kInvalidArgument, "letterboxToTensor requires device tensors"};
    }
    if (src.dtype() != DType::kUInt8) {
        return Status{StatusCode::kDtypeMismatch, "letterboxToTensor source must be uint8 HWC"};
    }
    if (dst.dtype() != DType::kFloat32 && dst.dtype() != DType::kFloat16) {
        return Status{StatusCode::kDtypeMismatch, "letterboxToTensor destination must be float32 or float16"};
    }
    if (dst.shape().rank() != 4 || dst.shape()[0] != 1) {
        return Status{StatusCode::kInvalidArgument, "letterboxToTensor destination must be NCHW with N==1"};
    }
    int srcH = 0, srcW = 0, channels = 0;
    if (!readHwc(src, srcH, srcW, channels) || srcH <= 0 || srcW <= 0 || channels <= 0) {
        return Status{StatusCode::kInvalidArgument, "letterboxToTensor source must be HWC ([H,W,C] or [1,H,W,C])"};
    }
    const int outC = static_cast<int>(dst.shape()[1]);
    const int outH = static_cast<int>(dst.shape()[2]);
    const int outW = static_cast<int>(dst.shape()[3]);

    int newW = outW;
    int newH = outH;
    float invScaleX = static_cast<float>(srcW) / outW;
    float invScaleY = static_cast<float>(srcH) / outH;
    if (spec.keepAspectRatioPad) {
        const float scale = std::min(static_cast<float>(outW) / srcW, static_cast<float>(outH) / srcH);
        newW = static_cast<int>(std::lround(srcW * scale));
        newH = static_cast<int>(std::lround(srcH * scale));
        newW = std::min(newW, outW);
        newH = std::min(newH, outH);
        invScaleX = invScaleY = 1.0f / scale;
    }

    Norm norm{};
    for (int i = 0; i < 4; ++i) {
        norm.mean[i] = spec.mean[static_cast<std::size_t>(i)];
        norm.scale[i] = spec.scale[static_cast<std::size_t>(i)];
    }

    // Guard the element count against 32-bit overflow: both the grid size here and the kernel's
    // internal flat index are int, so an outC*outH*outW exceeding INT_MAX would silently launch
    // zero blocks (leaving the output unwritten). Reject it cleanly instead.
    const long long total64 = static_cast<long long>(outC) * outH * outW;
    if (total64 > 2147483647LL) {
        return Status{StatusCode::kInvalidArgument, "letterboxToTensor output tensor has too many elements (> INT_MAX)"};
    }
    const int total = static_cast<int>(total64);
    const int blockSize = 256;
    const int gridSize = (total + blockSize - 1) / blockSize;
    const auto *srcPtr = static_cast<const unsigned char *>(src.data());

    if (dst.dtype() == DType::kFloat32) {
        letterboxKernel<float><<<gridSize, blockSize, 0, stream.handle()>>>(srcPtr, srcH, srcW, channels, static_cast<float *>(dst.data()),
                                                                            outC, outH, outW, newW, newH, invScaleX, invScaleY, norm,
                                                                            spec.swapRB, spec.padValue);
    } else {
        letterboxKernel<__half><<<gridSize, blockSize, 0, stream.handle()>>>(srcPtr, srcH, srcW, channels,
                                                                             static_cast<__half *>(dst.data()), outC, outH, outW, newW,
                                                                             newH, invScaleX, invScaleY, norm, spec.swapRB, spec.padValue);
    }
    return cudaToStatus(cudaGetLastError(), "letterboxToTensor kernel launch");
}

} // namespace trtcpp::preproc
