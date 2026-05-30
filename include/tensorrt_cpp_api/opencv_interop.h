#pragma once

// Optional OpenCV interop (TRT_CPP_API_WITH_OPENCV). Strictly opt-in: the engine core knows
// nothing about OpenCV, and this header is empty unless the option is on (which links OpenCV
// and defines the macro). Adapters between cv::Mat / cv::cuda::GpuMat and the library types.
#ifdef TRT_CPP_API_WITH_OPENCV

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

#include "tensorrt_cpp_api/cuda.h"
#include "tensorrt_cpp_api/device_tensor.h"
#include "tensorrt_cpp_api/dtype.h"
#include "tensorrt_cpp_api/status.h"
#include "tensorrt_cpp_api/tensor.h"

namespace trtcpp::opencv {

/// Map an OpenCV element depth (CV_8U, CV_32F, ...) to a DType.
Result<DType> dtypeOfCvDepth(int cvDepth);

/// Zero-copy device view over a CONTINUOUS GpuMat as an HWC tensor [rows, cols, channels].
/// Errors (kUnsupported) on a padded (non-continuous) GpuMat -- GpuMat rows are usually pitched,
/// and a TensorView has no row stride, so copy into a cv::cuda::createContinuous() buffer first
/// (clone() does not help; it is pitched too). Device is the current CUDA device.
Result<TensorView> viewOf(const cv::cuda::GpuMat &mat, Layout layout = Layout::kNHWC);

/// Host view over a continuous cv::Mat as an HWC tensor [rows, cols, channels].
Result<TensorView> viewOf(const cv::Mat &mat, Layout layout = Layout::kNHWC);

/// Upload a cv::Mat to a freshly allocated device Tensor (async on `stream`).
Result<Tensor> upload(const cv::Mat &mat, const Stream &stream);

} // namespace trtcpp::opencv

#endif // TRT_CPP_API_WITH_OPENCV
