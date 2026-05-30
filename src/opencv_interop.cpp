#include "tensorrt_cpp_api/opencv_interop.h"

#ifdef TRT_CPP_API_WITH_OPENCV

#include <cstddef>
#include <utility>

namespace trtcpp::opencv {

Result<DType> dtypeOfCvDepth(int cvDepth) {
    switch (cvDepth) {
    case CV_8U:
        return DType::kUInt8;
    case CV_8S:
        return DType::kInt8;
    case CV_32S:
        return DType::kInt32;
    case CV_32F:
        return DType::kFloat32;
    case CV_16F:
        return DType::kFloat16;
    default:
        return Status{StatusCode::kUnsupported, "unsupported OpenCV element depth"};
    }
}

Result<TensorView> viewOf(const cv::cuda::GpuMat &mat, Layout layout) {
    if (mat.empty()) {
        return Status{StatusCode::kInvalidArgument, "empty GpuMat"};
    }
    auto dtype = dtypeOfCvDepth(mat.depth());
    if (!dtype) {
        return dtype.status();
    }
    const std::size_t rowBytes = static_cast<std::size_t>(mat.cols) * mat.elemSize();
    if (mat.step != rowBytes) {
        // GpuMat rows are usually pitched (padded for alignment); a TensorView has no row stride.
        // Note clone() does NOT help (it is also pitched) -- copy into a cv::cuda::createContinuous
        // buffer first.
        return Status{StatusCode::kUnsupported, "non-continuous GpuMat (padded rows); copy into a cv::cuda::createContinuous buffer first"};
    }
    Shape shape{mat.rows, mat.cols, mat.channels()};
    // The GpuMat lives on the current CUDA device (OpenCV allocates on it); record that rather
    // than hardcoding device 0, so a multi-GPU caller's downstream copyFrom/enqueue targets the
    // right device.
    int device = 0;
    cudaGetDevice(&device);
    return TensorView{mat.cudaPtr(), dtype.value(), std::move(shape), Device::kCuda, device, layout};
}

Result<TensorView> viewOf(const cv::Mat &mat, Layout layout) {
    if (mat.empty()) {
        return Status{StatusCode::kInvalidArgument, "empty cv::Mat"};
    }
    if (!mat.isContinuous()) {
        return Status{StatusCode::kUnsupported, "non-continuous cv::Mat; clone() to a continuous buffer first"};
    }
    auto dtype = dtypeOfCvDepth(mat.depth());
    if (!dtype) {
        return dtype.status();
    }
    Shape shape{mat.rows, mat.cols, mat.channels()};
    return TensorView{const_cast<unsigned char *>(mat.data), dtype.value(), std::move(shape), Device::kHost, 0, layout};
}

Result<Tensor> upload(const cv::Mat &mat, const Stream &stream) {
    auto hostView = viewOf(mat);
    if (!hostView) {
        return hostView.status();
    }
    auto device = Tensor::allocate(hostView.value().dtype(), hostView.value().shape(), Device::kCuda);
    if (!device) {
        return device.status();
    }
    if (Status status = device.value().copyFrom(hostView.value(), stream); !status) {
        return status;
    }
    return std::move(device).value();
}

} // namespace trtcpp::opencv

#endif // TRT_CPP_API_WITH_OPENCV
