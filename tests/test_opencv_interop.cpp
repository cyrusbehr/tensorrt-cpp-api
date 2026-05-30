#include "tensorrt_cpp_api/opencv_interop.h"

#ifdef TRT_CPP_API_WITH_OPENCV

#include "tensorrt_cpp_api/cuda.h"
#include "tensorrt_cpp_api/device_tensor.h"

#include <gtest/gtest.h>

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

using namespace trtcpp;

TEST(OpenCVInterop, MatHostViewIsHwc) {
    cv::Mat mat(4, 5, CV_8UC3, cv::Scalar(1, 2, 3));
    auto view = opencv::viewOf(mat);
    ASSERT_TRUE(view.ok()) << view.status().message();
    EXPECT_EQ(view.value().dtype(), DType::kUInt8);
    EXPECT_EQ(view.value().shape(), (Shape{4, 5, 3}));
    EXPECT_EQ(view.value().device(), Device::kHost);
    EXPECT_EQ(view.value().data(), mat.data);
}

TEST(OpenCVInterop, UnsupportedDepthRejected) {
    cv::Mat mat(2, 2, CV_64FC1);
    auto view = opencv::viewOf(mat);
    EXPECT_FALSE(view.ok());
    EXPECT_EQ(view.status().code(), StatusCode::kUnsupported);
}

TEST(OpenCVInterop, UploadRoundtrip) {
    cv::Mat mat(2, 2, CV_32FC1);
    mat.at<float>(0, 0) = 1.0f;
    mat.at<float>(0, 1) = 2.0f;
    mat.at<float>(1, 0) = 3.0f;
    mat.at<float>(1, 1) = 4.0f;

    Stream stream;
    auto device = opencv::upload(mat, stream);
    ASSERT_TRUE(device.ok()) << device.status().message();
    EXPECT_EQ(device.value().device(), Device::kCuda);
    EXPECT_EQ(device.value().shape(), (Shape{2, 2, 1}));

    auto host = device.value().toHost(stream);
    ASSERT_TRUE(host.ok());
    auto values = host.value().as<float>();
    ASSERT_TRUE(values.ok());
    ASSERT_EQ(values.value().size(), 4u);
    EXPECT_FLOAT_EQ(values.value()[0], 1.0f);
    EXPECT_FLOAT_EQ(values.value()[3], 4.0f);
}

TEST(OpenCVInterop, ContinuousGpuMatView) {
    cv::cuda::GpuMat gpu(4, 8, CV_8UC3);
    auto view = opencv::viewOf(gpu);
    // A padded GpuMat is rejected (kUnsupported); a continuous one yields an HWC device view.
    if (view.ok()) {
        EXPECT_EQ(view.value().device(), Device::kCuda);
        EXPECT_EQ(view.value().shape(), (Shape{4, 8, 3}));
    } else {
        EXPECT_EQ(view.status().code(), StatusCode::kUnsupported);
    }
}

#endif // TRT_CPP_API_WITH_OPENCV
