#include "tensorrt_cpp_api/cuda.h"
#include "tensorrt_cpp_api/device_tensor.h"
#include "tensorrt_cpp_api/preproc.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

using namespace trtcpp;
using namespace trtcpp::preproc;

namespace {

// Build an HWC uint8 image with a constant per-channel value, uploaded to the device.
Result<Tensor> constantImage(int h, int w, int c, const std::vector<std::uint8_t> &channelValues) {
    std::vector<std::uint8_t> host(static_cast<std::size_t>(h * w * c));
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            for (int ch = 0; ch < c; ++ch) {
                host[static_cast<std::size_t>((y * w + x) * c + ch)] = channelValues[static_cast<std::size_t>(ch)];
            }
        }
    }
    auto device = Tensor::allocate(DType::kUInt8, Shape{h, w, c}, Device::kCuda);
    if (!device) {
        return device.status();
    }
    Stream stream;
    TensorView view{host.data(), DType::kUInt8, Shape{h, w, c}, Device::kHost};
    if (Status status = device.value().copyFrom(view, stream); !status) {
        return status;
    }
    if (Status status = stream.synchronize(); !status) { // keep host alive until uploaded
        return status;
    }
    return std::move(device).value();
}

} // namespace

TEST(Preproc, ConstantImageNoResize) {
    auto src = constantImage(4, 4, 3, {10, 20, 30});
    ASSERT_TRUE(src.ok()) << src.status().message();
    auto dst = Tensor::allocate(DType::kFloat32, Shape{1, 3, 4, 4}, Device::kCuda);
    ASSERT_TRUE(dst.ok());
    Stream stream;
    ASSERT_TRUE(letterboxToTensor(src.value().view(), dst.value().view(), PreprocSpec{}, stream).ok());

    auto host = dst.value().toHost(stream);
    ASSERT_TRUE(host.ok());
    auto values = host.value().as<float>();
    ASSERT_TRUE(values.ok());
    for (int x = 0; x < 16; ++x) {
        EXPECT_FLOAT_EQ(values.value()[0 * 16 + x], 10.0f); // channel 0
        EXPECT_FLOAT_EQ(values.value()[1 * 16 + x], 20.0f); // channel 1
        EXPECT_FLOAT_EQ(values.value()[2 * 16 + x], 30.0f); // channel 2
    }
}

TEST(Preproc, SwapRB) {
    auto src = constantImage(4, 4, 3, {10, 20, 30});
    ASSERT_TRUE(src.ok());
    auto dst = Tensor::allocate(DType::kFloat32, Shape{1, 3, 4, 4}, Device::kCuda);
    ASSERT_TRUE(dst.ok());
    Stream stream;
    PreprocSpec spec;
    spec.swapRB = true;
    ASSERT_TRUE(letterboxToTensor(src.value().view(), dst.value().view(), spec, stream).ok());

    auto host = dst.value().toHost(stream);
    ASSERT_TRUE(host.ok());
    auto values = host.value().as<float>();
    ASSERT_TRUE(values.ok());
    EXPECT_FLOAT_EQ(values.value()[0 * 16], 30.0f); // channel 0 now holds the old channel 2
    EXPECT_FLOAT_EQ(values.value()[2 * 16], 10.0f);
}

TEST(Preproc, PerChannelNormalize) {
    auto src = constantImage(4, 4, 3, {10, 20, 30});
    ASSERT_TRUE(src.ok());
    auto dst = Tensor::allocate(DType::kFloat32, Shape{1, 3, 4, 4}, Device::kCuda);
    ASSERT_TRUE(dst.ok());
    Stream stream;
    PreprocSpec spec;
    spec.mean = {10.0f, 20.0f, 30.0f, 0.0f};
    spec.scale = {1.0f, 1.0f, 1.0f, 1.0f};
    ASSERT_TRUE(letterboxToTensor(src.value().view(), dst.value().view(), spec, stream).ok());

    auto host = dst.value().toHost(stream);
    ASSERT_TRUE(host.ok());
    auto values = host.value().as<float>();
    ASSERT_TRUE(values.ok());
    for (float v : values.value()) {
        EXPECT_FLOAT_EQ(v, 0.0f); // (channel - mean[channel]) == 0
    }
}

TEST(Preproc, LetterboxPadsRightBottom) {
    auto src = constantImage(2, 4, 3, {50, 50, 50}); // H=2, W=4 -> scale 1, pads rows 2..3
    ASSERT_TRUE(src.ok());
    auto dst = Tensor::allocate(DType::kFloat32, Shape{1, 3, 4, 4}, Device::kCuda);
    ASSERT_TRUE(dst.ok());
    Stream stream;
    ASSERT_TRUE(letterboxToTensor(src.value().view(), dst.value().view(), PreprocSpec{}, stream).ok());

    auto host = dst.value().toHost(stream);
    ASSERT_TRUE(host.ok());
    auto values = host.value().as<float>();
    ASSERT_TRUE(values.ok());
    auto at = [&](int c, int y, int x) { return values.value()[c * 16 + y * 4 + x]; };
    EXPECT_FLOAT_EQ(at(0, 0, 0), 50.0f); // resized region
    EXPECT_FLOAT_EQ(at(0, 1, 3), 50.0f);
    EXPECT_FLOAT_EQ(at(0, 2, 0), 0.0f); // padded region (rows 2..3)
    EXPECT_FLOAT_EQ(at(0, 3, 3), 0.0f);
}

TEST(Preproc, Float16Runs) {
    auto src = constantImage(4, 4, 3, {10, 20, 30});
    ASSERT_TRUE(src.ok());
    auto dst = Tensor::allocate(DType::kFloat16, Shape{1, 3, 4, 4}, Device::kCuda);
    ASSERT_TRUE(dst.ok());
    Stream stream;
    EXPECT_TRUE(letterboxToTensor(src.value().view(), dst.value().view(), PreprocSpec{}, stream).ok());
    EXPECT_TRUE(stream.synchronize().ok());
}
