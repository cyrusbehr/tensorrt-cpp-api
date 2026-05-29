#include "tensorrt_cpp_api/device_tensor.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <utility>

using namespace trtcpp;

TEST(Tensor, AllocateDeviceAndHost) {
    auto device = Tensor::allocate(DType::kFloat32, Shape{1, 3, 4, 4}, Device::kCuda);
    ASSERT_TRUE(device.ok()) << device.status().message();
    EXPECT_FALSE(device.value().empty());
    EXPECT_EQ(device.value().nbytes(), 48u * 4);
    EXPECT_EQ(device.value().device(), Device::kCuda);

    auto host = Tensor::allocate(DType::kFloat32, Shape{4}, Device::kHost);
    ASSERT_TRUE(host.ok()) << host.status().message();
    EXPECT_EQ(host.value().device(), Device::kHost);
}

TEST(Tensor, RejectsDynamicShape) {
    auto t = Tensor::allocate(DType::kFloat32, Shape{-1, 3}, Device::kCuda);
    EXPECT_FALSE(t.ok());
    EXPECT_EQ(t.status().code(), StatusCode::kInvalidArgument);
}

TEST(Tensor, HostToDeviceToHostRoundtrip) {
    std::array<float, 6> src{1, 2, 3, 4, 5, 6};
    TensorView srcView{src.data(), DType::kFloat32, Shape{2, 3}, Device::kHost};
    Stream stream;

    auto device = Tensor::allocate(DType::kFloat32, Shape{2, 3}, Device::kCuda);
    ASSERT_TRUE(device.ok());
    ASSERT_TRUE(device.value().copyFrom(srcView, stream).ok());

    auto host = device.value().toHost(stream);
    ASSERT_TRUE(host.ok());
    auto span = host.value().as<float>();
    ASSERT_TRUE(span.ok());
    ASSERT_EQ(span.value().size(), 6u);
    for (std::size_t i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(span.value()[i], src[i]);
    }
}

TEST(Tensor, MoveTransfersBuffer) {
    auto a = Tensor::allocate(DType::kFloat32, Shape{8}, Device::kCuda);
    ASSERT_TRUE(a.ok());
    void *ptr = a.value().data();
    Tensor b = std::move(a.value());
    EXPECT_EQ(b.data(), ptr);
    EXPECT_TRUE(a.value().empty()); // moved-from is empty
}

TEST(Tensor, CopyFromRejectsSizeMismatch) {
    std::array<float, 4> src{};
    TensorView srcView{src.data(), DType::kFloat32, Shape{4}, Device::kHost};
    Stream stream;
    auto device = Tensor::allocate(DType::kFloat32, Shape{2}, Device::kCuda);
    ASSERT_TRUE(device.ok());
    Status status = device.value().copyFrom(srcView, stream);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kShapeMismatch);
}

TEST(Tensor, CopyFromRejectsDtypeMismatch) {
    std::array<std::int32_t, 4> src{};
    TensorView srcView{src.data(), DType::kInt32, Shape{4}, Device::kHost};
    Stream stream;
    auto device = Tensor::allocate(DType::kFloat32, Shape{4}, Device::kCuda);
    ASSERT_TRUE(device.ok());
    Status status = device.value().copyFrom(srcView, stream);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kDtypeMismatch);
}

TEST(Tensor, CopyFromRejectsCrossDevice) {
    Stream stream;
    auto device = Tensor::allocate(DType::kFloat32, Shape{3}, Device::kCuda); // deviceId 0
    ASSERT_TRUE(device.ok());
    // A (synthetic) view claiming to live on a different CUDA device; rejected before any copy.
    TensorView otherDevice{device.value().data(), DType::kFloat32, Shape{3}, Device::kCuda, 1};
    Status status = device.value().copyFrom(otherDevice, stream);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kUnsupported);
}

TEST(Tensor, MoveAssignmentTransfersBuffer) {
    auto a = Tensor::allocate(DType::kFloat32, Shape{8}, Device::kCuda);
    auto b = Tensor::allocate(DType::kFloat32, Shape{2}, Device::kCuda);
    ASSERT_TRUE(a.ok());
    ASSERT_TRUE(b.ok());
    void *ptr = a.value().data();
    b.value() = std::move(a.value()); // must free b's old buffer and take a's
    EXPECT_EQ(b.value().data(), ptr);
    EXPECT_TRUE(a.value().empty());
    EXPECT_EQ(b.value().shape(), (Shape{8}));
}

TEST(Tensor, MoveFromEmptyIsSafe) {
    Tensor empty; // default-constructed, no buffer
    EXPECT_TRUE(empty.empty());
    Tensor moved = std::move(empty);
    EXPECT_TRUE(moved.empty());
    // both destruct here without freeing a null buffer
}

TEST(Tensor, ToDeviceToDevice) {
    std::array<float, 3> src{7, 8, 9};
    TensorView srcView{src.data(), DType::kFloat32, Shape{3}, Device::kHost};
    Stream stream;
    auto device = Tensor::allocate(DType::kFloat32, Shape{3}, Device::kCuda);
    ASSERT_TRUE(device.ok());
    ASSERT_TRUE(device.value().copyFrom(srcView, stream).ok());

    auto device2 = device.value().to(Device::kCuda, 0, stream); // device-to-device
    ASSERT_TRUE(device2.ok());
    auto host = device2.value().toHost(stream);
    ASSERT_TRUE(host.ok());
    auto span = host.value().as<float>();
    ASSERT_TRUE(span.ok());
    EXPECT_FLOAT_EQ(span.value()[2], 9.f);
}
