#include "tensorrt_cpp_api/tensor.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>

using namespace trtcpp;

TEST(TensorView, DefaultIsEmpty) {
    TensorView v;
    EXPECT_EQ(v.data(), nullptr);
    EXPECT_EQ(v.dtype(), DType::kFloat32);
    EXPECT_EQ(v.shape().rank(), 0);
}

TEST(TensorView, PositionalCtorAndAccessors) {
    int dummy = 0;
    TensorView v{&dummy, DType::kFloat32, Shape{1, 3, 4, 4}, Device::kCuda, 0, Layout::kNCHW};
    EXPECT_EQ(v.data(), &dummy);
    EXPECT_EQ(v.dtype(), DType::kFloat32);
    EXPECT_EQ(v.shape(), (Shape{1, 3, 4, 4}));
    EXPECT_TRUE(v.isCuda());
    EXPECT_EQ(v.layout(), Layout::kNCHW);
    EXPECT_EQ(v.nbytes(), 1u * 3 * 4 * 4 * 4); // 48 floats * 4 bytes
}

TEST(TensorView, DescCtor) {
    float buf[6] = {};
    TensorView v{TensorView::Desc{.data = buf, .dtype = DType::kFloat32, .shape = Shape{2, 3}, .device = Device::kHost}};
    EXPECT_EQ(v.device(), Device::kHost);
    EXPECT_FALSE(v.isCuda());
    EXPECT_EQ(v.nbytes(), 24u);
}

TEST(TensorView, NbytesPacksInt4) {
    TensorView v{nullptr, DType::kInt4, Shape{1, 10}, Device::kCuda};
    EXPECT_EQ(v.nbytes(), 5u); // 10 int4 -> 5 bytes
}

TEST(TensorView, AsHostTypedSpan) {
    std::array<float, 4> buf{1.f, 2.f, 3.f, 4.f};
    TensorView v{buf.data(), DType::kFloat32, Shape{4}, Device::kHost};
    auto span = v.as<float>();
    ASSERT_TRUE(span.ok());
    ASSERT_EQ(span.value().size(), 4u);
    EXPECT_FLOAT_EQ(span.value()[2], 3.f);
}

TEST(TensorView, AsRejectsDeviceTensor) {
    int dummy = 0;
    TensorView v{&dummy, DType::kFloat32, Shape{1}, Device::kCuda};
    auto span = v.as<float>();
    EXPECT_FALSE(span.ok());
    EXPECT_EQ(span.status().code(), StatusCode::kInvalidArgument);
}

TEST(TensorView, AsRejectsDtypeMismatch) {
    std::array<float, 2> buf{1.f, 2.f};
    TensorView v{buf.data(), DType::kFloat32, Shape{2}, Device::kHost};
    auto span = v.as<std::int32_t>();
    EXPECT_FALSE(span.ok());
    EXPECT_EQ(span.status().code(), StatusCode::kDtypeMismatch);
}

TEST(TensorView, AsBoolHostSpan) {
    std::array<bool, 3> buf{true, false, true};
    TensorView v{buf.data(), DType::kBool, Shape{3}, Device::kHost};
    auto span = v.as<bool>();
    ASSERT_TRUE(span.ok());
    ASSERT_EQ(span.value().size(), 3u);
    EXPECT_TRUE(span.value()[0]);
    EXPECT_FALSE(span.value()[1]);
}

TEST(TensorView, DynamicShapeIsUnresolved) {
    std::array<float, 4> buf{};
    TensorView v{buf.data(), DType::kFloat32, Shape{-1, 4}, Device::kHost};
    EXPECT_EQ(v.nbytes(), 0u); // unresolved dynamic shape -> 0 bytes
    auto span = v.as<float>();
    ASSERT_TRUE(span.ok()); // ok, but a length-0 span until the shape is resolved
    EXPECT_EQ(span.value().size(), 0u);
}
