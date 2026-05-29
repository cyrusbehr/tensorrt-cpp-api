#include "detail/buffers.h"

#include <gtest/gtest.h>

using namespace trtcpp;
using namespace trtcpp::detail;

TEST(NamedBuffers, AllocatesAndReports) {
    NamedBuffers buffers;
    auto view = buffers.ensure("out", DType::kFloat32, Shape{1, 100});
    ASSERT_TRUE(view.ok()) << view.status().message();
    EXPECT_NE(view.value().data(), nullptr);
    EXPECT_EQ(view.value().dtype(), DType::kFloat32);
    EXPECT_EQ(view.value().shape(), (Shape{1, 100}));
    EXPECT_TRUE(buffers.has("out"));
    EXPECT_EQ(buffers.capacity("out"), 400u);
    EXPECT_EQ(buffers.address("out"), view.value().data());
}

TEST(NamedBuffers, ReusesWhenNotGrowing) {
    NamedBuffers buffers;
    auto first = buffers.ensure("out", DType::kFloat32, Shape{1, 100});
    ASSERT_TRUE(first.ok());
    void *ptr = first.value().data();
    std::size_t cap = buffers.capacity("out");

    // Smaller request reuses the same allocation (no realloc, capacity unchanged).
    auto second = buffers.ensure("out", DType::kFloat32, Shape{1, 50});
    ASSERT_TRUE(second.ok());
    EXPECT_EQ(second.value().data(), ptr);
    EXPECT_EQ(buffers.capacity("out"), cap);
    EXPECT_EQ(second.value().shape(), (Shape{1, 50}));
}

TEST(NamedBuffers, GrowsWhenLarger) {
    NamedBuffers buffers;
    ASSERT_TRUE(buffers.ensure("out", DType::kFloat32, Shape{1, 50}).ok());
    EXPECT_EQ(buffers.capacity("out"), 200u);
    ASSERT_TRUE(buffers.ensure("out", DType::kFloat32, Shape{1, 200}).ok());
    EXPECT_EQ(buffers.capacity("out"), 800u); // grew
}

TEST(NamedBuffers, RejectsDynamicShape) {
    NamedBuffers buffers;
    auto view = buffers.ensure("out", DType::kFloat32, Shape{-1, 100});
    EXPECT_FALSE(view.ok());
    EXPECT_EQ(view.status().code(), StatusCode::kInvalidArgument);
}

TEST(NamedBuffers, RejectsOverflowingShape) {
    NamedBuffers buffers;
    auto view = buffers.ensure("out", DType::kFloat32, Shape{1LL << 40, 1LL << 40});
    EXPECT_FALSE(view.ok());
    EXPECT_EQ(view.status().code(), StatusCode::kInvalidArgument);
}

TEST(NamedBuffers, ZeroSizeShapeAllocatesNothing) {
    NamedBuffers buffers;
    auto view = buffers.ensure("z", DType::kFloat32, Shape{0});
    ASSERT_TRUE(view.ok());
    EXPECT_EQ(view.value().shape().numel(), 0);
    EXPECT_EQ(buffers.capacity("z"), 0u);
    EXPECT_EQ(buffers.address("z"), nullptr);
}

TEST(NamedBuffers, DeviceChangeReallocates) {
    auto count = deviceCount();
    if (!count.ok() || count.value() < 2) {
        GTEST_SKIP() << "requires >= 2 CUDA devices";
    }
    NamedBuffers buffers;
    auto onZero = buffers.ensure("x", DType::kFloat32, Shape{4}, 0);
    ASSERT_TRUE(onZero.ok());
    void *ptr0 = onZero.value().data();
    auto onOne = buffers.ensure("x", DType::kFloat32, Shape{4}, 1);
    ASSERT_TRUE(onOne.ok());
    EXPECT_NE(onOne.value().data(), ptr0); // reallocated on the other device
}

TEST(NamedBuffers, MultipleNamesAndClear) {
    NamedBuffers buffers;
    ASSERT_TRUE(buffers.ensure("a", DType::kFloat32, Shape{4}).ok());
    ASSERT_TRUE(buffers.ensure("b", DType::kInt8, Shape{8}).ok());
    EXPECT_EQ(buffers.names().size(), 2u);
    EXPECT_NE(buffers.address("a"), buffers.address("b"));
    EXPECT_EQ(buffers.address("missing"), nullptr);
    buffers.clear();
    EXPECT_FALSE(buffers.has("a"));
    EXPECT_EQ(buffers.capacity("a"), 0u);
}
