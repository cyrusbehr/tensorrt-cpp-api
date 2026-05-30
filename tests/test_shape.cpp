#include "tensorrt_cpp_api/shape.h"

#include <gtest/gtest.h>

#include <stdexcept>

using namespace trtcpp;

TEST(Shape, DefaultIsScalar) {
    Shape s;
    EXPECT_EQ(s.rank(), 0);
    EXPECT_FALSE(s.isDynamic());
    EXPECT_EQ(s.numel(), 1); // rank-0 scalar
    EXPECT_EQ(s.toString(), "[]");
}

TEST(Shape, InitializerListAndAccess) {
    Shape s{1, 3, 640, 640};
    EXPECT_EQ(s.rank(), 4);
    EXPECT_EQ(s[0], 1);
    EXPECT_EQ(s[3], 640);
    EXPECT_EQ(s.at(1), 3);
    EXPECT_EQ(s.numel(), 1LL * 3 * 640 * 640);
    EXPECT_EQ(s.toString(), "[1,3,640,640]");
}

TEST(Shape, AtThrowsOutOfRange) {
    Shape s{2, 2};
    EXPECT_THROW((void)s.at(-1), std::out_of_range);
    EXPECT_THROW((void)s.at(2), std::out_of_range);
    EXPECT_NO_THROW((void)s.at(0));
}

TEST(Shape, DynamicDims) {
    Shape s{-1, 3, -1, -1};
    EXPECT_TRUE(s.isDynamic());
    EXPECT_EQ(s.numel(), 0); // dynamic -> 0
    EXPECT_EQ(s.toString(), "[-1,3,-1,-1]");
}

TEST(Shape, Equality) {
    EXPECT_EQ((Shape{1, 3, 224, 224}), (Shape{1, 3, 224, 224}));
    EXPECT_NE((Shape{1, 3, 224, 224}), (Shape{1, 3, 225, 224}));
    EXPECT_NE((Shape{1, 3, 224}), (Shape{1, 3, 224, 224})); // different rank
}

TEST(Shape, SpanConstruction) {
    const std::int64_t dims[] = {8, 16};
    Shape s{std::span<const std::int64_t>(dims, 2)};
    EXPECT_EQ(s.rank(), 2);
    EXPECT_EQ(s.numel(), 128);
    EXPECT_EQ(s.dims().size(), 2u);
}

TEST(Shape, MaxRankEightDims) {
    Shape s{1, 2, 3, 4, 5, 6, 7, 8};
    EXPECT_EQ(s.rank(), Shape::kMaxRank);
    EXPECT_EQ(s.rank(), 8);
    EXPECT_EQ(s[7], 8);
    EXPECT_EQ(s.numel(), 1LL * 2 * 3 * 4 * 5 * 6 * 7 * 8);
}
