#include "tensorrt_cpp_api/dtype.h"

#include <gtest/gtest.h>

using namespace trtcpp;

TEST(DType, BitsPerElement) {
    EXPECT_EQ(bitsPerElement(DType::kFloat32), 32);
    EXPECT_EQ(bitsPerElement(DType::kInt64), 64);
    EXPECT_EQ(bitsPerElement(DType::kFloat16), 16);
    EXPECT_EQ(bitsPerElement(DType::kBFloat16), 16);
    EXPECT_EQ(bitsPerElement(DType::kInt8), 8);
    EXPECT_EQ(bitsPerElement(DType::kUInt8), 8);
    EXPECT_EQ(bitsPerElement(DType::kBool), 8);
    EXPECT_EQ(bitsPerElement(DType::kFp8), 8);
    EXPECT_EQ(bitsPerElement(DType::kInt4), 4);
}

TEST(DType, ByteSizeRoundsUp) {
    EXPECT_EQ(byteSize(DType::kFloat32), 4u);
    EXPECT_EQ(byteSize(DType::kFloat16), 2u);
    EXPECT_EQ(byteSize(DType::kInt8), 1u);
    EXPECT_EQ(byteSize(DType::kInt4), 1u); // a single sub-byte element rounds up to 1
}

TEST(DType, ByteSizeOfPacksSubByte) {
    EXPECT_EQ(byteSizeOf(DType::kFloat32, 10), 40u);
    EXPECT_EQ(byteSizeOf(DType::kInt8, 10), 10u);
    EXPECT_EQ(byteSizeOf(DType::kInt4, 10), 5u); // two int4 per byte
    EXPECT_EQ(byteSizeOf(DType::kInt4, 9), 5u);  // 9 elements -> ceil(36/8) = 5 bytes
    EXPECT_EQ(byteSizeOf(DType::kInt4, 0), 0u);
}

TEST(DType, ToString) {
    EXPECT_EQ(toString(DType::kFloat32), "float32");
    EXPECT_EQ(toString(DType::kInt4), "int4");
    EXPECT_EQ(toString(DType::kBool), "bool");
}

TEST(DType, DTypeOfMapping) {
    EXPECT_EQ(DTypeOf<float>::value, DType::kFloat32);
    EXPECT_EQ(DTypeOf<std::int32_t>::value, DType::kInt32);
    EXPECT_EQ(DTypeOf<std::int64_t>::value, DType::kInt64);
    EXPECT_EQ(DTypeOf<std::int8_t>::value, DType::kInt8);
    EXPECT_EQ(DTypeOf<std::uint8_t>::value, DType::kUInt8);
    EXPECT_EQ(DTypeOf<bool>::value, DType::kBool);
}
