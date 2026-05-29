#include "tensorrt_cpp_api/allocator.h"
#include "tensorrt_cpp_api/cuda.h"

#include <gtest/gtest.h>

#include <utility>

using namespace trtcpp;

TEST(Cuda, DeviceCountAndQuery) {
    auto count = deviceCount();
    ASSERT_TRUE(count.ok()) << count.status().message();
    ASSERT_GE(count.value(), 1);

    auto info = queryDevice(0);
    ASSERT_TRUE(info.ok()) << info.status().message();
    EXPECT_FALSE(info.value().name.empty());
    EXPECT_EQ(info.value().uuid.size(), 32u);
    EXPECT_GE(info.value().computeMajor, 1);
    EXPECT_GT(info.value().totalMemoryBytes, 0u);
}

TEST(Cuda, DefaultStreamSynchronizes) {
    Stream stream;
    EXPECT_TRUE(stream.synchronize().ok());
}

TEST(Cuda, CreateStreamOwns) {
    auto stream = Stream::create();
    ASSERT_TRUE(stream.ok()) << stream.status().message();
    EXPECT_TRUE(stream.value().owns());
    EXPECT_NE(stream.value().handle(), nullptr);
    EXPECT_TRUE(stream.value().synchronize().ok());
}

TEST(Cuda, WrapStreamIsNonOwning) {
    auto owned = Stream::create();
    ASSERT_TRUE(owned.ok());
    Stream wrapped = Stream::wrap(owned.value().handle());
    EXPECT_FALSE(wrapped.owns());
    EXPECT_EQ(wrapped.handle(), owned.value().handle());
    // wrapped's destructor must not destroy the stream still owned by `owned`.
}

TEST(Cuda, StreamMoveTransfersOwnership) {
    auto a = Stream::create();
    ASSERT_TRUE(a.ok());
    cudaStream_t handle = a.value().handle();
    Stream b = std::move(a.value());
    EXPECT_TRUE(b.owns());
    EXPECT_EQ(b.handle(), handle);
    EXPECT_FALSE(a.value().owns()); // moved-from no longer owns
}

TEST(Cuda, StreamOrderedAllocatorRoundtrip) {
    auto allocator = defaultDeviceAllocator(0);
    ASSERT_NE(allocator, nullptr);
    Stream stream;
    void *ptr = allocator->allocate(1024, 256, stream);
    EXPECT_NE(ptr, nullptr);
    allocator->deallocate(ptr, stream);
    EXPECT_TRUE(stream.synchronize().ok());
}
