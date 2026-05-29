#include "tensorrt_cpp_api/status.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>

using namespace trtcpp;

TEST(Status, DefaultIsOk) {
    Status s;
    EXPECT_TRUE(s.ok());
    EXPECT_TRUE(static_cast<bool>(s));
    EXPECT_EQ(s.code(), StatusCode::kOk);
}

TEST(Status, ErrorCarriesMessage) {
    Status s{StatusCode::kInvalidArgument, "bad arg"};
    EXPECT_FALSE(s.ok());
    EXPECT_EQ(s.code(), StatusCode::kInvalidArgument);
    EXPECT_EQ(s.message(), "bad arg");
    EXPECT_EQ(toString(s.code()), "invalid_argument");
}

TEST(Result, HoldsValue) {
    Result<int> r{42};
    ASSERT_TRUE(r.ok());
    EXPECT_EQ(r.value(), 42);
    EXPECT_EQ(*r, 42);
    EXPECT_EQ(r.value_or(-1), 42);
}

TEST(Result, HoldsError) {
    Result<int> r{Status{StatusCode::kNotFound, "missing"}};
    EXPECT_FALSE(r.ok());
    EXPECT_EQ(r.status().code(), StatusCode::kNotFound);
    EXPECT_EQ(r.value_or(-1), -1);
}

TEST(Result, ArrowOperator) {
    Result<std::string> r{std::string("hello")};
    ASSERT_TRUE(r.ok());
    EXPECT_EQ(r->size(), 5u);
}

TEST(Result, AndThenChains) {
    Result<int> r{10};
    auto doubled = r.and_then([](int &v) { return Result<int>{v * 2}; });
    ASSERT_TRUE(doubled.ok());
    EXPECT_EQ(doubled.value(), 20);

    Result<int> err{Status{StatusCode::kInternal, "boom"}};
    auto chained = err.and_then([](int &v) { return Result<int>{v * 2}; });
    EXPECT_FALSE(chained.ok());
    EXPECT_EQ(chained.status().code(), StatusCode::kInternal);
}

TEST(Result, TransformMaps) {
    Result<int> r{7};
    auto s = r.transform([](int &v) { return std::to_string(v); });
    ASSERT_TRUE(s.ok());
    EXPECT_EQ(s.value(), "7");

    Result<int> err{Status{StatusCode::kIoError, "io"}};
    auto s2 = err.transform([](int &v) { return std::to_string(v); });
    EXPECT_FALSE(s2.ok());
    EXPECT_EQ(s2.status().code(), StatusCode::kIoError);
}

namespace {
Status consumePositive(Result<int> r) {
    TRTCPP_TRY(int v, std::move(r));
    if (v < 0) {
        return Status{StatusCode::kInvalidArgument, "negative"};
    }
    return Status{};
}
} // namespace

TEST(Result, TryMacroPropagatesError) {
    EXPECT_TRUE(consumePositive(Result<int>{5}).ok());

    Status propagated = consumePositive(Result<int>{Status{StatusCode::kCudaError, "cuda"}});
    EXPECT_FALSE(propagated.ok());
    EXPECT_EQ(propagated.code(), StatusCode::kCudaError);
}

TEST(Result, MoveOnlyValue) {
    Result<std::unique_ptr<int>> r{std::make_unique<int>(7)};
    ASSERT_TRUE(r.ok());
    EXPECT_EQ(*r.value(), 7);
    auto ptr = std::move(r).value(); // rvalue value() moves the unique_ptr out
    ASSERT_TRUE(ptr);
    EXPECT_EQ(*ptr, 7);
}

TEST(Result, TransformOnMoveOnly) {
    Result<std::unique_ptr<int>> r{std::make_unique<int>(5)};
    auto deref = r.transform([](std::unique_ptr<int> &p) { return *p; });
    ASSERT_TRUE(deref.ok());
    EXPECT_EQ(deref.value(), 5);
}
