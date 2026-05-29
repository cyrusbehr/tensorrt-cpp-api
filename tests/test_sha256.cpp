#include "detail/sha256.h"

#include <gtest/gtest.h>

#include <string>

using trtcpp::detail::sha256Hex;

TEST(Sha256, KnownVectors) {
    EXPECT_EQ(sha256Hex("", 0), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");

    const std::string abc = "abc";
    EXPECT_EQ(sha256Hex(abc.data(), abc.size()), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");

    // NIST two-block (>56 byte) vector, exercising multi-block padding.
    const std::string two = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    EXPECT_EQ(sha256Hex(two.data(), two.size()), "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
}

TEST(Sha256, DiffersByContent) {
    const std::string a = "model-a";
    const std::string b = "model-b";
    EXPECT_NE(sha256Hex(a.data(), a.size()), sha256Hex(b.data(), b.size()));
}
