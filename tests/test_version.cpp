// Also exercises that the umbrella header is self-contained and compiles standalone.
#include "tensorrt_cpp_api/all.h"

#include <gtest/gtest.h>

using namespace trtcpp;

TEST(Version, LibraryVersionIsSeven) {
    EXPECT_EQ(libraryVersion().major, 7);
    EXPECT_EQ(libraryVersion().minor, 0);
}

TEST(Version, TensorRTBuildVersionRecorded) {
    EXPECT_GE(tensorrtBuildMajor(), 10); // built against TensorRT >= 10
    EXPECT_LT(tensorrtBuildMajor(), 12);
}

TEST(Version, StringMentionsBothVersions) {
    const std::string s = versionString();
    EXPECT_NE(s.find("7.0.0"), std::string::npos);
    EXPECT_NE(s.find("TensorRT"), std::string::npos);
}
