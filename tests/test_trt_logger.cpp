#include "detail/trt_common.h"

#include <gtest/gtest.h>

#include <NvInfer.h>

#include <memory>
#include <string>
#include <vector>

using namespace trtcpp;
using namespace trtcpp::detail;

namespace {
class RecordingLogger final : public ILogger {
public:
    struct Record {
        Severity severity;
        std::string message;
    };
    void log(Severity severity, std::string_view message) noexcept override { records.push_back({severity, std::string(message)}); }
    std::vector<Record> records;
};
} // namespace

TEST(TrtLogger, SeverityMapsByNameNotInteger) {
    EXPECT_EQ(fromNvSeverity(nvinfer1::ILogger::Severity::kINTERNAL_ERROR), Severity::kInternalError);
    EXPECT_EQ(fromNvSeverity(nvinfer1::ILogger::Severity::kERROR), Severity::kError);
    EXPECT_EQ(fromNvSeverity(nvinfer1::ILogger::Severity::kWARNING), Severity::kWarning);
    EXPECT_EQ(fromNvSeverity(nvinfer1::ILogger::Severity::kINFO), Severity::kInfo);
    EXPECT_EQ(fromNvSeverity(nvinfer1::ILogger::Severity::kVERBOSE), Severity::kVerbose);

    // Direct forward assertions (a round-trip alone could mask a two-to-one toNv bug).
    EXPECT_EQ(toNvSeverity(Severity::kInternalError), nvinfer1::ILogger::Severity::kINTERNAL_ERROR);
    EXPECT_EQ(toNvSeverity(Severity::kError), nvinfer1::ILogger::Severity::kERROR);
    EXPECT_EQ(toNvSeverity(Severity::kWarning), nvinfer1::ILogger::Severity::kWARNING);
    EXPECT_EQ(toNvSeverity(Severity::kInfo), nvinfer1::ILogger::Severity::kINFO);
    EXPECT_EQ(toNvSeverity(Severity::kVerbose), nvinfer1::ILogger::Severity::kVERBOSE);

    for (Severity s : {Severity::kVerbose, Severity::kInfo, Severity::kWarning, Severity::kError, Severity::kInternalError}) {
        EXPECT_EQ(fromNvSeverity(toNvSeverity(s)), s);
    }

    // The reason the mapping must be by-name: the integer values genuinely differ, so a
    // plain static_cast would invert every level.
    EXPECT_NE(static_cast<int>(nvinfer1::ILogger::Severity::kINTERNAL_ERROR), static_cast<int>(Severity::kInternalError));
}

TEST(TrtLogger, BridgeForwardsToTrtcppLogger) {
    auto recorder = std::make_shared<RecordingLogger>();
    TrtLoggerBridge bridge(recorder);
    bridge.log(nvinfer1::ILogger::Severity::kWARNING, "trt-warn");
    bridge.log(nvinfer1::ILogger::Severity::kVERBOSE, "trt-verbose");
    ASSERT_EQ(recorder->records.size(), 2u);
    EXPECT_EQ(recorder->records[0].severity, Severity::kWarning);
    EXPECT_EQ(recorder->records[0].message, "trt-warn");
    EXPECT_EQ(recorder->records[1].severity, Severity::kVerbose);
}

TEST(TrtLogger, BridgeNullLoggerFallsBackToDefault) {
    TrtLoggerBridge bridge(nullptr); // must not crash
    bridge.log(nvinfer1::ILogger::Severity::kINFO, "ok");
    bridge.log(nvinfer1::ILogger::Severity::kERROR, nullptr); // null message tolerated
    SUCCEED();
}
