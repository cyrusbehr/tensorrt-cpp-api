#include "tensorrt_cpp_api/logger.h"

#include <gtest/gtest.h>

#include <cstdio>
#include <functional>
#include <string>
#include <unistd.h>
#include <vector>

using namespace trtcpp;

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

// Capture everything written to the C stderr stream during fn() (Linux-only test util).
std::string captureStderr(const std::function<void()> &fn) {
    std::fflush(stderr);
    int saved = dup(fileno(stderr));
    FILE *tmp = std::tmpfile();
    dup2(fileno(tmp), fileno(stderr));
    fn();
    std::fflush(stderr);
    dup2(saved, fileno(stderr));
    close(saved);
    std::rewind(tmp);
    std::string out;
    char buf[512];
    size_t n = 0;
    while ((n = std::fread(buf, 1, sizeof(buf), tmp)) > 0) {
        out.append(buf, n);
    }
    std::fclose(tmp);
    return out;
}

} // namespace

TEST(Logger, SeverityToString) {
    EXPECT_EQ(toString(Severity::kVerbose), "VERBOSE");
    EXPECT_EQ(toString(Severity::kInfo), "INFO");
    EXPECT_EQ(toString(Severity::kInternalError), "INTERNAL_ERROR");
}

TEST(Logger, InjectableAndPolymorphic) {
    RecordingLogger recorder;
    ILogger &logger = recorder; // used through the interface, as components will
    logger.log(Severity::kWarning, "hello");
    logger.log(Severity::kError, "world");
    ASSERT_EQ(recorder.records.size(), 2u);
    EXPECT_EQ(recorder.records[0].severity, Severity::kWarning);
    EXPECT_EQ(recorder.records[0].message, "hello");
    EXPECT_EQ(recorder.records[1].severity, Severity::kError);
    EXPECT_EQ(recorder.records[1].message, "world");
}

TEST(Logger, DefaultLoggerIsNonNullSingleton) {
    auto logger = defaultLogger();
    ASSERT_NE(logger, nullptr);
    EXPECT_EQ(logger, defaultLogger());
}

TEST(Logger, StderrThresholdFilters) {
    auto logger = makeStderrLogger(Severity::kWarning);
    std::string out = captureStderr([&] {
        logger->log(Severity::kInfo, "dropped-info");
        logger->log(Severity::kWarning, "kept-warning");
        logger->log(Severity::kError, "kept-error");
    });
    EXPECT_EQ(out.find("dropped-info"), std::string::npos);
    EXPECT_NE(out.find("kept-warning"), std::string::npos);
    EXPECT_NE(out.find("kept-error"), std::string::npos);
    EXPECT_NE(out.find("[WARNING]"), std::string::npos);
}

TEST(Logger, StderrTopThresholdKeepsOnlyInternalError) {
    auto logger = makeStderrLogger(Severity::kInternalError);
    std::string out = captureStderr([&] {
        logger->log(Severity::kError, "dropped-error");
        logger->log(Severity::kInternalError, "kept-internal");
    });
    EXPECT_EQ(out.find("dropped-error"), std::string::npos);
    EXPECT_NE(out.find("kept-internal"), std::string::npos);
}
