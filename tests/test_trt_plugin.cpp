#include "detail/trt_common.h"

#include "tensorrt_cpp_api/logger.h"

#include <gtest/gtest.h>

#include <NvInfer.h>

using namespace trtcpp;
using namespace trtcpp::detail;

TEST(TrtPlugin, EmptyListIsOk) { EXPECT_TRUE(loadPluginLibraries({}).ok()); }

TEST(TrtPlugin, BadPathErrors) {
    Status status = loadPluginLibraries({"/nonexistent/path/libfake_trt_plugin.so"});
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kTensorRtError);
}

TEST(TrtUniquePtr, BuilderCreateSmoke) {
    TrtLoggerBridge bridge(defaultLogger());
    TrtUniquePtr<nvinfer1::IBuilder> builder{nvinfer1::createInferBuilder(bridge.nv())};
    ASSERT_NE(builder, nullptr);
    // RAII: builder is deleted here via the default deleter (TRT 10/11 use `delete`).
}
