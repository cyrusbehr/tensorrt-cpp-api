#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include "NvInfer.h"

// Precision used for GPU inference
enum class Precision {
    FP32,
    FP16
};

// Options for the network
struct Options {
    bool doesSupportDynamicBatchSize = true;
    // Precision to use for GPU inference. 16 bit is faster but may reduce accuracy.
    Precision precision = Precision::FP16;
    // The batch size which should be optimized for.
    int32_t optBatchSize = 1;
    // Maximum allowable batch size
    int32_t maxBatchSize = 16;
    // Max allowable GPU memory to be used for model conversion, in bytes.
    // Applications should allow the engine builder as much workspace as they can afford;
    // at runtime, the SDK allocates no more than this and typically less.
    size_t maxWorkspaceSize = 4000000000;
    // GPU device index
    int deviceIndex = 0;
};

// Class to extend TensorRT logger
class Logger : public nvinfer1::ILogger {
    void log (Severity severity, const char* msg) noexcept override;
};

class Engine {
public:
    Engine(const Options& options);
    ~Engine();
    // Build the network
    bool build(std::string onnxModelPath);
    // Load and prepare the network for inference
    bool loadNetwork();
    // Run inference.
    // Input format [input][batch][image]
    // Output format [batch][output][feature_vector]
    bool runInference(const std::vector<std::vector<cv::cuda::GpuMat>>& inputs, std::vector<std::vector<std::vector<float>>>& featureVectors, const std::array<float, 3>& subVals = {0.f, 0.f, 0.f},
                      const std::array<float, 3>& divVals = {1.f, 1.f, 1.f}, bool normalize = true);

    // Utility method
    static cv::cuda::GpuMat resizeKeepAspectRatioPadRightBottom(const cv::cuda::GpuMat& input, size_t height, size_t width, const cv::Scalar& bgcolor = cv::Scalar(0, 0, 0));

    const std::vector<nvinfer1::Dims3>& getInputDims() const { return m_inputDims; };
    const std::vector<nvinfer1::Dims>& getOutputDims() const { return m_outputDims ;};
private:
    // Converts the engine options into a string
    std::string serializeEngineOptions(const Options& options, const std::string& onnxModelPath);

    void getDeviceNames(std::vector<std::string>& deviceNames);

    bool doesFileExist(const std::string& filepath);

    // Holds pointers to the input and output GPU buffers
    std::vector<void*> m_buffers;
    std::vector<uint32_t> m_outputLengthsFloat{};
    std::vector<nvinfer1::Dims3> m_inputDims;
    std::vector<nvinfer1::Dims> m_outputDims;

    // Must keep IRuntime around for inference, see: https://forums.developer.nvidia.com/t/is-it-safe-to-deallocate-nvinfer1-iruntime-after-creating-an-nvinfer1-icudaengine-but-before-running-inference-with-said-icudaengine/255381/2?u=cyruspk4w6
    std::unique_ptr<nvinfer1::IRuntime> m_runtime = nullptr;

    std::unique_ptr<nvinfer1::ICudaEngine> m_engine = nullptr;
    std::unique_ptr<nvinfer1::IExecutionContext> m_context = nullptr;
    Options m_options;
    Logger m_logger;
    std::string m_engineName;

    inline void checkCudaErrorCode(cudaError_t code);
};
