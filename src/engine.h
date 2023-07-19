#pragma once

#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include "NvInfer.h"

// Utility Timer
template <typename Clock = std::chrono::high_resolution_clock>
class Stopwatch
{
    typename Clock::time_point start_point;
public:
    Stopwatch() :start_point(Clock::now()){}

    // Returns elapsed time
    template <typename Rep = typename Clock::duration::rep, typename Units = typename Clock::duration>
    Rep elapsedTime() const {
        std::atomic_thread_fence(std::memory_order_relaxed);
        auto counted_time = std::chrono::duration_cast<Units>(Clock::now() - start_point).count();
        std::atomic_thread_fence(std::memory_order_relaxed);
        return static_cast<Rep>(counted_time);
    }
};

using preciseStopwatch = Stopwatch<>;

// Precision used for GPU inference
enum class Precision {
    FP32,
    FP16
};

// Options for the network
struct Options {
    // Precision to use for GPU inference. 16 bit is faster but may reduce accuracy.
    Precision precision = Precision::FP16;
    // The batch size which should be optimized for.
    int32_t optBatchSize = 1;
    // Maximum allowable batch size
    int32_t maxBatchSize = 16;
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
    // Input format [input][batch][cv::cuda::GpuMat]
    // Output format [batch][output][feature_vector]
    bool runInference(const std::vector<std::vector<cv::cuda::GpuMat>>& inputs, std::vector<std::vector<std::vector<float>>>& featureVectors, const std::array<float, 3>& subVals = {0.f, 0.f, 0.f},
                      const std::array<float, 3>& divVals = {1.f, 1.f, 1.f}, bool normalize = true);

    // Utility method for resizing an image while maintaining the aspect ratio by adding padding to smaller dimension after scaling
    // While letterbox padding normally adds padding to top & bottom, or left & right sides, this implementation only adds padding to the right or bottom side
    // This is done so that it's easier to convert detected coordinates (ex. YOLO model) back to the original reference frame.
    static cv::cuda::GpuMat resizeKeepAspectRatioPadRightBottom(const cv::cuda::GpuMat& input, size_t height, size_t width, const cv::Scalar& bgcolor = cv::Scalar(0, 0, 0));

    const std::vector<nvinfer1::Dims3>& getInputDims() const { return m_inputDims; };
    const std::vector<nvinfer1::Dims>& getOutputDims() const { return m_outputDims ;};

    static bool doesFileExist(const std::string& filepath);

    // Utility method for transforming triple nested output array into 2D array
    // Should be used when the output batch size is 1, but there are multiple output feature vectors
    static void transformOutput(std::vector<std::vector<std::vector<float>>>& input, std::vector<std::vector<float>>& output);

    // Utility method for transforming triple nested output array into single array
    // Should be used when the output batch size is 1, and there is only a single output feature vector
    static void transformOutput(std::vector<std::vector<std::vector<float>>>& input, std::vector<float>& output);
private:
    // Converts the engine options into a string
    std::string serializeEngineOptions(const Options& options, const std::string& onnxModelPath);

    void getDeviceNames(std::vector<std::string>& deviceNames);

    // Holds pointers to the input and output GPU buffers
    std::vector<void*> m_buffers;
    std::vector<uint32_t> m_outputLengthsFloat{};
    std::vector<nvinfer1::Dims3> m_inputDims;
    std::vector<nvinfer1::Dims> m_outputDims;
    std::vector<std::string> m_IOTensorNames;

    // Must keep IRuntime around for inference, see: https://forums.developer.nvidia.com/t/is-it-safe-to-deallocate-nvinfer1-iruntime-after-creating-an-nvinfer1-icudaengine-but-before-running-inference-with-said-icudaengine/255381/2?u=cyruspk4w6
    std::unique_ptr<nvinfer1::IRuntime> m_runtime = nullptr;

    std::unique_ptr<nvinfer1::ICudaEngine> m_engine = nullptr;
    std::unique_ptr<nvinfer1::IExecutionContext> m_context = nullptr;
    Options m_options;
    Logger m_logger;
    std::string m_engineName;

    static inline void checkCudaErrorCode(cudaError_t code);
};
