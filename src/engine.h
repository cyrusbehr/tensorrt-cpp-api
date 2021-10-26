#pragma once

#include <opencv2/opencv.hpp>

#include "NvInfer.h"
#include "buffers.h"

// Options for the network
struct Options {
    // Use 16 bit floating point type for inference
    bool FP16 = false;
    // Batch sizes to optimize for.
    std::vector<int32_t> optBatchSizes;
    // Maximum allowable batch size
    int32_t maxBatchSize = 16;
    // Max allowable GPU memory to be used for model conversion, in bytes
    size_t maxWorkspaceSize = 4000000000;
};

// Class to extend TensorRT logger
class Logger : public nvinfer1::ILogger {
    void log (Severity severity, const char* msg) noexcept override;
};

class Engine {
public:
    Engine(const Options& options);
    // Build the network
    bool build(std::string onnxModelPath);
    // Load and prepare the network for inference
    bool loadNetwork();
    // Run inference.
    bool runInference(const std::vector<cv::Mat>& inputFaceChips, std::vector<std::vector<float>>& featureVectors);
private:
    // Converts the engine options into a string
    std::string serializeEngineOptions(const Options& options);

    bool doesFileExist(const std::string& filepath);

    std::unique_ptr<nvinfer1::ICudaEngine> m_engine = nullptr;
    std::unique_ptr<nvinfer1::IExecutionContext> m_context = nullptr;
    const Options& m_options;
    Logger m_logger;
    samplesCommon::ManagedBuffer m_inputBuff;
    samplesCommon::ManagedBuffer m_outputBuff;
    size_t m_prevBatchSize = 0;
    std::string m_engineName;

    int32_t m_inputW;
    int32_t m_inputH;
    int32_t m_inputC;
    int32_t m_outputL;
};
