#pragma once

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include <chrono>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <fstream>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/opencv.hpp>

// Utility methods
namespace Util {
inline bool doesFileExist(const std::string &filepath) {
    std::ifstream f(filepath.c_str());
    return f.good();
}

inline void checkCudaErrorCode(cudaError_t code) {
    if (code != 0) {
        std::string errMsg = "CUDA operation failed with code: " + std::to_string(code) + "(" + cudaGetErrorName(code) +
                             "), with message: " + cudaGetErrorString(code);
        std::cout << errMsg << std::endl;
        throw std::runtime_error(errMsg);
    }
}

std::vector<std::string> getFilesInDirectory(const std::string &dirPath);
} // namespace Util
// Utility Timer
template <typename Clock = std::chrono::high_resolution_clock> class Stopwatch {
    typename Clock::time_point start_point;

public:
    Stopwatch() : start_point(Clock::now()) {}

    // Returns elapsed time
    template <typename Rep = typename Clock::duration::rep, typename Units = typename Clock::duration> Rep elapsedTime() const {
        std::atomic_thread_fence(std::memory_order_relaxed);
        auto counted_time = std::chrono::duration_cast<Units>(Clock::now() - start_point).count();
        std::atomic_thread_fence(std::memory_order_relaxed);
        return static_cast<Rep>(counted_time);
    }
};

using preciseStopwatch = Stopwatch<>;

// Precision used for GPU inference
enum class Precision {
    // Full precision floating point value
    FP32,
    // Half prevision floating point value
    FP16,
    // Int8 quantization.
    // Has reduced dynamic range, may result in slight loss in accuracy.
    // If INT8 is selected, must provide path to calibration dataset directory.
    INT8,
};

// Options for the network
struct Options {
    // Precision to use for GPU inference.
    Precision precision = Precision::FP16;
    // If INT8 precision is selected, must provide path to calibration dataset
    // directory.
    std::string calibrationDataDirectoryPath;
    // The batch size to be used when computing calibration data for INT8
    // inference. Should be set to as large a batch number as your GPU will
    // support.
    int32_t calibrationBatchSize = 128;
    // The batch size which should be optimized for.
    int32_t optBatchSize = 1;
    // Maximum allowable batch size
    int32_t maxBatchSize = 16;
    // GPU device index
    int deviceIndex = 0;
};

// Class used for int8 calibration
class Int8EntropyCalibrator2 : public nvinfer1::IInt8EntropyCalibrator2 {
public:
    Int8EntropyCalibrator2(int32_t batchSize, int32_t inputW, int32_t inputH, const std::string &calibDataDirPath,
                           const std::string &calibTableName, const std::string &inputBlobName,
                           const std::array<float, 3> &subVals = {0.f, 0.f, 0.f}, const std::array<float, 3> &divVals = {1.f, 1.f, 1.f},
                           bool normalize = true, bool readCache = true);
    virtual ~Int8EntropyCalibrator2();
    // Abstract base class methods which must be implemented
    int32_t getBatchSize() const noexcept override;
    bool getBatch(void *bindings[], char const *names[], int32_t nbBindings) noexcept override;
    void const *readCalibrationCache(std::size_t &length) noexcept override;
    void writeCalibrationCache(void const *ptr, std::size_t length) noexcept override;

private:
    const int32_t m_batchSize;
    const int32_t m_inputW;
    const int32_t m_inputH;
    int32_t m_imgIdx;
    std::vector<std::string> m_imgPaths;
    size_t m_inputCount;
    const std::string m_calibTableName;
    const std::string m_inputBlobName;
    const std::array<float, 3> m_subVals;
    const std::array<float, 3> m_divVals;
    const bool m_normalize;
    const bool m_readCache;
    void *m_deviceInput;
    std::vector<char> m_calibCache;
};

// Class to extend TensorRT logger
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char *msg) noexcept override;
};

template <typename T> class Engine {
public:
    Engine(const Options &options);
    ~Engine();

    // Build the onnx model into a TensorRT engine file, cache the model to disk
    // (to avoid rebuilding in future), and then load the model into memory The
    // default implementation will normalize values between [0.f, 1.f] Setting the
    // normalize flag to false will leave values between [0.f, 255.f] (some
    // converted models may require this). If the model requires values to be
    // normalized between [-1.f, 1.f], use the following params:
    //    subVals = {0.5f, 0.5f, 0.5f};
    //    divVals = {0.5f, 0.5f, 0.5f};
    //    normalize = true;
    bool buildLoadNetwork(std::string onnxModelPath, const std::array<float, 3> &subVals = {0.f, 0.f, 0.f},
                          const std::array<float, 3> &divVals = {1.f, 1.f, 1.f}, bool normalize = true);

    // Load a TensorRT engine file from disk into memory
    // The default implementation will normalize values between [0.f, 1.f]
    // Setting the normalize flag to false will leave values between [0.f, 255.f]
    // (some converted models may require this). If the model requires values to
    // be normalized between [-1.f, 1.f], use the following params:
    //    subVals = {0.5f, 0.5f, 0.5f};
    //    divVals = {0.5f, 0.5f, 0.5f};
    //    normalize = true;
    bool loadNetwork(std::string trtModelPath, const std::array<float, 3> &subVals = {0.f, 0.f, 0.f},
                     const std::array<float, 3> &divVals = {1.f, 1.f, 1.f}, bool normalize = true);

    // Run inference.
    // Input format [input][batch][cv::cuda::GpuMat]
    // Output format [batch][output][feature_vector]
    bool runInference(const std::vector<std::vector<cv::cuda::GpuMat>> &inputs, std::vector<std::vector<std::vector<T>>> &featureVectors);

    // Utility method for resizing an image while maintaining the aspect ratio by
    // adding padding to smaller dimension after scaling While letterbox padding
    // normally adds padding to top & bottom, or left & right sides, this
    // implementation only adds padding to the right or bottom side This is done
    // so that it's easier to convert detected coordinates (ex. YOLO model) back
    // to the original reference frame.
    static cv::cuda::GpuMat resizeKeepAspectRatioPadRightBottom(const cv::cuda::GpuMat &input, size_t height, size_t width,
                                                                const cv::Scalar &bgcolor = cv::Scalar(0, 0, 0));

    [[nodiscard]] const std::vector<nvinfer1::Dims3> &getInputDims() const { return m_inputDims; };
    [[nodiscard]] const std::vector<nvinfer1::Dims> &getOutputDims() const { return m_outputDims; };

    // Utility method for transforming triple nested output array into 2D array
    // Should be used when the output batch size is 1, but there are multiple
    // output feature vectors
    static void transformOutput(std::vector<std::vector<std::vector<T>>> &input, std::vector<std::vector<T>> &output);

    // Utility method for transforming triple nested output array into single
    // array Should be used when the output batch size is 1, and there is only a
    // single output feature vector
    static void transformOutput(std::vector<std::vector<std::vector<T>>> &input, std::vector<T> &output);
    // Convert NHWC to NCHW and apply scaling and mean subtraction
    static cv::cuda::GpuMat blobFromGpuMats(const std::vector<cv::cuda::GpuMat> &batchInput, const std::array<float, 3> &subVals,
                                            const std::array<float, 3> &divVals, bool normalize);

private:
    // Build the network
    bool build(std::string onnxModelPath, const std::array<float, 3> &subVals, const std::array<float, 3> &divVals, bool normalize);

    // Converts the engine options into a string
    std::string serializeEngineOptions(const Options &options, const std::string &onnxModelPath);

    void getDeviceNames(std::vector<std::string> &deviceNames);

    void clearGpuBuffers();

    // Normalization, scaling, and mean subtraction of inputs
    std::array<float, 3> m_subVals{};
    std::array<float, 3> m_divVals{};
    bool m_normalize;

    // Holds pointers to the input and output GPU buffers
    std::vector<void *> m_buffers;
    std::vector<uint32_t> m_outputLengths{};
    std::vector<nvinfer1::Dims3> m_inputDims;
    std::vector<nvinfer1::Dims> m_outputDims;
    std::vector<std::string> m_IOTensorNames;
    int32_t m_inputBatchSize;

    // Must keep IRuntime around for inference, see:
    // https://forums.developer.nvidia.com/t/is-it-safe-to-deallocate-nvinfer1-iruntime-after-creating-an-nvinfer1-icudaengine-but-before-running-inference-with-said-icudaengine/255381/2?u=cyruspk4w6
    std::unique_ptr<nvinfer1::IRuntime> m_runtime = nullptr;
    std::unique_ptr<Int8EntropyCalibrator2> m_calibrator = nullptr;
    std::unique_ptr<nvinfer1::ICudaEngine> m_engine = nullptr;
    std::unique_ptr<nvinfer1::IExecutionContext> m_context = nullptr;
    const Options m_options;
    Logger m_logger;
};

template <typename T> Engine<T>::Engine(const Options &options) : m_options(options) {}

template <typename T> Engine<T>::~Engine() { clearGpuBuffers(); }

template <typename T> void Engine<T>::clearGpuBuffers() {
    if (!m_buffers.empty()) {
        // Free GPU memory of outputs
        const auto numInputs = m_inputDims.size();
        for (int32_t outputBinding = numInputs; outputBinding < m_engine->getNbIOTensors(); ++outputBinding) {
            Util::checkCudaErrorCode(cudaFree(m_buffers[outputBinding]));
        }
        m_buffers.clear();
    }
}

template <typename T>
bool Engine<T>::buildLoadNetwork(std::string onnxModelPath, const std::array<float, 3> &subVals, const std::array<float, 3> &divVals,
                                 bool normalize) {
    // Only regenerate the engine file if it has not already been generated for
    // the specified options, otherwise load cached version from disk
    const auto engineName = serializeEngineOptions(m_options, onnxModelPath);
    std::cout << "Searching for engine file with name: " << engineName << std::endl;

    if (Util::doesFileExist(engineName)) {
        std::cout << "Engine found, not regenerating..." << std::endl;
    } else {
        if (!Util::doesFileExist(onnxModelPath)) {
            throw std::runtime_error("Could not find onnx model at path: " + onnxModelPath);
        }

        // Was not able to find the engine file, generate...
        std::cout << "Engine not found, generating. This could take a while..." << std::endl;

        // Build the onnx model into a TensorRT engine
        auto ret = build(onnxModelPath, subVals, divVals, normalize);
        if (!ret) {
            return false;
        }
    }

    // Load the TensorRT engine file into memory
    return loadNetwork(engineName, subVals, divVals, normalize);
}

template <typename T>
bool Engine<T>::loadNetwork(std::string trtModelPath, const std::array<float, 3> &subVals, const std::array<float, 3> &divVals,
                            bool normalize) {
    m_subVals = subVals;
    m_divVals = divVals;
    m_normalize = normalize;

    // Read the serialized model from disk
    if (!Util::doesFileExist(trtModelPath)) {
        std::cout << "Error, unable to read TensorRT model at path: " + trtModelPath << std::endl;
        return false;
    } else {
        std::cout << "Loading TensorRT engine file at path: " << trtModelPath << std::endl;
    }

    std::ifstream file(trtModelPath, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        throw std::runtime_error("Unable to read engine file");
    }

    // Create a runtime to deserialize the engine file.
    m_runtime = std::unique_ptr<nvinfer1::IRuntime>{nvinfer1::createInferRuntime(m_logger)};
    if (!m_runtime) {
        return false;
    }

    // Set the device index
    auto ret = cudaSetDevice(m_options.deviceIndex);
    if (ret != 0) {
        int numGPUs;
        cudaGetDeviceCount(&numGPUs);
        auto errMsg = "Unable to set GPU device index to: " + std::to_string(m_options.deviceIndex) + ". Note, your device has " +
                      std::to_string(numGPUs) + " CUDA-capable GPU(s).";
        throw std::runtime_error(errMsg);
    }

    // Create an engine, a representation of the optimized model.
    m_engine = std::unique_ptr<nvinfer1::ICudaEngine>(m_runtime->deserializeCudaEngine(buffer.data(), buffer.size()));
    if (!m_engine) {
        return false;
    }

    // The execution context contains all of the state associated with a
    // particular invocation
    m_context = std::unique_ptr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());
    if (!m_context) {
        return false;
    }

    // Storage for holding the input and output buffers
    // This will be passed to TensorRT for inference
    clearGpuBuffers();
    m_buffers.resize(m_engine->getNbIOTensors());

    m_outputLengths.clear();
    m_inputDims.clear();
    m_outputDims.clear();
    m_IOTensorNames.clear();

    // Create a cuda stream
    cudaStream_t stream;
    Util::checkCudaErrorCode(cudaStreamCreate(&stream));

    // Allocate GPU memory for input and output buffers
    m_outputLengths.clear();
    for (int i = 0; i < m_engine->getNbIOTensors(); ++i) {
        const auto tensorName = m_engine->getIOTensorName(i);
        m_IOTensorNames.emplace_back(tensorName);
        const auto tensorType = m_engine->getTensorIOMode(tensorName);
        const auto tensorShape = m_engine->getTensorShape(tensorName);
        const auto tensorDataType = m_engine->getTensorDataType(tensorName);

        if (tensorType == nvinfer1::TensorIOMode::kINPUT) {
            // The implementation currently only supports inputs of type float
            if (m_engine->getTensorDataType(tensorName) != nvinfer1::DataType::kFLOAT) {
                throw std::runtime_error("Error, the implementation currently only supports float inputs");
            }

            // Don't need to allocate memory for inputs as we will be using the OpenCV
            // GpuMat buffer directly.

            // Store the input dims for later use
            m_inputDims.emplace_back(tensorShape.d[1], tensorShape.d[2], tensorShape.d[3]);
            m_inputBatchSize = tensorShape.d[0];
        } else if (tensorType == nvinfer1::TensorIOMode::kOUTPUT) {
            // Ensure the model output data type matches the template argument
            // specified by the user
            if (tensorDataType == nvinfer1::DataType::kFLOAT && !std::is_same<float, T>::value) {
                throw std::runtime_error("Error, the model has expected output of type float. Engine class "
                                         "template parameter must be adjusted.");
            } else if (tensorDataType == nvinfer1::DataType::kHALF && !std::is_same<__half, T>::value) {
                throw std::runtime_error("Error, the model has expected output of type __half. Engine class "
                                         "template parameter must be adjusted.");
            } else if (tensorDataType == nvinfer1::DataType::kINT8 && !std::is_same<int8_t, T>::value) {
                throw std::runtime_error("Error, the model has expected output of type int8_t. Engine class "
                                         "template parameter must be adjusted.");
            } else if (tensorDataType == nvinfer1::DataType::kINT32 && !std::is_same<int32_t, T>::value) {
                throw std::runtime_error("Error, the model has expected output of type int32_t. Engine "
                                         "class template parameter must be adjusted.");
            } else if (tensorDataType == nvinfer1::DataType::kBOOL && !std::is_same<bool, T>::value) {
                throw std::runtime_error("Error, the model has expected output of type bool. Engine class "
                                         "template parameter must be adjusted.");
            } else if (tensorDataType == nvinfer1::DataType::kUINT8 && !std::is_same<uint8_t, T>::value) {
                throw std::runtime_error("Error, the model has expected output of type uint8_t. Engine "
                                         "class template parameter must be adjusted.");
            } else if (tensorDataType == nvinfer1::DataType::kFP8) {
                throw std::runtime_error("Error, model has unsupported output type");
            }

            // The binding is an output
            uint32_t outputLength = 1;
            m_outputDims.push_back(tensorShape);

            for (int j = 1; j < tensorShape.nbDims; ++j) {
                // We ignore j = 0 because that is the batch size, and we will take that
                // into account when sizing the buffer
                outputLength *= tensorShape.d[j];
            }

            m_outputLengths.push_back(outputLength);
            // Now size the output buffer appropriately, taking into account the max
            // possible batch size (although we could actually end up using less
            // memory)
            Util::checkCudaErrorCode(cudaMallocAsync(&m_buffers[i], outputLength * m_options.maxBatchSize * sizeof(T), stream));
        } else {
            throw std::runtime_error("Error, IO Tensor is neither an input or output!");
        }
    }

    // Synchronize and destroy the cuda stream
    Util::checkCudaErrorCode(cudaStreamSynchronize(stream));
    Util::checkCudaErrorCode(cudaStreamDestroy(stream));

    return true;
}

template <typename T>
bool Engine<T>::build(std::string onnxModelPath, const std::array<float, 3> &subVals, const std::array<float, 3> &divVals, bool normalize) {
    // Create our engine builder.
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(m_logger));
    if (!builder) {
        return false;
    }

    // Define an explicit batch size and then create the network (implicit batch
    // size is deprecated). More info here:
    // https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#explicit-implicit-batch
    auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network) {
        return false;
    }

    // Create a parser for reading the onnx file.
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, m_logger));
    if (!parser) {
        return false;
    }

    // We are going to first read the onnx file into memory, then pass that buffer
    // to the parser. Had our onnx model file been encrypted, this approach would
    // allow us to first decrypt the buffer.
    std::ifstream file(onnxModelPath, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        throw std::runtime_error("Unable to read engine file");
    }

    // Parse the buffer we read into memory.
    auto parsed = parser->parse(buffer.data(), buffer.size());
    if (!parsed) {
        return false;
    }

    // Ensure that all the inputs have the same batch size
    const auto numInputs = network->getNbInputs();
    if (numInputs < 1) {
        throw std::runtime_error("Error, model needs at least 1 input!");
    }
    const auto input0Batch = network->getInput(0)->getDimensions().d[0];
    for (int32_t i = 1; i < numInputs; ++i) {
        if (network->getInput(i)->getDimensions().d[0] != input0Batch) {
            throw std::runtime_error("Error, the model has multiple inputs, each "
                                     "with differing batch sizes!");
        }
    }

    // Check to see if the model supports dynamic batch size or not
    bool doesSupportDynamicBatch = false;
    if (input0Batch == -1) {
        doesSupportDynamicBatch = true;
        std::cout << "Model supports dynamic batch size" << std::endl;
    } else {
        std::cout << "Model only supports fixed batch size of " << input0Batch << std::endl;
        // If the model supports a fixed batch size, ensure that the maxBatchSize
        // and optBatchSize were set correctly.
        if (m_options.optBatchSize != input0Batch || m_options.maxBatchSize != input0Batch) {
            throw std::runtime_error("Error, model only supports a fixed batch size of " + std::to_string(input0Batch) +
                                     ". Must set Options.optBatchSize and Options.maxBatchSize to 1");
        }
    }

    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) {
        return false;
    }

    // Register a single optimization profile
    nvinfer1::IOptimizationProfile *optProfile = builder->createOptimizationProfile();
    for (int32_t i = 0; i < numInputs; ++i) {
        // Must specify dimensions for all the inputs the model expects.
        const auto input = network->getInput(i);
        const auto inputName = input->getName();
        const auto inputDims = input->getDimensions();
        int32_t inputC = inputDims.d[1];
        int32_t inputH = inputDims.d[2];
        int32_t inputW = inputDims.d[3];

        // Specify the optimization profile`
        if (doesSupportDynamicBatch) {
            optProfile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, inputC, inputH, inputW));
        } else {
            optProfile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMIN,
                                      nvinfer1::Dims4(m_options.optBatchSize, inputC, inputH, inputW));
        }
        optProfile->setDimensions(inputName, nvinfer1::OptProfileSelector::kOPT,
                                  nvinfer1::Dims4(m_options.optBatchSize, inputC, inputH, inputW));
        optProfile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMAX,
                                  nvinfer1::Dims4(m_options.maxBatchSize, inputC, inputH, inputW));
    }
    config->addOptimizationProfile(optProfile);

    // Set the precision level
    const auto engineName = serializeEngineOptions(m_options, onnxModelPath);
    if (m_options.precision == Precision::FP16) {
        // Ensure the GPU supports FP16 inference
        if (!builder->platformHasFastFp16()) {
            throw std::runtime_error("Error: GPU does not support FP16 precision");
        }
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    } else if (m_options.precision == Precision::INT8) {
        if (numInputs > 1) {
            throw std::runtime_error("Error, this implementation currently only supports INT8 "
                                     "quantization for single input models");
        }

        // Ensure the GPU supports INT8 Quantization
        if (!builder->platformHasFastInt8()) {
            throw std::runtime_error("Error: GPU does not support INT8 precision");
        }

        // Ensure the user has provided path to calibration data directory
        if (m_options.calibrationDataDirectoryPath.empty()) {
            throw std::runtime_error("Error: If INT8 precision is selected, must provide path to "
                                     "calibration data directory to Engine::build method");
        }

        config->setFlag((nvinfer1::BuilderFlag::kINT8));

        const auto input = network->getInput(0);
        const auto inputName = input->getName();
        const auto inputDims = input->getDimensions();
        const auto calibrationFileName = engineName + ".calibration";

        m_calibrator = std::make_unique<Int8EntropyCalibrator2>(m_options.calibrationBatchSize, inputDims.d[3], inputDims.d[2],
                                                                m_options.calibrationDataDirectoryPath, calibrationFileName, inputName,
                                                                subVals, divVals, normalize);
        config->setInt8Calibrator(m_calibrator.get());
    }

    // CUDA stream used for profiling by the builder.
    cudaStream_t profileStream;
    Util::checkCudaErrorCode(cudaStreamCreate(&profileStream));
    config->setProfileStream(profileStream);

    // Build the engine
    // If this call fails, it is suggested to increase the logger verbosity to
    // kVERBOSE and try rebuilding the engine. Doing so will provide you with more
    // information on why exactly it is failing.
    std::unique_ptr<nvinfer1::IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan) {
        return false;
    }

    // Write the engine to disk
    std::ofstream outfile(engineName, std::ofstream::binary);
    outfile.write(reinterpret_cast<const char *>(plan->data()), plan->size());

    std::cout << "Success, saved engine to " << engineName << std::endl;

    Util::checkCudaErrorCode(cudaStreamDestroy(profileStream));
    return true;
}

template <typename T>
bool Engine<T>::runInference(const std::vector<std::vector<cv::cuda::GpuMat>> &inputs,
                             std::vector<std::vector<std::vector<T>>> &featureVectors) {
    // First we do some error checking
    if (inputs.empty() || inputs[0].empty()) {
        std::cout << "===== Error =====" << std::endl;
        std::cout << "Provided input vector is empty!" << std::endl;
        return false;
    }

    const auto numInputs = m_inputDims.size();
    if (inputs.size() != numInputs) {
        std::cout << "===== Error =====" << std::endl;
        std::cout << "Incorrect number of inputs provided!" << std::endl;
        return false;
    }

    // Ensure the batch size does not exceed the max
    if (inputs[0].size() > static_cast<size_t>(m_options.maxBatchSize)) {
        std::cout << "===== Error =====" << std::endl;
        std::cout << "The batch size is larger than the model expects!" << std::endl;
        std::cout << "Model max batch size: " << m_options.maxBatchSize << std::endl;
        std::cout << "Batch size provided to call to runInference: " << inputs[0].size() << std::endl;
        return false;
    }

    // Ensure that if the model has a fixed batch size that is greater than 1, the
    // input has the correct length
    if (m_inputBatchSize != -1 && inputs[0].size() != static_cast<size_t>(m_inputBatchSize)) {
        std::cout << "===== Error =====" << std::endl;
        std::cout << "The batch size is different from what the model expects!" << std::endl;
        std::cout << "Model batch size: " << m_inputBatchSize << std::endl;
        std::cout << "Batch size provided to call to runInference: " << inputs[0].size() << std::endl;
        return false;
    }

    const auto batchSize = static_cast<int32_t>(inputs[0].size());
    // Make sure the same batch size was provided for all inputs
    for (size_t i = 1; i < inputs.size(); ++i) {
        if (inputs[i].size() != static_cast<size_t>(batchSize)) {
            std::cout << "===== Error =====" << std::endl;
            std::cout << "The batch size needs to be constant for all inputs!" << std::endl;
            return false;
        }
    }

    // Create the cuda stream that will be used for inference
    cudaStream_t inferenceCudaStream;
    Util::checkCudaErrorCode(cudaStreamCreate(&inferenceCudaStream));

    std::vector<cv::cuda::GpuMat> preprocessedInputs;

    // Preprocess all the inputs
    for (size_t i = 0; i < numInputs; ++i) {
        const auto &batchInput = inputs[i];
        const auto &dims = m_inputDims[i];

        auto &input = batchInput[0];
        if (input.channels() != dims.d[0] || input.rows != dims.d[1] || input.cols != dims.d[2]) {
            std::cout << "===== Error =====" << std::endl;
            std::cout << "Input does not have correct size!" << std::endl;
            std::cout << "Expected: (" << dims.d[0] << ", " << dims.d[1] << ", " << dims.d[2] << ")" << std::endl;
            std::cout << "Got: (" << input.channels() << ", " << input.rows << ", " << input.cols << ")" << std::endl;
            std::cout << "Ensure you resize your input image to the correct size" << std::endl;
            return false;
        }

        nvinfer1::Dims4 inputDims = {batchSize, dims.d[0], dims.d[1], dims.d[2]};
        m_context->setInputShape(m_IOTensorNames[i].c_str(),
                                 inputDims); // Define the batch size

        // OpenCV reads images into memory in NHWC format, while TensorRT expects
        // images in NCHW format. The following method converts NHWC to NCHW. Even
        // though TensorRT expects NCHW at IO, during optimization, it can
        // internally use NHWC to optimize cuda kernels See:
        // https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#data-layout
        // Copy over the input data and perform the preprocessing
        auto mfloat = blobFromGpuMats(batchInput, m_subVals, m_divVals, m_normalize);
        preprocessedInputs.push_back(mfloat);
        m_buffers[i] = mfloat.ptr<void>();
    }

    // Ensure all dynamic bindings have been defined.
    if (!m_context->allInputDimensionsSpecified()) {
        throw std::runtime_error("Error, not all required dimensions specified.");
    }

    // Set the address of the input and output buffers
    for (size_t i = 0; i < m_buffers.size(); ++i) {
        bool status = m_context->setTensorAddress(m_IOTensorNames[i].c_str(), m_buffers[i]);
        if (!status) {
            return false;
        }
    }

    // Run inference.
    bool status = m_context->enqueueV3(inferenceCudaStream);
    if (!status) {
        return false;
    }

    // Copy the outputs back to CPU
    featureVectors.clear();

    for (int batch = 0; batch < batchSize; ++batch) {
        // Batch
        std::vector<std::vector<T>> batchOutputs{};
        for (int32_t outputBinding = numInputs; outputBinding < m_engine->getNbIOTensors(); ++outputBinding) {
            // We start at index m_inputDims.size() to account for the inputs in our
            // m_buffers
            std::vector<T> output;
            auto outputLength = m_outputLengths[outputBinding - numInputs];
            output.resize(outputLength);
            // Copy the output
            Util::checkCudaErrorCode(cudaMemcpyAsync(output.data(),
                                                     static_cast<char *>(m_buffers[outputBinding]) + (batch * sizeof(T) * outputLength),
                                                     outputLength * sizeof(T), cudaMemcpyDeviceToHost, inferenceCudaStream));
            batchOutputs.emplace_back(std::move(output));
        }
        featureVectors.emplace_back(std::move(batchOutputs));
    }

    // Synchronize the cuda stream
    Util::checkCudaErrorCode(cudaStreamSynchronize(inferenceCudaStream));
    Util::checkCudaErrorCode(cudaStreamDestroy(inferenceCudaStream));
    return true;
}

template <typename T>
cv::cuda::GpuMat Engine<T>::blobFromGpuMats(const std::vector<cv::cuda::GpuMat> &batchInput, const std::array<float, 3> &subVals,
                                            const std::array<float, 3> &divVals, bool normalize) {
    cv::cuda::GpuMat gpu_dst(1, batchInput[0].rows * batchInput[0].cols * batchInput.size(), CV_8UC3);

    size_t width = batchInput[0].cols * batchInput[0].rows;
    for (size_t img = 0; img < batchInput.size(); img++) {
        std::vector<cv::cuda::GpuMat> input_channels{
            cv::cuda::GpuMat(batchInput[0].rows, batchInput[0].cols, CV_8U, &(gpu_dst.ptr()[0 + width * 3 * img])),
            cv::cuda::GpuMat(batchInput[0].rows, batchInput[0].cols, CV_8U, &(gpu_dst.ptr()[width + width * 3 * img])),
            cv::cuda::GpuMat(batchInput[0].rows, batchInput[0].cols, CV_8U, &(gpu_dst.ptr()[width * 2 + width * 3 * img]))};
        cv::cuda::split(batchInput[img], input_channels); // HWC -> CHW
    }

    cv::cuda::GpuMat mfloat;
    if (normalize) {
        // [0.f, 1.f]
        gpu_dst.convertTo(mfloat, CV_32FC3, 1.f / 255.f);
    } else {
        // [0.f, 255.f]
        gpu_dst.convertTo(mfloat, CV_32FC3);
    }

    // Apply scaling and mean subtraction
    cv::cuda::subtract(mfloat, cv::Scalar(subVals[0], subVals[1], subVals[2]), mfloat, cv::noArray(), -1);
    cv::cuda::divide(mfloat, cv::Scalar(divVals[0], divVals[1], divVals[2]), mfloat, 1, -1);

    return mfloat;
}

template <typename T> std::string Engine<T>::serializeEngineOptions(const Options &options, const std::string &onnxModelPath) {
    const auto filenamePos = onnxModelPath.find_last_of('/') + 1;
    std::string engineName = onnxModelPath.substr(filenamePos, onnxModelPath.find_last_of('.') - filenamePos) + ".engine";

    // Add the GPU device name to the file to ensure that the model is only used
    // on devices with the exact same GPU
    std::vector<std::string> deviceNames;
    getDeviceNames(deviceNames);

    if (static_cast<size_t>(options.deviceIndex) >= deviceNames.size()) {
        throw std::runtime_error("Error, provided device index is out of range!");
    }

    auto deviceName = deviceNames[options.deviceIndex];
    // Remove spaces from the device name
    deviceName.erase(std::remove_if(deviceName.begin(), deviceName.end(), ::isspace), deviceName.end());

    engineName += "." + deviceName;

    // Serialize the specified options into the filename
    if (options.precision == Precision::FP16) {
        engineName += ".fp16";
    } else if (options.precision == Precision::FP32) {
        engineName += ".fp32";
    } else {
        engineName += ".int8";
    }

    engineName += "." + std::to_string(options.maxBatchSize);
    engineName += "." + std::to_string(options.optBatchSize);

    return engineName;
}

template <typename T> void Engine<T>::getDeviceNames(std::vector<std::string> &deviceNames) {
    int numGPUs;
    cudaGetDeviceCount(&numGPUs);

    for (int device = 0; device < numGPUs; device++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);

        deviceNames.push_back(std::string(prop.name));
    }
}

template <typename T>
cv::cuda::GpuMat Engine<T>::resizeKeepAspectRatioPadRightBottom(const cv::cuda::GpuMat &input, size_t height, size_t width,
                                                                const cv::Scalar &bgcolor) {
    float r = std::min(width / (input.cols * 1.0), height / (input.rows * 1.0));
    int unpad_w = r * input.cols;
    int unpad_h = r * input.rows;
    cv::cuda::GpuMat re(unpad_h, unpad_w, CV_8UC3);
    cv::cuda::resize(input, re, re.size());
    cv::cuda::GpuMat out(height, width, CV_8UC3, bgcolor);
    re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
    return out;
}

template <typename T>
void Engine<T>::transformOutput(std::vector<std::vector<std::vector<T>>> &input, std::vector<std::vector<T>> &output) {
    if (input.size() != 1) {
        throw std::logic_error("The feature vector has incorrect dimensions!");
    }

    output = std::move(input[0]);
}

template <typename T> void Engine<T>::transformOutput(std::vector<std::vector<std::vector<T>>> &input, std::vector<T> &output) {
    if (input.size() != 1 || input[0].size() != 1) {
        throw std::logic_error("The feature vector has incorrect dimensions!");
    }

    output = std::move(input[0][0]);
}
