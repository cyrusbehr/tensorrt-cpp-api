#pragma once

#include <fstream>
#include <chrono>
#include <atomic>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include "NvInfer.h"

#if defined(__clang__) || defined(__GNUC__)
#define CPP_STANDARD __cplusplus
#elif defined(_MSC_VER)
#define CPP_STANDARD _MSVC_LANG
#endif

#if CPP_STANDARD >= 199711L
#define HAS_CPP_03 1
#else
#define HAS_CPP_03 0
#endif
#if CPP_STANDARD >= 201103L
#define HAS_CPP_11 1
#else
#define HAS_CPP_11 0
#endif
#if CPP_STANDARD >= 201402L
#define HAS_CPP_14 1
#else
#define HAS_CPP_14 0
#endif
#if CPP_STANDARD >= 201703L
#define HAS_CPP_17 1
#else
#define HAS_CPP_17 0
#endif

// Utility Timer
template <typename Clock = std::chrono::high_resolution_clock>
class Stopwatch
{
    typename Clock::time_point start_point;
public:
    Stopwatch() :start_point(Clock::now()) {}

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
    // Full precision floating point value
    FP32,
    // Half prevision floating point value
    FP16,
    // Int8 quantization.
    // Has reduced dynamic range, may result in slight loss in accuracy.
    // If INT8 is selected, must provide path to calibration dataset directory.
    INT8,
};

// EngineOptions for the network
struct EngineOptions {
    // Precision to use for GPU inference.
    Precision precision = Precision::FP16;
    // If INT8 precision is selected, must provide path to calibration dataset directory.
    std::string calibrationDataDirectoryPath;
    // The batch size to be used when computing calibration data for INT8 inference.
    // Should be set to as large a batch number as your GPU will support.
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
    Int8EntropyCalibrator2(int32_t batchSize,
                           int32_t inputW,
                           int32_t inputH,
                           const std::string& calibDataDirPath,
                           std::string calibTableName,
                           std::string inputBlobName,
                           const std::array<float, 3>& subVals = {0.f, 0.f, 0.f},
                           const std::array<float, 3>& divVals = {1.f, 1.f, 1.f},
                           bool normalize = true, 
                           bool readCache = true);
    virtual ~Int8EntropyCalibrator2() override;
    // Abstract base class methods which must be implemented
    virtual int32_t getBatchSize() const noexcept override;
    virtual bool getBatch(void* bindings[], char const* names[], int32_t nbBindings) noexcept override;
    virtual void const* readCalibrationCache(std::size_t& length) noexcept override;
    virtual void writeCalibrationCache(void const* ptr, std::size_t length) noexcept override;
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
    void* m_deviceInput;
    std::vector<char> m_calibCache;
};

// Class to extend TensorRT logger
class Logger : public nvinfer1::ILogger {
    virtual void log(Severity severity, const char* msg) noexcept override;
};

class Engine {
public:
    Engine();
    // Engine(const EngineOptions& options);
    ~Engine();
    // Build the network
    // The default implementation will normalize values between [0.f, 1.f]
    // Setting the normalize flag to false will leave values between [0.f, 255.f] (some converted models may require this).
    // If the model requires values to be normalized between [-1.f, 1.f], use the following params:
    //    subVals = {0.5f, 0.5f, 0.5f};
    //    divVals = {0.5f, 0.5f, 0.5f};
    //    normalize = true;
    static bool build(const std::string& onnxModelPath,
                      const EngineOptions& engineOptions,
                      const std::array<float, 3>& subVals = {0.f, 0.f, 0.f},
                      const std::array<float, 3>& divVals = {1.f, 1.f, 1.f},
                      bool normalize = true);
    // Load and prepare the network for inference
    bool loadNetwork(const std::string& enginePath,
                     const EngineOptions& engineOptions,
                     const std::array<float, 3>& subVals = {0.f, 0.f, 0.f},
                     const std::array<float, 3>& divVals = {1.f, 1.f, 1.f},
                     bool normalize = true);
    // Run inference.
    // Input format [input][batch][cv::cuda::GpuMat]
    // Output format [batch][output][feature_vector]
    bool runInference(const std::vector<std::vector<cv::cuda::GpuMat>>& inputs, std::vector<std::vector<std::vector<float>>>& featureVectors);

    // Utility method for resizing an image while maintaining the aspect ratio by adding padding to smaller dimension after scaling
    // While letterbox padding normally adds padding to top & bottom, or left & right sides, this implementation only adds padding to the right or bottom side
    // This is done so that it's easier to convert detected coordinates (ex. YOLO model) back to the original reference frame.
    static cv::cuda::GpuMat resizeKeepAspectRatioPadRightBottom(const cv::cuda::GpuMat& input, size_t height, size_t width, const cv::Scalar& bgcolor = cv::Scalar(0, 0, 0));

#if HAS_CPP_17
    [[nodiscard]] const std::vector<nvinfer1::Dims3>& getInputDims() const { return m_inputDims; }
    [[nodiscard]] const std::vector<nvinfer1::Dims>& getOutputDims() const { return m_outputDims; }
#else
    const std::vector<nvinfer1::Dims3>& getInputDims() const { return m_inputDims; }
    const std::vector<nvinfer1::Dims>& getOutputDims() const { return m_outputDims; }
#endif

    // Utility method for transforming triple nested output array into 2D array
    // Should be used when the output batch size is 1, but there are multiple output feature vectors
    static void transformOutput(std::vector<std::vector<std::vector<float>>>& input, std::vector<std::vector<float>>& output);

    // Utility method for transforming triple nested output array into single array
    // Should be used when the output batch size is 1, and there is only a single output feature vector
    static void transformOutput(std::vector<std::vector<std::vector<float>>>& input, std::vector<float>& output);
    // Convert NHWC to NCHW and apply scaling and mean subtraction
    static cv::cuda::GpuMat blobFromGpuMats(const std::vector<cv::cuda::GpuMat>& batchInput, const std::array<float, 3>& subVals, const std::array<float, 3>& divVals, bool normalize);

private:
    static Logger m_logger;

    // Normalization, scaling, and mean subtraction of inputs
    std::array<float, 3> m_subVals{};
    std::array<float, 3> m_divVals{};
    bool m_normalize;

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

    EngineOptions m_engineOptions;
};

// Utility methods
namespace EngineUtil {
    inline bool doesFileExist(const std::string& filepath) {
        const std::ifstream f(filepath.c_str());
        return f.good();
    }

    inline bool doseDirectoryExist(const std::string& dirPath)
    {
        struct stat info{};
        return stat(dirPath.c_str(), &info) == 0 && info.st_mode & S_IFDIR;
    }

    inline void checkCudaErrorCode(cudaError_t code) {
        if (code != 0) {
            const std::string errMsg = "CUDA operation failed with code: " + std::to_string(code) + "(" + cudaGetErrorName(code) + "), with message: " + cudaGetErrorString(code);
            std::cout << errMsg << std::endl;
            throw std::runtime_error(errMsg);
        }
    }

    void getFilesInDirectory(const std::string& dirPath, std::vector<std::string>& filepaths);

    inline void getGpuDeviceNames(std::vector<std::string>& gpuDeviceNames)
    {
        int numGPUs;
        cudaGetDeviceCount(&numGPUs);

        for (int device = 0; device < numGPUs; device++)
        {
            cudaDeviceProp prop{};
            cudaGetDeviceProperties(&prop, device);

            gpuDeviceNames.emplace_back(prop.name);
        }
    }

    // Converts the engine options into a string
    inline std::string serializeEngineOptions(const EngineOptions& options, const std::string& onnxModelPath)
    {
        std::string engineName = onnxModelPath.substr(0, onnxModelPath.find_last_of('.')) + ".engine";

        // Add the GPU device name to the file to ensure that the model is only used on devices with the exact same GPU
        std::vector<std::string> deviceNames;
        getGpuDeviceNames(deviceNames);

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
        }
        else if (options.precision == Precision::FP32) {
            engineName += ".fp32";
        }
        else {
            engineName += ".int8";
        }

        engineName += "." + std::to_string(options.maxBatchSize);
        engineName += "." + std::to_string(options.optBatchSize);

        return engineName;
    }
}