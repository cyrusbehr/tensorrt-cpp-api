#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <iterator>
#include <opencv2/cudaimgproc.hpp>
#include "engine.h"
#include "NvOnnxParser.h"

using namespace nvinfer1;
using namespace EngineUtil;

void Logger::log(Severity severity, const char* msg) noexcept {
    // Would advise using a proper logging utility such as https://github.com/gabime/spdlog
    // For the sake of this tutorial, will just log to the console.

    // Only log Warnings or more important.
    if (severity <= Severity::kWARNING) {
        std::cout << msg << std::endl;
    }
}

Logger Engine::m_logger{};

Engine::Engine()
    : m_subVals({0.f, 0.f, 0.f})
    , m_divVals({1.f, 1.f, 1.f})
    , m_normalize(true)
{
    
}

Engine::~Engine() {
    // Free the GPU memory
    for (const auto& buffer : m_buffers) {
        checkCudaErrorCode(cudaFree(buffer));
    }

    m_buffers.clear();
}

bool Engine::build(const std::string& onnxModelPath,
                   const EngineOptions& engineOptions,
                   const std::array<float, 3>& subVals,
                   const std::array<float, 3>& divVals,
                   bool normalize)
{
    // Only regenerate the engine file if it has not already been generated for the specified options
    std::string engineName = serializeEngineOptions(engineOptions, onnxModelPath);
    std::cout << "Searching for engine file with name: " << engineName << std::endl;

    if (doesFileExist(engineName)) {
        std::cout << "Engine found, not regenerating..." << std::endl;
        return true;
    }

    if (!doesFileExist(onnxModelPath)) {
        throw std::runtime_error("Could not find model at path: " + onnxModelPath);
    }

    // Was not able to find the engine file, generate...
    std::cout << "Engine not found, generating. This could take a while..." << std::endl;

    // Create our engine builder.
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(m_logger));
    if (!builder) {
        return false;
    }

    // Define an explicit batch size and then create the network (implicit batch size is deprecated).
    // More info here: https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#explicit-implicit-batch
    auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network) {
        return false;
    }

    // Create a parser for reading the onnx file.
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, m_logger));
    if (!parser) {
        return false;
    }

    // We are going to first read the onnx file into memory, then pass that buffer to the parser.
    // Had our onnx model file been encrypted, this approach would allow us to first decrypt the buffer.
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
            throw std::runtime_error("Error, the model has multiple inputs, each with differing batch sizes!");
        }
    }

    // Check to see if the model supports dynamic batch size or not
    if (input0Batch == -1) {
        std::cout << "Model supports dynamic batch size" << std::endl;
    }
    else if (input0Batch == 1) {
        std::cout << "Model only supports fixed batch size of 1" << std::endl;
        // If the model supports a fixed batch size, ensure that the maxBatchSize and optBatchSize were set correctly.
        if (engineOptions.optBatchSize != input0Batch || engineOptions.maxBatchSize != input0Batch) {
            throw std::runtime_error("Error, model only supports a fixed batch size of 1. Must set EngineOptions.optBatchSize and EngineOptions.maxBatchSize to 1");
        }
    }
    else {
        throw std::runtime_error("Implementation currently only supports dynamic batch sizes or a fixed batch size of 1 (your batch size is fixed to "
            + std::to_string(input0Batch) + ")");
    }

    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) {
        return false;
    }

    // Register a single optimization profile
    IOptimizationProfile* optProfile = builder->createOptimizationProfile();
    for (int32_t i = 0; i < numInputs; ++i) {
        // Must specify dimensions for all the inputs the model expects.
        const auto input = network->getInput(i);
        const auto inputName = input->getName();
        const auto inputDims = input->getDimensions();
        int32_t inputC = inputDims.d[1];
        int32_t inputH = inputDims.d[2];
        int32_t inputW = inputDims.d[3];

        // Specify the optimization profile`
        optProfile->setDimensions(inputName, OptProfileSelector::kMIN, Dims4(1, inputC, inputH, inputW));
        optProfile->setDimensions(inputName, OptProfileSelector::kOPT, Dims4(engineOptions.optBatchSize, inputC, inputH, inputW));
        optProfile->setDimensions(inputName, OptProfileSelector::kMAX, Dims4(engineOptions.maxBatchSize, inputC, inputH, inputW));
    }
    config->addOptimizationProfile(optProfile);

    std::unique_ptr<Int8EntropyCalibrator2> calibrator;

    // Set the precision level
    if (engineOptions.precision == Precision::FP16) {
        // Ensure the GPU supports FP16 inference
        if (!builder->platformHasFastFp16()) {
            throw std::runtime_error("Error: GPU does not support FP16 precision");
        }
        config->setFlag(BuilderFlag::kFP16);
    }
    else if (engineOptions.precision == Precision::INT8) {
        if (numInputs > 1) {
            throw std::runtime_error("Error, this implementation currently only supports INT8 quantization for single input models");
        }

        // Ensure the GPU supports INT8 Quantization
        if (!builder->platformHasFastInt8()) {
            throw std::runtime_error("Error: GPU does not support INT8 precision");
        }

        // Ensure the user has provided path to calibration data directory
        if (engineOptions.calibrationDataDirectoryPath.empty()) {
            throw std::runtime_error("Error: If INT8 precision is selected, must provide path to calibration data directory to Engine::build method");
        }

        config->setFlag((BuilderFlag::kINT8));

        const auto input = network->getInput(0);
        const auto inputName = input->getName();
        const auto inputDims = input->getDimensions();
        const auto calibrationFileName = engineName + ".calibration";

        calibrator.reset(new Int8EntropyCalibrator2(engineOptions.calibrationBatchSize,
                                                    inputDims.d[3],
                                                    inputDims.d[2],
                                                    engineOptions.calibrationDataDirectoryPath,
                                                    calibrationFileName,
                                                    inputName,
                                                    subVals,
                                                    divVals,
                                                    normalize));
        config->setInt8Calibrator(calibrator.get());
    }

    // CUDA stream used for profiling by the builder.
    cudaStream_t profileStream;
    checkCudaErrorCode(cudaStreamCreate(&profileStream));
    config->setProfileStream(profileStream);

    // Build the engine
    // If this call fails, it is suggested to increase the logger verbosity to kVERBOSE and try rebuilding the engine.
    // Doing so will provide you with more information on why exactly it is failing.
    std::unique_ptr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan) {
        return false;
    }

    // Write the engine to disk
    std::ofstream outfile(engineName, std::ofstream::binary);
    outfile.write(static_cast<const char*>(plan->data()), plan->size());

    std::cout << "Success, saved engine to " << engineName << std::endl;

    checkCudaErrorCode(cudaStreamDestroy(profileStream));
    return true;
}

bool Engine::loadNetwork(const std::string& enginePath,
                         const EngineOptions& engineOptions,
                         const std::array<float, 3>& subVals,
                         const std::array<float, 3>& divVals,
                         bool normalize)
{
    m_subVals = subVals;
    m_divVals = divVals;
    m_normalize = normalize;

    for (const auto& buffer : m_buffers) {
        checkCudaErrorCode(cudaFree(buffer));
    }
    m_buffers.clear();
    m_outputLengthsFloat.clear();
    m_inputDims.clear();
    m_outputDims.clear();
    m_IOTensorNames.clear();

    m_context.reset();
    m_engine.reset();
    m_runtime.reset();

    m_engineOptions = engineOptions;

    if (!doesFileExist(enginePath))
    {
        throw std::runtime_error("Could not find engine at path: " + enginePath);
    }

    // Read the serialized model from disk
    std::ifstream file(enginePath, std::ios::binary | std::ios::ate);
    const std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        throw std::runtime_error("Unable to read engine file");
    }

    // Create a runtime to deserialize the engine file.
    m_runtime.reset(createInferRuntime(m_logger));
    if (!m_runtime) {
        return false;
    }

    // Set the device index
    const auto ret = cudaSetDevice(m_engineOptions.deviceIndex);
    if (ret != 0) {
        int numGPUs;
        cudaGetDeviceCount(&numGPUs);
        const auto errMsg = "Unable to set GPU device index to: " + std::to_string(m_engineOptions.deviceIndex) +
                            ". Note, your device has " + std::to_string(numGPUs) + " CUDA-capable GPU(s).";
        throw std::runtime_error(errMsg);
    }

    // Create an engine, a representation of the optimized model.
    m_engine.reset(m_runtime->deserializeCudaEngine(buffer.data(), buffer.size()));
    if (!m_engine) {
        return false;
    }

    // The execution context contains all of the state associated with a particular invocation
    m_context.reset(m_engine->createExecutionContext());
    if (!m_context) {
        return false;
    }

    // Storage for holding the input and output buffers
    // This will be passed to TensorRT for inference
    m_buffers.resize(m_engine->getNbIOTensors());

    // Create a cuda stream
    cudaStream_t stream;
    checkCudaErrorCode(cudaStreamCreate(&stream));

    // Allocate GPU memory for input and output buffers
    m_outputLengthsFloat.clear();
    for (int i = 0; i < m_engine->getNbIOTensors(); ++i) {
        const auto tensorName = m_engine->getIOTensorName(i);
        m_IOTensorNames.emplace_back(tensorName);
        const auto tensorType = m_engine->getTensorIOMode(tensorName);
        const auto tensorShape = m_engine->getTensorShape(tensorName);
        if (tensorType == TensorIOMode::kINPUT) {
            // Allocate memory for the input
            // Allocate enough to fit the max batch size (we could end up using less later)
            checkCudaErrorCode(cudaMallocAsync(&m_buffers[i], m_engineOptions.maxBatchSize * tensorShape.d[1] * tensorShape.d[2] * tensorShape.d[3] * sizeof(float), stream));

            // Store the input dims for later use
            m_inputDims.emplace_back(tensorShape.d[1], tensorShape.d[2], tensorShape.d[3]);
        }
        else if (tensorType == TensorIOMode::kOUTPUT) {
            // The binding is an output
            uint32_t outputLenFloat = 1;
            m_outputDims.push_back(tensorShape);

            for (int j = 1; j < tensorShape.nbDims; ++j) {
                // We ignore j = 0 because that is the batch size, and we will take that into account when sizing the buffer
                outputLenFloat *= tensorShape.d[j];
            }

            m_outputLengthsFloat.push_back(outputLenFloat);
            // Now size the output buffer appropriately, taking into account the max possible batch size (although we could actually end up using less memory)
            checkCudaErrorCode(cudaMallocAsync(&m_buffers[i], outputLenFloat * m_engineOptions.maxBatchSize * sizeof(float), stream));
        }
        else {
            throw std::runtime_error("Error, IO Tensor is neither an input or output!");
        }
    }

    // Synchronize and destroy the cuda stream
    checkCudaErrorCode(cudaStreamSynchronize(stream));
    checkCudaErrorCode(cudaStreamDestroy(stream));

    return true;
}

bool Engine::runInference(const std::vector<std::vector<cv::cuda::GpuMat>>& inputs, std::vector<std::vector<std::vector<float>>>& featureVectors) {
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
    if (inputs[0].size() > static_cast<size_t>(m_engineOptions.maxBatchSize)) {
        std::cout << "===== Error =====" << std::endl;
        std::cout << "The batch size is larger than the model expects!" << std::endl;
        std::cout << "Model max batch size: " << m_engineOptions.maxBatchSize << std::endl;
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
    checkCudaErrorCode(cudaStreamCreate(&inferenceCudaStream));

    // Preprocess all the inputs
    for (size_t i = 0; i < numInputs; ++i) {
        const auto& batchInput = inputs[i];
        const auto& dims = m_inputDims[i];

        auto& input = batchInput[0];
        if (input.channels() != dims.d[0] ||
            input.rows != dims.d[1] ||
            input.cols != dims.d[2]) {
            std::cout << "===== Error =====" << std::endl;
            std::cout << "Input does not have correct size!" << std::endl;
            std::cout << "Expected: (" << dims.d[0] << ", " << dims.d[1] << ", "
                << dims.d[2] << ")" << std::endl;
            std::cout << "Got: (" << input.channels() << ", " << input.rows << ", " << input.cols << ")" << std::endl;
            std::cout << "Ensure you resize your input image to the correct size" << std::endl;
            return false;
        }

        nvinfer1::Dims4 inputDims = { batchSize, dims.d[0], dims.d[1], dims.d[2] };
        m_context->setInputShape(m_IOTensorNames[i].c_str(), inputDims); // Define the batch size

        // OpenCV reads images into memory in NHWC format, while TensorRT expects images in NCHW format. 
        // The following method converts NHWC to NCHW.
        // Even though TensorRT expects NCHW at IO, during optimization, it can internally use NHWC to optimize cuda kernels
        // See: https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#data-layout
        // Copy over the input data and perform the preprocessing
        auto mfloat = blobFromGpuMats(batchInput, m_subVals, m_divVals, m_normalize);
        auto* dataPointer = mfloat.ptr<void>();

        checkCudaErrorCode(cudaMemcpyAsync(m_buffers[i], dataPointer,
                                           mfloat.cols * mfloat.rows * mfloat.channels() * sizeof(float),
                                           cudaMemcpyDeviceToDevice, inferenceCudaStream));
    }

    // Ensure all dynamic bindings have been defined.
    // TODO: Should use allInputShapesSpecified()
    if (!m_context->allInputDimensionsSpecified()) {
        throw std::runtime_error("Error, not all required dimensions specified.");
    }

    // Set the address of the input and output buffers
    for (size_t i = 0; i < m_buffers.size(); ++i) {
        const bool status = m_context->setTensorAddress(m_IOTensorNames[i].c_str(), m_buffers[i]);
        if (!status) {
            return false;
        }
    }

    // Run inference.
    const bool status = m_context->enqueueV3(inferenceCudaStream);
    if (!status) {
        return false;
    }

    // Copy the outputs back to CPU
    featureVectors.clear();

    for (int batch = 0; batch < batchSize; ++batch) {
        // Batch
        std::vector<std::vector<float>> batchOutputs{};
        for (int32_t outputBinding = numInputs; outputBinding < m_engine->getNbBindings(); ++outputBinding) {
            // We start at index m_inputDims.size() to account for the inputs in our m_buffers
            std::vector<float> output;
            auto outputLenFloat = m_outputLengthsFloat[outputBinding - numInputs];
            output.resize(outputLenFloat);
            // Copy the output
            checkCudaErrorCode(cudaMemcpyAsync(output.data(), static_cast<char*>(m_buffers[outputBinding]) + (batch * sizeof(float) * outputLenFloat), outputLenFloat * sizeof(float), cudaMemcpyDeviceToHost, inferenceCudaStream));
            batchOutputs.emplace_back(std::move(output));
        }
        featureVectors.emplace_back(std::move(batchOutputs));
    }

    // Synchronize the cuda stream
    checkCudaErrorCode(cudaStreamSynchronize(inferenceCudaStream));
    checkCudaErrorCode(cudaStreamDestroy(inferenceCudaStream));
    return true;
}

cv::cuda::GpuMat Engine::blobFromGpuMats(const std::vector<cv::cuda::GpuMat>& batchInput, const std::array<float, 3>& subVals, const std::array<float, 3>& divVals, bool normalize) {
    cv::cuda::GpuMat gpu_dst(1, batchInput[0].rows * batchInput[0].cols * batchInput.size(), CV_8UC3);

    size_t width = batchInput[0].cols * batchInput[0].rows;
    for (size_t img = 0; img < batchInput.size(); img++) {
        std::vector<cv::cuda::GpuMat> input_channels{
            cv::cuda::GpuMat(batchInput[0].rows, batchInput[0].cols, CV_8U, &(gpu_dst.ptr()[0 + width * 3 * img])),
                cv::cuda::GpuMat(batchInput[0].rows, batchInput[0].cols, CV_8U, &(gpu_dst.ptr()[width + width * 3 * img])),
                cv::cuda::GpuMat(batchInput[0].rows, batchInput[0].cols, CV_8U,
                    &(gpu_dst.ptr()[width * 2 + width * 3 * img]))
        };
        cv::cuda::split(batchInput[img], input_channels);  // HWC -> CHW
    }

    cv::cuda::GpuMat mfloat;
    if (normalize) {
        // [0.f, 1.f]
        gpu_dst.convertTo(mfloat, CV_32FC3, 1.f / 255.f);
    }
    else {
        // [0.f, 255.f]
        gpu_dst.convertTo(mfloat, CV_32FC3);
    }

    // Apply scaling and mean subtraction
    cv::cuda::subtract(mfloat, cv::Scalar(subVals[0], subVals[1], subVals[2]), mfloat, cv::noArray(), -1);
    cv::cuda::divide(mfloat, cv::Scalar(divVals[0], divVals[1], divVals[2]), mfloat, 1, -1);

    return mfloat;
}

cv::cuda::GpuMat Engine::resizeKeepAspectRatioPadRightBottom(const cv::cuda::GpuMat& input, size_t height, size_t width, const cv::Scalar& bgcolor) {
    float r = std::min(width / (input.cols * 1.0), height / (input.rows * 1.0));
    int unpad_w = r * input.cols;
    int unpad_h = r * input.rows;
    cv::cuda::GpuMat re(unpad_h, unpad_w, CV_8UC3);
    cv::cuda::resize(input, re, re.size());
    cv::cuda::GpuMat out(height, width, CV_8UC3, bgcolor);
    re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
    return out;
}

void Engine::transformOutput(std::vector<std::vector<std::vector<float>>>& input, std::vector<std::vector<float>>& output) {
    if (input.size() != 1) {
        throw std::logic_error("The feature vector has incorrect dimensions!");
    }

    output = std::move(input[0]);
}

void Engine::transformOutput(std::vector<std::vector<std::vector<float>>>& input, std::vector<float>& output) {
    if (input.size() != 1 || input[0].size() != 1) {
        throw std::logic_error("The feature vector has incorrect dimensions!");
    }

    output = std::move(input[0][0]);
}

Int8EntropyCalibrator2::Int8EntropyCalibrator2(int32_t batchSize, int32_t inputW, int32_t inputH,
                                               const std::string& calibDataDirPath,
                                               std::string calibTableName,
                                               std::string inputBlobName,
                                               const std::array<float, 3>& subVals,
                                               const std::array<float, 3>& divVals,
                                               bool normalize,
                                               bool readCache)
    : m_batchSize(batchSize)
    , m_inputW(inputW)
    , m_inputH(inputH)
    , m_imgIdx(0)
    , m_calibTableName(std::move(calibTableName))
    , m_inputBlobName(std::move(inputBlobName))
    , m_subVals(subVals)
    , m_divVals(divVals)
    , m_normalize(normalize)
    , m_readCache(readCache) {

    // Allocate GPU memory to hold the entire batch
    m_inputCount = 3 * inputW * inputH * batchSize;
    checkCudaErrorCode(cudaMalloc(&m_deviceInput, m_inputCount * sizeof(float)));

    // Read the name of all the files in the specified directory.
    if (!doseDirectoryExist(calibDataDirPath))
    {
        throw std::runtime_error("Error, directory at provided path does not exist: " + calibDataDirPath);
    }

    getFilesInDirectory(calibDataDirPath, m_imgPaths);
    if (m_imgPaths.size() < static_cast<size_t>(batchSize)) {
        throw std::runtime_error("There are fewer calibration images than the specified batch size!");
    }

    // Randomize the calibration data
    std::random_device rd{};
    auto rng = std::default_random_engine{ rd() };
    std::shuffle(std::begin(m_imgPaths), std::end(m_imgPaths), rng);
}

Int8EntropyCalibrator2::~Int8EntropyCalibrator2() {
    checkCudaErrorCode(cudaFree(m_deviceInput));
};

int32_t Int8EntropyCalibrator2::getBatchSize() const noexcept {
    // Return the batch size
    return m_batchSize;
}

bool Int8EntropyCalibrator2::getBatch(void** bindings, const char** names, int32_t nbBindings) noexcept {
    // This method will read a batch of images into GPU memory, and place the pointer to the GPU memory in the bindings variable.

    if (m_imgIdx + m_batchSize > static_cast<int>(m_imgPaths.size())) {
        // There are not enough images left to satisfy an entire batch
        return false;
    }

    // Read the calibration images into memory for the current batch
    std::vector<cv::cuda::GpuMat> inputImgs;
    for (int i = m_imgIdx; i < m_imgIdx + m_batchSize; i++) {
        std::cout << "Reading image " << i << ": " << m_imgPaths[i] << std::endl;
        auto cpuImg = cv::imread(m_imgPaths[i]);
        if (cpuImg.empty()) {
            std::cout << "Fatal error: Unable to read image at path: " << m_imgPaths[i] << std::endl;
            return false;
        }

        cv::cuda::GpuMat gpuImg;
        gpuImg.upload(cpuImg);
        cv::cuda::cvtColor(gpuImg, gpuImg, cv::COLOR_BGR2RGB);

        // TODO: Define any preprocessing code here, such as resizing
        auto resized = Engine::resizeKeepAspectRatioPadRightBottom(gpuImg, m_inputH, m_inputW);

        inputImgs.emplace_back(std::move(resized));
    }

    // Convert the batch from NHWC to NCHW
    // ALso apply normalization, scaling, and mean subtraction
    auto mfloat = Engine::blobFromGpuMats(inputImgs, m_subVals, m_divVals, m_normalize);
    auto* dataPointer = mfloat.ptr<void>();

    // Copy the GPU buffer to member variable so that it persists
    checkCudaErrorCode(cudaMemcpyAsync(m_deviceInput, dataPointer, m_inputCount * sizeof(float), cudaMemcpyDeviceToDevice));

    m_imgIdx += m_batchSize;
    if (std::string(names[0]) != m_inputBlobName) {
        std::cout << "Error: Incorrect input name provided!" << std::endl;
        return false;
    }
    bindings[0] = m_deviceInput;
    return true;
}

void const* Int8EntropyCalibrator2::readCalibrationCache(size_t& length) noexcept {
    std::cout << "Searching for calibration cache: " << m_calibTableName << std::endl;
    m_calibCache.clear();
    std::ifstream input(m_calibTableName, std::ios::binary);
    input >> std::noskipws;
    if (m_readCache && input.good()) {
        std::cout << "Reading calibration cache: " << m_calibTableName << std::endl;
        std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(), std::back_inserter(m_calibCache));
    }
    length = m_calibCache.size();
    return length ? m_calibCache.data() : nullptr;
}

void Int8EntropyCalibrator2::writeCalibrationCache(const void* ptr, std::size_t length) noexcept {
    std::cout << "Writing calib cache: " << m_calibTableName << " Size: " << length << " bytes" << std::endl;
    std::ofstream output(m_calibTableName, std::ios::binary);
    output.write(reinterpret_cast<const char*>(ptr), length);
}