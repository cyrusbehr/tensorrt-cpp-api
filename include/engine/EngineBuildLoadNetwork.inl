#pragma once
#include <filesystem>
#include <spdlog/spdlog.h>
#include "util/Util.h"

template <typename T>
bool Engine<T>::buildLoadNetwork(std::string onnxModelPath, const std::array<float, 3> &subVals, const std::array<float, 3> &divVals,
                                 bool normalize) {
    const auto engineName = serializeEngineOptions(m_options, onnxModelPath);
    const auto engineDir = std::filesystem::path(m_options.engineFileDir);
    std::filesystem::path enginePath = engineDir / engineName;
    spdlog::info("Searching for engine file with name: {}", enginePath.string());

    if (Util::doesFileExist(enginePath)) {
        spdlog::info("Engine found, not regenerating...");
    } else {
        if (!Util::doesFileExist(onnxModelPath)) {
            auto msg = "Could not find ONNX model at path: " + onnxModelPath;
            spdlog::error(msg);
            throw std::runtime_error(msg);
        }

        spdlog::info("Engine not found, generating. This could take a while...");
        if (!std::filesystem::exists(engineDir)) {
            std::filesystem::create_directories(engineDir);
            spdlog::info("Created directory: {}", engineDir.string());
        }

        auto ret = build(onnxModelPath, subVals, divVals, normalize);
        if (!ret) {
            return false;
        }
    }

    return loadNetwork(enginePath, subVals, divVals, normalize);
}

template <typename T>
bool Engine<T>::loadNetwork(std::string trtModelPath, const std::array<float, 3> &subVals, const std::array<float, 3> &divVals,
                            bool normalize) {
    m_subVals = subVals;
    m_divVals = divVals;
    m_normalize = normalize;

    // Read the serialized model from disk
    if (!Util::doesFileExist(trtModelPath)) {
        auto msg = "Error, unable to read TensorRT model at path: " + trtModelPath;
        spdlog::error(msg);
        return false;
    } else {
        auto msg = "Loading TensorRT engine file at path: " + trtModelPath;
        spdlog::info(msg);
    }

    std::ifstream file(trtModelPath, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        auto msg = "Error, unable to read engine file";
        spdlog::error(msg);
        throw std::runtime_error(msg);
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
        spdlog::error(errMsg);
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
                auto msg = "Error, the implementation currently only supports float inputs";
                spdlog::error(msg);
                throw std::runtime_error(msg);
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
                auto msg = "Error, the model has expected output of type float. Engine class template parameter must be adjusted.";
                spdlog::error(msg);
                throw std::runtime_error(msg);
            } else if (tensorDataType == nvinfer1::DataType::kHALF && !std::is_same<__half, T>::value) {
                auto msg = "Error, the model has expected output of type __half. Engine class template parameter must be adjusted.";
                spdlog::error(msg);
                throw std::runtime_error(msg);
            } else if (tensorDataType == nvinfer1::DataType::kINT8 && !std::is_same<int8_t, T>::value) {
                auto msg = "Error, the model has expected output of type int8_t. Engine class template parameter must be adjusted.";
                spdlog::error(msg);
                throw std::runtime_error(msg);
            } else if (tensorDataType == nvinfer1::DataType::kINT32 && !std::is_same<int32_t, T>::value) {
                auto msg = "Error, the model has expected output of type int32_t. Engine class template parameter must be adjusted.";
                spdlog::error(msg);
                throw std::runtime_error(msg);
            } else if (tensorDataType == nvinfer1::DataType::kBOOL && !std::is_same<bool, T>::value) {
                auto msg = "Error, the model has expected output of type bool. Engine class template parameter must be adjusted.";
                spdlog::error(msg);
                throw std::runtime_error(msg);
            } else if (tensorDataType == nvinfer1::DataType::kUINT8 && !std::is_same<uint8_t, T>::value) {
                auto msg = "Error, the model has expected output of type uint8_t. Engine class template parameter must be adjusted.";
                spdlog::error(msg);
                throw std::runtime_error(msg);
            } else if (tensorDataType == nvinfer1::DataType::kFP8) {
                auto msg = "Error, the model has expected output of type kFP8. This is not supported by the Engine class.";
                spdlog::error(msg);
                throw std::runtime_error(msg);
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
            auto msg = "Error, IO Tensor is neither an input or output!";
            spdlog::error(msg);
            throw std::runtime_error(msg);
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
        auto msg = "Error, unable to read engine file";
        spdlog::error(msg);
        throw std::runtime_error(msg);
    }

    // Parse the buffer we read into memory.
    auto parsed = parser->parse(buffer.data(), buffer.size());
    if (!parsed) {
        return false;
    }

    // Ensure that all the inputs have the same batch size
    const auto numInputs = network->getNbInputs();
    if (numInputs < 1) {
        auto msg = "Error, model needs at least 1 input!";
        spdlog::error(msg);
        throw std::runtime_error(msg);
    }
    const auto input0Batch = network->getInput(0)->getDimensions().d[0];
    for (int32_t i = 1; i < numInputs; ++i) {
        if (network->getInput(i)->getDimensions().d[0] != input0Batch) {
            auto msg = "Error, the model has multiple inputs, each with differing batch sizes!";
            spdlog::error(msg);
            throw std::runtime_error(msg);
        }
    }

    // Check to see if the model supports dynamic batch size or not
    bool doesSupportDynamicBatch = false;
    if (input0Batch == -1) {
        doesSupportDynamicBatch = true;
        spdlog::info("Model supports dynamic batch size");
    } else {
        spdlog::info("Model only supports fixed batch size of {}", input0Batch);
        // If the model supports a fixed batch size, ensure that the maxBatchSize
        // and optBatchSize were set correctly.
        if (m_options.optBatchSize != input0Batch || m_options.maxBatchSize != input0Batch) {
            auto msg = "Error, model only supports a fixed batch size of " + std::to_string(input0Batch) +
                       ". Must set Options.optBatchSize and Options.maxBatchSize to 1";
            spdlog::error(msg);
            throw std::runtime_error(msg);
        }
    }

    const auto input3Batch = network->getInput(0)->getDimensions().d[3];
    bool doesSupportDynamicWidth = false;
    if (input3Batch == -1) {
        doesSupportDynamicWidth = true;
        spdlog::info("Model supports dynamic width. Using Options.maxInputWidth, Options.minInputWidth, and Options.optInputWidth to set the input width.");

        // Check that the values of maxInputWidth, minInputWidth, and optInputWidth are valid
        if (m_options.maxInputWidth < m_options.minInputWidth || m_options.maxInputWidth < m_options.optInputWidth ||
            m_options.minInputWidth > m_options.optInputWidth
            || m_options.maxInputWidth < 1 || m_options.minInputWidth < 1 || m_options.optInputWidth < 1) {
            auto msg = "Error, invalid values for Options.maxInputWidth, Options.minInputWidth, and Options.optInputWidth";
            spdlog::error(msg);
            throw std::runtime_error(msg);
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
    
        int32_t minInputWidth = std::max(m_options.minInputWidth, inputW);

        // Specify the optimization profile`
        if (doesSupportDynamicBatch) {
            optProfile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, inputC, inputH, minInputWidth));
        } else {
            optProfile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMIN,
                                      nvinfer1::Dims4(m_options.optBatchSize, inputC, inputH, minInputWidth));
        }

        if (doesSupportDynamicWidth) {
            optProfile->setDimensions(inputName, nvinfer1::OptProfileSelector::kOPT,
                                      nvinfer1::Dims4(m_options.optBatchSize, inputC, inputH, m_options.optInputWidth));
            optProfile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMAX,
                                      nvinfer1::Dims4(m_options.maxBatchSize, inputC, inputH, m_options.maxInputWidth));
        } else {
            optProfile->setDimensions(inputName, nvinfer1::OptProfileSelector::kOPT,
                                    nvinfer1::Dims4(m_options.optBatchSize, inputC, inputH, inputW));
            optProfile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMAX,
                                    nvinfer1::Dims4(m_options.maxBatchSize, inputC, inputH, inputW));
        }
    }
    config->addOptimizationProfile(optProfile);

    // Set the precision level
    const auto engineName = serializeEngineOptions(m_options, onnxModelPath);
    if (m_options.precision == Precision::FP16) {
        // Ensure the GPU supports FP16 inference
        if (!builder->platformHasFastFp16()) {
            auto msg = "Error: GPU does not support FP16 precision";
            spdlog::error(msg);
            throw std::runtime_error(msg);
        }
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    } else if (m_options.precision == Precision::INT8) {
        if (numInputs > 1) {
            auto msg = "Error, this implementation currently only supports INT8 "
                       "quantization for single input models";
            spdlog::error(msg);
            throw std::runtime_error(msg);
        }

        // Ensure the GPU supports INT8 Quantization
        if (!builder->platformHasFastInt8()) {
            auto msg = "Error: GPU does not support INT8 precision";
            spdlog::error(msg);
            throw std::runtime_error(msg);
        }

        // Ensure the user has provided path to calibration data directory
        if (m_options.calibrationDataDirectoryPath.empty()) {
            auto msg = "Error: If INT8 precision is selected, must provide path to "
                       "calibration data directory to Engine::build method";
            throw std::runtime_error(msg);
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
    const auto enginePath = std::filesystem::path(m_options.engineFileDir) / engineName;
    std::ofstream outfile(enginePath, std::ofstream::binary);
    outfile.write(reinterpret_cast<const char *>(plan->data()), plan->size());
    spdlog::info("Success, saved engine to {}", enginePath.string());

    Util::checkCudaErrorCode(cudaStreamDestroy(profileStream));
    return true;
}