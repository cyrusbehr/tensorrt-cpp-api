#include <iostream>
#include <fstream>

#include "engine.h"
#include "NvOnnxParser.h"

void Logger::log(Severity severity, const char *msg) noexcept {
    // Would advise using a proper logging utility such as https://github.com/gabime/spdlog
    // For the sake of this tutorial, will just log to the console.

    // Only log Warnings or more important.
    if (severity <= Severity::kWARNING) {
        std::cout << msg << std::endl;
    }
}

bool Engine::doesFileExist(const std::string &filepath) {
    std::ifstream f(filepath.c_str());
    return f.good();
}

Engine::Engine(const Options &options)
    : m_options(options) {}

bool Engine::build(std::string onnxModelPath) {
    // Create our engine builder.
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(m_logger));
    if (!builder) {
        return false;
    }

    // Set the max supported batch size
    builder->setMaxBatchSize(m_options.maxBatchSize);

    // Define an explicit batch size and then create the network.
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

    auto parsed = parser->parse(buffer.data(), buffer.size());
    if (!parsed) {
        return false;
    }

    // Save the input height, width, and channels.
    // Require this info for inference.
    const auto input = network->getInput(0);
    const auto output = network->getOutput(0);
    const auto inputName = input->getName();
    const auto inputDims = input->getDimensions();
    m_inputC = inputDims.d[1];
    m_inputH = inputDims.d[2];
    m_inputW = inputDims.d[3];
    m_outputL = output->getDimensions().d[1];

    // Only regenerate the engine file if it has not already been generated for the specified options
    m_engineName = serializeEngineOptions(m_options);
    std::cout << "Searching for engine file with name: " << m_engineName << std::endl;

    if (doesFileExist(m_engineName)) {
        std::cout << "Engine found, not regenerating..." << std::endl;
        return true;
    }

    // Was not able to find the engine file, generate...
    std::cout << "Engine not found, generating..." << std::endl;

    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) {
        return false;
    }

    // Specify the optimization profiles and the
    IOptimizationProfile* defaultProfile = builder->createOptimizationProfile();
    defaultProfile->setDimensions(inputName, OptProfileSelector::kMIN, Dims4(1, m_inputC, m_inputH, m_inputW));
    defaultProfile->setDimensions(inputName, OptProfileSelector::kOPT, Dims4(1, m_inputC, m_inputH, m_inputW));
    defaultProfile->setDimensions(inputName, OptProfileSelector::kMAX, Dims4(m_options.maxBatchSize, m_inputC, m_inputH, m_inputW));
    config->addOptimizationProfile(defaultProfile);

    // Specify all the optimization profiles.
    for (const auto& optBatchSize: m_options.optBatchSizes) {
        if (optBatchSize == 1) {
            continue;
        }

        if (optBatchSize > m_options.maxBatchSize) {
            throw std::runtime_error("optBatchSize cannot be greater than maxBatchSize!");
        }

        IOptimizationProfile* profile = builder->createOptimizationProfile();
        profile->setDimensions(inputName, OptProfileSelector::kMIN, Dims4(1, m_inputC, m_inputH, m_inputW));
        profile->setDimensions(inputName, OptProfileSelector::kOPT, Dims4(optBatchSize, m_inputC, m_inputH, m_inputW));
        profile->setDimensions(inputName, OptProfileSelector::kMAX, Dims4(m_options.maxBatchSize, m_inputC, m_inputH, m_inputW));
        config->addOptimizationProfile(profile);
    }

    config->setMaxWorkspaceSize(m_options.maxWorkspaceSize);

    if (m_options.FP16) {
        config->setFlag(BuilderFlag::kFP16);
    }

    // CUDA stream used for profiling by the builder.
    auto profileStream = samplesCommon::makeCudaStream();
    if (!profileStream) {
        return false;
    }
    config->setProfileStream(*profileStream);

    // Build the engine
    std::unique_ptr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan) {
        return false;
    }

    // Write the engine to disk
    std::ofstream outfile(m_engineName, std::ofstream::binary);
    outfile.write(reinterpret_cast<const char*>(plan->data()), plan->size());

    std::cout << "Success, saved engine to " << m_engineName << std::endl;

    return true;
}

bool Engine::loadNetwork() {
    // Read the serialized model from disk
    std::ifstream file(m_engineName, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        throw std::runtime_error("Unable to read engine file");
    }

    std::unique_ptr<IRuntime> runtime{createInferRuntime(m_logger)};
    if (!runtime) {
        return false;
    }


    m_engine = std::unique_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(buffer.data(), buffer.size()));
    if (!m_engine) {
        return false;
    }

    m_context = std::unique_ptr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());
    if (!m_context) {
        return false;
    }

    return true;
}

bool Engine::runInference(const std::vector<cv::Mat> &inputFaceChips, std::vector<std::vector<float>>& featureVectors) {
    Dims4 inputDims = {static_cast<int32_t>(inputFaceChips.size()), m_inputC, m_inputH, m_inputW};
    m_context->setBindingDimensions(0, inputDims);

    if (!m_context->allInputDimensionsSpecified()) {
        throw std::runtime_error("Error, not all input dimensions specified.");
    }

    auto batchSize = static_cast<int32_t>(inputFaceChips.size());
    // Only reallocate buffers if the batch size has changed
    if (m_prevBatchSize != inputFaceChips.size()) {

        m_inputBuff.hostBuffer.resize(inputDims);
        m_inputBuff.deviceBuffer.resize(inputDims);

        Dims2 outputDims {batchSize, m_outputL};
        m_outputBuff.hostBuffer.resize(outputDims);
        m_outputBuff.deviceBuffer.resize(outputDims);

        m_prevBatchSize = batchSize;
    }

    auto* hostDataBuffer = static_cast<float*>(m_inputBuff.hostBuffer.data());

    for (size_t batch = 0; batch < inputFaceChips.size(); ++batch) {
        auto image = inputFaceChips[batch];

        // Preprocess code
        image.convertTo(image, CV_32FC3, 1.f / 255.f);
        cv::subtract(image, cv::Scalar(0.5f, 0.5f, 0.5f), image, cv::noArray(), -1);
        cv::divide(image, cv::Scalar(0.5f, 0.5f, 0.5f), image, 1, -1);

        // NHWC to NCHW conversion
        // NHWC: For each pixel, its 3 colors are stored together in RGB order.
        // For a 3 channel image, say RGB, pixels of the R channel are stored first, then the G channel and finally the B channel.
        // https://user-images.githubusercontent.com/20233731/85104458-3928a100-b23b-11ea-9e7e-95da726fef92.png
        int offset = m_inputC * m_inputH * m_inputW * batch;
        int r = 0 , g = 0, b = 0;
        for (int i = 0; i < m_inputH * m_inputW * m_inputC; ++i) {
            if (i % 3 == 0) {
                hostDataBuffer[offset + r++] = *(reinterpret_cast<float*>(image.data) + i);
            } else if (i % 3 == 1) {
                hostDataBuffer[offset + g++ + 112*112] = *(reinterpret_cast<float*>(image.data) + i);
            } else {
                hostDataBuffer[offset + b++ + 112*112*2] = *(reinterpret_cast<float*>(image.data) + i);
            }
        }
    }

    // Copy from CPU to GPU
    auto ret = cudaMemcpy(m_inputBuff.deviceBuffer.data(), m_inputBuff.hostBuffer.data(), m_inputBuff.hostBuffer.nbBytes(), cudaMemcpyHostToDevice);
    if (ret != 0) {
        return false;
    }

    std::vector<void*> predicitonBindings = {m_inputBuff.deviceBuffer.data(), m_outputBuff.deviceBuffer.data()};

    // Run inference.
    bool status = m_context->executeV2(predicitonBindings.data());
    if (!status) {
        return false;
    }

    // Copy the results back to CPU memory
    ret = cudaMemcpy(m_outputBuff.hostBuffer.data(), m_outputBuff.deviceBuffer.data(), m_outputBuff.deviceBuffer.nbBytes(), cudaMemcpyDeviceToHost);
    if (ret != 0) {
        std::cout << "Unable to copy buffer from GPU back to CPU" << std::endl;
        return false;
    }

    // Copy to output
    for (int batch = 0; batch < batchSize; ++batch) {
        std::vector<float> featureVector;
        featureVector.resize(m_outputL);

        memcpy(featureVector.data(), reinterpret_cast<const char*>(m_outputBuff.hostBuffer.data()) +
        batch * m_outputL * sizeof(float), m_outputL * sizeof(float ));
        featureVectors.emplace_back(std::move(featureVector));
    }

    return true;
}

std::string Engine::serializeEngineOptions(const Options &options) {
    std::string engineName = "trt.engine";
    // Serialize the specified options into the filename
    if (options.FP16) {
        engineName += ".fp16";
    } else {
        engineName += ".fp32";
    }

    engineName += "." + std::to_string(options.maxBatchSize) + ".";
    for (size_t i = 0; i < m_options.optBatchSizes.size(); ++i) {
        engineName += std::to_string(m_options.optBatchSizes[i]);
        if (i != m_options.optBatchSizes.size() - 1) {
            engineName += "_";
        }
    }

    engineName += "." + std::to_string(options.maxWorkspaceSize);

    return engineName;
}

