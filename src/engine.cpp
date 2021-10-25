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

    // Only regenerate the engine file if it has not already been generated for the specified options
    auto engineFilename = serializeEngineOptions(m_options);

    std::cout << "Searching for engine file with name: " << engineFilename << std::endl;

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

    engineName += "." + std::to_string(options.maxBatchSize) + "." + std::to_string(options.optBatchSize) +
            "." + std::to_string(options.maxWorkspaceSize);

    return engineName;
}