#include "engine.h"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <random>

using namespace nvinfer1;
using namespace Util;

void Logger::log(Severity severity, const char *msg) noexcept {
    switch (severity) {
        case Severity::kVERBOSE:
            spdlog::debug(msg);
            break;
        case Severity::kINFO:
            spdlog::info(msg);
            break;
        case Severity::kWARNING:
            spdlog::warn(msg);
            break;
        case Severity::kERROR:
            spdlog::error(msg);
            break;
        case Severity::kINTERNAL_ERROR:
            spdlog::critical(msg);
            break;
        default:
            spdlog::info("Unexpected severity level");
    }
}

Int8EntropyCalibrator2::Int8EntropyCalibrator2(int32_t batchSize, int32_t inputW, int32_t inputH, const std::string &calibDataDirPath,
                                               const std::string &calibTableName, const std::string &inputBlobName,
                                               const std::array<float, 3> &subVals, const std::array<float, 3> &divVals, bool normalize,
                                               bool readCache)
    : m_batchSize(batchSize), m_inputW(inputW), m_inputH(inputH), m_imgIdx(0), m_calibTableName(calibTableName),
      m_inputBlobName(inputBlobName), m_subVals(subVals), m_divVals(divVals), m_normalize(normalize), m_readCache(readCache) {

    // Allocate GPU memory to hold the entire batch
    m_inputCount = 3 * inputW * inputH * batchSize;
    checkCudaErrorCode(cudaMalloc(&m_deviceInput, m_inputCount * sizeof(float)));

    // Read the name of all the files in the specified directory.
    if (!doesFileExist(calibDataDirPath)) {
        auto msg = "Error, directory at provided path does not exist: " + calibDataDirPath;
        spdlog::error(msg);
        throw std::runtime_error(msg);
    }

    m_imgPaths = getFilesInDirectory(calibDataDirPath);
    if (m_imgPaths.size() < static_cast<size_t>(batchSize)) {
        auto msg = "Error, there are fewer calibration images than the specified batch size!";
        spdlog::error(msg);
        throw std::runtime_error(msg);
    }

    // Randomize the calibration data
    auto rd = std::random_device{};
    auto rng = std::default_random_engine{rd()};
    std::shuffle(std::begin(m_imgPaths), std::end(m_imgPaths), rng);
}

int32_t Int8EntropyCalibrator2::getBatchSize() const noexcept {
    // Return the batch size
    return m_batchSize;
}

bool Int8EntropyCalibrator2::getBatch(void **bindings, const char **names, int32_t nbBindings) noexcept {
    // This method will read a batch of images into GPU memory, and place the
    // pointer to the GPU memory in the bindings variable.

    if (m_imgIdx + m_batchSize > static_cast<int>(m_imgPaths.size())) {
        // There are not enough images left to satisfy an entire batch
        return false;
    }

    // Read the calibration images into memory for the current batch
    std::vector<cv::cuda::GpuMat> inputImgs;
    for (int i = m_imgIdx; i < m_imgIdx + m_batchSize; i++) {
        spdlog::info("Reading image {}: {}", i, m_imgPaths[i]);
        auto cpuImg = cv::imread(m_imgPaths[i]);
        if (cpuImg.empty()) {
            spdlog::error("Fatal error: Unable to read image at path: " + m_imgPaths[i]);
            return false;
        }

        cv::cuda::GpuMat gpuImg;
        gpuImg.upload(cpuImg);
        //cv::cuda::cvtColor(gpuImg, gpuImg, cv::COLOR_BGR2RGB);

        // TODO: Define any preprocessing code here, such as resizing
        auto resized = Engine<float>::resizeKeepAspectRatioPadRightBottom(gpuImg, m_inputH, m_inputW);

        inputImgs.emplace_back(std::move(resized));
    }

    // Convert the batch from NHWC to NCHW
    // ALso apply normalization, scaling, and mean subtraction
    auto mfloat = Engine<float>::blobFromGpuMats(inputImgs, m_subVals, m_divVals, m_normalize, true);
    auto *dataPointer = mfloat.ptr<void>();

    // Copy the GPU buffer to member variable so that it persists
    checkCudaErrorCode(cudaMemcpyAsync(m_deviceInput, dataPointer, m_inputCount * sizeof(float), cudaMemcpyDeviceToDevice));

    m_imgIdx += m_batchSize;
    if (std::string(names[0]) != m_inputBlobName) {
        spdlog::error("Error: Incorrect input name provided!");
        return false;
    }
    bindings[0] = m_deviceInput;
    return true;
}

void const *Int8EntropyCalibrator2::readCalibrationCache(size_t &length) noexcept {
    spdlog::info("Searching for calibration cache: {}", m_calibTableName);
    m_calibCache.clear();
    std::ifstream input(m_calibTableName, std::ios::binary);
    input >> std::noskipws;
    if (m_readCache && input.good()) {
        spdlog::info("Reading calibration cache: {}", m_calibTableName);
        std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(), std::back_inserter(m_calibCache));
    }
    length = m_calibCache.size();
    return length ? m_calibCache.data() : nullptr;
}

void Int8EntropyCalibrator2::writeCalibrationCache(const void *ptr, std::size_t length) noexcept {
    spdlog::info("Writing calibration cache: {}", m_calibTableName);
    spdlog::info("Calibration cache size: {} bytes", length);
    std::ofstream output(m_calibTableName, std::ios::binary);
    output.write(reinterpret_cast<const char *>(ptr), length);
}

Int8EntropyCalibrator2::~Int8EntropyCalibrator2() { checkCudaErrorCode(cudaFree(m_deviceInput)); };
