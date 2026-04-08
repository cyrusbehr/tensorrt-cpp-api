#pragma once
#include <filesystem>
#include <spdlog/spdlog.h>

template <typename T>
void Engine<T>::transformOutput(std::vector<std::vector<std::vector<T>>> &input, std::vector<std::vector<T>> &output) {
    if (input.size() != 1) {
        auto msg = "The feature vector has incorrect dimensions!";
        spdlog::error(msg);
        throw std::logic_error(msg);
    }

    output = std::move(input[0]);
}

template <typename T> void Engine<T>::transformOutput(std::vector<std::vector<std::vector<T>>> &input, std::vector<T> &output) {
    if (input.size() != 1 || input[0].size() != 1) {
        auto msg = "The feature vector has incorrect dimensions!";
        spdlog::error(msg);
        throw std::logic_error(msg);
    }

    output = std::move(input[0][0]);
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
    //? this function position image at 0,0
    //? if you need center image, do change here and in post process indexes logic in your implementation
    re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
    return out;
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

template <typename T> std::string Engine<T>::serializeEngineOptions(const Options &options, const std::string &onnxModelPath) {
    const auto filenamePos = onnxModelPath.find_last_of('/') + 1;
    std::string engineName = onnxModelPath.substr(filenamePos, onnxModelPath.find_last_of('.') - filenamePos) + ".engine";

    // Add the GPU device name to the file to ensure that the model is only used
    // on devices with the exact same GPU
    std::vector<std::string> deviceNames;
    getDeviceNames(deviceNames);

    if (static_cast<size_t>(options.deviceIndex) >= deviceNames.size()) {
        auto msg = "Error, provided device index is out of range!";
        spdlog::error(msg);
        throw std::runtime_error(msg);
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
    engineName += "." + std::to_string(options.minInputWidth);
    engineName += "." + std::to_string(options.optInputWidth);
    engineName += "." + std::to_string(options.maxInputWidth);

    spdlog::info("Engine name: {}", engineName);
    return engineName;
}

template <typename T>
cv::cuda::GpuMat Engine<T>::blobFromGpuMats(const std::vector<cv::cuda::GpuMat> &batchInput, const std::array<float, 3> &subVals,
                                            const std::array<float, 3> &divVals, bool normalize, bool swapRB) {
   
    CHECK(!batchInput.empty())
    CHECK(batchInput[0].channels() == 3)

    //! rewrite was tested on real world nn
    //! nn behave like wanted, but dataset is different to real data
    //! code logic produce expected result, but expected result is not final valid output (it jumps around target position)
    //! if you see error in rewrite, note it and change

    bool const&  invert_channels = swapRB;

    std::vector<cv::cuda::GpuMat> const& individualMats = batchInput;
    auto const& indMref = individualMats[0];
    size_t size = 
            individualMats.size()
            *
            indMref.cols //? assumed mats have valid size yet
            *
            indMref.rows
            *
            3 //? assumed rgb
        ;

    //? there gpu_dst is JUST memory, it is not valid cv mat
    //? you cannot have size > 1<<31 since mats size is int variable
    //? you cant set (height,width), you have to set (1,height*width) //* because of padding //%% if you really want bigger number, if depth (or other dim) < max allowed by opencv, use CV_32F(depth); result layout will be same
    //cv::cuda::GpuMat gpu_dst(1, batchInput[0].rows * batchInput[0].cols * batchInput.size(), CV_8UC3);
    cv::cuda::GpuMat gpu_dst(1, (int)size, CV_32FC(1));
    const size_t result_size_one_batch = size_t(indMref.cols) * size_t(indMref.rows);

    for(uint batch_index = 0; batch_index < individualMats.size(); ++batch_index)
    {
        cv::cuda::GpuMat mfloat;
        if (normalize) {
            // [0.f, 1.f]
            individualMats[batch_index].convertTo(mfloat, CV_32FC3, 1.f / 255.f); //NOLINT magic 255.f
        } else {
            // [0.f, 255.f]
            individualMats[batch_index].convertTo(mfloat, CV_32FC3);
        }
        // Apply scaling and mean subtraction
        //NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index) //is it some sort of a joke? it is constant lol
        cv::cuda::subtract(mfloat, cv::Scalar(subVals[invert_channels ? 2 : 0], subVals[1], subVals[invert_channels ? 0 : 2]), mfloat, cv::noArray(), -1);
        //NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index) //is it some sort of a joke? it is constant lol
        cv::cuda::divide(mfloat, cv::Scalar(divVals[invert_channels ? 2 : 0], divVals[1], divVals[invert_channels ? 0 : 2]), mfloat, 1, -1);

        ///
        //////
        ///

        //? basically what happening here:
        //? opencv provide split function, that should copy channels to 3 different regions efficiently
        //? here this regions are pointers instead of standalone buffers
        //? so basically just split image's channels to 3 given pointers
        std::vector<cv::cuda::GpuMat> input_channels{
            cv::cuda::GpuMat(
                    mfloat.rows,
                    mfloat.cols,
                    CV_32FC1,
                    gpu_dst.ptr<float>() //* note that it is float pointer
                        +//NOLINT pointer arithmetic
                        (result_size_one_batch * (invert_channels ? 2 : 0) /*R*/)
                        + //NOLINT pointer arithmetic
                        (result_size_one_batch * 3 * batch_index)
                ),
            cv::cuda::GpuMat(
                    mfloat.rows,
                    mfloat.cols,
                    CV_32FC1,
                    gpu_dst.ptr<float>() 
                        +//NOLINT pointer arithmetic
                        (result_size_one_batch *1 /*G*/)
                        + //NOLINT pointer arithmetic
                        (result_size_one_batch * 3 * batch_index)
                ),
            cv::cuda::GpuMat(
                    mfloat.rows,
                    mfloat.cols,
                    CV_32FC1,
                    gpu_dst.ptr<float>() 
                        +//NOLINT pointer arithmetic
                        (result_size_one_batch * (invert_channels ? 0 : 2) /*B*/)
                        + //NOLINT pointer arithmetic
                        (result_size_one_batch * 3 * batch_index)
                )
        };
        cv::cuda::split(mfloat, input_channels);  // HWC -> CHW
        // ^^^ by batch index it also fill blocks by batch dimension
    }//for all batches

    return gpu_dst;

    //* end of function, that was final return

    /*
     //? original for compare
     
     //width here == result_size_batch
    size_t width = batchInput[0].cols * batchInput[0].rows;
    if (swapRB) {
        for (size_t img = 0; img < batchInput.size(); ++img) {
            std::vector<cv::cuda::GpuMat> input_channels{
                cv::cuda::GpuMat(batchInput[0].rows, batchInput[0].cols, CV_8U, &(gpu_dst.ptr()[width * 2 + width * 3 * img])),
                cv::cuda::GpuMat(batchInput[0].rows, batchInput[0].cols, CV_8U, &(gpu_dst.ptr()[width + width * 3 * img])),
                cv::cuda::GpuMat(batchInput[0].rows, batchInput[0].cols, CV_8U, &(gpu_dst.ptr()[0 + width * 3 * img]))};
            cv::cuda::split(batchInput[img], input_channels); // HWC -> CHW
        }
    } else {
        for (size_t img = 0; img < batchInput.size(); ++img) {
            std::vector<cv::cuda::GpuMat> input_channels{
                cv::cuda::GpuMat(batchInput[0].rows, batchInput[0].cols, CV_8U, &(gpu_dst.ptr()[0 + width * 3 * img])),
                cv::cuda::GpuMat(batchInput[0].rows, batchInput[0].cols, CV_8U, &(gpu_dst.ptr()[width + width * 3 * img])),
                cv::cuda::GpuMat(batchInput[0].rows, batchInput[0].cols, CV_8U, &(gpu_dst.ptr()[width * 2 + width * 3 * img]))};
            cv::cuda::split(batchInput[img], input_channels); // HWC -> CHW
        }
    }
    */
}

template <typename T> void Engine<T>::clearGpuBuffers() {
    if (!m_buffers.empty()) {
        // Free GPU memory of outputs
        const auto numInputs = m_inputDims.size();
        //FIX REQUIRE rewrite to work with trt10
        //fix here assumed that position is <input0,input1,output0,output1>, but it isn't in trt10
        //fix and that repeats everywhere!
        for (int32_t outputBinding = numInputs; outputBinding < m_engine->getNbIOTensors(); ++outputBinding) {
            Util::checkCudaErrorCode(cudaFree(m_buffers[outputBinding]));
        }
        m_buffers.clear();
    }
}
