#include "engine.h"
#include <opencv2/opencv.hpp>
#include <chrono>

typedef std::chrono::high_resolution_clock Clock;

cv::cuda::GpuMat resizeKeepAspectRatioPadRightBottom(const cv::cuda::GpuMat& input, size_t newDim, const cv::Scalar& bgcolor = cv::Scalar(0, 0, 0)) {
    float r = std::min(newDim / (input.cols * 1.0), newDim / (input.rows * 1.0));
    int unpad_w = r * input.cols;
    int unpad_h = r * input.rows;
    cv::cuda::GpuMat re(unpad_h, unpad_w, CV_8UC3);
    cv::cuda::resize(input, re, re.size());
    cv::cuda::GpuMat out(newDim, newDim, CV_8UC3, bgcolor);
    re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
    return out;
}

int main() {
    // Specify our GPU inference configuration options
    Options options;
    // TODO: If your model only supports a static batch size
    options.doesSupportDynamicBatchSize = false;
    options.precision = Precision::FP16; // Use fp16 precision for faster inference.
    options.optBatchSizes = {2, 4, 8};

    Engine engine(options);

    // TODO: Specify your model here.
    // Must specify a dynamic batch size when exporting the model from onnx.
    // If model only specifies a static batch size, must set the above variable doesSupportDynamicBatchSize to false.
    const std::string onnxModelpath = "../models/scrfd_10g_F0E1_TensorRT.onnx";

    bool succ = engine.build(onnxModelpath);
    if (!succ) {
        throw std::runtime_error("Unable to build TRT engine.");
    }

    succ = engine.loadNetwork();
    if (!succ) {
        throw std::runtime_error("Unable to load TRT engine.");
    }

    size_t batchSize = 4;
    if (!options.doesSupportDynamicBatchSize) {
        batchSize = 1;
    }


    const std::string inputImage = "/home/cyrus/work/c-sdks/truefaceSDK/benchmarks/images/headshot.jpg";
    auto cpuImg = cv::imread(inputImage);
    if (cpuImg.empty()) {
        throw std::runtime_error("Unable to read image at path: " + inputImage);
    }

    cv::cvtColor(cpuImg, cpuImg, cv::COLOR_BGR2RGB);

    cv::cuda::GpuMat img;
    img.upload(cpuImg);

    // Apply our static resize algorithm.
    img = resizeKeepAspectRatioPadRightBottom(img, engine.getInputHeight());

    if (img.cols != engine.getInputWidth() ||
        img.rows != engine.getInputHeight()) {
        std::cout << "The image is not the right size of the model!" << std::endl;
        std::cout << "The model expects: (" << engine.getInputHeight() << "x" << engine.getInputWidth() << ")" << std::endl;
        std::cout << "Provided input image: (" << img.rows << "x" << img.cols << ")" << std::endl;
        std::cout << "You must either resize your image, add padding, or source images of the correct input size" << std::endl;
        // TODO: At this point, you'd want to resize the image appropriately.
        // I have deliberately left this part empty as it depends on your implementation.
        // You may want to resize while maintaining the aspect ratio (with use of padding).
        // You may want to only add padding without resizing
        // Or you may want to only use inputs which are already sized correctly (this is the case
        // for us as we are using face chips from a face detector pipeline).
        return -1;
    }


    std::vector<cv::cuda::GpuMat> images;
    for (size_t i = 0; i < batchSize; ++i) {
        images.push_back(img);
    }

    std::array<float, 3> subVals {0.5f, 0.5f, 0.5f};
    std::array<float, 3> divVals {0.5f, 0.5f, 0.5f};

    // Discard the first inference time as it takes longer
    std::vector<std::vector<std::vector<float>>> featureVectors;
    succ = engine.runInference(images, featureVectors, subVals, divVals);
    if (!succ) {
        throw std::runtime_error("Unable to run inference.");
    }

    size_t numIterations = 100;

    auto t1 = Clock::now();
    for (size_t i = 0; i < numIterations; ++i) {
        featureVectors.clear();
        engine.runInference(images, featureVectors, subVals, divVals);
    }
    auto t2 = Clock::now();
    double totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

    std::cout << "Success! Average time per inference: " << totalTime / numIterations / static_cast<float>(images.size()) <<
    " ms, for batch size of: " << images.size() << std::endl;

    // Print the feature vector
//    for (const auto& e: featureVectors[0]) {
//        std::cout << e << " ";
//    }
//    std::cout << "\n" << std::flush;

    return 0;
}
