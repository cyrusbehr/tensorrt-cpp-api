#include "engine.h"
#include <opencv2/opencv.hpp>
#include <chrono>

typedef std::chrono::high_resolution_clock Clock;


int main() {
    Options options;
    options.optBatchSizes = {2, 4, 8};

    Engine engine(options);

    // TODO: Specify your model here.
    // Must specify a dynamic batch size when exporting the model from onnx.
    const std::string onnxModelpath = "../model.dynamic_batch.onnx";

    bool succ = engine.build(onnxModelpath);
    if (!succ) {
        throw std::runtime_error("Unable to build TRT engine.");
    }

    succ = engine.loadNetwork();
    if (!succ) {
        throw std::runtime_error("Unable to load TRT engine.");
    }

    const size_t batchSize = 4;
    std::vector<cv::Mat> images;


    const std::string inputImage = "../img.jpg";
    auto img = cv::imread(inputImage);
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    for (size_t i = 0; i < batchSize; ++i) {
        images.push_back(img);
    }

    // Discard the first inference time as it takes longer
    std::vector<std::vector<float>> featureVectors;
    succ = engine.runInference(images, featureVectors);
    if (!succ) {
        throw std::runtime_error("Unable to run inference.");
    }

    size_t numIterations = 100;

    auto t1 = Clock::now();
    for (size_t i = 0; i < numIterations; ++i) {
        featureVectors.clear();
        engine.runInference(images, featureVectors);
    }
    auto t2 = Clock::now();
    double totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

    std::cout << "Success! Average time per inference: " << totalTime / numIterations / static_cast<float>(images.size()) <<
    " ms, for batch size of: " << images.size() << std::endl;

    return 0;
}
