#include "engine.h"
#include <opencv2/opencv.hpp>
#include <chrono>

typedef std::chrono::high_resolution_clock Clock;

int main() {
    // Specify our GPU inference configuration options
    Options options;
    // TODO: If your model only supports a static batch size
    options.doesSupportDynamicBatchSize = false;
    options.precision = Precision::FP16; // Use fp16 precision for faster inference.

    if (options.doesSupportDynamicBatchSize) {
        options.optBatchSize = 4;
        options.maxBatchSize = 16;
    }

    Engine engine(options);

    // TODO: Specify your model here.
    // Must specify a dynamic batch size when exporting the model from onnx.
    // If model only specifies a static batch size, must set the above variable doesSupportDynamicBatchSize to false.
    const std::string onnxModelpath = "../models/arcfaceresnet100-8.onnx";

    bool succ = engine.build(onnxModelpath);
    if (!succ) {
        throw std::runtime_error("Unable to build TRT engine.");
    }

    succ = engine.loadNetwork();
    if (!succ) {
        throw std::runtime_error("Unable to load TRT engine.");
    }

    // Lets use a batch size which matches that which we set the Options::optBatchSize option
    size_t batchSize = options.optBatchSize;

    const std::string inputImage = "../inputs/face_chip.jpg";
    auto cpuImg = cv::imread(inputImage);
    if (cpuImg.empty()) {
        throw std::runtime_error("Unable to read image at path: " + inputImage);
    }

    // The model expects RGB input
    cv::cvtColor(cpuImg, cpuImg, cv::COLOR_BGR2RGB);

    // Upload to GPU memory
    cv::cuda::GpuMat img;
    img.upload(cpuImg);

    // TODO: If the model expects a different input size, resize it here.
    // You can choose to resize by scaling, adding padding, or a conbination of the two in order to maintain the aspect ratio
    // You can use the Engine::resizeKeepAspectRatioPadRightBottom to resize to a square while maintain the aspect ratio (adds padding where necessary to achieve this).
    // If you are running the sample code using the suggested model, then the input image already has the correct size.

    if (img.cols != engine.getInputWidth() ||
        img.rows != engine.getInputHeight()) {
        std::cout << "The image is not the right size of the model!" << std::endl;
        std::cout << "The model expects: (" << engine.getInputHeight() << "x" << engine.getInputWidth() << ")" << std::endl;
        std::cout << "Provided input image: (" << img.rows << "x" << img.cols << ")" << std::endl;
        std::cout << "You must either resize your image, add padding, or source images of the correct input size" << std::endl;
        // TODO: This means you forgot to resize your image.
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

    // Define our preprocessing code
    // Default values are between 0 --> 1
    // The following values with transform the input range to -1 --> 1
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

    // Print the feature vectors
    for (int batch = 0; batch < featureVectors.size(); ++batch) {
        for (int outputNum = 0; outputNum < featureVectors[batch].size(); ++outputNum) {
            std::cout << "Batch " << batch << ", " << "output " << outputNum << std::endl;
            int i = 0;
            for (const auto &e:  featureVectors[batch][outputNum]) {
                std::cout << e << " ";
                if (++i == 10) {
                    std::cout << "...";
                    break;
                }
            }
            std::cout << "\n" << std::endl;
        }
    }

    return 0;
}
