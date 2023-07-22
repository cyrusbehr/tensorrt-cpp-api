#include "engine.h"
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <chrono>

int main(int argc, char *argv[]) {
    // Parse the command line arguments
    // Must pass the model path as a command line argument to the executable
    if (argc < 2) {
        std::cout << "Error: Must specify the model path" << std::endl;
        std::cout << "Usage: " << argv[0] << " /path/to/onnx/model.onnx" << std::endl;
        return -1;
    }

    if (argc > 3) {
        std::cout << "Error: Too many arguments provided" << std::endl;
        std::cout << "Usage: " << argv[0] << " /path/to/onnx/model.onnx" << std::endl;
    }

    // Ensure the onnx model exists
    const std::string onnxModelPath = argv[1];
    if (!Util::doesFileExist(onnxModelPath)) {
        std::cout << "Error: Unable to find file at path: " << onnxModelPath << std::endl;
        return -1;
    }

    // Specify our GPU inference configuration options
    Options options;
    // Specify what precision to use for inference
    // FP16 is approximately twice as fast as FP32.
    options.precision = Precision::FP16;
    // If using INT8 precision, must specify path to directory containing calibration data.
    options.calibrationDataDirectoryPath = "";
    // If the model does not support dynamic batch size, then the below two parameters must be set to 1.
    // Specify the batch size to optimize for.
    options.optBatchSize = 1;
    // Specify the maximum batch size we plan on running.
    options.maxBatchSize = 1;

    Engine engine(options);

    // Define our preprocessing code
    // The default Engine::build method will normalize values between [0.f, 1.f]
    // Setting the normalize flag to false will leave values between [0.f, 255.f] (some converted models may require this).

    // For our YoloV8 model, we need the values to be normalized between [0.f, 1.f] so we use the following params
    std::array<float, 3> subVals {0.f, 0.f, 0.f};
    std::array<float, 3> divVals {1.f, 1.f, 1.f};
    bool normalize = true;
    // Note, we could have also used the default values.

    // If the model requires values to be normalized between [-1.f, 1.f], use the following params:
    //    subVals = {0.5f, 0.5f, 0.5f};
    //    divVals = {0.5f, 0.5f, 0.5f};
    //    normalize = true;
    
    // Build the onnx model into a TensorRT engine file.
    bool succ = engine.build(onnxModelPath, subVals, divVals, normalize);
    if (!succ) {
        throw std::runtime_error("Unable to build TRT engine.");
    }

    // Load the TensorRT engine file from disk
    succ = engine.loadNetwork();
    if (!succ) {
        throw std::runtime_error("Unable to load TRT engine.");
    }

    // Read the input image
    // TODO: You will need to read the input image required for your model
    const std::string inputImage = "../inputs/team.jpg";
    auto cpuImg = cv::imread(inputImage);
    if (cpuImg.empty()) {
        throw std::runtime_error("Unable to read image at path: " + inputImage);
    }

    // Upload the image GPU memory
    cv::cuda::GpuMat img;
    img.upload(cpuImg);

    // The model expects RGB input
    cv::cuda::cvtColor(img, img, cv::COLOR_BGR2RGB);

    // In the following section we populate the input vectors to later pass for inference
    const auto& inputDims = engine.getInputDims();
    std::vector<std::vector<cv::cuda::GpuMat>> inputs;

    // Let's use a batch size which matches that which we set the Options.optBatchSize option
    size_t batchSize = options.optBatchSize;

    // TODO:
    // For the sake of the demo, we will be feeding the same image to all the inputs
    // You should populate your inputs appropriately.
    for (const auto & inputDim : inputDims) { // For each of the model inputs...
        std::vector<cv::cuda::GpuMat> input;
        for (size_t j = 0; j < batchSize; ++j) { // For each element we want to add to the batch...
            // TODO:
            // You can choose to resize by scaling, adding padding, or a combination of the two in order to maintain the aspect ratio
            // You can use the Engine::resizeKeepAspectRatioPadRightBottom to resize to a square while maintain the aspect ratio (adds padding where necessary to achieve this).
            auto resized = Engine::resizeKeepAspectRatioPadRightBottom(img, inputDim.d[1], inputDim.d[2]);
            // You could also perform a resize operation without maintaining aspect ratio with the use of padding by using the following instead:
//            cv::cuda::resize(img, resized, cv::Size(inputDim.d[2], inputDim.d[1])); // TRT dims are (height, width) whereas OpenCV is (width, height)
            input.emplace_back(std::move(resized));
        }
        inputs.emplace_back(std::move(input));
    }

    // Warm up the network before we begin the benchmark
    std::cout << "\nWarming up the network..." << std::endl;
    std::vector<std::vector<std::vector<float>>> featureVectors;
    for (int i = 0; i < 100; ++i) {
        succ = engine.runInference(inputs, featureVectors);
        if (!succ) {
            throw std::runtime_error("Unable to run inference.");
        }
    }

    // Benchmark the inference time
    size_t numIterations = 1000;
    std::cout << "Warmup done. Running benchmarks (" << numIterations << " iterations)...\n" << std::endl;
    preciseStopwatch stopwatch;
    for (size_t i = 0; i < numIterations; ++i) {
        featureVectors.clear();
        engine.runInference(inputs, featureVectors);
    }
    auto totalElapsedTimeMs = stopwatch.elapsedTime<float, std::chrono::milliseconds>();
    auto avgElapsedTimeMs = totalElapsedTimeMs / numIterations / static_cast<float>(inputs[0].size());

    std::cout << "Benchmarking complete!" << std::endl;
    std::cout << "======================" << std::endl;
    std::cout << "Avg time per sample: " << std::endl;
    std::cout << avgElapsedTimeMs << " ms" << std::endl;
    std::cout << "Batch size: " << std::endl;
    std::cout << inputs[0].size() << std::endl;
    std::cout << "Avg FPS: " << std::endl;
    std::cout << static_cast<int>(1000 / avgElapsedTimeMs) << " fps" << std::endl;
    std::cout << "======================\n" << std::endl;

    // Print the feature vectors
    for (size_t batch = 0; batch < featureVectors.size(); ++batch) {
        for (size_t outputNum = 0; outputNum < featureVectors[batch].size(); ++outputNum) {
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

    // TODO: If your model requires post processing (ex. convert feature vector into bounding boxes) then you would do so here.

    return 0;
}
