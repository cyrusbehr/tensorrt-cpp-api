#include <engine.h>

void runBenchmark(const EngineOptions& engineOptions, const cv::cuda::GpuMat& img, Engine& engine)
{
    // In the following section we populate the input vectors to later pass for inference
    const auto& inputDims = engine.getInputDims();
    std::vector<std::vector<cv::cuda::GpuMat>> inputs;

    // Let's use a batch size which matches that which we set the Options.optBatchSize option
    const size_t batchSize = engineOptions.optBatchSize;

    // TODO:
    // For the sake of the demo, we will be feeding the same image to all the inputs
    // You should populate your inputs appropriately.
    for (const auto& inputDim : inputDims) { // For each of the model inputs...
        std::vector<cv::cuda::GpuMat> input;
        for (size_t j = 0; j < batchSize; ++j) { // For each element we want to add to the batch...
            // TODO:
            // You can choose to resize by scaling, adding padding, or a combination of the two in order to maintain the aspect ratio
            // You can use the Engine::resizeKeepAspectRatioPadRightBottom to resize to a square while maintain the aspect ratio (adds padding where necessary to achieve this).
            auto resized = Engine::resizeKeepAspectRatioPadRightBottom(img, inputDim.d[1], inputDim.d[2]);
            // You could also perform a resize operation without maintaining aspect ratio with the use of padding by using the following instead:
            // cv::cuda::resize(img, resized, cv::Size(inputDim.d[2], inputDim.d[1])); // TRT dims are (height, width) whereas OpenCV is (width, height)
            input.emplace_back(std::move(resized));
        }
        inputs.emplace_back(std::move(input));
    }

    // Warm up the network before we begin the benchmark
    std::cout << "\nWarming up the network..." << std::endl;
    std::vector<std::vector<std::vector<float>>> featureVectors;
    for (int i = 0; i < 100; ++i) {
        bool succ = engine.runInference(inputs, featureVectors);
        if (!succ) {
            throw std::runtime_error("Unable to run inference.");
        }
    }

    // Benchmark the inference time
    constexpr size_t numIterations = 1000;
    std::cout << "Warmup done. Running benchmarks (" << numIterations << " iterations)...\n" << std::endl;
    const preciseStopwatch stopwatch;
    for (size_t i = 0; i < numIterations; ++i) {
        featureVectors.clear();
        engine.runInference(inputs, featureVectors);
    }
    const auto totalElapsedTimeMs = stopwatch.elapsedTime<float, std::chrono::milliseconds>();
    const auto avgElapsedTimeMs = totalElapsedTimeMs / numIterations / static_cast<float>(inputs[0].size());

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
            for (const auto& e : featureVectors[batch][outputNum]) {
                std::cout << e << " ";
                if (++i == 10) {
                    std::cout << "...";
                    break;
                }
            }
            std::cout << "\n" << std::endl;
        }
    }
}

int main(int argc, char* argv[])
{
    try
    {
        // View your GPU indexes
        std::vector<std::string> gpuDeviceNames;
        EngineUtil::getGpuDeviceNames(gpuDeviceNames);
        std::cout << "The computer has " << gpuDeviceNames.size() << " GPUs, and their names are as follows: \n";
        for (size_t i = 0; i < gpuDeviceNames.size(); ++i)
        {
            std::cout << "- Index " << i << ": " << gpuDeviceNames.at(i) << "\n";
        }

        // Build ONNX to TensorRT engine
        const std::string yolov8nOnnxPath = "E:/Github/tensorrt-cpp-api/build/Benchmark/Release/yolov8n.onnx";

        EngineOptions yolov8nINT8Options;
        yolov8nINT8Options.precision = Precision::INT8;
        yolov8nINT8Options.optBatchSize = 1;
        yolov8nINT8Options.maxBatchSize = 1;
        yolov8nINT8Options.deviceIndex = 0;
        yolov8nINT8Options.calibrationBatchSize = 128;
        yolov8nINT8Options.calibrationDataDirectoryPath = "E:/Github/tensorrt-cpp-api/build/Benchmark/Release/CocoVal";
        Engine::build(yolov8nOnnxPath, yolov8nINT8Options);

        EngineOptions yolov8nFP16Options;
        yolov8nFP16Options.precision = Precision::FP16;
        yolov8nFP16Options.optBatchSize = 1;
        yolov8nFP16Options.maxBatchSize = 1;
        yolov8nFP16Options.deviceIndex = 0;
        Engine::build(yolov8nOnnxPath, yolov8nFP16Options);

        // Read image
        const std::string inputImagePath = "../../inputs/team.jpg";
        cv::Mat cpuImage = cv::imread(inputImagePath);
        cv::cvtColor(cpuImage, cpuImage, cv::COLOR_BGR2RGB);

        cv::cuda::GpuMat img;
        img.upload(cpuImage);

        // Load a TensorRT engine
        Engine engine{};
        engine.loadNetwork(EngineUtil::serializeEngineOptions(yolov8nINT8Options, yolov8nOnnxPath), yolov8nINT8Options);

        runBenchmark(yolov8nINT8Options, img, engine);

        // Switch to another one
        engine.loadNetwork(EngineUtil::serializeEngineOptions(yolov8nFP16Options, yolov8nOnnxPath), yolov8nFP16Options);

        runBenchmark(yolov8nFP16Options, img, engine);
    }
    catch (const std::exception& e)
    {
        std::cout << e.what() << "\n";
    }
}
