// Minimal C++ inference-throughput benchmark, the baseline for the Python zero-copy
// perf-parity check (examples/python/benchmark_parity.py). Uses the zero-copy enqueue path with
// preallocated device IO so it measures inference + launch overhead, not allocation.
//
// Output (stdout): a single line  "cpp_latency_ms <value>"  plus a human-readable summary on
// stderr, so the Python harness can parse the number.
//
// Usage: benchmark <model.onnx|engine> [iters=200]

#include <chrono>
#include <cstdio>
#include <string>
#include <unordered_map>

#include <tensorrt_cpp_api/all.h>

using namespace trtcpp;

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model.onnx|engine> [iters=200] [precision=fp16|fp32|int8]\n", argv[0]);
        return 2;
    }
    const std::string modelPath = argv[1];
    const int iters = argc > 2 ? std::atoi(argv[2]) : 200;
    const std::string prec = argc > 3 ? argv[3] : "fp16";

    BuildOptions bo;
    bo.precision = prec == "fp32" ? Precision::kFp32 : prec == "int8" ? Precision::kInt8Qdq : Precision::kFp16;
    bo.engineCacheDir = "engines";
    auto engine = EngineBuilder{}.buildAndLoad(modelPath, bo);
    if (!engine) {
        std::fprintf(stderr, "engine: %s\n", engine.status().message().c_str());
        return 1;
    }

    const std::string inName = engine->inputNames().front();
    const std::string outName = engine->outputNames().front();
    auto inShape = engine->tensorShape(inName).value();
    auto outShape = engine->tensorShape(outName).value();

    Stream stream;
    auto in = Tensor::allocate(DType::kFloat32, inShape, Device::kCuda).value();
    auto out = Tensor::allocate(engine->tensorDType(outName).value(), outShape, Device::kCuda).value();
    std::unordered_map<std::string, TensorView> inputs{{inName, in.view()}};
    std::unordered_map<std::string, TensorView> outputs{{outName, out.view()}};

    auto once = [&]() {
        if (auto s = engine->enqueue(inputs, outputs, stream); !s) {
            std::fprintf(stderr, "enqueue: %s\n", s.message().c_str());
            std::exit(1);
        }
        stream.synchronize();
    };

    for (int i = 0; i < 20; ++i) {
        once(); // warmup
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        once();
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;

    std::fprintf(stderr, "C++  : %.3f ms/infer  (%.0f inf/s) over %d iters\n", ms, 1000.0 / ms, iters);
    std::printf("cpp_latency_ms %.4f\n", ms);
    return 0;
}
