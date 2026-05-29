// Image classification (ImageNet) with tensorrt_cpp_api.
//
// Pipeline: decode (stb, CPU) -> upload HWC-uint8 to GPU -> fused resize/normalize/NCHW
// (preproc sublib) -> infer -> softmax + top-5. Works with any ImageNet ONNX classifier whose
// input is [1,3,H,W] and output is [1,N] logits (e.g. MobileNetV2, ResNet50).
//
// Usage: classification <model.onnx|engine> <image> [topk]

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

#include "../common/image_io.h"

using namespace trtcpp;

int main(int argc, char **argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <model.onnx|engine> <image> [topk]\n", argv[0]);
        return 2;
    }
    const std::string modelPath = argv[1];
    const std::string imagePath = argv[2];
    const int topk = argc > 3 ? std::atoi(argv[3]) : 5;

    BuildOptions bo;
    bo.precision = Precision::kFp16;
    bo.engineCacheDir = "engines";
    auto engine = EngineBuilder{}.buildAndLoad(modelPath, bo);
    if (!engine) {
        std::fprintf(stderr, "engine: %s\n", engine.status().message().c_str());
        return 1;
    }

    const std::string inName = engine->inputNames().front();
    auto inShape = engine->tensorShape(inName).value(); // [1,3,H,W]
    const int inH = static_cast<int>(inShape[2]);
    const int inW = static_cast<int>(inShape[3]);

    examples::Image img = examples::decodeImage(imagePath);
    if (img.empty()) {
        std::fprintf(stderr, "could not read image: %s\n", imagePath.c_str());
        return 1;
    }

    Stream stream;
    auto src = examples::uploadHWC(img, stream);
    if (!src) {
        std::fprintf(stderr, "upload: %s\n", src.status().message().c_str());
        return 1;
    }
    auto dst = Tensor::allocate(DType::kFloat32, Shape{1, 3, inH, inW}, Device::kCuda).value();
    if (auto s = preproc::letterboxToTensor(src->view(), dst.view(), examples::imagenetSpec(), stream); !s) {
        std::fprintf(stderr, "preproc: %s\n", s.message().c_str());
        return 1;
    }

    auto output = engine->inferSingle({{inName, dst.view()}}, stream);
    if (!output) {
        std::fprintf(stderr, "infer: %s\n", output.status().message().c_str());
        return 1;
    }
    auto host = output->toHost(stream).value();
    auto logits = host.as<float>().value(); // [1, N]

    // numerically-stable softmax + top-k
    const float maxLogit = *std::max_element(logits.begin(), logits.end());
    std::vector<float> probs(logits.size());
    double sum = 0.0;
    for (std::size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp(logits[i] - maxLogit);
        sum += probs[i];
    }
    std::vector<int> idx(logits.size());
    std::iota(idx.begin(), idx.end(), 0);
    const int k = std::min<int>(topk, static_cast<int>(idx.size()));
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) { return probs[a] > probs[b]; });

    std::printf("top-%d of %zu classes:\n", k, logits.size());
    for (int i = 0; i < k; ++i) {
        std::printf("  class %4d  %6.2f%%\n", idx[i], 100.0 * probs[idx[i]] / sum);
    }
    return 0;
}
