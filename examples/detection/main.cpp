// Object detection with YOLOv8 + tensorrt_cpp_api.
//
// Pipeline: decode (stb) -> upload -> letterbox/normalize/NCHW (preproc) -> infer ->
// decode the [1, 84, 8400] anchor-free output -> class-aware NMS -> draw boxes.
//
// The model is a stock ultralytics export:  yolo export model=yolov8n.pt format=onnx imgsz=640
// Output layout: 84 = 4 box (cx,cy,w,h, in 640-input pixels) + 80 COCO class scores (already
// sigmoid-activated); 8400 anchors. Box coords/scores need no extra activation -- only NMS.
//
// Usage: detection <yolov8n.onnx|engine> <image> [conf] [out.jpg]

#include <algorithm>
#include <array>
#include <cstdio>
#include <string>
#include <vector>

#include "../common/image_io.h"

using namespace trtcpp;

namespace {

constexpr int kInput = 640;
constexpr int kNumClasses = 80;

const std::array<const char *, kNumClasses> kCoco = {
    "person",         "bicycle",    "car",           "motorcycle",    "airplane",     "bus",           "train",
    "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",    "parking meter", "bench",
    "bird",           "cat",        "dog",           "horse",         "sheep",        "cow",           "elephant",
    "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",     "handbag",       "tie",
    "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball",  "kite",          "baseball bat",
    "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",       "wine glass",    "cup",
    "fork",           "knife",      "spoon",         "bowl",          "banana",       "apple",         "sandwich",
    "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",        "donut",         "cake",
    "chair",          "couch",      "potted plant",  "bed",           "dining table", "toilet",        "tv",
    "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",   "microwave",     "oven",
    "toaster",        "sink",       "refrigerator",  "book",          "clock",        "vase",          "scissors",
    "teddy bear",     "hair drier", "toothbrush"};

struct Detection {
    int x0, y0, x1, y1;
    float score;
    int cls;
};

float iou(const Detection &a, const Detection &b) {
    const int ix0 = std::max(a.x0, b.x0), iy0 = std::max(a.y0, b.y0);
    const int ix1 = std::min(a.x1, b.x1), iy1 = std::min(a.y1, b.y1);
    const float inter = static_cast<float>(std::max(0, ix1 - ix0)) * std::max(0, iy1 - iy0);
    const float areaA = static_cast<float>(a.x1 - a.x0) * (a.y1 - a.y0);
    const float areaB = static_cast<float>(b.x1 - b.x0) * (b.y1 - b.y0);
    const float uni = areaA + areaB - inter;
    return uni > 0 ? inter / uni : 0.f;
}

// Greedy class-aware NMS.
std::vector<Detection> nms(std::vector<Detection> dets, float iouThresh) {
    std::sort(dets.begin(), dets.end(), [](const Detection &a, const Detection &b) { return a.score > b.score; });
    std::vector<Detection> keep;
    std::vector<bool> removed(dets.size(), false);
    for (std::size_t i = 0; i < dets.size(); ++i) {
        if (removed[i]) {
            continue;
        }
        keep.push_back(dets[i]);
        for (std::size_t j = i + 1; j < dets.size(); ++j) {
            if (!removed[j] && dets[j].cls == dets[i].cls && iou(dets[i], dets[j]) > iouThresh) {
                removed[j] = true;
            }
        }
    }
    return keep;
}

} // namespace

int main(int argc, char **argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <yolov8n.onnx|engine> <image> [conf=0.25] [out.jpg]\n", argv[0]);
        return 2;
    }
    const std::string modelPath = argv[1];
    const std::string imagePath = argv[2];
    const float conf = argc > 3 ? static_cast<float>(std::atof(argv[3])) : 0.25f;
    const std::string outPath = argc > 4 ? argv[4] : "detections.jpg";

    BuildOptions bo;
    bo.precision = Precision::kFp16;
    bo.engineCacheDir = "engines";
    auto engine = EngineBuilder{}.buildAndLoad(modelPath, bo);
    if (!engine) {
        std::fprintf(stderr, "engine: %s\n", engine.status().message().c_str());
        return 1;
    }

    examples::Image img = examples::decodeImage(imagePath);
    if (img.empty()) {
        std::fprintf(stderr, "could not read image: %s\n", imagePath.c_str());
        return 1;
    }

    // --- inference (the ergonomic core) ---
    Stream stream;
    auto src = examples::uploadHWC(img, stream).value();
    auto dst = Tensor::allocate(DType::kFloat32, Shape{1, 3, kInput, kInput}, Device::kCuda).value();
    preproc::PreprocSpec spec; // letterbox: keep aspect, pad right/bottom, /255 (stb is already RGB)
    spec.keepAspectRatioPad = true;
    spec.scale = {1.f / 255.f, 1.f / 255.f, 1.f / 255.f, 1.f};
    if (auto s = preproc::letterboxToTensor(src.view(), dst.view(), spec, stream); !s) {
        std::fprintf(stderr, "preproc: %s\n", s.message().c_str());
        return 1;
    }
    auto out = engine->inferSingle({{engine->inputNames().front(), dst.view()}}, stream);
    if (!out) {
        std::fprintf(stderr, "infer: %s\n", out.status().message().c_str());
        return 1;
    }
    auto host = out->toHost(stream).value();
    auto data = host.as<float>().value(); // [1, 84, 8400], channels-first
    // --- end inference core ---

    const int nAnchors = static_cast<int>(host.shape()[2]); // 8400
    const float r = std::min(static_cast<float>(kInput) / img.width, static_cast<float>(kInput) / img.height);

    std::vector<Detection> dets;
    for (int a = 0; a < nAnchors; ++a) {
        int bestCls = 0;
        float bestScore = 0.f;
        for (int c = 0; c < kNumClasses; ++c) {
            const float s = data[(4 + c) * nAnchors + a];
            if (s > bestScore) {
                bestScore = s;
                bestCls = c;
            }
        }
        if (bestScore < conf) {
            continue;
        }
        const float cx = data[0 * nAnchors + a], cy = data[1 * nAnchors + a];
        const float w = data[2 * nAnchors + a], h = data[3 * nAnchors + a];
        // 640-input pixels -> original pixels (letterbox is top-left aligned, pad right/bottom)
        dets.push_back({static_cast<int>((cx - w / 2) / r), static_cast<int>((cy - h / 2) / r), static_cast<int>((cx + w / 2) / r),
                        static_cast<int>((cy + h / 2) / r), bestScore, bestCls});
    }

    const auto kept = nms(std::move(dets), 0.45f);
    std::printf("%zu detections (conf >= %.2f):\n", kept.size(), conf);
    for (const auto &d : kept) {
        std::printf("  %-14s %5.1f%%  [%d,%d %dx%d]\n", kCoco[d.cls], 100.f * d.score, d.x0, d.y0, d.x1 - d.x0, d.y1 - d.y0);
        examples::drawRect(img, d.x0, d.y0, d.x1, d.y1, 0, 220, 0, 3);
    }
    examples::writeJpg(outPath, img);
    std::printf("wrote %s\n", outPath.c_str());
    return 0;
}
