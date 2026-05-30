#pragma once

// Dependency-free image I/O for the reference examples, via the vendored stb single-header
// libraries. The library itself has no image-codec dependency; the examples decode on the CPU,
// upload raw HWC-uint8 to the GPU, do all tensor work there (preproc sublib + engine), and write
// results back. This keeps the examples buildable anywhere -- no OpenCV / cuDNN / system-codec
// version coupling. (The library separately offers an optional OpenCV interop header for callers
// who already have cv::Mat / cv::cuda::GpuMat.)
//
// stb_image decodes to RGB, so the preproc specs here use swapRB=false (the ONNX classifiers /
// detectors expect RGB).

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "stb_image.h"
#include "stb_image_write.h"

#include <tensorrt_cpp_api/all.h>
#include <tensorrt_cpp_api/preproc.h>

namespace examples {

/// An 8-bit RGB image in HWC layout (channels = 3).
struct Image {
    std::vector<std::uint8_t> data;
    int width = 0;
    int height = 0;
    int channels = 3;
    bool empty() const { return data.empty(); }
};

inline Image decodeImage(const std::string &path) {
    int w = 0, h = 0, c = 0;
    std::uint8_t *pixels = stbi_load(path.c_str(), &w, &h, &c, 3); // force RGB
    Image img;
    if (pixels) {
        img.width = w;
        img.height = h;
        img.channels = 3;
        img.data.assign(pixels, pixels + static_cast<std::size_t>(w) * h * 3);
        stbi_image_free(pixels);
    }
    return img;
}

inline bool writeJpg(const std::string &path, const Image &img, int quality = 92) {
    return stbi_write_jpg(path.c_str(), img.width, img.height, img.channels, img.data.data(), quality) != 0;
}

/// Upload a decoded HWC-uint8 RGB image to a device Tensor [H, W, 3]. Async on `stream`.
inline trtcpp::Result<trtcpp::Tensor> uploadHWC(const Image &img, const trtcpp::Stream &stream) {
    if (img.empty()) {
        return trtcpp::Status{trtcpp::StatusCode::kInvalidArgument, "image is empty"};
    }
    const trtcpp::Shape shape{img.height, img.width, 3};
    auto dev = trtcpp::Tensor::allocate(trtcpp::DType::kUInt8, shape, trtcpp::Device::kCuda);
    if (!dev) {
        return dev.status();
    }
    const trtcpp::TensorView host{const_cast<std::uint8_t *>(img.data.data()), trtcpp::DType::kUInt8, shape, trtcpp::Device::kHost};
    if (auto s = dev->copyFrom(host, stream); !s) {
        return s;
    }
    return dev;
}

/// ImageNet preprocessing for torchvision/ONNX-zoo models: /255 then (x-mean)/std with the
/// standard ImageNet statistics. The preproc kernel computes out = (pixel - mean)*scale, so
/// mean = 255*imagenet_mean and scale = 1/(255*imagenet_std), in RGB channel order.
inline trtcpp::preproc::PreprocSpec imagenetSpec() {
    trtcpp::preproc::PreprocSpec spec;
    spec.swapRB = false;             // stb already gives RGB
    spec.keepAspectRatioPad = false; // plain resize to the square input
    spec.mean = {0.485f * 255.f, 0.456f * 255.f, 0.406f * 255.f, 0.f};
    spec.scale = {1.f / (0.229f * 255.f), 1.f / (0.224f * 255.f), 1.f / (0.225f * 255.f), 1.f};
    return spec;
}

/// Draw an axis-aligned rectangle outline (clamped to the image) in RGB.
inline void drawRect(Image &img, int x0, int y0, int x1, int y1, std::uint8_t r, std::uint8_t g, std::uint8_t b, int thickness = 2) {
    x0 = std::clamp(x0, 0, img.width - 1);
    y0 = std::clamp(y0, 0, img.height - 1);
    x1 = std::clamp(x1, 0, img.width - 1);
    y1 = std::clamp(y1, 0, img.height - 1);
    auto put = [&](int x, int y) {
        if (x < 0 || y < 0 || x >= img.width || y >= img.height) {
            return;
        }
        auto *p = &img.data[(static_cast<std::size_t>(y) * img.width + x) * 3];
        p[0] = r;
        p[1] = g;
        p[2] = b;
    };
    for (int t = 0; t < thickness; ++t) {
        for (int x = x0; x <= x1; ++x) {
            put(x, y0 + t);
            put(x, y1 - t);
        }
        for (int y = y0; y <= y1; ++y) {
            put(x0 + t, y);
            put(x1 - t, y);
        }
    }
}

} // namespace examples
