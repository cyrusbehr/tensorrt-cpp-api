# Examples

Four reference programs covering the common CNN inference tasks, each built against the
**installed** `tensorrt_cpp_api` package via `find_package`. They share one trait: the library
does the GPU work (build/cache the engine, fused preprocessing, inference) and the example does
only I/O and task-specific post-processing.

| Example | Model | Pipeline |
|---|---|---|
| `classification/` | MobileNetV2 / ResNet50 (ImageNet) | decode → preproc → infer → softmax top-5 |
| `detection/` | YOLOv8n | decode → letterbox preproc → infer → decode `[1,84,8400]` → NMS → draw |
| `segmentation/` | DeepLabV3-MobileNetV3 | decode → preproc → infer → per-pixel argmax → colorize/blend |
| `benchmark/` | any | preallocated zero-copy `enqueue` loop, throughput baseline |

Image decode/encode uses the vendored **stb** single-header libraries (`common/`), so the
examples have **no third-party image dependency** — they build wherever the library does. (If you
prefer OpenCV, the library ships an optional `opencv_interop.h`; swap `common/image_io.h`
accordingly.)

## Get the models

```sh
./download_models.sh        # writes ../models/{mobilenetv2-7,yolov8n,deeplabv3_mobilenetv3}.onnx
```

Detection/segmentation export needs `ultralytics` and `torch`/`torchvision` (CPU is fine — see the
script header). Classification needs only `curl`.

## Build (against an installed library)

```sh
# 1. install the library somewhere
cmake -S . -B build -DTRT_CPP_API_BUILD_PREPROC=ON
cmake --build build -j && cmake --install build --prefix /opt/trtcpp

# 2. build the examples against it
cmake -S examples -B build/examples -DCMAKE_PREFIX_PATH=/opt/trtcpp
cmake --build build/examples -j
```

(For a tarball TensorRT, add `-DTensorRT_DIR=/path/to/TensorRT-10.x` to both.)

Alternatively build them in-tree with the library: `cmake -S . -B build -DTRT_CPP_API_BUILD_EXAMPLES=ON`.

## Run

```sh
./build/examples/classification models/mobilenetv2-7.onnx image.jpg
./build/examples/detection      models/yolov8n.onnx image.jpg 0.25 out.jpg
./build/examples/segmentation   models/deeplabv3_mobilenetv3.onnx image.jpg out.jpg
```

The first run builds and caches a TensorRT engine next to the working directory (`engines/`);
later runs load the cache. See `python/` for the zero-copy CuPy example and the perf-parity check.
