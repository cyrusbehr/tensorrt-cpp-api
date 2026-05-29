# Upgrading from v6

v7 is a **clean break**: there is no source-compatibility shim. v6's API — a templated
`Engine<T>`, OpenCV `cv::cuda::GpuMat` in the signatures, triply-nested `std::vector` outputs,
`bool`/throw error handling, and preprocessing baked into the engine — is fundamentally
incompatible with v7's leak-free, name-keyed, no-throw design. This guide maps every v6 concept to
its v7 equivalent and shows a full before/after.

## What changed, and why

| v6 | v7 | Why |
|---|---|---|
| `Engine<T>` (templated on output type) | `Engine` (non-templated) + runtime `DType` on `Tensor`/`TensorView` | One type handles mixed-dtype, multi-output models; no template bloat. |
| `Options` (one struct for everything) | `BuildOptions` (build) + `EngineOptions` (load) | Build-time vs runtime concerns are separate. |
| `Precision::FP16` / `INT8` | `Precision::kFp16` / `kInt8Qdq` (or `kInt8CalibLegacy`, TRT < 11) | INT8 defaults to explicit-QDQ; legacy calibration is gated out on TensorRT 11. |
| `buildLoadNetwork(onnx, subVals, divVals, normalize)` | `EngineBuilder::buildAndLoad(onnx, BuildOptions)` + separate `preproc::letterboxToTensor(...)` | Preprocessing is no longer fused into the engine; it's an explicit, reusable GPU kernel. |
| `loadNetwork(trtPath, …)` | `Engine::loadFromFile(path, EngineOptions)` / `loadFromMemory(bytes, …)` | — |
| `runInference(inputs: vector<vector<GpuMat>>, out: vector<vector<vector<T>>>)` | `engine.infer(map<string,TensorView>, Stream)` → `map<string,Tensor>`, or `enqueue(in, out, Stream)` for caller-allocated zero-copy | Name-keyed, zero-copy, single-allocation; no nested vectors. |
| Engine creates/destroys a stream per call | Caller passes a `Stream` (own it, or `Stream::wrap(handle)`) | Caller controls stream ordering and overlap. |
| Implicit device→host copy of outputs | Explicit `Tensor::toHost(stream)` (copies **and** syncs) | No hidden D2H; you decide when to pay for it. |
| `bool` returns / exceptions | `Status` / `Result<T>` (+ `TRTCPP_TRY`) | Uniform, no-throw error handling. |
| `cv::cuda::GpuMat` inputs | `TensorView` over device memory (or `opencv_interop::viewOf(gpuMat)` with `-DTRT_CPP_API_WITH_OPENCV=ON`) | OpenCV is optional, never in the core API. |
| `subVals` / `divVals` / `normalize` | `preproc::PreprocSpec{ mean, scale, swapRB, keepAspectRatioPad }` (`scale = 1/divVals`, `mean = subVals`) | Per-channel, with letterbox, in one fused kernel. |
| `optBatchSize` / `maxBatchSize` / input-width options | `BuildOptions.profiles` = `OptimizationProfile`/`ProfileShape` (per-input min/opt/max) | Real dynamic-shape support, not just batch. |
| `getInputDims()` / `getOutputDims()` | `engine.inputNames()` / `outputNames()` / `tensorShape(name)` / `tensors()` | Name-keyed introspection. |
| Global spdlog logger | Injectable `EngineOptions.logger` (`ILogger`); spdlog adapter behind `-DTRT_CPP_API_WITH_SPDLOG=ON` | No forced logging dependency. |
| `#include "engine.h"`, `Engine`/`Util`/`Stopwatch` in the global namespace | `#include <tensorrt_cpp_api/all.h>`, everything under `namespace trtcpp` | Clean include root + namespace. |
| Header `.inl` implementation (TRT/CUDA leak into every TU) | PImpl; no `nvinfer1`/OpenCV/spdlog in public headers | Fast, decoupled compiles. |

## Before / after

**v6:**
```cpp
#include "engine.h"

Options options;
options.precision = Precision::FP16;
options.optBatchSize = 1;
options.maxBatchSize = 1;

Engine<float> engine(options);
std::array<float, 3> subVals{0.f, 0.f, 0.f}, divVals{1.f, 1.f, 1.f};
engine.buildLoadNetwork("model.onnx", subVals, divVals, /*normalize=*/true);

std::vector<std::vector<cv::cuda::GpuMat>> inputs = /* one GpuMat per input, per batch */;
std::vector<std::vector<std::vector<float>>> featureVectors;
engine.runInference(inputs, featureVectors);   // returns bool; D2H happens internally
float first = featureVectors[0][0][0];
```

**v7:**
```cpp
#include <tensorrt_cpp_api/all.h>
#include <tensorrt_cpp_api/preproc.h>
using namespace trtcpp;

BuildOptions opt;
opt.precision = Precision::kFp16;
opt.engineCacheDir = "engines";

auto engine = EngineBuilder{}.buildAndLoad("model.onnx", opt);
if (!engine) return /* handle */ engine.status();

Stream stream;
// Preprocess on the GPU into an NCHW tensor (replaces subVals/divVals/normalize):
auto input = Tensor::allocate(DType::kFloat32, Shape{1, 3, H, W}, Device::kCuda).value();
preproc::PreprocSpec spec;                       // out = (pixel - mean) * scale
spec.scale = {1.f / 255, 1.f / 255, 1.f / 255, 1.f};
preproc::letterboxToTensor(srcHWCDeviceView, input.view(), spec, stream);

auto out = engine->inferSingle({{engine->inputNames().front(), input.view()}}, stream);
if (!out) return out.status();
auto host = out->toHost(stream).value();         // explicit D2H + sync
std::span<const float> scores = host.as<float>().value();
float first = scores[0];
```

## INT8 / calibration

- **Preferred (works on TensorRT 10 and 11):** export an explicit Q/DQ ONNX and build with
  `Precision::kInt8Qdq`. No calibration data, no precision flags.
- **Legacy calibration** (v6 `calibrationDataDirectoryPath`): only available when the library is
  built against **TensorRT < 11**. Use `Precision::kInt8CalibLegacy` and set
  `BuildOptions.calibrator` to an `ICalibrator` (see `tensorrt_cpp_api/calibrator.h`). On
  TensorRT ≥ 11 this path is rejected with a clear error rather than silently downgrading.

## Dynamic shapes

v6 exposed only batch knobs. In v7, give one `OptimizationProfile` per concurrent context:
```cpp
ProfileShape p;
p.inputName = "images";
p.min = Shape{1, 3, 640, 640};
p.opt = Shape{1, 3, 640, 640};
p.max = Shape{8, 3, 640, 640};
opt.profiles = {OptimizationProfile{{p}}};
```
For concurrent multi-stream dynamic inference, build with ≥ N profiles and use `EnginePool`
(one profile per leased context).

## Migration checklist

1. Replace `#include "engine.h"` with `#include <tensorrt_cpp_api/all.h>`; add `using namespace trtcpp;`.
2. Split `Options` into `BuildOptions` (precision, profiles, cache dir) and `EngineOptions`
   (device, logger, plugins).
3. Drop the `Engine<T>` template parameter; read outputs through `Tensor`/`as<T>()`.
4. Move preprocessing out of the engine into `preproc::letterboxToTensor` (or your own kernel);
   translate `subVals → mean`, `divVals → 1/scale`.
5. Replace `runInference(GpuMat, nested-vector)` with `infer`/`inferSingle`/`enqueue` over a
   name-keyed `TensorView` map and a caller `Stream`; read back with `toHost`.
6. Replace `bool` checks with `if (!result) … result.status()`.
7. (Optional) Turn on `-DTRT_CPP_API_WITH_OPENCV=ON` and use `opencv_interop::viewOf` if you still
   feed `cv::cuda::GpuMat`.

See the [reference examples](../examples) for complete, runnable detection/classification/segmentation
pipelines built this way.
