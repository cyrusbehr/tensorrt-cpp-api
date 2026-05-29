# Quickstart & core concepts

This walks through the mental model and the common flows. For end-to-end programs see
[`examples/`](../examples); for installation see [install.md](install.md).

## The five types you'll use

| Type | Role |
|---|---|
| `Status` / `Result<T>` | No-throw error handling. `Status` is ok-or-error; `Result<T>` is value-or-error. `if (!r) … r.status()`. |
| `Shape` | A dynamic-aware shape (`-1` = dynamic). `Shape{1,3,640,640}`. |
| `Tensor` | An **owning** device or pinned-host buffer (RAII, move-only). The library returns these for outputs. |
| `TensorView` | A **non-owning** view over memory you (or another library) own. The zero-copy boundary type. |
| `Stream` | A CUDA stream you own (or wrap). Every async call takes one; you control synchronization. |

Everything lives in `namespace trtcpp` and is reachable via `#include <tensorrt_cpp_api/all.h>`
(the optional `preproc.h` / `opencv_interop.h` are included separately).

## Error handling

Every fallible call returns `Status` or `Result<T>` — never throws.

```cpp
auto engine = EngineBuilder{}.buildAndLoad("model.onnx", opt);
if (!engine) {
    std::fprintf(stderr, "%s\n", engine.status().message().c_str());
    return 1;
}
engine->inputNames();           // operator-> / operator* to reach the value

// Or propagate with the helper (the enclosing function must return Status or a Result):
TRTCPP_TRY(auto eng, EngineBuilder{}.buildAndLoad("model.onnx", opt));
```

## Build (or load) an engine

`EngineBuilder` turns an ONNX model into an optimized engine and caches it on disk. `buildAndLoad`
is the one-call entry point: it builds if there's no current cache, otherwise reuses it, then
deserializes into a ready `Engine`.

```cpp
BuildOptions opt;
opt.precision = Precision::kFp16;     // kFp32 / kFp16 / kInt8Qdq / kFp8 …
opt.engineCacheDir = "engines";       // where the .engine + .json sidecar live
auto engine = EngineBuilder{}.buildAndLoad("model.onnx", opt);
```

The cache key is the ONNX content hash + build options + TensorRT version + GPU UUID. Change any of
them (new model, different precision, driver upgrade, different GPU) and the stale cache is detected
and rebuilt — it is never silently reused. To build and deserialize separately, use
`buildOrLoad` (returns a path) + `Engine::loadFromFile`.

## Run inference

Inputs and outputs are **name-keyed** maps of `TensorView`. There are three entry points:

```cpp
Stream stream;   // owns a non-blocking stream; or Stream::wrap(yourCudaStreamHandle)

// 1) Library allocates outputs for you:
auto outputs = engine->infer({{"images", inputView}}, stream);   // Result<map<string,Tensor>>

// 2) Single-output shortcut (classifier/detector common case):
auto out = engine->inferSingle({{"images", inputView}}, stream); // Result<Tensor>

// 3) Caller-allocated, fully zero-copy (no allocation in the hot path):
engine->enqueue({{"images", inputView}}, {{"output", outView}}, stream);   // Status
```

None of these synchronize. Read results back explicitly:

```cpp
auto host = out->toHost(stream).value();          // D2H copy AND stream sync
std::span<const float> v = host.as<float>().value();
```

`as<T>()` dtype-checks and errors on a device tensor (it never does an implicit D2H). Use
`engine->outputShapes(inputs)` to size your own output buffers for the `enqueue` path.

## Preprocess on the GPU (optional)

The `preproc` sublibrary fuses the usual CNN front-end into one kernel — no intermediate buffers:

```cpp
#include <tensorrt_cpp_api/preproc.h>

// src: HWC uint8 device tensor [H,W,C]; dst: NCHW float device tensor [1,C,Hout,Wout]
preproc::PreprocSpec spec;             // out = (pixel - mean) * scale, per channel
spec.swapRB = true;                    // BGR -> RGB (if your decode gives BGR)
spec.keepAspectRatioPad = true;        // letterbox; pads right/bottom
spec.scale = {1.f/255, 1.f/255, 1.f/255, 1.f};
preproc::letterboxToTensor(src.view(), dst.view(), spec, stream);
```

## Concurrency

`Engine` is thread-compatible (use it from one thread at a time). For concurrent inference across
threads/streams, use `EnginePool`, which hands out execution-context leases:

```cpp
auto pool = EnginePool::create("model.engine", /*contexts=*/4).value();
auto lease = pool.acquire();                       // blocks until one is free
auto out = lease.infer({{"images", inputView}}, myStream);
```

For a **dynamic-shape** engine, each concurrently-used context needs its own optimization profile,
so build the engine with at least `contexts` profiles (TensorRT 11 rule; `create` enforces it).

## Dynamic shapes

```cpp
ProfileShape p;
p.inputName = "images";
p.min = Shape{1,3,640,640}; p.opt = Shape{1,3,640,640}; p.max = Shape{8,3,640,640};
opt.profiles = {OptimizationProfile{{p}}};
// at inference time, the input view's shape selects the actual dims within [min,max]
```

## Streams & lifetime — the contract

- Inference is **asynchronous** on the stream you pass and **never implicitly synchronizes**.
- Keep input buffers alive until you've synchronized the stream (`stream.synchronize()` or
  `Tensor::toHost`, which syncs).
- A `Stream` you construct owns its CUDA stream; `Stream::wrap(handle)` adopts an existing one
  (e.g. a PyTorch/CuPy stream) without owning it.

## Python

The same model, zero-copy, from Python:

```python
import cupy as cp, trtcpp
eng = trtcpp.EngineBuilder().build_and_load("model.onnx", trtcpp.BuildOptions())
stream = trtcpp.Stream.wrap(cp.cuda.get_current_stream().ptr)
out = eng.infer_single({"images": cp_gpu_array}, stream)   # cupy array in, zero-copy
y = cp.asarray(out); stream.synchronize()                  # cupy view of the output, zero-copy
```

See [`examples/python`](../examples/python) and the API reference (`doxygen Doxyfile`).
