# Phase A — Audit of the Current (v6) Implementation

Findings from five parallel code auditors (API surface, correctness/CUDA-safety,
TensorRT modernity, build system, quantization), cross-checked by the integrator
against the source. Ordered by severity: **correctness bugs > API/design >
deprecations > nits.** Every finding cites `file:line`.

The v6 code is small (~1450 LoC) and already on several modern idioms
(`enqueueV3` + `setTensorAddress` + name-based IO, `buildSerializedNetwork`,
`deserializeCudaEngine`). The problems are concentrated in: positional IO
assumptions, a broken dynamic-shape path, leaky public types, a `bool`/`throw`
error muddle, hard-coded 3-channel NCHW preprocessing, and a calibrator-only INT8
path that is dead in TensorRT 11.

## 1. Correctness bugs

- **Positional input-then-output IO assumption** — `EngineRunInference.inl:113-122`,
  `EngineUtilities.inl:138`. The output-copy loop iterates `outputBinding` from
  `numInputs..getNbIOTensors()` and indexes `m_buffers`/`m_outputLengths`
  positionally, and `clearGpuBuffers` frees `m_buffers[numInputs..]`. But
  `m_buffers`/`m_IOTensorNames` are filled in raw `getIOTensorName(i)` order
  (`EngineBuildLoadNetwork.inl:112-186`), which TensorRT does **not** guarantee
  places all inputs before all outputs. An interleaved IO order reads/frees the
  wrong buffers. Fix: build explicit `inputIndices`/`outputIndices` maps keyed by
  `getTensorIOMode` and index through them. (Also flagged by the community in PR #94.)

- **Dynamic H/W inference path is broken** — `EngineRunInference.inl:62` vs
  `EngineBuildLoadNetwork.inl:131`. For a dynamic-width model the engine shape is
  `-1`, stored verbatim into `m_inputDims`, so the validation
  `input.cols != dims.d[2]` rejects every legal input (`cols` can never be `-1`).
  The dynamic-width feature (added in #82) cannot actually run inference. Fix:
  skip/clamp the check for any axis whose stored extent is `-1` and set the
  per-call shape from the actual input size. (Community: #80, #86, #93.)

- **Output length computed from dynamic `-1` dims** — `EngineRunInference.inl:117,121`,
  `EngineBuildLoadNetwork.inl:170-173`. `outputLength` is the product of the
  *engine* output dims, which contain `-1` for dynamic output axes, corrupting the
  per-sample stride/size of the D2H copy. Fix: query `m_context->getTensorShape`
  after shapes are set and compute lengths from the concrete runtime shape.

- **Preprocessing not ordered onto the inference stream** —
  `EngineRunInference.inl:54-84,129`. `blobFromGpuMats` runs OpenCV-CUDA ops on the
  default per-thread stream, while `enqueueV3` runs on a freshly created
  `inferenceCudaStream`; the preprocessing kernels are not ordered before the
  enqueue → data race. Fix: thread one `cv::cuda::Stream`/`cudaStream_t` through
  preprocessing and inference.

- **Calibration `getBatch` async-copy lifetime bug** — `engine.cpp:101-105`.
  `cudaMemcpyAsync(DeviceToDevice)` is issued on the default stream with no
  synchronize before returning; the source `mfloat` GpuMat is destroyed at scope
  exit, so its device memory may be freed before the copy lands → nondeterministic
  calibration data. Fix: synchronous copy (or sync the stream) before return.

- **Calibration vs inference preprocessing mismatch** — `engine.cpp:101` (`swapRB=true`)
  vs `engine.h:122` / `EngineRunInference.inl:81` (`swapRB=false` default). The
  calibrator sees BGR→RGB-swapped data while inference does not, biasing every
  computed INT8 scale. Fix: plumb the inference color convention into the calibrator.

- **`cudaMallocAsync` paired with `cudaFree`** — `EngineBuildLoadNetwork.inl:180`
  vs `EngineUtilities.inl:139`. Output buffers are stream-allocated but freed
  synchronously, and the destructor path (`~Engine` → `clearGpuBuffers`) frees with
  no stream sync. Fix: pair `cudaMallocAsync` with `cudaFreeAsync` on a tracked
  stream, or use synchronous `cudaMalloc`/`cudaFree` consistently.

- **Uninitialized members read before load** — `engine.h:138,146`. `m_normalize`
  and `m_inputBatchSize` have no in-class initializer; reading them before a
  successful `loadNetwork` is UB. Fix: initialize (`bool m_normalize = true;
  int32_t m_inputBatchSize = -1;`) and gate inference on a loaded flag.

- **`getBatch` ignores `nbBindings` / assumes single input** — `engine.cpp:112`,
  `Int8Calibrator.h:14`. Writes only `bindings[0]` with no `nbBindings==1` assert
  and keys by index not name. Fix: assert/validate and match by `names[i]`.

## 2. API / design issues

- **`Engine<T>` templated on the output dtype** — `engine.h:65-66`,
  `EngineBuildLoadNetwork.inl:136-164`. Forces the caller to know the output type at
  compile time and recompile to switch models, yet the constraint is only checked at
  runtime, defeating the template. Fix: non-templated engine with type-erased,
  shape-carrying output tensors. (Community: #27, #47.)

- **Triply-nested `std::vector` IO with inconsistent index order** — `engine.h:97`,
  `IEngine.h:16-17`. Input is `[input][batch][GpuMat]`, output is
  `[batch][output][feature]` — opaque, allocation-heavy, easy to misuse. Fix: a
  small name-keyed `Tensor`/`TensorView` carrying shape + contiguous data.

- **OpenCV leaks through the public interface** — `engine.h:8-12,97,105,121`,
  `IEngine.h:5,16`. `cv::cuda::GpuMat` and the `opencv2/*` include tree are in the
  public/`IEngine` header, hard-coupling every consumer to a specific OpenCV-CUDA
  build (the #1 install pain). Fix: neutral device-pointer + shape at the boundary;
  OpenCV as an optional adapter.

- **`nvinfer1::*` leaks through public headers** — `engine.h:3,108-109,150-155`,
  `IEngine.h:6,18-19`, `Int8Calibrator.h:2-5`. `Dims`, `Dims3`, `ILogger`,
  `ICudaEngine`, `IInt8EntropyCalibrator2` exposed, leaking the TRT ABI. Fix:
  library-owned `Shape`; PImpl/forward-declare the nvinfer1 members.

- **`spdlog` baked into public headers, no injection** — `logger.h:5`, `macros.h:3`,
  `util/Util.h:7`. Consumers inherit a logging dependency they can't redirect or
  silence. Fix: an `ILogger` callback in `Options`; spdlog only in `.cpp`. (Community: #64.)

- **`bool` returns mixed with throws** — `IEngine.h:12-17`,
  `EngineBuildLoadNetwork.inl:18-21,124`, `EngineRunInference.inl:90`. Same call path
  signals failure two ways. Fix: one mechanism — a `Status`/`Result` (or
  `std::expected`/`tl::expected`, which PR #94's author is already moving toward).

- **No CUDA-stream injection, single context, single profile** — `engine.h:34-58`,
  `EngineRunInference.inl:51-52,130`. A stream is created+destroyed every call; no
  multi-context/multi-stream; only width is dynamic. Limits the library to
  single-threaded single-stream use. Fix: caller-provided stream, multi-context
  pool, multiple optimization profiles. (Community: #28, #57, #85, #43.)

- **Hard-wired rank-4 NCHW, 3 channels** — `engine.h:121-122,105`,
  `EngineBuildLoadNetwork.inl:131`, `EngineUtilities.inl:32-34,96`. `Dims3` input
  dims, `CHECK(channels()==3)`, `CV_8UC3`. Grayscale/RGBA/rank-3/5 silently break.
  Fix: derive rank and channels from the parsed tensor shape. (Community: #83, #87, #11.)

- **Per-channel normalization is wrong** — `EngineUtilities.inl:128-129`. `cv::Scalar`
  mean/std are applied across the interleaved blob, misapplying per-channel stats
  when they differ across channels (breaks ImageNet-style preprocessing). Fix: apply
  per-channel on the correct CHW slices. (Community: #92, PR #94 — highest-value user fix.)

- **Float-only input** — `EngineBuildLoadNetwork.inl:121-125`. Inputs restricted to
  `kFLOAT`. Fix: select the blob path from the actual input `DataType`.

- **Static helpers hang off `Engine<T>`** — `engine.h:105,114,119,121`. Callers must
  write `Engine<float>::resizeKeepAspectRatioPadRightBottom` for functions unrelated
  to `T`. Fix: free functions in a utility namespace.

- **Calibration cache: weak key + wrong directory** — `EngineBuildLoadNetwork.inl:366`,
  `engine.cpp:119,132`. The `.calibration` name omits the dataset, sub/div/normalize,
  and batch size (stale-cache → wrong scales) and is written to CWD, not
  `engineFileDir`. Fix: hash the calibration inputs into the name; write beside the engine.

- **Engine cache filename omits TRT version & ONNX hash** — `EngineUtilities.inl:51-88`.
  The name encodes precision/batch/width/GPU name but not the TRT version or a hash
  of the ONNX, so an engine built by a different TRT or from a changed ONNX is
  silently reused and fails to deserialize. Fix: content-hash + sidecar metadata
  (TRT version, GPU UUID, ONNX hash, build options) with an integrity check on load.
  (Community: version-skew issues #41, #17, #54, #10.)

## 3. Deprecations (TensorRT — see PHASE_B for the full TRT 11 picture)

- **Calibrator-only INT8 path is removed in TRT 11** — `Int8Calibrator.h:5`,
  `EngineBuildLoadNetwork.inl:368-371`. `IInt8EntropyCalibrator2` +
  `setInt8Calibrator` were deprecated in 10.6.0 and **removed in 11.0.0**; this code
  will not compile against TRT 11. Fix: explicit Q/DQ quantization (strongly-typed
  network, no calibrator), with the legacy path gated behind `#if NV_TENSORRT_MAJOR < 11`.

- **Per-precision BuilderFlags are removed in TRT 11** — `EngineBuildLoadNetwork.inl:338,361`.
  `kFP16`/`kINT8` (and the weak-typing setters) are gone in 11.0; strongly-typed is
  the default. Fix: adopt `kSTRONGLY_TYPED`; carry precision in the ONNX/QDQ graph.

- **`kEXPLICIT_BATCH` flag is a deprecated no-op** — `EngineBuildLoadNetwork.inl:207-208`.
  Networks are always explicit-batch in TRT 10+. Fix: `createNetworkV2(0)` and drop
  the stale "implicit batch is deprecated" comment.

- **No timing cache** — `EngineBuildLoadNetwork.inl:383`. Every cold build re-runs
  full tactic timing. Fix: `createTimingCache`/`setTimingCache` + serialize to disk.

- **`FindCUDA` module is deprecated** — `CMakeLists.txt:23`. `find_package(CUDA)`
  uses the legacy module. Fix: `find_package(CUDAToolkit)` + `CUDA::cudart` targets.

- **Parser errors discarded** — `EngineBuildLoadNetwork.inl:234`. `parser->parse`
  return is checked but `parser->getError(i)` is never logged. Fix: iterate and log
  parser errors on failure.

## 4. Nits

- `-Ofast -DNDEBUG` and `-Wno-deprecated-declarations` hard-coded into global CXX
  flags — `CMakeLists.txt:9`. Breaks Debug builds, alters float semantics, and hides
  the very TRT deprecations above. Fix: let `CMAKE_BUILD_TYPE` drive optimization.
- Hard-coded developer paths — `CMakeLists.txt:16` (`TensorRT_DIR=/home/cyrus/...`),
  `CMakeLists.txt:19` (`CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda`). (Community: #40.)
- spdlog used in ~15 TUs but never `find_package`d or linked — `CMakeLists.txt` (no
  `find_package(spdlog)`); relies on transitive inclusion.
- No install/export rules, no `tensorrt_cpp_api::` alias, no package config —
  `CMakeLists.txt:27-34`; cannot be consumed via `find_package`.
- C++17 via global `CMAKE_CXX_STANDARD` not `target_compile_features` —
  `CMakeLists.txt:8`; requirement not propagated; extensions left on.
- Library hard-coded `SHARED` — `CMakeLists.txt:27`; ignore `BUILD_SHARED_LIBS`.
- `cmake_minimum_required(VERSION 3.18)` — `CMakeLists.txt:1`; bump to >= 3.22.
- `FindTensorRT.cmake:73` references `TensorRT_NVPARSERS_LIBRARY` it never searches;
  `nvonnxparser` not attached to the imported target.
- ccache wired via legacy `RULE_LAUNCH_COMPILE` — `cmake/ccache.cmake:3`; prefer
  `CMAKE_<LANG>_COMPILER_LAUNCHER`.
- `getLogLevelFromEnvironment` reads a global `LOG_LEVEL` env var — `logger.h:20-28`;
  hidden global state. Fix: take level from `Options`.
- Dead commented `cvtColor` — `engine.cpp:91`; ambiguous color handling.
- `calibrationBatchSize` defaults to 128 but the ctor hard-errors with fewer images —
  `engine.h:43`, `engine.cpp:53`. Fix: clamp to `min(batchSize, imgPaths.size())`.
- No tests, no CI anywhere in the repo. (Community: #63.)

## Top issues (the rewrite must address these first)

1. **Leaky public boundary** — OpenCV, nvinfer1, and spdlog all cross the public
   interface; this is the root of the install/version-skew pain.
2. **Broken dynamic shapes + positional IO** — the dynamic path cannot run, and IO
   buffers are addressed by a positional assumption TRT does not guarantee.
3. **Calibrator-only INT8 is dead in TRT 11** — the whole quantization path needs to
   move to explicit Q/DQ behind a version gate.
