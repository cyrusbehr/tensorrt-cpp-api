# Changelog

## v7.0.0-rc1 (2026-05-29)

First release candidate of the v7 rewrite — a ground-up reimplementation into a reusable,
installable C++ library. **This is a clean break from v6 — the API is not source-compatible.**
See [docs/upgrading_from_v6.md](docs/upgrading_from_v6.md).

### Added
- No-throw API: every fallible call returns `Status` or `Result<T>` (no exceptions); `TRTCPP_TRY`
  sugar for propagation.
- Name-keyed IO (`unordered_map<string, TensorView>`) at the inference boundary; non-templated
  `Tensor`/`TensorView` with a runtime `DType` (replaces v6's `Engine<T>` and nested vectors).
- Caller-owned CUDA streams (`Stream`, including `Stream::wrap` for an external handle); explicit
  host/device transfers (`Tensor::toHost`/`to`/`copyFrom`) — never an implicit D2H copy.
- Safe engine cache: `EngineBuilder::buildOrLoad`/`buildAndLoad` keyed by ONNX content hash +
  build options + TensorRT version + GPU UUID, JSON sidecar, atomic write, stale-cache detection.
- Dynamic shapes via per-input min/opt/max optimization profiles and a `-1`-aware `Shape`.
- `EnginePool` for concurrent multi-stream inference (one optimization profile per context).
- Quantization that is version-aware and never a silent no-op (`Precision::kFp16`/`kInt8Qdq`/
  `kFp8`/…); legacy INT8 calibration is available only when built against TensorRT < 11.
- Optional fused preprocessing sublibrary (`tensorrt_cpp_api::preproc`): a single CUDA kernel for
  letterbox-resize → BGR↔RGB → per-channel normalize → HWC→NCHW → cast.
- Optional OpenCV interop header (zero-copy views over `cv::Mat` / `cv::cuda::GpuMat`), strictly
  opt-in behind `-DTRT_CPP_API_WITH_OPENCV`.
- Optional Python bindings (`trtcpp`, pybind11 + scikit-build-core) with zero-copy
  `__cuda_array_interface__`/DLPack interop, caller streams, and GIL release during inference.
- CMake install/export: `find_package(tensorrt_cpp_api)` with a relocatable package config and a
  bundled `FindTensorRT` module (apt or tarball).
- Reference examples (classification, detection, segmentation, zero-copy Python) and GitHub Actions
  CI (build + CPU tests, sanitizers, Python wheel, lint).

### Changed
- Minimum TensorRT is 10.0; the code is written to the TensorRT 11 surface and version-gates the
  removed-in-11 features (legacy calibrators, weak typing, `IPluginV2`).
- C++20 (was C++17); namespace `trtcpp`; include root `tensorrt_cpp_api/`; CMake target
  `tensorrt_cpp_api::tensorrt_cpp_api`.

### Removed
- The v6 API in its entirety — no source-compatibility shim (see the upgrade guide):
  - the monolithic templated `Engine<T>` and its `.inl`-in-header implementation;
  - the single `Options` struct, replaced by `BuildOptions` + `EngineOptions`;
  - OpenCV `cv::cuda::GpuMat` in the inference signatures and the triply-nested `std::vector`
    inputs/outputs, replaced by name-keyed `TensorView`/`Tensor`;
  - `bool`/exception error handling, replaced by `Status`/`Result<T>`;
  - per-call stream creation, replaced by caller-owned `Stream`;
  - the `run_inference_benchmark` CLI executable and the v6 `include/` header layout.

---

## v6.0
- Implementation now requires TensorRT >= 10.0.

## v5.0
- `Engine` became a class template parameterized on the model's output data type (`float`,
  `__half`, `int8_t`, `int32_t`, `bool`, `uint8_t`).
- Added loading a TensorRT engine file directly (without compiling from ONNX).
- Added a command-line parser.

## v4.1
- Support for fixed batch size > 1.

## v4.0
- Added INT8 precision support.

## v3.0
- Updated to the TensorRT 8.6 API (`IExecutionContext::enqueueV3()`).
- Benchmark executable renamed to `run_inference_benchmark`; takes the ONNX path as an argument.
- Auto-detect supported batch sizes; stop limiting workspace memory.

## v2.2
- Serialize the model name as part of the engine file.

## v2.1
- Support for models with multiple inputs.

## v2.0
- Requires OpenCV with CUDA.
- Support for models with more than one output and for non-batchable models; more error checking.
