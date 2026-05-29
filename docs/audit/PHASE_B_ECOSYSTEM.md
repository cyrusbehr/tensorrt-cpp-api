# Phase B — The 2026 Ecosystem

Synthesis of six parallel research agents (TensorRT, CUDA, cuDNN+OpenCV, GPU
preprocessing, reference projects, quantization), research date **2026-05-29**,
primary sources only. This locks the dependency versions and the API idioms the v7
design (Phase D) builds on.

## Headline: TensorRT 11.0 has shipped

**TensorRT 11.0.0 reached GA on/around 2026-05-27** — it is no longer a future
event (`tensorrt 11.0.0.114` on PyPI dated 2026-05-27; the "latest" docs channel now
resolves to the 11.0 release notes). The 10.x line tops out at **10.16.1** (the last
release before the 11.0 break). TRT 11.0 makes hard removals that dictate the design:

- **Implicit quantization removed.** `IInt8Calibrator` + all subclasses,
  `setInt8Calibrator()`, `setDynamicRange()`, `resetDynamicRange()` — gone.
  (deprecated 10.6.0 Nov 2024 → removed 11.0.0.)
- **Weak typing removed.** `ITensor::setType`, `ILayer::setPrecision`,
  `setOutputType`, and the precision `BuilderFlag`s `kFP16/kBF16/kFP8/kINT8/kINT4/kFP4`
  — gone. **Strongly-typed networks are the default.**
- **`IPluginV2` family removed** → `IPluginV3` + `addPluginV3()` only.
- **All static `.a` libraries removed** → shared-library only (a build-system break).
- TREx retired; legacy NMS plugins removed; min GPU is SM 7.5 (Turing); Ubuntu 20.04
  dropped.
- APIs deprecated *in* 11.0 are retained until **March 2027**.

Sources: [TRT 11.0.0 release notes](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/release-notes-11/11.0.0.html),
[support matrix](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html),
[work-quantized-types](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-quantized-types.html).

**Strategic consequence:** target the stable **10.16.x (`-cu12`)** line now, but write
every API call to the **11.0 surface** so the upgrade is a version bump, not a rewrite.
Version-gate the few things that legitimately differ (calibrator, precision flags)
behind `#if NV_TENSORRT_MAJOR < 11`.

## What v6 already does right

v6 is closer to the future than its version number suggests: it already uses
`createNetworkV2`, `enqueueV3` + `setTensorAddress` + `setInputShape`, and name-based
introspection (`getNbIOTensors`/`getIOTensorName`/`getTensorShape`/`getTensorIOMode`/
`getTensorDataType`) — no deprecated binding-index APIs. The modernization is mostly
about removing the calibrator/weak-typing path and the leaky boundary, not rewriting
the execution core.

## TensorRT

| Track | Latest GA | Use |
| --- | --- | --- |
| 11.x | 11.0.0 (2026-05-27) | write code to this surface; validate the upgrade |
| 10.x | 10.16.1 | **primary build target now** (`-cu12` build, CUDA 12.x) |

- NVIDIA ships parallel `-cu12` and `-cu13` builds; a TRT build is compatible only
  within its CUDA major family. For a CUDA 12.x deployment use the `-cu12` build.
- Login-free install paths (confirmed): the **CUDA network apt repo (`cuda-keyring`)**
  ships `libnvinfer-dev`/`tensorrt-dev` without a developer login; **PyPI
  `pip install tensorrt`** needs no login. Only the tarball/local-repo `.deb` downloads
  from the TensorRT download page are login-gated. (Don't mix network + local repos.)
- Standardize on: strongly-typed networks (`kSTRONGLY_TYPED`), `enqueueV3`,
  multi-context (N `IExecutionContext` per engine), `IPluginV3`, timing cache,
  `setMemoryPoolLimit(kWORKSPACE, ...)`, optional `kVERSION_COMPATIBLE` /
  hardware-compatibility for portable engines, weight streaming (large models).

**FP16 caveat for the design:** with strong typing (TRT 11 default), there is no
`kFP16` flag — precision is carried by the ONNX graph. v6's convenient "FP32 ONNX +
`Precision::FP16`" flow relies on weak typing, which is removed in 11. v7 must keep a
weak-typed builder for FP16/INT8-PTQ convenience **gated `#if NV_TENSORRT_MAJOR < 11`**,
and on 11.x require an FP16/QDQ ONNX (or provide a modelopt/onnx FP16-cast helper). The
explicit-QDQ + strongly-typed path is the one that compiles unchanged across the 10→11
boundary and is therefore the default.

## CUDA + driver

- **Current GA: CUDA 13.3** (release notes 2026-05-26); terminal 12.x is **12.9.2**.
- **Driver floors (Linux x86_64):** CUDA 12.6 ≥ 560.28.03; 12.8 ≥ 570.26; **13.0 ≥
  580.65.06**; 13.3 bundles ≥ 610.43.02.
- **This host (driver 565.57.01) cannot run CUDA 13** — below the 580 floor.
  **Target CUDA 12.6** here (12.9 works via minor-version compatibility); a fresh/CI
  host with driver ≥ 580 should use **CUDA 13.3**. Ampere `sm_86` is supported by both
  12.x and 13.x — the only 13.x blocker is the driver.
- CUDA 13 dropped offline support for Maxwell/Pascal/Volta; minimum arch is now Turing
  `sm_75`. Set `CMAKE_CUDA_ARCHITECTURES=86` (+ a virtual-PTX fallback like
  `86-real;90-virtual`); never rely on dropped arches.
- Use the **stream-ordered allocator** (`cudaMallocAsync`/`cudaFreeAsync`) with a
  **private pool** (`cudaMallocFromPoolAsync`, never mutate the device default pool in a
  library) and set `cudaMemPoolAttrReleaseThreshold = UINT64_MAX` so freed memory is
  retained across inference iterations. Adopt **CUDA Graphs** for the steady-state
  fixed-shape loop (capture once, relaunch); keep tensor addresses stable to enable it.
- NVTX v2 removed in 12.9 (use `nvtx3/`); non-`_Ctx` NPP APIs removed in 13.3.

Sources: [CUDA 13.0 release notes Table 3](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html),
[minor-version compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html),
[stream-ordered allocation](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html).

## cuDNN + OpenCV

- **cuDNN is NOT a runtime dependency for pure-TensorRT inference.** NVIDIA states it
  is optional, used only by a few deprecated layers, never by the lean/dispatch
  runtimes, and unsupported with TRT on Blackwell; it does not appear in the TRT 11.0
  prerequisites at all. The only way cuDNN re-enters is via OpenCV's DNN-CUDA backend.
  cuDNN latest is **9.22.0**. **Do not link cuDNN for the inference path.**
- **OpenCV stable is 4.13.0** (Dec 2025). **OpenCV 5.0 is alpha-only** (not production).
  Target **4.13.x**. Build on **C++17** so an eventual OpenCV-5 migration (C++17 min, C
  API removed) is low-friction.
- **`cv::cuda` still requires building OpenCV + opencv_contrib from source** with
  `WITH_CUDA=ON`; apt `libopencv-dev` ships **no** CUDA modules. Keep
  `OPENCV_DNN_CUDA=OFF`/`WITH_CUDNN=OFF` so cuDNN never enters the build.
- **Decision: OpenCV is optional, never a hard dependency of the engine core.** Default
  builds use apt CPU OpenCV (or none) for I/O; GPU image ops live behind a default-OFF
  CMake option. This directly addresses the largest community pain (install/OpenCV-CUDA).

Sources: [TRT prerequisites](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/prerequisites.html),
[OpenCV releases](https://opencv.org/releases/), [OpenCV config reference](https://docs.opencv.org/4.x/db/d05/tutorial_config_reference.html).

## GPU preprocessing

A CNN TRT pipeline needs ~5 ops: resize+letterbox, BGR↔RGB, mean-subtract+scale,
NHWC↔NCHW, dtype cast. Three GPU paths, all OpenCV-free:

- **Hand-rolled fused CUDA kernel (recommended for the core).** One kernel that
  letterbox-resizes, swaps channels, applies `(x-mean)*scale`, scatters to planar NCHW,
  and casts to fp16/int8 — writing TRT-ready tensors with no intermediate buffers and
  zero third-party runtime dep. Bilinear is sufficient for CNN preprocessing.
- **CV-CUDA 0.16.0-beta** (Apache-2.0, C/C++/Python, CUDA 12.2+/13.0+, SM 7.5+) — full
  operator coverage incl. `Reformat` (NHWC↔NCHW). **Pre-1.0, no ABI-stability
  guarantee** — pin exact versions; offer as an optional backend, not the core.
- **NPP** (free with the toolkit) — `_Ctx`-only (legacy APIs removed in CUDA 13.3); no
  native reformat op (compose it). Optional backend.

**Decision:** a separately-linkable `preproc` sublibrary with a backend-selecting
interface (Kernel | CVCUDA | NPP); the engine core links none of them by default.

Sources: [CV-CUDA releases](https://github.com/CVCUDA/CV-CUDA/releases),
[NPP introduction](https://docs.nvidia.com/cuda/npp/introduction.html).

## Quantization

- Implicit PTQ (calibrator) and pre-computed dynamic ranges are **removed in TRT 11**.
  The supported path is **explicit Q/DQ** (QuantizeLinear/DequantizeLinear nodes baked
  into the ONNX, built as a strongly-typed network — TRT folds them into INT8/FP8/INT4
  kernels). A QDQ ONNX builds on both TRT 10 and 11, so it is the format that survives
  the major bump.
- Produce QDQ ONNX with **NVIDIA Model Optimizer `nvidia-modelopt` 0.44.0** (2026-05-13):
  `python -m modelopt.onnx.quantization --onnx_path=m.onnx --quantize_mode=int8
  --calibration_data=calib.npy --calibration_method=entropy`. CNN/ViT: ≥500 calibration
  samples; INT8 opset ≥13. Treat quantization as an **offline asset step**; the C++
  engine just ingests the QDQ ONNX.
- **Precision by arch — for Ampere `sm_86` (this host), INT8 is the only
  hardware-accelerated low precision.** FP8 needs Ada (8.9+)/Hopper/Blackwell; FP4/NVFP4
  needs Blackwell (10.0+). The library must validate the requested mode against the
  detected compute capability at runtime and fail fast.
- **API design:** a single `QuantMode { FP16, INT8_QDQ, INT8_CALIB_LEGACY, FP8, NVFP4 }`;
  `INT8_QDQ` (explicit, strongly-typed) is the default and only path that survives TRT 11;
  `INT8_CALIB_LEGACY` (calibrator + `kINT8`) is gated `#if NV_TENSORRT_MAJOR < 11` behind a
  separate factory; never expose `setDynamicRange` as a long-term public API.

Sources: [TRT 11.0.0 notes](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/release-notes-11/11.0.0.html),
[work-quantized-types](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-quantized-types.html),
[nvidia-modelopt](https://pypi.org/project/nvidia-modelopt/), [TRT-Model-Optimizer onnx_ptq](https://github.com/NVIDIA/TensorRT-Model-Optimizer/tree/main/examples/onnx_ptq).

## Reference-project patterns to adopt

From NVIDIA/TensorRT samples (`buffers.h`, `sampleOnnxMNIST`), Polygraphy `TrtRunner`,
ONNX Runtime C++ (`IoBinding`), CV-CUDA, and community wrappers (tensorRT_Pro):

- **RAII smart-pointer ownership of every TRT object** (custom-deleter `unique_ptr`),
  no manual `destroy()`.
- **Name-keyed IO buffers sized from engine metadata** (`getTensorShape` ×
  `getTensorDataType` element size), with **owned output buffers lazily resized** from
  execution-context shapes after `setInputShape`. Drop all binding-index code.
- **Persistent, caller-injectable CUDA stream** owned by the engine object (not a fresh
  stream per call); stable tensor addresses to enable CUDA-graph capture (ORT pattern).
- **Stream-native preprocessing as a separate stage** producing GPU-resident tensors
  handed straight to `setTensorAddress` + `enqueueV3(stream)` with no host round-trip
  (CV-CUDA async-operator-on-stream model).
- **One error model at the boundary** (ORT uses exceptions; TRT samples use bool+logger
  — pick one; the design picks `Status`/`Result`).

## What to target NOW (the binding decisions)

| Dependency | Target | Notes |
| --- | --- | --- |
| TensorRT | build on **10.16.x `-cu12`**; code to the **11.0** surface | version-gate calibrator/precision-flags `#if NV_TENSORRT_MAJOR < 11`; shared-link only |
| CUDA | **12.6** on this host; **13.3** on driver ≥580 hosts | `CMAKE_CUDA_ARCHITECTURES=86` + PTX fallback; stream-ordered allocator + graphs |
| cuDNN | **not linked** for inference | only enters via optional OpenCV DNN-CUDA, which we keep OFF |
| OpenCV | **4.13.x, optional** | apt CPU default; CUDA build behind default-OFF option; never `OPENCV_DNN_CUDA` |
| Preproc | **hand-rolled fused CUDA kernel** core | optional CV-CUDA 0.16 / NPP `_Ctx` backends, separate sublib |
| Quantization | **explicit QDQ via nvidia-modelopt 0.44** | INT8 only on Ampere; FP8/NVFP4 runtime-gated; legacy calibrator gated `<11` |
| C++ / build | **C++17**, CMake **≥3.22**, shared+static via `BUILD_SHARED_LIBS` | exported `tensorrt_cpp_api::` targets, `find_package` config |
