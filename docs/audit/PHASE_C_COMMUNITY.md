# Phase C — Community Signal (Issues & PRs)

From two parallel triagers over `cyrusbehr/tensorrt-cpp-api` (73 issues, 7 open;
one open PR). Categories ordered by complaint volume; every category cites issue
numbers. The "non-negotiable" list at the end is anything recurring in >= 3 separate
issues — those are binding requirements for v7.

## Categories by volume

### 1. Install / build / version-mismatch pain (~20+ issues — dominates the tracker)

Two failure modes:

- **TensorRT API-version skew breaks the build.** `#41`, `#17`:
  `'nvinfer1::ICudaEngine' has no member named 'getNbIOTensors'` (TRT too old);
  `#54` (TRT 10) `no member getNbBindings`; `#13` (TRT 7) destructor errors; `#10`
  `no member buildSerializedNetwork`. The maintainer repeatedly answers "use 8.6+".
  Every major TRT bump silently breaks a cohort.
- **CUDA/library resolution.** `#45`, `#46`: `libcublas.so.12 => not found` (CUDA-11
  users on a CUDA-12 build) — *"I tried softlink the .so.11 to 12 ... the program
  detected it and complained."* `#14` `NvInfer.h: No such file`; `#24` missing
  `#include <cuda_runtime.h>`; `#40` `Could NOT find TensorRT`; `#44` undefined symbol;
  `#78` Windows DLL. OpenCV-with-CUDA is its own swamp: `#32`, `#52`, `#84`, `#37`,
  `#35`, `#89`, `#51`.

Signal: the library is fragile to environment and gives poor diagnostics.

### 2. Dynamic batch / dynamic shape bugs (~8 issues, highest comment counts)

`#80` (13 comments, the most-commented issue): batch > 1 → "Error, not all required
dimensions specified" — never resolved. `#86`: batch=2 → `enqueueV3 ... Cuda Runtime
(invalid argument)`; reporter concludes *"the current implementation ... does not
correctly support dynamic output sizes."* Dynamic H/W unsupported for years: `#29`,
`#20`. Dynamic width added in `#81`/`#82` but `#93` (open) reports the fix is buggy.
`#34` dynamic ONNX `buildSerializedNetwork` fails. (Confirmed in the Phase A audit:
the dynamic path literally cannot run — `EngineRunInference.inl:62`.)

### 3. Output type / layout issues (~6 issues)

Output hard-coded to `float`. `#27`: *"this code assumes 'float' output types ...
incompatible with networks that have an integer output type"* (Argmax → int32).
Tracked as `#38`, `#49` (claimed fixed in PR `#53` — verify it holds). `#47`: FP16
model returns all `nan` (IO buffers not converted FP32↔FP16). `#12`, `#48`: mapping
YOLO output to boxes / `--end2end` NMS confusion. `#3` D2H copy failure.

### 4. Preprocessing limitations (~5 issues)

`#92` (open) — real correctness bug: per-channel mean/std is wrong because
`blobFromGpuMats` applies a single `cv::Scalar` over the interleaved layout *"doesn't
work correctly when these values are different for different color channels"*
(MobileNetV3/ImageNet stats). Non-3-channel unsupported: `#83` (open, good-first-issue,
grayscale), `#87`, `#11` (NHWC/NCHW confusion). Hard-coded RGB assumption: `#93`
*"engine is built on assumption that it will be used with rgb image, but it is not."*

### 5. Plugin / multi-stream / async asks (~5 issues)

`#88`: custom ONNX plugin (`TRTBatchedNMS`) fails to deserialize — *"Cannot find
plugin ... IPluginCreator not found"*; only worked via a `dlopen()` hack. `#28`:
*"does enqueueV3 support multi-stream inference?"* — no; must make one `Engine` per
stream. `#57`, `#85`: multithread — `#85` reports memory grows until OOM under load.
`#43`: zero-copy — *"can we use the pointer to the GpuMat directly ... and remove the
memcopy?"* — maintainer: "Yes ... requires the code to be reworked."

### 6. Jetson / DLA gaps (~6 issues)

`#58` (open) Jetson Orin NX: dies on `opencv2/cudaarithm.hpp: No such file` and on
`nvinfer1::DataType::kFP8` (absent in JetPack 5.x TRT 8.5); a `TENSORRT_VERSION`-guard
proposal is unmerged. `#17` (AGX Xavier), `#1` (Jetson segfault), `#21` (TX2), `#59`
(wants a CPU-only OpenCV path), `#51`. Theme: JetPack ships old TRT + non-CUDA OpenCV.

### 7. INT8 / quantization (~3 issues)

`#91` (open): *"FP16 and INT8 have almost the same inference speed ... why?"* — no
answer. `#56`: calibration-table confusion. `#18`: INT8 segmentation failing.

### 8. API ergonomics (~10 issues, mostly maintainer-driven, healthy)

`#63` (open) unit tests + CI (stalled on no-GPU GH runners); `#69` IEngine interface;
`#64` spdlog; `#15` enqueueV3 upgrade; `#72`/`#75` refactors. `#93`'s author is
independently rewriting toward an OOP / `tl::expected` API.

## Open PR

- **PR `#94`** (CharaVerKys, +115/-18) bundles: (1) the `minInputWidth` profile fix for
  `#93` (trivial, correct); (2) a `blobFromGpuMats` rewrite for correct per-channel
  preprocessing (`#92`) — the single most impactful user fix; (3) comments noting the
  TRT-10 binding-order fragility; (4) a `build_opencv.sh` tweak removing cuDNN flags.
  It is messy (commented-out blocks, informal notes) and sits unreviewed — but the
  underlying bugs are real and have open issues. v7 implements all of these cleanly.

## Merged-PR trajectory (what the project has been converging toward)

Each merge chips away at hard-coded YOLO assumptions: `#39` fixed batch>1; `#82`
dynamic input width; `#60` `swapRB`; `#53` templated output type + load-engine-direct +
CLI. Refactors toward decoupling: `#72` split into `.inl`/`.h`; `#75` `IEngine`
interface; `#74` inline logging. Polish: `#68` `LOG_LEVEL`; `#77` auto-create engine
dir. The maintainer merges focused, issue-linked PRs and closes sprawling multi-concern
ones (`#16`, `#19`, `#62`). v7 is the natural endpoint of this arc.

## NON-NEGOTIABLE for v7 (each recurs in >= 3 separate issues)

1. **TensorRT version compatibility & guarding** — `#41`, `#17`, `#54`, `#13`, `#10`
   (+ `#45`/`#46`/`#34`). Explicit `NV_TENSORRT_MAJOR` compile guards, a documented
   supported-version matrix, and a clear CMake-time error naming the required TRT.
2. **Robust install / CMake diagnostics** — `#14`, `#24`, `#40`, `#44`, `#45`, `#46`
   (+ OpenCV `#32`/`#52`/`#84`/`#37`). Auto-detect TRT/CUDA, emit "found X, need Y",
   no manual CMake edits. (The `install_deps.sh`/`verify_deps.sh` scripts + the
   optional-OpenCV decision directly target this.)
3. **Correct, configurable preprocessing** — `#92` (broken per-channel norm), `#93`
   (hard-coded RGB), `#83`/`#87`/`#11` (non-3-channel / layout). Per-channel mean/std on
   the right slices; arbitrary channel counts; configurable colorspace.
4. **Dynamic batch & shape, end-to-end** — `#80`, `#86`, `#29`, `#20`, `#93`, `#34`.
   Correct opt-profile setup AND dynamic-aware output sizing, with tests (dynamic batch
   is the single most-reported bug).
5. **Non-float output types** — `#27`, `#38`, `#47` (+ `#12`/`#3`). Use
   `getTensorDataType` + typed/byte output buffers; handle int32 (Argmax) and FP16.
6. **First-class Jetson support** — `#58`, `#17`, `#1`, `#59`, `#21` (+ `#51`). Guard
   JetPack-absent symbols, support a CPU-only/optional-OpenCV path, handle older-TRT
   Jetson. (In v7 scope as best-effort ARM64; x86_64 is the primary target.)

Additional strong (≥2) signals folded into the design: **multi-stream / multithread**
(`#28`, `#57`, `#85`), **zero-copy device input** (`#43`), **first-class plugin loading**
(`#88`), **unit tests + CI** (`#63`).
