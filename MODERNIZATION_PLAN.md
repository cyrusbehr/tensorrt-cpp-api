# TensorRT-CPP-API Modernization — Autonomous Execution Spec

> **You** are an autonomous coding agent on a GPU-equipped Linux box. **This document is your mission.** Read it end-to-end before doing anything. Execute the process described here from start to finish. Do not ask the human to confirm your plan, review intermediate work, or pick between options. The only legitimate reason to stop and prompt the human is a `sudo` password or NVIDIA-developer login — see §7.

> **STATUS UPDATE (2026-05-29, from Phase B research).** This plan was written
> anticipating TensorRT 11 as a future event. **TensorRT 11.0.0 reached GA on
> 2026-05-27.** Therefore, wherever the plan says the calibrator family / weak typing
> / `IPluginV2` are "slated for removal in TRT 11" or talks about "when TRT 11 ships,"
> read those as **already removed** (TRT 11 removed `IInt8Calibrator*`,
> `setInt8Calibrator`, `setDynamicRange`, the per-precision `BuilderFlag`s, the
> `IPluginV2` family, and all static libs; strongly-typed networks are the default).
> Locked targets (see `docs/audit/PHASE_B_ECOSYSTEM.md`): **build on TensorRT 10.16.x
> (`-cu12`) but write all code to the TRT 11.0 surface**, version-gating the few
> legacy calls behind `#if NV_TENSORRT_MAJOR < 11`; explicit Q/DQ (via
> `nvidia-modelopt`) is the default quantization path. **CUDA:** the dev host's driver
> (565.57.01) caps CUDA at 12.7, so this host targets **CUDA 12.6** and CUDA 13.x is
> not used here (it needs driver >= 580); fresh/CI hosts target CUDA 13.3. **OpenCV**
> (4.13.x) and **cuDNN** (9.22, unused for pure inference) are optional, not hard deps.
> The plan's process and operating rules are otherwise unchanged.

---

## 1. The Mission (verbatim from the maintainer)

> Consider the following repository. I wrote it back before the day and age of agentic coding, so I think a lot of it was poorly designed / suboptimal / may have bugs. Despite that, it got nearly 800 stars and a lot of people are making use of it. My intention is to update the project fully using AI, I'm ok with a huge refactor if it means improved API and more robust design. Ultimately I want to convert this into a good project into the BEST resource for using TensorRT for CNN type models. So that will include improving the code and API, but also bringing it up to date with latest API and CUDA and OpenCV and TensorRT if you feel that's worthwhile, because I think a lot of things have changed. Ultimately we could support various cuda versions, including the newest, and have automatic support, but also guide the user to which version of what to install, etc. So start by becoming familiar with the code fully, then understanding any changes in CUDA, OpenCV, TensorRT, then installing the latest stable versions (you can compile them from source), to understand and learn about the APIs, and then from there we can start to design the improved version of this library. You can also query github to see what issues are open, what suggestions people have, what problems they are running into. We don't need to support Windows, I'm ok just supporting Linux. You don't need to implement what people have opened as PRs, but it will just give an idea of the features people are looking at or possible bugs. But do a very deep dive of the code and audit against documentation and known examples. I know in the current state the code works, but maybe it's brittle, or could be improved, and maybe things have changed in the newer APIs. Also in the plan should be steps to pull the libraries we need, to compile them, install deps, ex. executing parallel subagents where it makes sense. The intention is to require minimal input from me. I trust your judgement. I don't need gates to review your plan, or implementation. I want everything to be autonomous. So that's why the plan needs to be flawless, to ensure everything is reviewed and cross audited. The only time you can ask for me input is if you require a sudo command or something you can't install yourself. Otherwise it should run to completion without user intervention.

**Repository:** `cyrusbehr/tensorrt-cpp-api` (~800 stars, currently V6.0, TensorRT >= 10.0 required).
**Sibling repos that consume this library:** `cyrusbehr/YOLOv8-TensorRT-CPP` and `cyrusbehr/YOLOv9-TensorRT-CPP`. These are migration targets in Phase J; the API change is not done until both build green against v7.
**Goal:** become the canonical open-source C++ reference for high-performance TensorRT inference on CNN-class models, with optional first-class Python bindings so the Python ML community can use the same engine without a perf cliff.
**Out of scope:** Windows support, LLM/transformer-specialized features.

---

## 2. Operating Rules

1. **Autonomous.** Run the full process below to completion without checkpoints with the human. Do not ask "should I proceed?", "does this look right?", or "which option do you prefer?" Make the call and document the rationale in code or commit messages.
2. **Stop only for sudo / credentials.** See §7. Any other blocker is for you to resolve by retrying, picking a sane default, or routing around it.
3. **Cross-audit replaces human review.** Every implementation phase ends with parallel review subagents (§8). A phase is only "done" when at least one independent reviewer signs off and you've addressed any structural issues raised.
4. **Trust but verify.** When a subagent returns "looks good", spot-check its claim against actual files. Subagents hallucinate; you are the integrator.
5. **Commit per phase, not per file.** Each phase below should land as 1–3 commits with messages prefixed by the phase letter (e.g., `[E1] add Tensor type`).
6. **Branch from `main`.** First action of §3.0 is `git checkout -b v7-rewrite`. Keep `main` clean so the maintainer can revert.
7. **Never `--force-push`, never amend commits**, never skip git hooks, never modify `.git/config`. Standard repo discipline.
8. **No half-finished features.** If you start a feature and discover it's out of scope, revert — don't leave a stub.
9. **Comments are for non-obvious WHY only.** Don't narrate the code; the API names should describe themselves.
10. **No time estimates anywhere.** The maintainer doesn't want them.

---

## 3. The Process

The process is nine phases, A through I. Phases A–C are read-only research; D is design; E–H are implementation; I is validation and release. The phases form a strict order, but each phase parallelizes its internal work via subagents.

### Phase A — Onboard and audit the existing code

**Goal:** know every line of the current codebase, its design problems, and its outright bugs.

Steps:

1. `git log --oneline -100` and read README.md, CHANGELOG (if any), CMakeLists.txt, every file under `src/` and `include/`, every script under `scripts/` and `cmake/`. Do this with the `Read` tool, not `cat`.
2. Build a mental map: who owns memory, who creates the CUDA stream, what's templated, what's hard-coded.
3. Dispatch **3–5 parallel audit subagents** with these distinct angles. Run them concurrently:
   - **API-surface auditor** — read every public symbol the library exports. Note awkward signatures, missing const-correctness, leaked dependencies (e.g., spdlog/OpenCV types showing through public headers), hard-coded assumptions (3-channel, fixed rank, NCHW only).
   - **Correctness auditor** — read the implementation files looking for: index-by-position assumptions about TensorRT IO ordering, lifetime bugs, double-free / leak patterns, stream/sync correctness, error-path bool/throw inconsistencies, missing validation of user input.
   - **Modernity auditor** — for each TensorRT call, check the current TensorRT docs for the recommended idiom (e.g., `enqueueV3` + `setTensorAddress` + name-based IO, `IPluginV3`, strongly-typed networks). Flag every deprecated call.
   - **Build-system auditor** — read CMakeLists.txt and `cmake/*.cmake`. Note hard-coded paths, missing exports, missing version detection, `-Ofast -DNDEBUG` defaults, lack of installed-target support, no tests, no CI.
   - **Quantization auditor** — read `Int8Calibrator.h`/`.cpp` and the build path that uses it. Note that `IInt8EntropyCalibrator2` and the entire `IInt8Calibrator*` family are slated for removal in TensorRT 11 — the audit should call out what currently uses calibrator-only patterns and what would need to change to support explicit-QDQ ONNX.
4. Synthesize subagent outputs into a single `docs/audit/PHASE_A_FINDINGS.md` ordered by severity (correctness bugs > API design > deprecations > nits). Cite file:line for every finding.
5. Commit: `[A] audit current implementation`.

**Pass condition:** PHASE_A_FINDINGS.md exists, every finding has a file:line citation, and you can speak to each finding without re-reading the source.

### Phase B — Research the 2026 ecosystem

**Goal:** know what's current and what's deprecated for TensorRT, CUDA, cuDNN, OpenCV, quantization, and reference projects. **Use the current date** to scope searches — anything more than two minor releases old may be deprecated.

Dispatch **6 parallel research subagents** with these scopes (run all simultaneously):

1. **TensorRT ecosystem** — current GA major.minor, announced next-major timeline, full diff of API changes since TRT 10.0, deprecation list (calibrator API, `IPluginV2*`, weakly-typed networks, implicit batch/quant), explicit-quantization recommended workflow, weight streaming, refit, multi-context, CUDA-graph capture, DLA path. Cite primary sources (developer.nvidia.com, github.com/NVIDIA/TensorRT releases).
2. **CUDA ecosystem** — current GA version, driver compatibility matrix, supported compute capabilities (note CUDA 13 dropped Maxwell/Pascal/Volta), stream-ordered allocator status, CUDA Graphs maturity, memory-pool best practices, what's deprecated since 12.0.
3. **cuDNN + OpenCV** — current cuDNN minor, whether cuDNN is needed at runtime for pure-TRT inference (it isn't; only OpenCV-cuda pulls it in), current OpenCV stable release, OpenCV 5 status, OpenCV-CUDA build flags for current CUDA, whether `cv::cuda` still requires `opencv_contrib`.
4. **Vision pre/post on GPU alternatives** — survey NVIDIA CV-CUDA (current version, API stability, CUDA support, operator coverage for resize/cvtcolor/normalize/reformat), NVIDIA NPP, and the cost/benefit of hand-rolled CUDA kernels for just the 4–5 ops a TRT preproc pipeline needs.
5. **Reference project survey** — read the public C++ API of NVIDIA/TensorRT samples (`samples/common/buffers.h`, `samples/sampleOnnxMNIST`, `samples/sampleOptions.h`), NVIDIA Polygraphy, ONNX Runtime C++ API, CV-CUDA, NVIDIA/cuda-samples, and 2–3 popular community wrappers. For each, capture: buffer-management pattern, stream ownership, error model, dynamic-shape API, plugin loading, multi-context, preprocessing separation.
6. **Quantization landscape** — implicit PTQ via calibrator (status: deprecated → removed in TRT 11), pre-computed dynamic ranges, QDQ-aware ONNX produced by NVIDIA Model Optimizer (PyPI `nvidia-modelopt`), FP8/INT4/NVFP4 status by GPU architecture, recommended workflow for a new C++ library that should work today and not break when TRT 11 ships.

Each subagent returns a <=2000-word report with primary-source citations. Synthesize all six into `docs/audit/PHASE_B_ECOSYSTEM.md`. Note ambiguities and pick a path through them — do not leave open questions for the human.

Commit: `[B] research 2026 ecosystem state`.

**Pass condition:** PHASE_B_ECOSYSTEM.md has a concrete recommendation for every dependency version, with cited reasoning, and clearly states what to target now and what's about to break.

### Phase C — Mine the community

**Goal:** know what real users complain about and what features they want, so the redesign addresses them.

Steps:

1. With `gh` CLI, list **every** issue and PR (open and closed):
   - `gh issue list -R cyrusbehr/tensorrt-cpp-api --state all --limit 500 --json number,title,state,createdAt,closedAt,labels,comments,author`
   - `gh pr list -R cyrusbehr/tensorrt-cpp-api --state all --limit 200 --json number,title,state,createdAt,mergedAt,author,additions,deletions`
2. Dispatch **2 parallel subagents**:
   - **Issue triage** — read the top 30 most-commented and all open issues with `gh issue view <n> --comments`. Categorize: install pain, dynamic-batch/shape bugs, preprocessing limitations (channels, color space, normalization), plugin/multi-stream/async asks, output-type/layout issues, Jetson/DLA gaps, version-mismatch errors. For each category, cite issue numbers and quote the most representative user phrasing.
   - **PR triage** — read every open PR and the 10 most recent merged PRs with `gh pr view` + `gh pr diff`. Note what unmerged work is sitting unreviewed (it's almost certainly relevant signal) and what patterns the historical PRs converge on.
3. Synthesize into `docs/audit/PHASE_C_COMMUNITY.md` with a prioritized "things the rewrite must address" section. **Anything that has >=3 separate issues filed against it is non-negotiable to fix.**
4. Commit: `[C] mine community issues and PRs`.

**Pass condition:** every category in PHASE_C_COMMUNITY.md has cited issue numbers, and the prioritized list reads as an actionable backlog.

### Phase D — Design

**Goal:** lock the public API and internal architecture. No code yet, just decisions and signatures.

Steps:

1. Synthesize A + B + C into a design document at `docs/design/PHASE_D_DESIGN.md`. Cover, at minimum, the headings below. Do not enumerate options for the human — pick one and justify it inline:

   - **Scope.** What's in (build/load/run engines for CNN ONNX models on Linux x86_64 with optional ARM64/Jetson) and what's not (Windows, LLM-specific features, Python-first APIs).
   - **Supported versions.** Target TensorRT major.minor range, CUDA range, GCC/clang range, CMake minimum, C++ standard. Be specific about what's required, what's optional, and what's auto-detected.
   - **Public API surface.** A minimal set of types and entry points. The maintainer's current API (templated `Engine<T>` with triply-nested-vector inputs/outputs and OpenCV-CUDA inputs) has shipped pain across many issues — design something better. At minimum, decide: how is a tensor represented at the API boundary; how is the CUDA stream provided; how are dynamic shapes specified for multi-input multi-dim models with multiple profiles; how are errors returned; how is the logger injected; how are plugins loaded; how is multi-context multi-stream inference exposed; how is preprocessing separated from engine execution.
   - **Internal architecture.** Module split (engine core, buffer management, calibration, plugins, preprocessing helpers, optional OpenCV interop). Header layout. Link-time vs compile-time dependencies. Header-only vs compiled library — pick one and commit.
   - **Quantization story.** PTQ-via-calibrator vs explicit-QDQ vs both. If both, how are they exposed in the API without confusing users. Make sure the design survives `IInt8EntropyCalibrator2` removal in TRT 11 — gate cleanly behind version checks.
   - **Engine cache.** What goes in the filename, what goes in a sidecar, how to detect corruption / stale caches across TRT versions and GPUs.
   - **Build and packaging.** CMake project layout, exported targets (`tensorrt_cpp_api::tensorrt_cpp_api`), `find_package` support, install rules, optional vcpkg/conan recipe stubs.
   - **Testing.** Framework (GoogleTest or Catch2 — pick one), unit-test conventions, integration-test corpus (small models per shape/layout combination), GPU-required vs no-GPU tests.
   - **Examples.** Three reference examples covering the three major CNN tasks: detection (YOLOv8n is canonical), classification (ResNet50 or MobileNetV3), segmentation (DeepLabV3 or U-Net). Each example must end-to-end run on a fresh checkout.
   - **CI.** GitHub Actions matrix: which CUDA × TRT combinations to test; how to handle GPU-required tests (self-hosted runner stub OR CPU-only build + lint pass with GPU tests gated to manual workflow).
   - **Compatibility.** Whether and how to maintain a v6-compatible shim header so YOLOv8-TensorRT-CPP / YOLOv9-TensorRT-CPP keep building. Pick a stance; don't both-ways.

2. Produce **complete header-file sketches** (not just descriptions) for every public type, with `///` doc comments. The other phases will treat these sketches as binding.

3. Dispatch **3 parallel design-review subagents** before locking:
   - **API ergonomics reviewer** — does the API read naturally for the YOLOv8 use case? Hello-world should be <=30 lines.
   - **TRT correctness reviewer** — does the design avoid the known-bad patterns (binding-index ordering assumptions, calibrator/explicit-quant conflation, single-context bottleneck)?
   - **Build-system reviewer** — is the CMake design idiomatic enough for vcpkg/conan/system-install consumers?

4. Address review feedback inline; revise the design doc. Re-run reviewers only if you made structural changes.

5. Commit: `[D] lock public API and architecture`.

**Pass condition:** PHASE_D_DESIGN.md contains complete header sketches and rationale for every design choice; all 3 reviewer agents have explicitly signed off.

### Phase E — Implement the core library

**Goal:** working C++ library compiling against the latest TRT + CUDA on the target system.

Decompose the implementation into **8–14 small phases (E1, E2, …)** following the dependency order in your design. Suggested rough decomposition (the agent should refine):

- **E1** — Core types: `Tensor`, `Shape`, `DType`, `Layout`, `Status`/`Result`. No CUDA dep yet.
- **E2** — `ILogger` interface + default stderr logger + optional spdlog adapter behind a CMake option.
- **E3** — CUDA helpers: stream wrappers, error -> Status converters, allocator abstraction (default stream-ordered).
- **E4** — TRT smart-pointer wrappers, logger bridge from `ILogger` to `nvinfer1::ILogger`, plugin-registry loader.
- **E5** — Buffer management: per-tensor-name device buffer + optional pinned host buffer + name-keyed lookups.
- **E6** — Engine builder: ONNX -> engine, options for precision, optimization profiles (multi-input, multi-dim, multi-profile), strongly-typed networks, workspace memory pool, DLA core selection, timing-cache save/load, plugin loading.
- **E7** — Engine cache management: content-hashed filename, sidecar JSON metadata (ONNX hash, TRT version, GPU UUID, build options), atomic write, integrity check on load.
- **E8** — Engine runtime: load from disk, deserialize, name-based binding allocation, `enqueueV3` + `setTensorAddress` per IO tensor, dynamic-shape input-shape setting, output-shape inference for caller-side allocation.
- **E9** — Multi-context pool: one engine, N execution contexts, lease-based acquisition, caller-provides-stream model.
- **E10** — Calibration: a `ICalibrator` interface plus a concrete `ImageDirectoryCalibrator` that wraps the legacy `IInt8EntropyCalibrator2` API. Gate clean behind TRT-version macros so explicit-QDQ engines compile without it.
- **E11** — Preprocessing sublibrary: hand-rolled CUDA kernels for the small ops needed by typical CNN inputs (resize + letterbox, BGR<->RGB swap, mean subtraction + scale, NHWC<->NCHW reformat). Linkable independently. The engine core does **not** depend on this sublibrary.
- **E12** — Optional OpenCV interop header (only built if `find_package(OpenCV)` succeeds and a CMake option is on): adapter functions between `cv::cuda::GpuMat` and the library's `Tensor`. Strictly optional; engine core knows nothing about OpenCV.
- **E13** — Optional `cv::Mat` (host) interop helpers behind the same opt-in. Convenience only.
- **E14** — `tensorrt_cpp_api/version.h` with compile-time version constants; `tensorrt_cpp_api/all.h` umbrella header for users who want one include.
- **E15** — Optional Python bindings behind a CMake option `TRT_CPP_API_BUILD_PYTHON=ON` (default OFF; ON in CI for the dedicated bindings job). Use **pybind11** (header-only, ABI-stable, well-tested with CUDA types) wrapped by **scikit-build-core** for wheel building. The bindings must be **performant by construction**:
  - Wrap `Tensor`, `TensorView`, `Shape`, `DType`, `Layout`, `Status`/`Result`, `IEngine`, `IEngineBuilder`, `EnginePool`, `ICalibrator`, `ILogger`.
  - **Zero-copy device interop:** every `Tensor` exposes `__cuda_array_interface__` (legacy CuPy/Numba) **and** `__dlpack__` + `__dlpack_device__` (PyTorch >= 1.10, CuPy >= 10, JAX, etc.). Accept DLPack capsules and `__cuda_array_interface__` dicts on input — no `cudaMemcpy` between framework and library.
  - **Caller-provided CUDA stream:** the Python `enqueue()` accepts an integer stream handle (PyTorch `torch.cuda.current_stream().cuda_stream`, CuPy `cupy.cuda.get_current_stream().ptr`, `int(stream)`) — never creates its own.
  - **GIL released on every long-running call:** `enqueue`, `build_from_onnx`, `build_or_load`, `load_engine_file`, `EnginePool::acquire`. Use `py::call_guard<py::gil_scoped_release>()`.
  - **No implicit D2H copies.** `Tensor.numpy()` and `Tensor.cpu()` exist and explicitly copy. Cross-device implicit copies raise.
  - **Pythonic naming** in the binding layer: `engine.enqueue(inputs={"images": t}, outputs={"output": o}, stream=...)` reads naturally; the underlying C++ method names are wrapped, not exposed verbatim.
  - Wheel built and tested via scikit-build-core; ship `pyproject.toml` with platform-specific CUDA / TRT wheel suffixes (e.g., `tensorrt_cpp_api-cu13`). PyPI publishing is **not** done by the agent — that's a maintainer-controlled step.

**Per-phase protocol (mandatory):**

a. Implement the phase in 1–3 commits.
b. Write unit tests for the phase before declaring it done. Unit tests run on no GPU where possible (most type/buffer/logging tests). Integration tests requiring a GPU are tagged.
c. `cmake --build` and `ctest` both pass (including the GPU-required subset on this machine).
d. Dispatch parallel cross-audit subagents (§8). At minimum: code reviewer + API consistency reviewer. For phases touching CUDA: add a CUDA correctness reviewer.
e. Address any blocking review feedback. Nits can be deferred to a single end-of-phase cleanup commit.
f. Commit: `[E<n>] <short title>`.

**Pass condition for Phase E:** every E-subphase has merged commits + green tests + at least one positive cross-audit. A `cmake --install` produces an installable artifact other projects can `find_package`. If E15 is enabled, `pip install .` from the repo root produces an importable `tensorrt_cpp_api` Python package.

### Phase F — Reference examples

**Goal:** four example apps — three C++ (detection, classification, segmentation) plus one Python (detection via the bindings) — each <=300 lines, each consuming the **installed** library (not in-tree headers).

Steps:

1. For each task, pick a small public ONNX model:
   - **Detection:** YOLOv8n (the de-facto sanity-check from the existing repo).
   - **Classification:** ResNet50 or MobileNetV3 from torchvision (export with `opset>=17`).
   - **Segmentation:** DeepLabV3 MobileNetV3 or a small U-Net.
2. For each C++ example, place under `examples/<task>/`: `main.cpp`, `CMakeLists.txt`, `README.md` (5–10 lines on what the example does, model download URL, run command).
3. Each C++ example must use the **public installed-package** interface, not in-tree paths. Build each via a tiny standalone CMake project that does `find_package(tensorrt_cpp_api)`. This is your "is the install actually usable" smoke test.
4. **F4 — Python example.** Place under `examples/python/yolov8_detect.py` (+ a one-page `examples/python/README.md`). It must:
   - Import `tensorrt_cpp_api` from the installed wheel (not from the source tree).
   - Build-or-load the YOLOv8n engine via the bindings.
   - Read an image with `cv2`, upload to GPU via either **`torch.from_numpy(...).cuda()`** or **`cupy.asarray(...)`** (demonstrate one path, comment the other).
   - Pass the GPU tensor into `engine.enqueue(...)` via DLPack — **prove zero-copy** by asserting the device pointer matches before and after.
   - Use the caller's current CUDA stream (`torch.cuda.current_stream()` or equivalent).
   - Read the output back, run NMS in NumPy or torch, draw boxes with `cv2`.
   - **Benchmark mode flag** (`--benchmark`) that prints FPS and asserts the Python path is within **20%** of the C++ FPS for the same model on the same GPU. If it isn't, that's a binding-layer bug — fail loudly.
5. Dispatch a **single review subagent** per example to confirm it reads as canonical (no leftover demo cruft, no hard-coded paths that aren't environment variables, no model-specific magic numbers without comments explaining them). For F4, the reviewer must specifically verify zero-copy interop, GIL release, and caller-stream usage.
6. Commit each example separately: `[F1] yolov8 example`, `[F2] classification example`, `[F3] segmentation example`, `[F4] python yolov8 example`.

**Pass condition:** all four examples build, run, and produce sensible output against the supplied model. F4 specifically passes the zero-copy assertion and the 20% perf-parity check.

### Phase G — Tests, CI, sanitizers

Steps:

1. Confirm every E-phase has unit tests. Add integration tests covering: dynamic batch >1 (the #1 community bug), dynamic H+W simultaneously, multi-input multi-output models, mixed-dtype outputs, multi-context concurrent inference (>=4 streams), plugin loading roundtrip, INT8 calibration cache roundtrip, FP16 precision build, engine cache hit / miss / stale, atomic engine-file write under simulated crash mid-write.
2. Add ASan + UBSan builds. Add an optional CUDA-MEMCHECK target.
3. `.github/workflows/ci.yml`: matrix on Ubuntu 22.04 + 24.04, two CUDA versions, two TRT versions. CPU-only build + format + lint always; GPU tests gated to manual `workflow_dispatch` because GitHub-hosted runners lack NVIDIA GPUs. Document how a downstream user can plug in a self-hosted Jetson or x86 GPU runner.
4. `.github/workflows/format.yml`: clang-format check, clang-tidy on changed files.
5. Add `pre-commit` hooks for clang-format and `cmake-format`.
6. Dispatch **2 reviewers**: a CI-config reviewer and a test-coverage reviewer.
7. Commit: `[G] add tests, sanitizers, and CI`.

**Pass condition:** CI green on a fresh push to the branch (CPU portions); test coverage subagent reports >70% line coverage on the engine and buffer modules.

### Phase H — Documentation

Steps:

1. Rewrite `README.md` from scratch. Sections in this order:
   - One-line pitch and badges.
   - **Install section, prominently placed near the top.** Three documented paths, in this order: (a) the **recommended** one-liner — `bash scripts/install_deps.sh` (link to the script); (b) a manual apt block for users who don't trust scripts (the same apt commands the script runs, copy-pasteable); (c) a from-tarball path for hosts that can't use the NVIDIA apt repo (CUDA runfile + TensorRT tarball, with the NVIDIA download links and a note that the tarball requires a free NVIDIA developer account).
   - 30-line quickstart (the YOLOv8 example, end-to-end), assuming the install section ran.
   - Feature list, supported precision/quantization modes.
   - **Python install + quickstart:** `pip install .` from the repo root or `pip install tensorrt_cpp_api` (when wheel is published), plus a 15-line Python quickstart that mirrors the C++ one. Note the `--with-python` flag to `scripts/install_deps.sh`.
   - Links to per-task examples (C++ and Python), API docs, "upgrading from v6" guide, contributors, license.
   - **No `<table>` HTML — use markdown tables only.**
2. Polish `scripts/install_deps.sh` (drafted in Phase 0). Add per-flag `--help` output, clearer error messages with remediation pointers, and a section banner before each phase ("Adding NVIDIA apt repo…", "Installing CUDA toolkit…", etc.). Also add a `scripts/verify_deps.sh` companion that runs a non-destructive check ("can the host build the library?") and exits 0/1 — useful for CI and for users diagnosing install problems.
3. Update `AGENTS.md` (drafted in Phase 0) to reflect the final v7 state: final API entry points, where the design decisions are documented, links to the upgrade guide. AGENTS.md and README must agree; if they diverge, README wins for humans and AGENTS.md wins for LLM agents — keep them in lockstep.
4. Write `docs/upgrading_from_v6.md` mapping every v6 public symbol to its v7 equivalent. If a v6 symbol has no direct replacement, explain the conceptual swap (e.g., "v6's `std::vector<std::vector<std::vector<T>>>` output -> v7's name-keyed `std::unordered_map<std::string, Tensor>`").
5. Doxygen-generate `docs/api/` from the public headers. Commit the generated HTML to a `gh-pages` branch via a `.github/workflows/docs.yml` workflow, not to `main`.
6. Write `docs/tutorial/` markdown lessons (4–6 short pages): engine lifecycle, dynamic shapes, INT8 PTQ, INT8/FP8 explicit-quant, plugins, multi-stream, Python interop (zero-copy DLPack patterns).
7. Write `docs/install/`:
   - `docs/install/quick.md` — one page: "run `scripts/install_deps.sh`, that's it." Links to the next pages for the curious.
   - `docs/install/manual_apt.md` — the apt commands, line by line, with what each package provides.
   - `docs/install/from_source.md` — for users who need to build OpenCV with CUDA support, or who need a CUDA/TRT version not in the apt repo. Keep `scripts/build_opencv.sh` (the existing one, modernized) and document its flags.
   - `docs/install/compatibility_matrix.md` — recommended NVIDIA driver / CUDA / cuDNN / TRT / OpenCV combos with citation links to NVIDIA's compatibility pages. Include a "tested on" table from CI.
   - `docs/install/troubleshooting.md` — top-10 install errors from the Phase C community survey, each with the exact error message and the fix.
8. Dispatch a **single doc-quality reviewer** (§6.9). Treat its output as advisory; the maintainer's voice is more important than reviewer suggestions.
9. Commit: `[H] rewrite docs, install scripts, and upgrade guide`.

**Pass condition:** a fresh reader can install, build, and run the YOLOv8 example following only the README; the upgrade guide accounts for every public v6 symbol.

### Phase I — Final validation and release

Steps:

1. **Benchmark parity check.** Run the v7 YOLOv8n FP32 / FP16 / INT8 benchmarks on the same GPU the existing README cites (RTX 3050 Ti Laptop) or document the GPU used. v7 must be **<= 5% slower** than v6 on the same model and ideally faster. Investigate any regression > 5%.
2. **Multi-thread stress.** Drive the multi-context pool with 4–8 threads for 60 s; assert no leaks (`compute-sanitizer --leak-check full`), no race-condition crashes, and per-thread FPS within 10% of single-thread × N.
3. **Sanitizer run.** ASan + UBSan + LSan + the CUDA-MEMCHECK target on the test suite. Zero unsuppressed reports.
4. **Cold-cache build smoke test.** `rm -rf build && cmake -B build && cmake --build build -j` from a clean checkout. Should succeed without manual editing.
5. **Install smoke test.** `cmake --install build --prefix /tmp/trtcpp` then build the YOLOv8 example as an external project linked against the installed package.
6. **Engine-cache round-trip.** Build engine, modify ONNX hash on disk, confirm v7 detects the stale cache and rebuilds.
7. **Dispatch a final 4-agent parallel review:**
   - Security/CUDA-safety reviewer.
   - API consistency reviewer (spot-checks for naming, const-correctness, noexcept-correctness across the public surface).
   - Documentation reviewer (does the README actually work?).
   - Migration reviewer (does the upgrade-from-v6 guide cover every breaking change?).
8. Bump version to **7.0.0** in `CMakeLists.txt` and `tensorrt_cpp_api/version.h`. Tag `v7.0.0-rc1`. Do **not** push tags or merge to `main` — leave the branch for the maintainer to ship.
9. Write `CHANGELOG.md` summarizing every breaking change and every fix tied back to a community issue number from Phase C.
10. Commit: `[I] v7.0.0-rc1 — final validation`.

**Pass condition:** benchmark within 5% (or better) of v6, sanitizers clean, fresh-clone build works without editing, and all 4 final reviewers signed off.

### Phase J — Sibling-repo migration PRs

**Goal:** prove the new API works for real downstream consumers by migrating both maintainer-owned sibling repos to v7 and opening PRs against them.

This phase is **the most meaningful API validation** in the whole plan: the two sibling repos were written by the same maintainer against v6, so if v7's API can't drive them cleanly, the v7 API needs revision (loop back to Phase D). Do not skip this phase or downgrade it.

The sibling repos (both owned by `cyrusbehr`):

- **YOLOv8-TensorRT-CPP** — https://github.com/cyrusbehr/YOLOv8-TensorRT-CPP
- **YOLOv9-TensorRT-CPP** — https://github.com/cyrusbehr/YOLOv9-TensorRT-CPP

Steps (run for each sibling in parallel, two subagents):

1. `gh repo clone cyrusbehr/<repo>` into `/tmp/sibling-<repo>`. Read its `README.md`, top-level `CMakeLists.txt`, and `src/`/`include/` to map every call into the v6 API.
2. Cross-reference each v6 call against `docs/upgrading_from_v6.md` (Phase H deliverable). Build a per-repo migration checklist before touching code.
3. Create a branch named `update-to-tensorrt-cpp-api-v7` in the sibling repo. Update the sibling's CMake to pull v7 either via `find_package(tensorrt_cpp_api 7.0 REQUIRED)` against a local install **or** via `FetchContent_Declare` from the `v7-rewrite` branch of tensorrt-cpp-api (pick `find_package` if the install was already validated in Phase I.5; otherwise FetchContent).
4. Migrate the call sites. Compile cleanly. Run the sibling repo's existing smoke test (typically: build engine from the bundled YOLO ONNX, run inference on `inputs/team.jpg`, assert non-empty detections / non-NaN feature vector).
5. **Run the sibling repo's benchmark** if it has one. The v7-migrated sibling must be within **5%** of its v6 self on the same GPU. Any larger regression is a bug in v7's API ergonomics or runtime — fix it in tensorrt-cpp-api first, then re-migrate.
6. Push the branch to the sibling repo and open a PR (`gh pr create`). PR title: `Migrate to tensorrt-cpp-api v7`. PR body must include:
   - Link to the v7.0.0-rc1 tag of tensorrt-cpp-api.
   - A bullet list of the API changes affecting this repo (drawn from the upgrade guide).
   - Benchmark numbers: v6 vs v7-migrated, same model, same GPU.
   - The smoke-test output proving correctness.
   - Note that this PR is intended to merge **after** tensorrt-cpp-api v7.0.0 final ships.
7. Do **not** request review or @-mention the maintainer; just open the PR. The maintainer will pick it up.
8. Update `CHANGELOG.md` in tensorrt-cpp-api to link to both sibling PRs.

**Pass condition:** both sibling repos have an open PR on a branch named `update-to-tensorrt-cpp-api-v7`, both builds pass, both smoke tests pass, both benchmarks within 5% of v6, and the PR descriptions are complete. Commit (in tensorrt-cpp-api): `[J] sibling repo PRs opened`.

---

## 4. Phase Dependency Graph

```
A --+
B --+--> D --> E (E1 -> E2 -> ... -> E15) --> F --+
C --+                                             +--> I --> J
                                          G ------+
                                          H ------+
```

A, B, C run in parallel (dispatched as subagents from the orchestrator).
D is a serial synthesis step.
E phases run sequentially (each depends on prior E phases). E15 (Python bindings) is conditionally enabled but should be on by default for v7 — see §11. Inside each E phase, sub-tasks parallelize: implementation in one thread, cross-audit dispatched in parallel.
F, G, H all depend on E completing; they can run in parallel.
I is the final serial validation.
J runs after I and dispatches two parallel sibling-repo migration agents.

---

## 5. When To Parallelize, When To Serialize

**Parallelize** when:
- Information-gathering across disjoint sources (Phase A audit angles, Phase B research scopes, Phase C issue vs PR triage).
- Cross-audit reviewers with disjoint perspectives (code, perf, API, docs).
- Building independent examples (Phase F has 3 examples -> 3 parallel subagents).

**Serialize** when:
- Each step's output is the next step's input (the whole A->B->C->D->E pipeline).
- One subagent writes a file another reads (CMakeLists.txt + module sources).
- You're integrating; merge conflicts compound when parallel subagents edit the same files.

When in doubt, parallelize the **reading** and serialize the **writing**.

Hard rule: never have two subagents write to the same file at the same time.

---

## 6. Subagent Prompt Templates

(See the original spec for the full text of templates 6.1–6.9: Auditor, Ecosystem
researcher, Community triage, Code reviewer, API ergonomics reviewer, CUDA
correctness reviewer, Python-bindings reviewer, Sibling-repo migration reviewer,
Documentation reviewer. Each prompt states: what to do, what context the parent
already has, the output format, and a word cap.)

---

## 7. Sudo / Human-Gate Protocol

You may not run `sudo`, `dpkg -i` of a system package, NVIDIA-developer-login–gated downloads, or anything else that requires interactive human authentication. When you hit one of these:

1. **Stop the current phase cleanly** — commit any work-in-progress with prefix `[WIP <phase>]`.
2. **Emit a single, copy-pasteable command block** to the human's terminal, prefixed by the exact reason.
3. **Wait for confirmation.** When the human says "done", verify with a non-interactive probe (`dpkg -l | grep libnvinfer-dev`) and proceed. If verification fails, output the failure and ask again.
4. **Batch your gates.** If you can foresee a second sudo step, list it in the same block. One human-touch per phase, not many.

Never run `sudo -n` to "test"; never embed `sudo` in a script and hope; never `pip install --user` something to work around a missing system lib that the design genuinely needs.

---

## 8. Cross-Audit Methodology

After every implementation phase (E1…E14, F, G, H, I):

1. Identify 2–4 disjoint review angles relevant to that phase.
2. Dispatch one subagent per angle **in parallel**.
3. Each reviewer returns a structured report: **blocking issues** vs **nits**.
4. Apply all blocking-issue fixes. Defer nits to a single `[<phase>] nits and cleanup` commit at the end of the phase.
5. If a reviewer's blocking issue turns out to be wrong on inspection, write a one-line note in the commit explaining why. Do not just ignore.
6. If two reviewers contradict each other, **you** are the integrator — pick the path that better matches the design doc, and document the call.

Cross-audit is not optional. A phase without a cross-audit is not done.

---

## 9. Environment Bootstrap (Phase 0)

Before Phase A, in a single `[0] env bootstrap` commit (or 2–3 commits):

1. Detect the toolchain: `gcc --version`, `cmake --version`, `nvidia-smi`, `nvcc --version`, `dpkg -l | grep -E 'libnvinfer|libopencv|libcudnn'`.
2. Detect the GPU and record it in `docs/audit/PHASE_0_HOST.md`.
3. **Write `scripts/install_deps.sh`** — the authoritative dependency installer for fresh Ubuntu 22.04/24.04. `set -euo pipefail`, distro detection, idempotent NVIDIA apt-repo add, installs CUDA/TensorRT/OpenCV/spdlog/fmt/build tooling (+ Python deps with `--with-python`). Flags: `--cuda-version`, `--with-python`, `--dry-run`, `--no-sudo`. Verifies post-install. Does NOT install the driver; verifies a compatible driver is present and fails fast otherwise.
4. **Use the script for this machine's bootstrap.** Minimums: CMake >= 3.22, GCC >= 11, driver supporting the installed CUDA, CUDA toolkit (latest GA), TensorRT (latest GA), OpenCV >= 4.9 (only if the design keeps OpenCV interop). Override only with explicit justification.
5. **Write `AGENTS.md`** at the repo root pointing at this `MODERNIZATION_PLAN.md` as the master spec.
6. Verify by building the **current v6** repo end-to-end against the installed toolchain. Fix the build if needed (likely a `TensorRT_DIR` path edit).
7. `git checkout -b v7-rewrite` and commit Phase 0.

---

## 10. Definition of Done

The v7.0.0-rc1 branch is "done" when every item in the original spec's checklist is
true: Phase 0 commit + working v6 build; `install_deps.sh` + `verify_deps.sh`;
`AGENTS.md` in lockstep with README; `PHASE_A/B/C` audit docs; `PHASE_D_DESIGN.md`
with header sketches + 3 sign-offs; every E-subphase committed + tested +
cross-audited; installable via `find_package`; three C++ examples + one Python
example (zero-copy verified, FPS within 20% of C++); CI green on CPU matrix;
sanitizers clean; multi-thread stress clean; YOLOv8 benchmark within 5% of v6;
README rewritten; upgrade guide covers every v6 symbol; CHANGELOG ties fixes to
community issues; final 4-agent review signs off; branch tagged `v7.0.0-rc1`
locally (not pushed, not merged); Phase J sibling PRs open and green.

The maintainer ships v7.0.0 themselves from the branch.

---

## 11. Operational Notes for the Executing Agent

- Read this whole document before any tool call. Re-read §2 when tempted to ask the human anything that isn't a sudo/credential gate.
- You have memory. Save anything about the repo or maintainer preferences future work benefits from.
- Commit messages matter — `[<phase>] <short title>` consistently.
- Don't optimize prematurely. Simplest thing that passes design + tests first; optimize in Phase I if benchmarks come in slow.
- Read the maintainer's `.clang-format` before writing C++.
- Don't gold-plate. Deliverables: the C++ library (E1–E14), optional pybind11 bindings (E15), four reference examples (F), tests/CI (G), docs (H), validation (I), sibling-repo PRs (J). No GUI, no server, no "while we're at it" features.
- Python bindings are first-class but optional, and MUST be performant: zero-copy DLPack / `__cuda_array_interface__`, caller-provided stream, GIL released on long ops, no implicit D2H copies. The 20% perf-parity assertion in F4 is a hard gate.
- When in doubt, look at how `samples/common/` in NVIDIA/TensorRT does it.
- No emojis anywhere.
- No `.md` files outside `docs/`, `examples/<task>/README.md`, and the repo root's `README.md` / `CHANGELOG.md` / `MODERNIZATION_PLAN.md`.
- If you discover this plan is wrong about something concrete, fix the plan first in a separate commit (`[meta] correct MODERNIZATION_PLAN.md`), then proceed.

---

## 12. Recovery and Resumption

If the executing agent crashes, runs out of context, or is interrupted:

1. The next agent reads `git log --oneline -50` to see the last committed phase.
2. Reads this document.
3. Reads the phase docs already committed under `docs/audit/`, `docs/design/`.
4. Picks up at the next uncommitted phase.

Per-phase commits with a `[<phase>]` prefix are the resumption protocol.
