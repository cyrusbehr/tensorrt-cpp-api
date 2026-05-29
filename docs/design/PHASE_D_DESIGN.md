# Phase D — v7 Public API and Architecture (LOCKED)

Synthesis of Phase A (audit), B (ecosystem), C (community) into the binding design
for v7. Header sketches below are the contract the E-phases implement. Decisions are
made, not enumerated; rationale is inline. Reviewer sign-offs are recorded at the end.

Namespace: **`trtcpp`**. Include root: **`tensorrt_cpp_api/`** (e.g.
`#include <tensorrt_cpp_api/engine.h>`). Exported CMake target:
**`tensorrt_cpp_api::tensorrt_cpp_api`**.

## Scope

**In:** build, cache, load, and run TensorRT engines for CNN-class ONNX models on
**Linux x86_64** (best-effort ARM64/Jetson). Name-keyed multi-input/multi-output,
dynamic shapes with multiple profiles, FP32/FP16/INT8, explicit-QDQ quantization,
multi-context multi-stream execution, caller-provided CUDA streams, optional GPU
preprocessing, optional OpenCV interop, optional Python bindings.

**Out:** Windows; LLM/transformer features (KV-cache, attention plugins); a
Python-first API (Python is a thin binding over the C++ core); training; model export
(quantization is an offline `nvidia-modelopt` step the library only consumes).

## Supported versions

| Axis | Required | Notes |
| --- | --- | --- |
| TensorRT | **10.6 – 11.0** | build on 10.16.x today; code written to the 11.0 surface; legacy paths gated `#if NV_TENSORRT_MAJOR < 11`. A CMake-time error names the required range (community #41/#54). |
| CUDA | **12.4 – 13.x** | dev host uses 12.6 (driver 565 caps at 12.7); CI/fresh hosts 13.3. |
| Compute capability | **>= 7.5** (Turing) | matches TRT 11 / CUDA 13 floor. Build `sm_86` here. |
| Compiler | GCC >= 11, Clang >= 14 | **C++20** (corrected from C++17 in E1: the public API uses `std::span`, which is C++20; all target compilers support it, it is a superset of C++17 so the "OpenCV 5 needs C++17 min" forward-compat rationale still holds, and nvcc 12.x supports `-std=c++20`). |
| CMake | **>= 3.22** | `CUDAToolkit`, `FetchContent`, export helpers. |
| OpenCV | 4.9 – 4.13 (**optional**) | only for interop module; never `OPENCV_DNN_CUDA`. |
| Python | 3.9 – 3.13 (**optional**) | pybind11 + scikit-build-core. |

Auto-detected: CUDA toolkit (`CUDAToolkit`), TensorRT (improved `FindTensorRT` +
version extraction), OpenCV (`find_package(OpenCV)`), GPU arch. Everything optional is
behind a default-OFF CMake option.

## Architecture

Compiled shared+static library (honor `BUILD_SHARED_LIBS`), **not header-only** — the
v6 `.inl`-in-headers approach leaks TRT/CUDA into every consumer TU and bloats compile
times. The public headers expose only library-owned types; TRT/CUDA/OpenCV are
implementation details behind PImpl or compiled `.cpp`.

Module split (each a separate translation-unit group; the engine core links none of
the optional ones):

```
include/tensorrt_cpp_api/
  dtype.h        layout.h      shape.h        status.h      tensor.h      # E1 core types (no CUDA)
  logger.h                                                               # E2 ILogger
  cuda.h         allocator.h   device_tensor.h                           # E3 stream/device-query/alloc + owning Tensor (CUDA)
                                                                          #   (Device query folded into cuda.h, no separate device.h;
                                                                          #    owning Tensor split here to keep tensor.h dependency-free)
  build_options.h  quant.h     engine_builder.h                          # E6 build
  engine.h       engine_pool.h                                           # E8/E9 runtime
  calibrator.h                                                            # E10 (gated < TRT 11)
  preproc.h                                                               # E11 optional sublib header
  opencv_interop.h                                                        # E12/E13 optional
  version.h      all.h                                                    # E14
src/...                                                                   # impls + internal TRT/CUDA glue
```

Dependency direction: `tensor`/`status`/`shape`/`dtype`/`logger` depend on nothing
external. `cuda`/`allocator` depend on the CUDA runtime only. `engine*`/`builder`
depend on CUDA + TensorRT (hidden). `preproc` depends on CUDA only. `opencv_interop`
depends on OpenCV. **The engine core never includes OpenCV, spdlog, or nvinfer1 in a
public header** — this is the fix for the largest class of community pain (install /
version skew, Phase C #1/#2).

## Public API — header sketches (binding)

### E1 — `dtype.h`, `layout.h`, `shape.h`, `status.h`, `tensor.h`

```cpp
namespace trtcpp {

/// Element type at the API boundary. Mirrors nvinfer1::DataType but is library-owned
/// so consumers never include NvInfer.h. kFp8/kInt4 require recent TRT + new HW.
enum class DType { kFloat32, kFloat16, kBFloat16, kInt32, kInt64, kInt8, kUInt8, kBool, kFp8, kInt4 };

/// Bytes per element (kInt4 reports 1 with a sub-byte note; callers handle packing).
constexpr std::size_t byteSize(DType) noexcept;
std::string_view toString(DType) noexcept;

/// Memory layout hint for interop/preprocessing. TensorRT IO is layout-agnostic
/// (it sees raw dims); Layout drives preprocessing and OpenCV adapters only.
enum class Layout { kNCHW, kNHWC, kLinear, kUnknown };

/// Where a tensor's data lives.
enum class Device { kHost, kCuda };

/// A dynamic-aware shape (max rank 8). A dim of -1 means "dynamic / unresolved".
class Shape {
public:
    Shape() = default;
    Shape(std::initializer_list<int64_t> dims);
    explicit Shape(std::span<const int64_t> dims);
    int rank() const noexcept;
    int64_t operator[](int i) const noexcept;            ///< no bounds check (hot path)
    int64_t at(int i) const;                              ///< throws std::out_of_range
    bool isDynamic() const noexcept;                      ///< any dim < 0
    /// Element count; returns 0 if any dim is dynamic (caller must resolve first).
    int64_t numel() const noexcept;
    std::span<const int64_t> dims() const noexcept;
    bool operator==(const Shape&) const noexcept;
    std::string toString() const;                         ///< e.g. "[1,3,640,640]"
};

/// A status code. The public API is NO-THROW: every fallible call returns Status or
/// Result<T>. (Internals may throw; the boundary catches and converts. Fixes the
/// v6 bool/throw muddle, audit A:correctness.)
enum class StatusCode {
    kOk, kInvalidArgument, kNotFound, kIoError, kCudaError, kTensorRtError,
    kShapeMismatch, kDtypeMismatch, kUnsupported, kStaleCache, kInternal
};

class Status {
public:
    Status() noexcept;                                    ///< kOk
    Status(StatusCode code, std::string message);
    bool ok() const noexcept;
    StatusCode code() const noexcept;
    const std::string& message() const noexcept;
    explicit operator bool() const noexcept { return ok(); }
};

/// Result<T> = value-or-Status. Lightweight std::expected-style (no exceptions).
template <class T> class Result {
public:
    Result(T value);
    Result(Status error);                                 ///< must be !ok()
    bool ok() const noexcept;
    explicit operator bool() const noexcept;
    T& value() &;  const T& value() const&;  T&& value() &&;   ///< asserts if !ok
    T* operator->();  const T* operator->() const;            ///< std::expected-like deref
    T& operator*() &;  const T& operator*() const&;
    const Status& status() const noexcept;
    T value_or(T fallback) const;
    /// Monadic chaining to cut `if (!x) return x.status();` noise at call sites.
    template <class F> auto and_then(F&&) -> std::invoke_result_t<F, T&>;     ///< F: T -> Result<U>
    template <class F> auto transform(F&&) -> Result<std::invoke_result_t<F, T&>>; ///< F: T -> U
};
/// Convenience macro for the common propagate-on-error pattern (header-only):
///   TRTCPP_TRY(auto engine, Engine::loadFromFile(path, {}));
/// expands to: auto _tmp = expr; if (!_tmp) return _tmp.status(); auto engine = std::move(_tmp).value();
#define TRTCPP_TRY(decl, expr) /* ... */

/// A view over device or host memory the library does NOT own. The primary input/
/// output type at the enqueue boundary — zero-copy by construction. Carries enough
/// to drive setTensorAddress + setInputShape and to build DLPack/__cuda_array_interface__.
class TensorView {
public:
    struct Desc { void* data = nullptr; DType dtype = DType::kFloat32; Shape shape;
                  Device device = Device::kCuda; int deviceId = 0; Layout layout = Layout::kLinear; };
    TensorView() = default;
    explicit TensorView(Desc d);                          ///< named-field ctor (avoids positional soup)
    TensorView(void* data, DType dtype, Shape shape, Device device,
               int deviceId = 0, Layout layout = Layout::kLinear);
    void* data() const noexcept;
    DType dtype() const noexcept;
    const Shape& shape() const noexcept;
    Device device() const noexcept;
    int deviceId() const noexcept;
    Layout layout() const noexcept;
    std::size_t nbytes() const noexcept;
    bool isCuda() const noexcept;
    /// Typed, dtype-checked host span. Errors (kDtypeMismatch) if T != dtype, or
    /// (kInvalidArgument) if this is a device view — never an implicit D2H copy.
    template <class T> Result<std::span<const T>> as() const;
};

/// An owning tensor (RAII device or pinned-host buffer). Returned when the library
/// allocates outputs for the caller; also the Python-side owned type. Move-only.
class Tensor {
public:
    Tensor() = default;
    static Result<Tensor> allocate(DType, Shape, Device, int deviceId = 0);
    Tensor(Tensor&&) noexcept;  Tensor& operator=(Tensor&&) noexcept;
    Tensor(const Tensor&) = delete;
    TensorView view() const noexcept;                     ///< non-owning view
    void* data() const noexcept;
    DType dtype() const noexcept;  const Shape& shape() const noexcept;
    Device device() const noexcept;  int deviceId() const noexcept;
    std::size_t nbytes() const noexcept;
    /// Explicit transfers only — no implicit D2H (Python contract too).
    Result<Tensor> to(Device, int deviceId, const struct Stream&) const;
    Status copyFrom(TensorView src, const struct Stream&);   ///< async on stream
    /// One-call host materialization: copies to host AND synchronizes the stream,
    /// so the result is immediately readable. The common readback path.
    Result<Tensor> toHost(const struct Stream&) const;
    /// Typed, dtype-checked host span (errors on a device tensor or dtype mismatch).
    /// Pair with toHost(): `auto h = det.toHost(s); auto v = h->as<float>();`
    template <class T> Result<std::span<const T>> as() const;
};

} // namespace trtcpp
```

Rationale: a non-templated `Tensor`/`TensorView` with runtime `DType` replaces v6's
`Engine<T>` (audit A:api-surface; community #27/#47). Name-keyed `TensorView`s at the
boundary replace the triply-nested vectors. `-1`-aware `Shape` is the dynamic-shape
fix (#80/#86). No OpenCV/nvinfer1/spdlog in any of these headers.

### E2 — `logger.h`

```cpp
namespace trtcpp {
enum class Severity { kVerbose, kInfo, kWarning, kError, kInternalError };

/// Injectable logger. Default = stderr. Optional spdlog adapter behind the
/// TRT_CPP_API_WITH_SPDLOG CMake option. Replaces v6's hard-wired global spdlog.
class ILogger {
public:
    virtual ~ILogger() = default;
    virtual void log(Severity, std::string_view msg) noexcept = 0;
};
std::shared_ptr<ILogger> defaultLogger();                 ///< thread-safe stderr logger
std::shared_ptr<ILogger> makeSpdlogLogger();              ///< only if built with spdlog
} // namespace trtcpp
```

### E3 — `cuda.h`, `device.h`, `allocator.h`

```cpp
namespace trtcpp {
/// RAII CUDA stream. Either OWNS a stream it created, or WRAPS a caller-provided one
/// (non-owning) — the caller-provided model the community asked for (#28/#43) and the
/// Python bindings require (int handle from torch/cupy). Replaces v6's per-call create/destroy.
class Stream {
public:
    Stream();                                             ///< owns a new non-blocking stream
    static Stream wrap(cudaStream_t existing) noexcept;   ///< non-owning (cudaStream_t = void*)
    static Stream wrap(uintptr_t existing) noexcept;      ///< for language bindings
    cudaStream_t handle() const noexcept;
    uintptr_t raw() const noexcept;
    Status synchronize() const noexcept;
    Stream(Stream&&) noexcept; Stream& operator=(Stream&&) noexcept;
    Stream(const Stream&) = delete;
};

struct DeviceInfo { int index; std::string name; std::string uuid; int major, minor; size_t totalMem; };
Result<DeviceInfo> queryDevice(int index);
Result<int> deviceCount();

/// Allocator abstraction; default is stream-ordered (cudaMallocAsync) from a private
/// pool with releaseThreshold=UINT64_MAX (Phase B CUDA best practice). Pluggable so a
/// host app can hand TensorRT its own arena.
class IDeviceAllocator {
public:
    virtual ~IDeviceAllocator() = default;
    virtual void* allocate(std::size_t bytes, std::size_t alignment, const Stream&) = 0;
    virtual void  deallocate(void* ptr, const Stream&) noexcept = 0;
};
std::shared_ptr<IDeviceAllocator> defaultDeviceAllocator(int deviceIndex = 0);
} // namespace trtcpp
```

### E6 — `quant.h`, `build_options.h`, `engine_builder.h`

```cpp
namespace trtcpp {

/// Precision/quantization mode. Default kFp16. INT8_QDQ (explicit, strongly-typed) is
/// the forward path that survives TRT 11. INT8_CALIB_LEGACY is the v6-style calibrator
/// path, only available when built against TRT < 11 (compile error otherwise). FP8/NVFP4
/// are validated against compute capability at build (Ampere = INT8 only — Phase B).
enum class Precision { kFp32, kFp16, kInt8Qdq, kInt8CalibLegacy, kFp8, kNvfp4 };

/// One input's min/opt/max extents for an optimization profile. Multi-input,
/// multi-dim, multi-profile (the dynamic-shape fix, community #80/#86/#29/#20).
struct ProfileShape { std::string inputName; Shape min, opt, max; };
struct OptimizationProfile { std::vector<ProfileShape> inputs; };

struct BuildOptions {
    Precision precision = Precision::kFp16;
    std::vector<OptimizationProfile> profiles;            ///< empty => static shapes from ONNX
    int deviceIndex = 0;
    int dlaCore = -1;                                     ///< -1 = GPU
    std::optional<std::size_t> workspaceBytes;            ///< setMemoryPoolLimit(kWORKSPACE)
    std::optional<bool> stronglyTyped;                    ///< nullopt => auto: true for kInt8Qdq and
                                                          ///  on TRT>=11; false for kFp16/kFp32 on
                                                          ///  TRT<11 so the precision flag is honored
    bool versionCompatible = false;                       ///< kVERSION_COMPATIBLE (relaxes cache staleness)
    bool hardwareCompatible = false;                      ///< (relaxes cache staleness — see Engine cache)
    std::string engineCacheDir = ".";
    std::string timingCachePath;                          ///< empty => engineCacheDir/<hash>.timing
    std::vector<std::string> pluginLibraries;             ///< dlopen + IPluginV3 register (community #88)
    // INT8_CALIB_LEGACY only (ignored otherwise). The field is forward-declared and
    // stays present (inert) on TRT>=11 so BuildOptions is ABI-stable across the gate:
    std::shared_ptr<class ICalibrator> calibrator;
};

/// Stateless builder. ONNX bytes/path -> serialized engine. Caching + sidecar metadata
/// handled here (E7): content-hashed filename + JSON sidecar (ONNX sha256, TRT version,
/// GPU UUID, build options); integrity check + stale detection on load (fixes v6 stale
/// cache across TRT versions/GPUs).
class EngineBuilder {
public:
    explicit EngineBuilder(std::shared_ptr<ILogger> = defaultLogger());
    Result<std::vector<std::byte>> buildFromOnnxFile(const std::string& onnxPath, const BuildOptions&);
    Result<std::vector<std::byte>> buildFromOnnxBytes(std::span<const std::byte>, const BuildOptions&);
    /// Build if no fresh cached engine exists, else load the cache. Returns the engine path.
    Result<std::string> buildOrLoad(const std::string& onnxPath, const BuildOptions&);
    /// One-shot: buildOrLoad + deserialize. The common case — skips the path string.
    /// (Engine is forward-declared here; engine_builder.h pulls in engine.h.)
    Result<Engine> buildAndLoad(const std::string& onnxPath, const BuildOptions&,
                                const EngineOptions& = {});
};
} // namespace trtcpp
```

### E8 / E9 — `engine.h`, `engine_pool.h`

```cpp
namespace trtcpp {

struct EngineOptions {
    int deviceIndex = 0;
    std::shared_ptr<ILogger> logger = defaultLogger();
    std::shared_ptr<IDeviceAllocator> allocator;          ///< null => default
    std::vector<std::string> pluginLibraries;
};

/// Metadata for one IO tensor (name-keyed; never index-keyed — fixes the positional
/// IO assumption, audit A:correctness).
struct TensorInfo { std::string name; bool isInput; DType dtype; Shape shape; };

/// A loaded engine. Thread-compatible (not thread-safe): for concurrent inference use
/// EnginePool. Owns the ICudaEngine + one default IExecutionContext.
class Engine {
public:
    static Result<Engine> loadFromFile(const std::string& enginePath, const EngineOptions&);
    static Result<Engine> loadFromMemory(std::span<const std::byte>, const EngineOptions&);

    std::vector<TensorInfo> tensors() const;
    std::vector<std::string> inputNames() const;
    std::vector<std::string> outputNames() const;
    int nbOptimizationProfiles() const;
    Result<Shape> tensorShape(const std::string& name) const;     ///< build-time (may be -1 dynamic)
    Result<DType> tensorDType(const std::string& name) const;

    /// Caller-allocated, zero-copy, no implicit sync. Selects optimization profile
    /// `profileIndex` (setOptimizationProfileAsync on `stream`) BEFORE setInputShape from
    /// the input views, then setTensorAddress + enqueueV3 — all on the SAME `stream`
    /// (TRT 11 requires the profile-set and enqueue to share the stream). `enqueue` is
    /// the TRT-native async submit; the caller owns the sync. An out-of-range
    /// `profileIndex` returns `kInvalidArgument` (not a cryptic kTensorRtError).
    Status enqueue(const std::unordered_map<std::string, TensorView>& inputs,
                   const std::unordered_map<std::string, TensorView>& outputs,
                   const Stream& stream, int profileIndex = 0);

    /// Library-allocated convenience: caller supplies inputs + stream; the library
    /// resolves output shapes on its OWN execution context (so concurrent infer() calls
    /// do not stomp shared state) and returns owned device Tensors sized from
    /// getTensorShape AFTER setInputShape (never build-time -1 dims). For high
    /// concurrency use EnginePool. Engine itself is thread-COMPATIBLE, not thread-safe.
    Result<std::unordered_map<std::string, Tensor>>
    infer(const std::unordered_map<std::string, TensorView>& inputs,
          const Stream& stream, int profileIndex = 0);

    /// Single-output no-throw shortcut (the classifier/embedding common case): returns
    /// the sole output, or kInvalidArgument if the engine has != 1 output. Avoids
    /// hard-coding an engine-specific output name (community #12/#48).
    Result<Tensor> inferSingle(const std::unordered_map<std::string, TensorView>& inputs,
                               const Stream& stream, int profileIndex = 0);

    /// Resolve ALL output shapes for a set of input views + profile, for caller-side
    /// buffer pre-allocation. NOT const: it sets input shapes on an execution context
    /// and reads getTensorShape (dynamic-aware; fixes community #86 and Phase A
    /// correctness #3). Reuses the input views already in hand — no duplicate bookkeeping.
    Result<std::unordered_map<std::string, Shape>>
    outputShapes(const std::unordered_map<std::string, TensorView>& inputs, int profileIndex = 0);

    Engine(Engine&&) noexcept; Engine& operator=(Engine&&) noexcept;
};

/// One engine, N execution contexts, ONE OPTIMIZATION PROFILE PER CONTEXT, lease-based
/// acquisition for multi-stream concurrent dynamic-shape inference (community
/// #28/#57/#85). TensorRT 11 requires each concurrently-used context to bind a distinct
/// optimization profile, so the engine MUST be built with >= `contexts` profiles —
/// create() returns kInvalidArgument otherwise (and the docs/tutorial shows building N
/// identical profiles for N streams). Each Lease owns one context+profile; enqueue/infer
/// call setOptimizationProfileAsync(itsProfile, stream) BEFORE setInputShape, on the
/// caller's per-lease stream. This is the runtime half of the dynamic-batch fix
/// (#80/#86): the profile/context coupling that v6 never handled.
class EnginePool {
public:
    static Result<EnginePool> create(const std::string& enginePath, int contexts, const EngineOptions&);
    class Lease {                                          ///< RAII; returns the context on destruction
    public:
        int profileIndex() const noexcept;                ///< the profile this lease is bound to
        /// Caller-allocated, zero-copy steady-state path (mirrors Engine::enqueue).
        Status enqueue(const std::unordered_map<std::string, TensorView>& inputs,
                       const std::unordered_map<std::string, TensorView>& outputs,
                       const Stream& stream);
        /// Library-allocated convenience (mirrors Engine::infer) — present where
        /// concurrency makes manual output allocation most error-prone.
        Result<std::unordered_map<std::string, Tensor>>
        infer(const std::unordered_map<std::string, TensorView>& inputs, const Stream& stream);
        Result<Tensor>                                    ///< single-output shortcut (mirrors Engine)
        inferSingle(const std::unordered_map<std::string, TensorView>& inputs, const Stream& stream);
        Result<std::unordered_map<std::string, Shape>>
        outputShapes(const std::unordered_map<std::string, TensorView>& inputs);
        ~Lease();
    };
    Result<Lease> acquire();                               ///< blocks until a context is free
    std::optional<Lease> tryAcquire();
    int size() const noexcept;                             ///< number of contexts == profiles
    const Engine& engine() const;                          ///< for tensor metadata
};
} // namespace trtcpp
```

### E10 — `calibrator.h` (gated `#if NV_TENSORRT_MAJOR < 11`)

```cpp
#if NV_TENSORRT_MAJOR < 11
namespace trtcpp {
/// Legacy INT8 PTQ. Removed from TensorRT in 11.0; entire header compiles out there.
/// Forward path is Precision::kInt8Qdq with a modelopt-quantized ONNX (no calibrator).
class ICalibrator {
public:
    virtual ~ICalibrator() = default;
    virtual int batchSize() const = 0;
    /// Fill the named input device buffers for the next batch; false when exhausted.
    virtual bool nextBatch(const std::unordered_map<std::string, TensorView>& inputs) = 0;
    virtual std::optional<std::vector<std::byte>> readCache() = 0;
    virtual void writeCache(std::span<const std::byte>) = 0;
};
/// Concrete: reads images from a directory, applies the SAME preprocessing as inference
/// (fixes the v6 swapRB calibration/inference mismatch, audit A:quantization).
std::shared_ptr<ICalibrator> makeImageDirectoryCalibrator(/* dir, preproc spec, cache path */);
} // namespace trtcpp
#endif
```

### E11 — `preproc.h` (separate sublibrary `tensorrt_cpp_api::preproc`)

```cpp
namespace trtcpp::preproc {
/// One fused CUDA kernel: letterbox-resize -> optional BGR<->RGB -> (x - mean)*scale
/// -> NHWC->NCHW -> dtype cast, writing a TRT-ready device tensor with no intermediate
/// buffers. Per-channel mean/scale (fixes community #92 broken per-channel norm).
/// Arbitrary channel count (fixes #83/#87). Engine core does NOT link this.
struct PreprocSpec {
    int outChannels = 3;
    std::array<float, 4> mean{0,0,0,0};                   ///< per-channel
    std::array<float, 4> scale{1,1,1,1};
    bool swapRB = false;
    bool keepAspectRatioPad = true;                       ///< letterbox vs stretch
    uint8_t padValue = 0;
    DType outDtype = DType::kFloat32;
};
/// src: HWC uint8 device buffer; dst: pre-allocated NCHW device tensor of target size.
Status letterboxToTensor(TensorView srcHwcU8, TensorView dstNchw, const PreprocSpec&, const Stream&);
} // namespace trtcpp::preproc
```

### E12 / E13 — `opencv_interop.h` (optional, `TRT_CPP_API_WITH_OPENCV`)

```cpp
#ifdef TRT_CPP_API_WITH_OPENCV
namespace trtcpp::opencv {
TensorView viewOf(const cv::cuda::GpuMat&, Layout = Layout::kNHWC);   ///< zero-copy device view
TensorView viewOf(const cv::Mat&);                                    ///< host view (convenience)
Result<Tensor> upload(const cv::Mat&, const Stream&);
} // namespace trtcpp::opencv
#endif
```

### E14 — `version.h`, `all.h`

```cpp
#define TRT_CPP_API_VERSION_MAJOR 7
#define TRT_CPP_API_VERSION_MINOR 0
#define TRT_CPP_API_VERSION_PATCH 0
namespace trtcpp { struct Version { int major, minor, patch; const char* tensorrtVersion; const char* cudaVersion; };
Version version() noexcept; }
// all.h: includes every public header for one-line consumption.
```

## Quantization story

Default **explicit QDQ** (`Precision::kInt8Qdq`) for quantized models: consume a
`nvidia-modelopt`-quantized ONNX as a strongly-typed network, set no precision flags —
the only INT8 path that compiles unchanged on both TRT 10 and 11.

`Precision` is **version-aware and never a silent no-op** (the reviewer's FP16 trap):

- `kFp32` — strongly-typed FP32 on any TRT.
- `kFp16` — the convenience "FP32 ONNX -> fast half engine" the 800-user base relies on.
  - **TRT < 11 (the current build target):** weak-typed network + the `kFP16` builder
    flag (`stronglyTyped` auto-false). Works exactly as users expect today.
  - **TRT >= 11:** weak typing and the `kFP16` flag are removed; a strongly-typed network
    derives precision from the graph. If the supplied ONNX is already FP16/mixed it is
    honored; if it is plain FP32, the builder returns **`kUnsupported`** with an
    actionable message and points at the bundled offline FP16-cast step (documented in
    `docs/tutorial`, same offline-asset philosophy as QDQ). It NEVER silently returns an
    FP32 engine.
- `kInt8Qdq` — strongly-typed, no precision flags; works on TRT 10 and 11.
- `kInt8CalibLegacy` — calibrator + `kINT8` flag; only compiles against TRT < 11
  (factory and `ICalibrator` are gated out on 11).
- `kFp8` / `kNvfp4` — explicit-QDQ ONNX with FP8/NVFP4 scales; rejected at build on
  insufficient hardware.

Requested precision is validated at build time against the device compute capability
(reject `kFp8` on < 8.9, `kNvfp4` on < 10.0 — Ampere `sm_86` accelerates INT8 only) and
against `NV_TENSORRT_MAJOR` (reject `kInt8CalibLegacy` on >= 11), failing fast with a
clear `kUnsupported` message rather than a cryptic builder error or a wrong-precision engine.

## Engine cache

Filename: `<onnxStem>.<sha8(onnxBytes)>.<trtMajorMinor>.<gpuShortUuid>.<precision>.engine`.
Sidecar `<engine>.json`: full ONNX sha256, TRT version, CUDA version, GPU name+UUID,
`BuildOptions` digest, the `versionCompatible`/`hardwareCompatible` flags, builder
timestamp. On load: recompute the ONNX hash and compare TRT version + GPU UUID; mismatch
=> `kStaleCache` and rebuild (fixes v6 silent stale reuse; community version-skew
#41/#54). Engine written atomically (temp file + rename).

**Staleness relaxes to honor portable engines** (the reviewer's contradiction), with the
exact direction the TRT semantics require: if the sidecar records `versionCompatible`,
the cache is valid only when the runtime TRT is the **same major AND >= the sidecar's
TRT** (TensorRT version compatibility is FORWARD-only — a version-compatible engine runs
on equal-or-newer minors, not older), and the `<trtMajorMinor>` filename component is
relaxed to the major for these engines so the filename and sidecar checks agree. If the
sidecar records `hardwareCompatible`, the **GPU-UUID check is skipped** but the load is
still gated on the loading GPU's compute capability being within the engine's compatible
set (e.g. >= 8.0 for the Ampere-plus level) — hardware compatibility spans a defined
arch range, not arbitrary GPUs. Otherwise the exact-match check stands. This keeps the
staleness guard from rebuilding the very engines those flags exist to make reusable,
without ever loading an engine the runtime cannot actually deserialize.

## Build & packaging

Targets: `tensorrt_cpp_api` (core) + `tensorrt_cpp_api_preproc` + optional
`tensorrt_cpp_api_opencv`, all under the `tensorrt_cpp_api::` namespace.
`target_compile_features(... PUBLIC cxx_std_20)`, `CXX_EXTENSIONS OFF`. Each shared
target sets `VERSION ${PROJECT_VERSION}` / `SOVERSION 7` for a proper distro soname.
Options (all **default OFF**, consistent `TRT_CPP_API_` prefix): `WITH_OPENCV`,
`WITH_SPDLOG`, `BUILD_PREPROC`, `BUILD_PYTHON`, `BUILD_TESTS`, `BUILD_EXAMPLES`.

**The exported package re-resolves its own dependencies** (build reviewer's #1 — without
this a downstream `find_package(tensorrt_cpp_api)` errors on missing imported targets).
We ship `cmake/tensorrt_cpp_api-config.cmake.in` that:

```cmake
@PACKAGE_INIT@
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}")   # installed FindTensorRT.cmake
include(CMakeFindDependencyMacro)
find_dependency(CUDAToolkit 12.4)
find_dependency(TensorRT 10.6)                               # custom module, installed alongside
if(@TRT_CPP_API_WITH_OPENCV@)  find_dependency(OpenCV 4.9)   endif()
if(@TRT_CPP_API_WITH_SPDLOG@)  find_dependency(spdlog)       endif()
find_dependency(Threads)
include("${CMAKE_CURRENT_LIST_DIR}/tensorrt_cpp_api-targets.cmake")
```

generated via `configure_package_config_file` + `write_basic_package_version_file(...
COMPATIBILITY SameMajorVersion)` (so `find_package(tensorrt_cpp_api 7.0 REQUIRED)`
works), with `install(TARGETS ... EXPORT)`, `install(EXPORT ... NAMESPACE
tensorrt_cpp_api::)`, and `install(FILES cmake/FindTensorRT.cmake DESTINATION
<cmake-dir>)`. The `WITH_*` option values are baked into the config via `@PACKAGE_*@` so
the `find_dependency` guards match the build. Own include dirs use
`$<BUILD_INTERFACE:>`/`$<INSTALL_INTERFACE:>`.

**Linkage / shared-only TRT** (build reviewer's #2): `BUILD_SHARED_LIBS` controls only
the linkage of *this project's* libraries. **TensorRT 11 ships no static archives**, so
`TensorRT::TensorRT` (nvinfer + nvonnxparser) is always an `UNKNOWN/SHARED IMPORTED`
target and a fully-static self-contained link is **not supported** against TRT >= 11 —
documented explicitly so `-DBUILD_SHARED_LIBS=OFF` does not set a false expectation. The
CUDA runtime defaults to shared (`CUDA::cudart`); `CUDA::cudart_static` is available for
those who want it.

**Improved `FindTensorRT.cmake`** (relocatable when installed; no build-tree paths):
searches the apt layout (`/usr/include/x86_64-linux-gnu`, `/usr/lib/...`) AND a
tarball root (`TensorRT_DIR`), extracts the version from `NvInferVersion.h`, **errors
outside the 10.6 – 11.0 range** with the exact required range in the message (community
#41/#54), and builds an imported `TensorRT::TensorRT` carrying the include dir +
`nvinfer` + `nvonnxparser` as INTERFACE properties. Drops the removed `nvparsers`
(unlike v6's `FindTensorRT.cmake:73`). Uses `find_package(CUDAToolkit)` — never the
legacy `FindCUDA`.

**CUDA architectures:** default to an explicit redistributable list
`75-real;80-real;86-real;89-real;90-virtual` (not `native` — that needs CMake 3.24 and
only probes the build host, wrong for CI/redistributable builds), overridable via
`CMAKE_CUDA_ARCHITECTURES`.

**vcpkg / conan** (build reviewer's #3): TensorRT and CUDA are **system / externally
provided** — they are not in the public registries and the recipes never fetch them. The
vcpkg overlay port `find_package`s a system-installed TRT/CUDA (it does not declare them
as vcpkg dependency edges); the conan recipe declares them via `system_requirements()` (or
a thin wrapper package exposing the absolute TRT/CUDA paths) — not `requires`/
`tool_requires`, which are the wrong category for an externally-provided link library —
and forwards the `WITH_*` flags as conan options. The `packaging/` recipes document this
so they work without hand-editing. Recipe stubs ship under `packaging/{vcpkg,conan}/`.

## Testing

**GoogleTest** (broad CMake/CI familiarity, parametric fixtures). Unit tests (no GPU):
`Shape`/`DType`/`Status`/`Result`, cache key/sidecar logic, logger, ProfileShape
validation, FindTensorRT version parse. GPU-tagged integration tests (CTest label
`gpu`): build+load roundtrip, dynamic batch > 1, simultaneous dynamic H+W, multi-input/
multi-output, mixed-dtype outputs, >=4-stream concurrent inference, plugin roundtrip,
INT8 QDQ build, FP16 build, cache hit/miss/stale, atomic-write crash sim. Tiny generated
ONNX fixtures per shape/layout combo. Target > 70% line coverage on engine + buffer.

## Examples

`examples/{detection,classification,segmentation}` (C++) + `examples/python`. Each
consumes the **installed** package via `find_package(tensorrt_cpp_api)`. Detection =
YOLOv8n; classification = ResNet50/MobileNetV3 (opset >= 17); segmentation = DeepLabV3
MobileNetV3. The detection hello-world target is <= 30 lines (ergonomics gate).

## CI

`.github/workflows/ci.yml`: matrix Ubuntu 22.04 + 24.04 × {CUDA 12.x, 13.x} × {TRT
10.16, 11.0}; CPU-only build + clang-format + clang-tidy always; GPU tests gated to
manual `workflow_dispatch` (GH runners have no NVIDIA GPU) with a documented self-hosted
runner stub. `format.yml`: clang-format + clang-tidy on changed files. Python wheel
build/import sanity in a dedicated job. pre-commit: clang-format + cmake-format.

## Compatibility with v6 / siblings

**No v6 source-compat shim.** v6's API (templated `Engine<T>`, OpenCV-in-the-signature,
triply-nested vectors) is fundamentally incompatible with the leak-free name-keyed
design, and a shim would re-import every leak it removes. Instead: a thorough
`docs/upgrading_from_v6.md` maps every v6 symbol to its v7 equivalent, and Phase J
migrates the sibling repos (`YOLOv8/YOLOv9-TensorRT-CPP`) directly as the proof the new
API drives real consumers. A clean break with a migration guide beats a leaky shim.

## Hello-world (ergonomics check — the YOLOv8 loop, full device->host->postprocess, ~24 lines)

```cpp
#include <tensorrt_cpp_api/all.h>
using namespace trtcpp;
int main() {
    BuildOptions bo; bo.precision = Precision::kFp16; bo.engineCacheDir = "engines";
    // kFp16 builds a half engine from the FP32 ONNX on the TRT 10.x target; on TRT>=11
    // supply an FP16/QDQ ONNX (the builder errors clearly otherwise — never silent FP32).
    auto engine = EngineBuilder{}.buildAndLoad("yolov8n.onnx", bo);   // build-or-load + deserialize
    if (!engine) return fprintf(stderr, "%s\n", engine.status().message().c_str()), 1;

    Stream stream;                                          // owns a stream (or Stream::wrap(yours))
    Tensor input = Tensor::allocate(DType::kFloat32, {1,3,640,640}, Device::kCuda).value();
    // preproc::letterboxToTensor(srcHwcU8, input.view(), spec, stream);  // GPU preprocessing

    auto det = engine->inferSingle({{"images", input.view()}}, stream);  // sole output, no name guess
    if (!det) return 1;

    auto host = det->toHost(stream);                        // D2H + sync in one call
    if (!host) return 1;
    auto data = host->as<float>();                          // dtype-checked typed span
    if (!data) return 1;
    // ... NMS over data->data() using det->shape() ...
}
```

## Implementation notes carried from review (for the E-phases)

Nits the reviewers raised that are implementation guidance, not design changes:

- E6: build with `createNetworkV2(0)` (flags=0) — never the deprecated `kEXPLICIT_BATCH`
  no-op flag v6 still sets (`EngineBuildLoadNetwork.inl:207`). Plugins via `IPluginV3` +
  `getPluginCreators` (name it in docs to preempt community #88 "IPluginCreator not found").
- E1: pin `Shape` max rank to `nvinfer1::Dims::MAX_DIMS` rather than a literal 8.
  `byteSize(kInt4)` is sub-byte — ensure `TensorView::nbytes()` handles packed INT4 IO.
- E2/E4: the internal `ILogger`->`nvinfer1::ILogger` adapter must map Severity 1:1
  (kVerbose/kInfo/kWarning/kError/kInternalError) so verbose filtering works.
- E10: `makeImageDirectoryCalibrator` must reuse the inference `PreprocSpec` (fixes the
  v6 swapRB calibration/inference mismatch, Phase A correctness).

## Reviewer sign-offs

Round 1 (draft) — all three returned blocking issues; none signed off. Addressed:

- **Ergonomics** (3 blocking): added typed host readback (`Tensor::toHost`,
  `Tensor`/`TensorView::as<T>()`); no-throw single-output `Engine::inferSingle`; mirrored
  `infer`/`outputShapes` onto `EnginePool::Lease`. Plus nits: `EngineBuilder::buildAndLoad`
  one-shot, `Result` `and_then`/`transform`/`operator*`/`operator->` + `TRTCPP_TRY`,
  named-field `TensorView::Desc`.
- **TRT correctness** (4 blocking): made optimization profiles first-class at runtime
  (one profile per context in `EnginePool`, `setOptimizationProfileAsync` before
  `setInputShape` on the lease stream, `create()` requires >= contexts profiles, profile
  index on `enqueue`/`infer`); replaced `const resolveOutputShape` with non-const
  `outputShapes(views, profile)` run on a context; resolved the `kFp16`-under-strong-typing
  no-op (version-aware, errors not silent); relaxed cache staleness for
  version/hardware-compatible engines.
- **Build system** (3 blocking): specified `Config.cmake.in` with `find_dependency` +
  installed relocatable `FindTensorRT` + module path + `ConfigVersion`/SOVERSION;
  clarified `BUILD_SHARED_LIBS` cannot statically absorb shared-only TRT 11; specified
  vcpkg/conan treat TRT/CUDA as system-provided; explicit CUDA arch list (not `native`).

Round 2 — re-review of the revised doc (structural changes warranted it): **all three
signed off** (no blocking issues remaining). Their refinement nits were folded in:
`Lease::inferSingle` for symmetry; version-compatibility is forward-only (runtime TRT >=
sidecar, same major) and hardware-compatibility keeps a compute-capability floor;
out-of-range `profileIndex` returns `kInvalidArgument`; Conan uses `system_requirements()`
not `tool_requires`.

- [x] API ergonomics reviewer — signed off
- [x] TRT correctness reviewer — signed off
- [x] Build-system reviewer — signed off
