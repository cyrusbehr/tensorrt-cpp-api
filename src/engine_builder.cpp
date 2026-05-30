#include "tensorrt_cpp_api/engine_builder.h"

#include "detail/calibrator_bridge.h"
#include "detail/engine_cache.h"
#include "detail/sha256.h"
#include "detail/trt_common.h"

#include "tensorrt_cpp_api/calibrator.h"

#include <memory>

#include <cuda_runtime.h>

#include <ctime>

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "tensorrt_cpp_api/cuda.h"

namespace trtcpp {

std::string_view toString(Precision p) noexcept {
    switch (p) {
    case Precision::kFp32:
        return "fp32";
    case Precision::kFp16:
        return "fp16";
    case Precision::kInt8Qdq:
        return "int8_qdq";
    case Precision::kInt8CalibLegacy:
        return "int8_calib_legacy";
    case Precision::kFp8:
        return "fp8";
    case Precision::kNvfp4:
        return "nvfp4";
    }
    return "unknown";
}

namespace {

static_assert(Shape::kMaxRank == nvinfer1::Dims::MAX_DIMS, "Shape::kMaxRank must match nvinfer1::Dims::MAX_DIMS");

nvinfer1::Dims toDims(const Shape &shape) {
    nvinfer1::Dims dims;
    dims.nbDims = shape.rank();
    for (int i = 0; i < shape.rank(); ++i) {
        dims.d[i] = shape[i];
    }
    return dims;
}

// Validate the requested precision against the device's compute capability and the linked
// TensorRT version, so unsupported requests fail fast with a clear message rather than a
// cryptic builder error or a wrong-precision engine.
Status validatePrecision(const BuildOptions &options) {
    if (options.precision == Precision::kInt8CalibLegacy) {
#if NV_TENSORRT_MAJOR >= 11
        return Status{StatusCode::kUnsupported, "kInt8CalibLegacy is removed in TensorRT 11; use kInt8Qdq with a modelopt-quantized ONNX"};
#else
        if (!options.calibrator) {
            return Status{StatusCode::kInvalidArgument, "kInt8CalibLegacy requires options.calibrator (see makeRawBatchCalibrator)"};
        }
#endif
    }
    if (options.precision == Precision::kFp8 || options.precision == Precision::kNvfp4) {
        cudaDeviceProp prop{};
        if (cudaError_t code = cudaGetDeviceProperties(&prop, options.deviceIndex); code != cudaSuccess) {
            return cudaToStatus(code, "cudaGetDeviceProperties");
        }
        const int sm = prop.major * 10 + prop.minor;
        if (options.precision == Precision::kFp8 && sm < 89) {
            return Status{StatusCode::kUnsupported, "FP8 requires compute capability >= 8.9 (Ada/Hopper/Blackwell)"};
        }
        if (options.precision == Precision::kNvfp4 && prop.major < 10) {
            return Status{StatusCode::kUnsupported, "NVFP4 requires compute capability >= 10.0 (Blackwell)"};
        }
    }
    return Status{};
}

bool resolveStronglyTyped(const BuildOptions &options) {
    if (options.stronglyTyped.has_value()) {
        return options.stronglyTyped.value();
    }
#if NV_TENSORRT_MAJOR >= 11
    return true; // strongly typed is mandatory/default in TRT 11
#else
    return options.precision == Precision::kInt8Qdq || options.precision == Precision::kFp8 || options.precision == Precision::kNvfp4;
#endif
}

std::vector<std::byte> readFileBytes(const std::string &path, Status &status) {
    std::error_code ec;
    if (!std::filesystem::is_regular_file(path, ec)) {
        // Rejects missing paths AND directories (whose tellg() returns a huge sentinel
        // that would otherwise trigger a multi-exabyte allocation).
        status = Status{StatusCode::kNotFound, "not a readable regular file: " + path};
        return {};
    }
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        status = Status{StatusCode::kNotFound, "cannot open file: " + path};
        return {};
    }
    const std::streamsize size = file.tellg();
    if (size < 0) {
        status = Status{StatusCode::kIoError, "failed to determine size of file: " + path};
        return {};
    }
    file.seekg(0, std::ios::beg);
    std::vector<std::byte> bytes(static_cast<std::size_t>(size));
    if (size > 0 && !file.read(reinterpret_cast<char *>(bytes.data()), size)) {
        status = Status{StatusCode::kIoError, "failed to read file: " + path};
        return {};
    }
    status = Status{};
    return bytes;
}

std::string trtVersionString() {
    return std::to_string(NV_TENSORRT_MAJOR) + "." + std::to_string(NV_TENSORRT_MINOR) + "." + std::to_string(NV_TENSORRT_PATCH);
}

std::string cudaVersionString() {
    int version = 0;
    cudaRuntimeGetVersion(&version);
    return std::to_string(version);
}

// A digest of the build options that affect the produced engine, so changing any of them
// invalidates the cache.
std::string buildOptionsDigest(const BuildOptions &options) {
    std::string s = "p=";
    s += toString(options.precision);
    s += ";dla=" + std::to_string(options.dlaCore);
    s += ";ws=" + std::to_string(options.workspaceBytes.value_or(0));
    s += ";st=" + std::string(options.stronglyTyped.has_value() ? (options.stronglyTyped.value() ? "1" : "0") : "auto");
    s += ";vc=" + std::to_string(static_cast<int>(options.versionCompatible));
    s += ";hc=" + std::to_string(static_cast<int>(options.hardwareCompatible));
    for (const std::string &lib : options.pluginLibraries) {
        s += ";plugin=" + lib;
    }
    for (const OptimizationProfile &profile : options.profiles) {
        s += "|prof";
        for (const ProfileShape &input : profile.inputs) {
            s += ":" + input.inputName + "[" + input.min.toString() + input.opt.toString() + input.max.toString() + "]";
        }
    }
    return detail::sha256Hex(s.data(), s.size());
}

} // namespace

EngineBuilder::EngineBuilder(std::shared_ptr<ILogger> logger) : logger_(std::move(logger)) {
    if (!logger_) {
        logger_ = defaultLogger();
    }
}

Result<std::vector<std::byte>> EngineBuilder::buildFromOnnxFile(const std::string &onnxPath, const BuildOptions &options) {
    auto bytes = detail::readFile(onnxPath);
    if (!bytes) {
        return bytes.status();
    }
    return buildFromOnnxBytes(bytes.value(), options);
}

Result<Engine> EngineBuilder::buildAndLoad(const std::string &onnxPath, const BuildOptions &options, const EngineOptions &engineOptions) {
    auto path = buildOrLoad(onnxPath, options);
    if (!path) {
        return path.status();
    }
    return Engine::loadFromFile(path.value(), engineOptions);
}

Result<std::string> EngineBuilder::buildOrLoad(const std::string &onnxPath, const BuildOptions &options) {
    auto onnxResult = detail::readFile(onnxPath);
    if (!onnxResult) {
        return onnxResult.status();
    }
    std::vector<std::byte> &onnx = onnxResult.value();

    detail::CacheMeta expected;
    expected.onnxSha256 = detail::sha256Hex(onnx.data(), onnx.size());
    expected.trtVersion = trtVersionString();
    expected.cudaVersion = cudaVersionString();
    auto device = queryDevice(options.deviceIndex);
    if (!device) {
        return device.status();
    }
    expected.gpuName = device.value().name;
    expected.gpuUuid = device.value().uuid;
    expected.precision = std::string(toString(options.precision));
    expected.buildOptionsDigest = buildOptionsDigest(options);
    expected.versionCompatible = options.versionCompatible;
    expected.hardwareCompatible = options.hardwareCompatible;

    const std::string stem = std::filesystem::path(onnxPath).stem().string();
    // Relax the filename for portable engines so the relaxed sidecar check can actually
    // find them: major-only TRT for version-compatible, a fixed token for hardware-compatible.
    const std::string fileTrtVersion = options.versionCompatible ? std::to_string(NV_TENSORRT_MAJOR) : expected.trtVersion;
    const std::string fileGpuUuid = options.hardwareCompatible ? std::string("hwcompat") : expected.gpuUuid;
    const std::string fileName = detail::cacheFileName(stem, expected.onnxSha256, fileTrtVersion, fileGpuUuid, expected.precision);
    const std::filesystem::path enginePath = std::filesystem::path(options.engineCacheDir) / fileName;
    std::filesystem::path sidecarPath = enginePath;
    sidecarPath += ".json";

    if (std::filesystem::is_regular_file(enginePath) && std::filesystem::is_regular_file(sidecarPath)) {
        auto meta = detail::readSidecar(sidecarPath.string());
        // A hardware-compatible (Ampere-plus) engine must not be reused on a pre-Ampere GPU
        // even though the UUID check is relaxed -- gate on the loading GPU's compute capability.
        // Gate on whether the CACHED engine was built hardware-compatible (the sidecar flag), not
        // the current request: the safety property belongs to the artifact on disk, independent of
        // the filename convention that led us here.
        const bool hardwareFloorOk = !meta || !meta.value().hardwareCompatible || device.value().computeMajor >= 8;
        if (meta && hardwareFloorOk && detail::isFresh(meta.value(), expected, options.versionCompatible, options.hardwareCompatible)) {
            return enginePath.string(); // fresh cache hit
        }
    }

    auto engine = buildFromOnnxBytes(onnx, options);
    if (!engine) {
        return engine.status();
    }
    if (Status written = detail::writeAtomic(enginePath.string(), engine.value()); !written) {
        return written;
    }
    expected.createdUnix = static_cast<long long>(std::time(nullptr));
    if (Status written = detail::writeSidecar(sidecarPath.string(), expected); !written) {
        return written;
    }
    return enginePath.string();
}

Result<std::vector<std::byte>> EngineBuilder::buildFromOnnxBytes(std::span<const std::byte> onnx, const BuildOptions &options) {
    using detail::TrtLoggerBridge;
    using detail::TrtUniquePtr;

    if (Status status = validatePrecision(options); !status) {
        return status;
    }
    if (Status status = detail::loadPluginLibraries(options.pluginLibraries); !status) {
        return status;
    }
    int previousDevice = 0;
    cudaGetDevice(&previousDevice);
    if (cudaError_t code = cudaSetDevice(options.deviceIndex); code != cudaSuccess) {
        return cudaToStatus(code, "cudaSetDevice");
    }
    // Restore the caller's current device on every exit path.
    struct DeviceRestore {
        int device;
        ~DeviceRestore() { cudaSetDevice(device); }
    } deviceRestore{previousDevice};

    TrtLoggerBridge bridge(logger_);
    TrtUniquePtr<nvinfer1::IBuilder> builder{nvinfer1::createInferBuilder(bridge.nv())};
    if (!builder) {
        return Status{StatusCode::kTensorRtError, "createInferBuilder failed"};
    }

    const bool stronglyTyped = resolveStronglyTyped(options);
    uint32_t networkFlags = 0;
    if (stronglyTyped) {
        networkFlags |= 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
    }
    TrtUniquePtr<nvinfer1::INetworkDefinition> network{builder->createNetworkV2(networkFlags)};
    if (!network) {
        return Status{StatusCode::kTensorRtError, "createNetworkV2 failed"};
    }

    TrtUniquePtr<nvonnxparser::IParser> parser{nvonnxparser::createParser(*network, bridge.nv())};
    if (!parser) {
        return Status{StatusCode::kTensorRtError, "createParser failed"};
    }
    if (!parser->parse(onnx.data(), onnx.size())) {
        std::string message = "ONNX parse failed";
        for (int i = 0; i < parser->getNbErrors(); ++i) {
            message += "\n  ";
            message += parser->getError(i)->desc();
        }
        return Status{StatusCode::kTensorRtError, std::move(message)};
    }

#if NV_TENSORRT_MAJOR >= 11
    if (options.precision == Precision::kFp16) {
        // Strongly typed (TRT 11): precision comes from the graph. Honor an FP16/mixed ONNX;
        // a plain FP32 ONNX cannot be downcast here -- fail loudly with the remedy.
        // Look for ANY kHALF tensor (inputs or layer outputs) -- keep_io_types FP16 graphs
        // keep FP32 IO with internal half tensors, so checking only inputs is too strict.
        bool hasHalf = false;
        for (int i = 0; i < network->getNbInputs() && !hasHalf; ++i) {
            hasHalf = network->getInput(i)->getType() == nvinfer1::DataType::kHALF;
        }
        for (int i = 0; i < network->getNbLayers() && !hasHalf; ++i) {
            nvinfer1::ILayer *layer = network->getLayer(i);
            for (int j = 0; j < layer->getNbOutputs() && !hasHalf; ++j) {
                hasHalf = layer->getOutput(j)->getType() == nvinfer1::DataType::kHALF;
            }
        }
        if (!hasHalf) {
            return Status{StatusCode::kUnsupported,
                          "kFp16 on TensorRT >= 11 requires an FP16/QDQ ONNX; cast the model offline (modelopt/onnx fp16)"};
        }
    }
#endif

    TrtUniquePtr<nvinfer1::IBuilderConfig> config{builder->createBuilderConfig()};
    if (!config) {
        return Status{StatusCode::kTensorRtError, "createBuilderConfig failed"};
    }

    if (options.workspaceBytes.has_value()) {
        config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, options.workspaceBytes.value());
    }

#if NV_TENSORRT_MAJOR < 11
    // Weak-typing path (precision flags). On TRT 11 these flags are removed; strong typing
    // carries precision instead. The calibrator bridge must outlive buildSerializedNetwork.
    std::unique_ptr<detail::Int8CalibratorBridge> calibratorBridge;
    if (!stronglyTyped) {
        if (options.precision == Precision::kFp16) {
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
        } else if (options.precision == Precision::kInt8CalibLegacy) {
            if (!builder->platformHasFastInt8()) {
                return Status{StatusCode::kUnsupported, "GPU does not support fast INT8"};
            }
            if (network->getNbInputs() != 1) {
                return Status{StatusCode::kUnsupported, "legacy INT8 calibration supports single-input models only"};
            }
            config->setFlag(nvinfer1::BuilderFlag::kINT8);
            const nvinfer1::ITensor *input = network->getInput(0);
            const nvinfer1::Dims inputDims = input->getDimensions();
            for (int i = 0; i < inputDims.nbDims; ++i) {
                if (inputDims.d[i] < 0) {
                    return Status{StatusCode::kUnsupported, "legacy INT8 calibration requires a static input shape"};
                }
            }
            std::vector<std::int64_t> batchDims;
            batchDims.push_back(options.calibrator->batchSize());
            for (int i = 1; i < inputDims.nbDims; ++i) {
                batchDims.push_back(inputDims.d[i]);
            }
            Shape batchShape{std::span<const std::int64_t>(batchDims.data(), batchDims.size())};
            calibratorBridge = std::make_unique<detail::Int8CalibratorBridge>(options.calibrator, input->getName(), batchShape);
            config->setInt8Calibrator(calibratorBridge.get());
        }
    }
#endif

    if (options.versionCompatible) {
        config->setFlag(nvinfer1::BuilderFlag::kVERSION_COMPATIBLE);
    }
    if (options.hardwareCompatible) {
        config->setHardwareCompatibilityLevel(nvinfer1::HardwareCompatibilityLevel::kAMPERE_PLUS);
    }
    if (options.dlaCore >= 0) {
        config->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
        config->setDLACore(options.dlaCore);
        config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
    }

    for (const OptimizationProfile &profile : options.profiles) {
        nvinfer1::IOptimizationProfile *trtProfile = builder->createOptimizationProfile();
        for (const ProfileShape &input : profile.inputs) {
            trtProfile->setDimensions(input.inputName.c_str(), nvinfer1::OptProfileSelector::kMIN, toDims(input.min));
            trtProfile->setDimensions(input.inputName.c_str(), nvinfer1::OptProfileSelector::kOPT, toDims(input.opt));
            trtProfile->setDimensions(input.inputName.c_str(), nvinfer1::OptProfileSelector::kMAX, toDims(input.max));
        }
        config->addOptimizationProfile(trtProfile);
    }

    // Timing cache: reuse tactic timings across cold builds.
    const std::string timingPath = options.timingCachePath.empty() ? (options.engineCacheDir + "/timing.cache") : options.timingCachePath;
    Status timingStatus;
    std::vector<std::byte> timingSeed = readFileBytes(timingPath, timingStatus); // missing is fine
    TrtUniquePtr<nvinfer1::ITimingCache> timingCache{
        config->createTimingCache(timingSeed.data(), timingStatus.ok() ? timingSeed.size() : 0)};
    if (timingCache) {
        config->setTimingCache(*timingCache, /*ignoreMismatch=*/true); // reuse cross-build seeds
    }

    Stream profileStream;
    config->setProfileStream(profileStream.handle());

    TrtUniquePtr<nvinfer1::IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan) {
        return Status{StatusCode::kTensorRtError, "buildSerializedNetwork failed"};
    }

    if (timingCache) {
        if (TrtUniquePtr<nvinfer1::IHostMemory> serialized{timingCache->serialize()}) {
            std::error_code mkec;
            std::filesystem::create_directories(std::filesystem::path(timingPath).parent_path(), mkec);
            std::ofstream out(timingPath, std::ios::binary);
            if (out) {
                out.write(static_cast<const char *>(serialized->data()), static_cast<std::streamsize>(serialized->size()));
            }
        }
    }

    const auto *data = static_cast<const std::byte *>(plan->data());
    return std::vector<std::byte>(data, data + plan->size());
}

} // namespace trtcpp
