#include "tensorrt_cpp_api/engine_pool.h"

#include "detail/engine_cache.h"
#include "detail/execution.h"
#include "detail/trt_common.h"

#include <cuda_runtime.h>

#include <NvInfer.h>

#include <condition_variable>
#include <mutex>
#include <utility>

namespace trtcpp {

namespace detail {

struct PoolState {
    EngineOptions options;
    TrtLoggerBridge bridge;
    TrtUniquePtr<nvinfer1::IRuntime> runtime;
    TrtUniquePtr<nvinfer1::ICudaEngine> engine;
    std::vector<TrtUniquePtr<nvinfer1::IExecutionContext>> contexts;
    EngineMeta meta;

    std::mutex mutex;
    std::condition_variable cv;
    std::vector<bool> inUse;

    explicit PoolState(EngineOptions opts) : options(std::move(opts)), bridge(options.logger) {}

    int profileForSlot(int slot) const { return meta.hasDynamic ? slot : 0; }

    void release(int slot) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            inUse[static_cast<std::size_t>(slot)] = false;
        }
        cv.notify_one();
    }
};

} // namespace detail

// ---- Lease ----------------------------------------------------------------------------

EnginePool::Lease::Lease(Lease &&other) noexcept : state_(std::move(other.state_)), slot_(other.slot_), profile_(other.profile_) {
    other.slot_ = -1;
}

EnginePool::Lease &EnginePool::Lease::operator=(Lease &&other) noexcept {
    if (this != &other) {
        if (slot_ >= 0 && state_) {
            state_->release(slot_);
        }
        state_ = std::move(other.state_);
        slot_ = other.slot_;
        profile_ = other.profile_;
        other.slot_ = -1;
    }
    return *this;
}

EnginePool::Lease::~Lease() {
    if (slot_ >= 0 && state_) {
        state_->release(slot_);
    }
}

Status EnginePool::Lease::enqueue(const std::unordered_map<std::string, TensorView> &inputs,
                                  const std::unordered_map<std::string, TensorView> &outputs, const Stream &stream) {
    if (!valid()) {
        return Status{StatusCode::kInvalidArgument, "lease is not valid (default-constructed or moved-from)"};
    }
    return detail::enqueue(*state_->contexts[static_cast<std::size_t>(slot_)], state_->meta, inputs, outputs, stream, profile_);
}

Result<std::unordered_map<std::string, Tensor>> EnginePool::Lease::infer(const std::unordered_map<std::string, TensorView> &inputs,
                                                                         const Stream &stream) {
    if (!valid()) {
        return Status{StatusCode::kInvalidArgument, "lease is not valid (default-constructed or moved-from)"};
    }
    return detail::infer(*state_->contexts[static_cast<std::size_t>(slot_)], state_->meta, inputs, stream, profile_);
}

Result<Tensor> EnginePool::Lease::inferSingle(const std::unordered_map<std::string, TensorView> &inputs, const Stream &stream) {
    if (!valid()) {
        return Status{StatusCode::kInvalidArgument, "lease is not valid (default-constructed or moved-from)"};
    }
    if (state_->meta.outputNames.size() != 1) {
        return Status{StatusCode::kInvalidArgument,
                      "inferSingle requires exactly one output; this engine has " + std::to_string(state_->meta.outputNames.size())};
    }
    auto outputs = infer(inputs, stream);
    if (!outputs) {
        return outputs.status();
    }
    return std::move(outputs.value().begin()->second);
}

Result<std::unordered_map<std::string, Shape>> EnginePool::Lease::outputShapes(const std::unordered_map<std::string, TensorView> &inputs) {
    if (!valid()) {
        return Status{StatusCode::kInvalidArgument, "lease is not valid (default-constructed or moved-from)"};
    }
    return detail::outputShapes(*state_->contexts[static_cast<std::size_t>(slot_)], state_->meta, inputs, profile_);
}

// ---- EnginePool -----------------------------------------------------------------------

EnginePool::EnginePool(EnginePool &&) noexcept = default;
EnginePool &EnginePool::operator=(EnginePool &&) noexcept = default;
EnginePool::~EnginePool() = default;

Result<EnginePool> EnginePool::create(const std::string &enginePath, int contexts, const EngineOptions &options) {
    if (contexts < 1) {
        return Status{StatusCode::kInvalidArgument, "EnginePool requires at least one context"};
    }
    auto data = detail::readFile(enginePath);
    if (!data) {
        return data.status();
    }
    if (cudaError_t code = cudaSetDevice(options.deviceIndex); code != cudaSuccess) {
        return cudaToStatus(code, "cudaSetDevice");
    }
    if (Status status = detail::loadPluginLibraries(options.pluginLibraries); !status) {
        return status;
    }

    auto state = std::make_shared<detail::PoolState>(options);
    state->runtime.reset(nvinfer1::createInferRuntime(state->bridge.nv()));
    if (!state->runtime) {
        return Status{StatusCode::kTensorRtError, "createInferRuntime failed"};
    }
    state->engine.reset(state->runtime->deserializeCudaEngine(data.value().data(), data.value().size()));
    if (!state->engine) {
        return Status{StatusCode::kTensorRtError, "deserializeCudaEngine failed (corrupt or incompatible engine)"};
    }
    state->meta = detail::introspectEngine(*state->engine, options.deviceIndex);

    // Dynamic-shape engines need a distinct optimization profile per concurrent context;
    // static engines share the default profile across any number of contexts.
    if (state->meta.hasDynamic && contexts > state->meta.nbProfiles) {
        return Status{StatusCode::kInvalidArgument, "EnginePool needs one optimization profile per context: engine has " +
                                                        std::to_string(state->meta.nbProfiles) + " profiles but " +
                                                        std::to_string(contexts) + " contexts were requested"};
    }

    for (int i = 0; i < contexts; ++i) {
        // Default (kSTATIC) allocation: each context pre-allocates its own activation memory,
        // which is what makes concurrent enqueueV3 on distinct contexts safe (no shared scratch).
        detail::TrtUniquePtr<nvinfer1::IExecutionContext> context{state->engine->createExecutionContext()};
        if (!context) {
            return Status{StatusCode::kTensorRtError, "createExecutionContext failed"};
        }
        state->contexts.push_back(std::move(context));
    }
    state->inUse.assign(static_cast<std::size_t>(contexts), false);

    EnginePool pool;
    pool.state_ = std::move(state);
    return pool;
}

Result<EnginePool::Lease> EnginePool::acquire() {
    std::unique_lock<std::mutex> lock(state_->mutex);
    int slot = -1;
    // The predicate records the free slot; safe because condition_variable evaluates it
    // under the held lock, and the lock is still held when it returns true.
    state_->cv.wait(lock, [&] {
        for (std::size_t i = 0; i < state_->inUse.size(); ++i) {
            if (!state_->inUse[i]) {
                slot = static_cast<int>(i);
                return true;
            }
        }
        return false;
    });
    state_->inUse[static_cast<std::size_t>(slot)] = true;
    return Lease{state_, slot, state_->profileForSlot(slot)};
}

std::optional<EnginePool::Lease> EnginePool::tryAcquire() {
    std::lock_guard<std::mutex> lock(state_->mutex);
    for (std::size_t i = 0; i < state_->inUse.size(); ++i) {
        if (!state_->inUse[i]) {
            state_->inUse[i] = true;
            const int slot = static_cast<int>(i);
            return Lease{state_, slot, state_->profileForSlot(slot)};
        }
    }
    return std::nullopt;
}

int EnginePool::size() const noexcept { return static_cast<int>(state_->contexts.size()); }
std::vector<TensorInfo> EnginePool::tensors() const { return state_->meta.tensors; }
std::vector<std::string> EnginePool::inputNames() const { return state_->meta.inputNames; }
std::vector<std::string> EnginePool::outputNames() const { return state_->meta.outputNames; }

} // namespace trtcpp
