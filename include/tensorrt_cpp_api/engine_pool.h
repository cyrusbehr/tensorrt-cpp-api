#pragma once

#include <memory>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorrt_cpp_api/cuda.h"
#include "tensorrt_cpp_api/device_tensor.h"
#include "tensorrt_cpp_api/engine.h" // EngineOptions, TensorInfo
#include "tensorrt_cpp_api/shape.h"
#include "tensorrt_cpp_api/status.h"
#include "tensorrt_cpp_api/tensor.h"

namespace trtcpp {

namespace detail {
struct PoolState; // defined in engine_pool.cpp
}

/// One deserialized engine, N execution contexts, ONE optimization profile per context.
/// Lease-based acquisition for concurrent multi-stream dynamic-shape inference. TensorRT
/// 11 requires each concurrently-used context to bind a distinct optimization profile, so a
/// dynamic-shape engine must be built with >= `contexts` profiles (create() rejects
/// otherwise). Each lease runs on the caller's stream.
class EnginePool {
public:
    /// A borrowed execution context (+ its bound profile). Returns the context to the pool
    /// on destruction. Move-only.
    class Lease {
    public:
        Lease() = default;
        Lease(Lease &&other) noexcept;
        Lease &operator=(Lease &&other) noexcept;
        Lease(const Lease &) = delete;
        Lease &operator=(const Lease &) = delete;
        ~Lease();

        bool valid() const noexcept { return slot_ >= 0; }
        int profileIndex() const noexcept { return profile_; }

        /// Caller-allocated, zero-copy (mirrors Engine::enqueue) on this lease's context+profile.
        Status enqueue(const std::unordered_map<std::string, TensorView> &inputs,
                       const std::unordered_map<std::string, TensorView> &outputs, const Stream &stream);
        /// Library-allocated convenience (mirrors Engine::infer).
        Result<std::unordered_map<std::string, Tensor>> infer(const std::unordered_map<std::string, TensorView> &inputs,
                                                              const Stream &stream);
        Result<Tensor> inferSingle(const std::unordered_map<std::string, TensorView> &inputs, const Stream &stream);
        Result<std::unordered_map<std::string, Shape>> outputShapes(const std::unordered_map<std::string, TensorView> &inputs);

    private:
        friend class EnginePool;
        Lease(std::shared_ptr<detail::PoolState> state, int slot, int profile) : state_(std::move(state)), slot_(slot), profile_(profile) {}
        std::shared_ptr<detail::PoolState> state_;
        int slot_ = -1;
        int profile_ = 0;
    };

    EnginePool(EnginePool &&) noexcept;
    EnginePool &operator=(EnginePool &&) noexcept;
    EnginePool(const EnginePool &) = delete;
    EnginePool &operator=(const EnginePool &) = delete;
    ~EnginePool();

    static Result<EnginePool> create(const std::string &enginePath, int contexts, const EngineOptions &options = {});

    Result<Lease> acquire();           ///< blocks until a context is free
    std::optional<Lease> tryAcquire(); ///< returns nullopt if none free
    int size() const noexcept;         ///< number of contexts (== profiles for dynamic engines)

    std::vector<TensorInfo> tensors() const;
    std::vector<std::string> inputNames() const;
    std::vector<std::string> outputNames() const;

private:
    EnginePool() = default;
    std::shared_ptr<detail::PoolState> state_;
};

} // namespace trtcpp
