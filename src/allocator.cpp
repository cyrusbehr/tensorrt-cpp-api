#include "tensorrt_cpp_api/allocator.h"

#include <cstdint>

namespace trtcpp {
namespace {

/// Stream-ordered allocator over a private CUDA memory pool. The release threshold is
/// UINT64_MAX so freed blocks are retained for reuse instead of being returned to the OS
/// on each sync. A private pool (never the device default pool) keeps the library from
/// perturbing process-wide allocation state.
class StreamOrderedAllocator final : public IDeviceAllocator {
public:
    explicit StreamOrderedAllocator(int deviceIndex) {
        cudaMemPoolProps props{};
        props.allocType = cudaMemAllocationTypePinned;
        props.handleTypes = cudaMemHandleTypeNone;
        props.location.type = cudaMemLocationTypeDevice;
        props.location.id = deviceIndex;
        if (cudaMemPoolCreate(&pool_, &props) == cudaSuccess && pool_ != nullptr) {
            std::uint64_t threshold = UINT64_MAX;
            cudaMemPoolSetAttribute(pool_, cudaMemPoolAttrReleaseThreshold, &threshold);
        } else {
            pool_ = nullptr;
        }
    }

    ~StreamOrderedAllocator() override {
        if (pool_ != nullptr) {
            cudaMemPoolDestroy(pool_);
        }
    }

    void *allocate(std::size_t bytes, std::size_t /*alignment*/, const Stream &stream) override {
        void *ptr = nullptr;
        // If pool creation failed (catastrophic only), degrade to the device default pool
        // -- a documented fallback rather than the private pool the class otherwise uses.
        cudaError_t code = (pool_ != nullptr) ? cudaMallocFromPoolAsync(&ptr, bytes, pool_, stream.handle())
                                              : cudaMallocAsync(&ptr, bytes, stream.handle());
        return (code == cudaSuccess) ? ptr : nullptr;
    }

    void deallocate(void *ptr, const Stream &stream) noexcept override {
        if (ptr != nullptr) {
            cudaFreeAsync(ptr, stream.handle());
        }
    }

private:
    cudaMemPool_t pool_ = nullptr;
};

} // namespace

std::shared_ptr<IDeviceAllocator> defaultDeviceAllocator(int deviceIndex) { return std::make_shared<StreamOrderedAllocator>(deviceIndex); }

} // namespace trtcpp
