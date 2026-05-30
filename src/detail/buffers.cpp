#include "detail/buffers.h"

#include <cstdint>
#include <utility>

namespace trtcpp::detail {

Result<TensorView> NamedBuffers::ensure(const std::string &name, DType dtype, Shape shape, int deviceId) {
    auto needBytes = checkedByteSize(dtype, shape); // rejects dynamic shapes and overflow
    if (!needBytes) {
        return needBytes.status();
    }
    const std::size_t need = needBytes.value();
    Slot &slot = slots_[name];
    const bool mustGrow = need > slot.capacity;
    const bool wrongDevice = need > 0 && (slot.buffer.empty() || slot.deviceId != deviceId);
    if (mustGrow || wrongDevice) {
        auto buffer = Tensor::allocate(DType::kUInt8, Shape{static_cast<std::int64_t>(need)}, Device::kCuda, deviceId);
        if (!buffer) {
            return buffer.status();
        }
        slot.buffer = std::move(buffer).value();
        slot.capacity = need;
        slot.deviceId = deviceId;
    }
    return TensorView{slot.buffer.data(), dtype, std::move(shape), Device::kCuda, deviceId, Layout::kLinear};
}

void *NamedBuffers::address(const std::string &name) const noexcept {
    auto it = slots_.find(name);
    return it == slots_.end() ? nullptr : it->second.buffer.data();
}

bool NamedBuffers::has(const std::string &name) const noexcept { return slots_.find(name) != slots_.end(); }

std::vector<std::string> NamedBuffers::names() const {
    std::vector<std::string> result;
    result.reserve(slots_.size());
    for (const auto &entry : slots_) {
        result.push_back(entry.first);
    }
    return result;
}

std::size_t NamedBuffers::capacity(const std::string &name) const noexcept {
    auto it = slots_.find(name);
    return it == slots_.end() ? 0 : it->second.capacity;
}

void NamedBuffers::clear() noexcept { slots_.clear(); }

} // namespace trtcpp::detail
