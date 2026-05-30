#pragma once

#include <array>
#include <cstdint>
#include <initializer_list>
#include <span>
#include <string>

namespace trtcpp {

/// A dynamic-aware tensor shape with fixed inline capacity (no heap). A dim of -1 means
/// "dynamic / unresolved" (TensorRT's convention). Rank 0 is a scalar (numel 1).
class Shape {
public:
    /// Matches nvinfer1::Dims::MAX_DIMS; static_assert'd against it once TRT is linked.
    static constexpr int kMaxRank = 8;

    Shape() = default;
    Shape(std::initializer_list<std::int64_t> dims);
    explicit Shape(std::span<const std::int64_t> dims);

    int rank() const noexcept { return rank_; }
    std::int64_t operator[](int i) const noexcept { return dims_[static_cast<std::size_t>(i)]; }
    std::int64_t at(int i) const; ///< throws std::out_of_range

    bool isDynamic() const noexcept; ///< any dim < 0
    /// Element count; returns 0 if any dim is dynamic (resolve the shape first). Assumes
    /// the product is representable in int64 (true for any real tensor on current GPUs).
    std::int64_t numel() const noexcept;

    std::span<const std::int64_t> dims() const noexcept { return {dims_.data(), static_cast<std::size_t>(rank_)}; }

    bool operator==(const Shape &other) const noexcept;
    bool operator!=(const Shape &other) const noexcept { return !(*this == other); }

    std::string toString() const; ///< e.g. "[1,3,640,640]"

private:
    std::array<std::int64_t, kMaxRank> dims_{};
    int rank_ = 0;
};

} // namespace trtcpp
