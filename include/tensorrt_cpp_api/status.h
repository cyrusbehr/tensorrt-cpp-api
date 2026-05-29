#pragma once

#include <cassert>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace trtcpp {

/// The public API is no-throw: every fallible call returns Status or Result<T>.
enum class StatusCode {
    kOk,
    kInvalidArgument,
    kNotFound,
    kIoError,
    kCudaError,
    kTensorRtError,
    kShapeMismatch,
    kDtypeMismatch,
    kUnsupported,
    kStaleCache,
    kInternal,
};

std::string_view toString(StatusCode code) noexcept;

class Status {
public:
    Status() noexcept = default; ///< default-constructed Status is kOk
    Status(StatusCode code, std::string message) : code_(code), message_(std::move(message)) {}

    bool ok() const noexcept { return code_ == StatusCode::kOk; }
    StatusCode code() const noexcept { return code_; }
    const std::string &message() const noexcept { return message_; }
    explicit operator bool() const noexcept { return ok(); }

private:
    StatusCode code_ = StatusCode::kOk;
    std::string message_;
};

/// Value-or-Status, std::expected-style but no exceptions. Move-friendly.
template <class T> class Result {
public:
    Result(T value) : value_(std::move(value)) {}
    Result(Status error) : status_(std::move(error)) { assert(!status_.ok() && "Result(Status) must be an error"); }

    bool ok() const noexcept { return status_.ok(); }
    explicit operator bool() const noexcept { return ok(); }

    T &value() & {
        assert(ok());
        return *value_;
    }
    const T &value() const & {
        assert(ok());
        return *value_;
    }
    T &&value() && {
        assert(ok());
        return std::move(*value_);
    }

    T *operator->() {
        assert(ok());
        return &*value_;
    }
    const T *operator->() const {
        assert(ok());
        return &*value_;
    }
    T &operator*() & {
        assert(ok());
        return *value_;
    }
    const T &operator*() const & {
        assert(ok());
        return *value_;
    }

    const Status &status() const noexcept { return status_; }
    T value_or(T fallback) const { return ok() ? *value_ : std::move(fallback); }

    /// F: T& -> Result<U>. Propagates this error if !ok().
    template <class F> auto and_then(F &&f) -> std::invoke_result_t<F, T &> {
        using R = std::invoke_result_t<F, T &>;
        if (ok()) {
            return std::forward<F>(f)(*value_);
        }
        return R(status_);
    }
    /// F: T& -> U. Wraps the result; propagates this error if !ok().
    template <class F> auto transform(F &&f) -> Result<std::invoke_result_t<F, T &>> {
        using U = std::invoke_result_t<F, T &>;
        if (ok()) {
            return Result<U>(std::forward<F>(f)(*value_));
        }
        return Result<U>(status_);
    }

private:
    Status status_; // default-constructed = ok
    std::optional<T> value_;
};

#define TRTCPP_CONCAT_(a, b) a##b
#define TRTCPP_CONCAT(a, b) TRTCPP_CONCAT_(a, b)
/// Propagate-on-error sugar: `TRTCPP_TRY(auto engine, Engine::loadFromFile(p, {}));`
/// On error, returns the Status from the enclosing function (which must return Status or
/// a Result<...>, since Result is constructible from Status). One use per source line.
#define TRTCPP_TRY(decl, expr)                                                                                                             \
    auto TRTCPP_CONCAT(trtcpp_try_, __LINE__) = (expr);                                                                                    \
    if (!TRTCPP_CONCAT(trtcpp_try_, __LINE__)) {                                                                                           \
        return TRTCPP_CONCAT(trtcpp_try_, __LINE__).status();                                                                              \
    }                                                                                                                                      \
    decl = std::move(TRTCPP_CONCAT(trtcpp_try_, __LINE__)).value()

} // namespace trtcpp
