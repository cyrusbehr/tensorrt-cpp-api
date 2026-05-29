#pragma once

#include <cstddef>
#include <span>
#include <string>

namespace trtcpp::detail {

// SHA-256 of `length` bytes at `data`, returned as a 64-char lowercase hex string. Used
// for engine-cache content hashing (not security); dependency-free so the core library
// pulls in no crypto library.
std::string sha256Hex(const void *data, std::size_t length);

inline std::string sha256Hex(std::span<const std::byte> bytes) { return sha256Hex(bytes.data(), bytes.size()); }

} // namespace trtcpp::detail
