#include "detail/engine_cache.h"

#include <array>
#include <charconv>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <tuple>

#include <unistd.h>

namespace trtcpp::detail {
namespace {

std::string jsonEscape(const std::string &value) {
    std::string out;
    out.reserve(value.size() + 2);
    for (char c : value) {
        if (c == '"' || c == '\\') {
            out.push_back('\\');
        }
        out.push_back(c);
    }
    return out;
}

// Targeted extractor for our own flat sidecar format ("key": "value" or "key": bool/num).
std::optional<std::string> jsonStringField(const std::string &content, const std::string &key) {
    const std::string needle = "\"" + key + "\"";
    std::size_t pos = content.find(needle);
    if (pos == std::string::npos) {
        return std::nullopt;
    }
    pos = content.find(':', pos + needle.size());
    if (pos == std::string::npos) {
        return std::nullopt;
    }
    pos = content.find('"', pos);
    if (pos == std::string::npos) {
        return std::nullopt;
    }
    std::string value;
    for (std::size_t i = pos + 1; i < content.size(); ++i) {
        if (content[i] == '\\' && i + 1 < content.size()) {
            value.push_back(content[++i]);
        } else if (content[i] == '"') {
            return value;
        } else {
            value.push_back(content[i]);
        }
    }
    return std::nullopt;
}

bool jsonBoolField(const std::string &content, const std::string &key) {
    const std::string needle = "\"" + key + "\"";
    std::size_t pos = content.find(needle);
    if (pos == std::string::npos) {
        return false;
    }
    pos = content.find(':', pos + needle.size());
    return pos != std::string::npos && content.find("true", pos) == content.find_first_not_of(" \t", pos + 1);
}

// No-throw: from_chars never throws (unlike stoi), and leaves the component 0 on an empty
// or overflowing field, so a corrupt sidecar version is treated as a mismatch (stale) -- the
// intended behavior -- rather than crashing the no-throw API.
std::tuple<int, int, int> parseVersion(const std::string &version) {
    int parts[3] = {0, 0, 0};
    int index = 0;
    std::string current;
    auto flush = [&]() {
        if (index < 3) {
            int value = 0;
            std::from_chars(current.data(), current.data() + current.size(), value);
            parts[index++] = value;
        }
        current.clear();
    };
    for (char c : version + ".") {
        if (c == '.') {
            flush();
        } else if (c >= '0' && c <= '9') {
            current.push_back(c);
        }
    }
    return {parts[0], parts[1], parts[2]};
}

} // namespace

Result<std::vector<std::byte>> readFile(const std::string &path) {
    std::error_code ec;
    if (!std::filesystem::is_regular_file(path, ec)) {
        return Status{StatusCode::kNotFound, "not a readable regular file: " + path};
    }
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        return Status{StatusCode::kNotFound, "cannot open file: " + path};
    }
    const std::streamsize size = file.tellg();
    if (size < 0) {
        return Status{StatusCode::kIoError, "cannot size file: " + path};
    }
    file.seekg(0, std::ios::beg);
    std::vector<std::byte> bytes(static_cast<std::size_t>(size));
    if (size > 0 && !file.read(reinterpret_cast<char *>(bytes.data()), size)) {
        return Status{StatusCode::kIoError, "cannot read file: " + path};
    }
    return bytes;
}

std::string cacheFileName(const std::string &onnxStem, const std::string &onnxSha256, const std::string &trtVersion,
                          const std::string &gpuUuid, const std::string &precision) {
    const std::string sha8 = onnxSha256.substr(0, std::min<std::size_t>(8, onnxSha256.size()));
    const std::string uuid8 = gpuUuid.substr(0, std::min<std::size_t>(8, gpuUuid.size()));
    return onnxStem + "." + sha8 + ".trt" + trtVersion + "." + uuid8 + "." + precision + ".engine";
}

Status writeAtomic(const std::string &path, std::span<const std::byte> bytes) {
    std::error_code ec;
    const std::filesystem::path target(path);
    if (target.has_parent_path()) {
        std::filesystem::create_directories(target.parent_path(), ec);
    }
    // PID-suffixed temp so concurrent builders writing the same cache entry don't collide.
    const std::filesystem::path tmp = target.string() + ".tmp." + std::to_string(::getpid());
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) {
            return Status{StatusCode::kIoError, "cannot open for write: " + tmp.string()};
        }
        if (!bytes.empty()) {
            out.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        }
        // Flush + close explicitly and check the result BEFORE the rename: a buffered write can
        // succeed here yet fail on the destructor's implicit flush (e.g. disk full), which would
        // otherwise rename a truncated temp file over the cache. Catch that failure first.
        out.close();
        if (!out) {
            std::filesystem::remove(tmp, ec);
            return Status{StatusCode::kIoError, "write/flush failed: " + tmp.string()};
        }
    }
    std::filesystem::rename(tmp, target, ec);
    if (ec) {
        std::filesystem::remove(tmp, ec);
        return Status{StatusCode::kIoError, "atomic rename failed: " + path};
    }
    return Status{};
}

Status writeSidecar(const std::string &path, const CacheMeta &meta) {
    std::string json = "{\n";
    json += "  \"onnx_sha256\": \"" + jsonEscape(meta.onnxSha256) + "\",\n";
    json += "  \"trt_version\": \"" + jsonEscape(meta.trtVersion) + "\",\n";
    json += "  \"cuda_version\": \"" + jsonEscape(meta.cudaVersion) + "\",\n";
    json += "  \"gpu_name\": \"" + jsonEscape(meta.gpuName) + "\",\n";
    json += "  \"gpu_uuid\": \"" + jsonEscape(meta.gpuUuid) + "\",\n";
    json += "  \"precision\": \"" + jsonEscape(meta.precision) + "\",\n";
    json += "  \"build_options_digest\": \"" + jsonEscape(meta.buildOptionsDigest) + "\",\n";
    json += "  \"version_compatible\": " + std::string(meta.versionCompatible ? "true" : "false") + ",\n";
    json += "  \"hardware_compatible\": " + std::string(meta.hardwareCompatible ? "true" : "false") + ",\n";
    json += "  \"created_unix\": " + std::to_string(meta.createdUnix) + "\n";
    json += "}\n";
    const auto *data = reinterpret_cast<const std::byte *>(json.data());
    return writeAtomic(path, std::span<const std::byte>(data, json.size()));
}

Result<CacheMeta> readSidecar(const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return Status{StatusCode::kNotFound, "cannot read sidecar: " + path};
    }
    const std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    CacheMeta meta;
    meta.onnxSha256 = jsonStringField(content, "onnx_sha256").value_or("");
    meta.trtVersion = jsonStringField(content, "trt_version").value_or("");
    meta.cudaVersion = jsonStringField(content, "cuda_version").value_or("");
    meta.gpuName = jsonStringField(content, "gpu_name").value_or("");
    meta.gpuUuid = jsonStringField(content, "gpu_uuid").value_or("");
    meta.precision = jsonStringField(content, "precision").value_or("");
    meta.buildOptionsDigest = jsonStringField(content, "build_options_digest").value_or("");
    meta.versionCompatible = jsonBoolField(content, "version_compatible");
    meta.hardwareCompatible = jsonBoolField(content, "hardware_compatible");
    if (meta.onnxSha256.empty()) {
        return Status{StatusCode::kStaleCache, "sidecar missing onnx_sha256: " + path};
    }
    return meta;
}

bool isFresh(const CacheMeta &sidecar, const CacheMeta &expected, bool versionCompatible, bool hardwareCompatible) {
    if (sidecar.onnxSha256 != expected.onnxSha256) {
        return false;
    }
    if (sidecar.buildOptionsDigest != expected.buildOptionsDigest) {
        return false;
    }
    if (versionCompatible) {
        const auto cached = parseVersion(sidecar.trtVersion);
        const auto runtime = parseVersion(expected.trtVersion);
        // Same major, and the runtime is the same or newer (version compatibility is forward-only).
        if (std::get<0>(cached) != std::get<0>(runtime) || runtime < cached) {
            return false;
        }
    } else if (sidecar.trtVersion != expected.trtVersion) {
        return false;
    }
    if (!hardwareCompatible && sidecar.gpuUuid != expected.gpuUuid) {
        return false;
    }
    return true;
}

} // namespace trtcpp::detail
