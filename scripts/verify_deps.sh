#!/usr/bin/env bash
set -uo pipefail

# verify_deps.sh — non-destructive readiness check for building tensorrt-cpp-api.
# Exits 0 if the host looks ready to build, 1 otherwise. Installs nothing.
# Useful in CI and for users diagnosing install problems.

ok()   { printf '  \033[0;32mOK\033[0m   %s\n' "$*"; }
bad()  { printf '  \033[0;31mFAIL\033[0m %s\n' "$*"; FAILED=1; }
note() { printf '  \033[0;33m..\033[0m   %s\n' "$*"; }
FAILED=0

echo "tensorrt-cpp-api dependency check"
echo "================================="

# Compilers / build tools -------------------------------------------------- #
echo "Toolchain:"
if command -v cmake >/dev/null 2>&1; then
    CM_VER="$(cmake --version | head -1 | awk '{print $3}')"
    if printf '%s\n3.22\n' "$CM_VER" | sort -V | head -1 | grep -qx "3.22"; then
        ok "cmake $CM_VER (>= 3.22)"
    else
        bad "cmake $CM_VER is older than the required 3.22"
    fi
else
    bad "cmake not found"
fi
command -v g++  >/dev/null 2>&1 && ok "g++ $(g++ -dumpversion)"            || bad "g++ not found"
command -v ninja >/dev/null 2>&1 && ok "ninja $(ninja --version)"          || note "ninja not found (optional; Make works too)"
command -v ccache >/dev/null 2>&1 && ok "ccache present"                   || note "ccache not found (optional)"

# CUDA --------------------------------------------------------------------- #
echo "CUDA:"
NVCC=""
if command -v nvcc >/dev/null 2>&1; then NVCC="nvcc";
elif [[ -x /usr/local/cuda/bin/nvcc ]]; then NVCC="/usr/local/cuda/bin/nvcc"; fi
if [[ -n "$NVCC" ]]; then
    ok "nvcc $($NVCC --version | sed -n 's/.*release \([0-9.]*\).*/\1/p' | head -1) ($NVCC)"
else
    bad "nvcc not found (add /usr/local/cuda/bin to PATH, or install CUDA toolkit)"
fi
if command -v nvidia-smi >/dev/null 2>&1; then
    ok "driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1), GPU $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
else
    note "nvidia-smi not found (no GPU visible; CPU-only build/lint still possible)"
fi

# TensorRT ----------------------------------------------------------------- #
echo "TensorRT:"
TRT_HDR=""
for d in /usr/include/x86_64-linux-gnu /usr/include "${TensorRT_DIR:-}/include"; do
    [[ -n "$d" && -f "$d/NvInferVersion.h" ]] && { TRT_HDR="$d/NvInferVersion.h"; break; }
done
if [[ -n "$TRT_HDR" ]]; then
    TRT_MAJ=$(sed -n 's/^#define NV_TENSORRT_MAJOR \([0-9]*\).*/\1/p' "$TRT_HDR")
    TRT_MIN=$(sed -n 's/^#define NV_TENSORRT_MINOR \([0-9]*\).*/\1/p' "$TRT_HDR")
    if [[ "${TRT_MAJ:-0}" -ge 10 ]]; then
        ok "TensorRT ${TRT_MAJ}.${TRT_MIN} headers ($TRT_HDR)"
    else
        bad "TensorRT ${TRT_MAJ}.${TRT_MIN} found, but >= 10.0 is required"
    fi
else
    bad "TensorRT headers not found (set TensorRT_DIR or install libnvinfer-headers-dev)"
fi

# OpenCV ------------------------------------------------------------------- #
echo "OpenCV (optional interop):"
OCV_CFG="$(find /usr/local/lib/cmake/opencv4 /usr/lib/*/cmake/opencv4 -name 'OpenCVConfig.cmake' 2>/dev/null | head -1)"
if [[ -n "$OCV_CFG" ]]; then
    ok "OpenCV cmake config: $OCV_CFG"
    if find "$(dirname "$OCV_CFG")/../../.." -name 'libopencv_cudaarithm*' 2>/dev/null | grep -q .; then
        ok "OpenCV has CUDA modules (cv::cuda available)"
    else
        note "OpenCV present but WITHOUT CUDA modules (cv::cuda interop unavailable; see from_source.md)"
    fi
else
    note "OpenCV not found (engine core does not require it; needed only for OpenCV interop)"
fi

# spdlog / fmt ------------------------------------------------------------- #
echo "Logging (optional spdlog adapter):"
if find /usr/include /usr/local/include -path '*spdlog/spdlog.h' 2>/dev/null | grep -q .; then
    ok "spdlog headers present"
else
    note "spdlog not found (engine core uses a built-in stderr logger by default)"
fi

echo "================================="
if [[ "$FAILED" -eq 0 ]]; then
    echo "Result: host is READY to build tensorrt-cpp-api."
    exit 0
else
    echo "Result: host is MISSING required dependencies. Run scripts/install_deps.sh"
    echo "        or see docs/install/troubleshooting.md."
    exit 1
fi
