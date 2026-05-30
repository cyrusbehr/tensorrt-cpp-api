#!/usr/bin/env bash
set -euo pipefail

# install_deps.sh — authoritative dependency installer for tensorrt-cpp-api (v7)
#
# Installs the system packages needed to BUILD the library on a fresh Ubuntu
# 22.04 or 24.04 x86_64 host: CUDA Toolkit, TensorRT dev, OpenCV dev, spdlog,
# fmt, and the usual build tooling. Optionally installs Python build deps.
#
# It does NOT install the NVIDIA kernel driver (too host-specific) and it does
# NOT build CUDA-enabled OpenCV from source (see docs/install/from_source.md and
# scripts/build_opencv.sh for that). It verifies a compatible driver is present
# and fails fast with a pointer if not.
#
# Safe to re-run: every apt operation is idempotent.

# --------------------------------------------------------------------------- #
# Defaults / flags
# --------------------------------------------------------------------------- #
CUDA_VERSION=""        # e.g. 12.6 or 13.0; empty => auto-detect from driver
WITH_PYTHON=0
DRY_RUN=0
NO_SUDO=0

usage() {
    cat <<'EOF'
Usage: scripts/install_deps.sh [OPTIONS]

Installs the build dependencies for tensorrt-cpp-api on Ubuntu 22.04/24.04.

Options:
  --cuda-version=<X.Y>   CUDA toolkit version to install (e.g. 12.6, 13.0).
                         Default: auto-detect the newest version the installed
                         NVIDIA driver supports.
  --with-python          Also install Python build deps (python3-dev, venv, pip,
                         pybind11-dev) for the optional Python bindings.
  --dry-run              Print the commands that would run; change nothing.
  --no-sudo              Print the privileged commands instead of running them
                         (for hosts with custom sudo policies). Implies the
                         caller will run them manually.
  -h, --help             Show this help and exit.

Examples:
  scripts/install_deps.sh                       # auto CUDA, no Python
  scripts/install_deps.sh --with-python          # + Python bindings deps
  scripts/install_deps.sh --cuda-version=12.6     # pin CUDA 12.6
  scripts/install_deps.sh --no-sudo               # print privileged commands

Notes:
  * Does NOT install the NVIDIA driver. CUDA 13.x needs driver >= 580; CUDA 12.x
    needs driver >= 525. If the driver is too old this script stops with a
    pointer to https://www.nvidia.com/Download/index.aspx .
  * apt `libopencv-dev` does NOT include the CUDA (cv::cuda) modules. For those,
    build OpenCV from source: docs/install/from_source.md.
EOF
}

for arg in "$@"; do
    case "$arg" in
        --cuda-version=*) CUDA_VERSION="${arg#*=}" ;;
        --with-python)    WITH_PYTHON=1 ;;
        --dry-run)        DRY_RUN=1 ;;
        --no-sudo)        NO_SUDO=1 ;;
        -h|--help)        usage; exit 0 ;;
        *) echo "Unknown option: $arg" >&2; usage >&2; exit 2 ;;
    esac
done

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
banner() { printf '\n\033[1;36m[%s] ==== %s ====\033[0m\n' "$(date '+%H:%M:%S')" "$*"; }
info()   { printf '\033[0;32m[%s] %s\033[0m\n' "$(date '+%H:%M:%S')" "$*"; }
warn()   { printf '\033[0;33m[%s] WARN: %s\033[0m\n' "$(date '+%H:%M:%S')" "$*" >&2; }
die()    { printf '\033[0;31m[%s] ERROR: %s\033[0m\n' "$(date '+%H:%M:%S')" "$*" >&2; exit 1; }

# Run a privileged command, honoring --dry-run / --no-sudo.
SUDO="sudo"
if [[ "$(id -u)" -eq 0 ]]; then SUDO=""; fi
priv() {
    if [[ "$DRY_RUN" -eq 1 || "$NO_SUDO" -eq 1 ]]; then
        echo "  + ${SUDO:+sudo }$*"
    else
        # shellcheck disable=SC2086
        $SUDO "$@"
    fi
}

# --------------------------------------------------------------------------- #
# Detect distro
# --------------------------------------------------------------------------- #
banner "Detecting host"
[[ -r /etc/os-release ]] || die "/etc/os-release not found; this script targets Ubuntu."
# shellcheck disable=SC1091
. /etc/os-release
[[ "${ID:-}" == "ubuntu" ]] || die "Unsupported distro '${ID:-?}'. This script supports Ubuntu 22.04/24.04."
case "${VERSION_ID:-}" in
    22.04) NV_DISTRO="ubuntu2204" ;;
    24.04) NV_DISTRO="ubuntu2404" ;;
    *) die "Unsupported Ubuntu ${VERSION_ID:-?}. Supported: 22.04, 24.04." ;;
esac
ARCH="$(uname -m)"
[[ "$ARCH" == "x86_64" ]] || warn "Architecture is $ARCH; the NVIDIA apt repo path below targets x86_64. For Jetson/ARM64 see docs/install/from_source.md."
info "Ubuntu ${VERSION_ID} (${NV_DISTRO}), arch ${ARCH}"

# --------------------------------------------------------------------------- #
# Detect driver, decide CUDA version, fail fast on incompatibility
# --------------------------------------------------------------------------- #
banner "Checking NVIDIA driver"
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found. Install an NVIDIA driver first (this script does not): https://www.nvidia.com/Download/index.aspx"

DRIVER_VER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)"
DRIVER_MAJOR="${DRIVER_VER%%.*}"
# Max CUDA the driver supports, as printed by nvidia-smi ("CUDA Version: X.Y").
DRIVER_MAX_CUDA="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1 || true)"
info "Driver ${DRIVER_VER:-unknown} supports up to CUDA ${DRIVER_MAX_CUDA:-unknown}"

if [[ -z "$CUDA_VERSION" ]]; then
    if [[ -n "$DRIVER_MAX_CUDA" ]]; then
        # Default to the driver's max supported CUDA (a known-good toolkit minor
        # for that major is chosen below).
        CUDA_VERSION="$DRIVER_MAX_CUDA"
    else
        CUDA_VERSION="12.6"
        warn "Could not read max CUDA from driver; defaulting to ${CUDA_VERSION}"
    fi
    info "Auto-selected CUDA ${CUDA_VERSION}"
fi
CUDA_MAJOR="${CUDA_VERSION%%.*}"
CUDA_MINOR="${CUDA_VERSION#*.}"

# Driver-major floor per CUDA major (Linux). Conservative.
case "$CUDA_MAJOR" in
    13) MIN_DRIVER=580 ;;
    12) MIN_DRIVER=525 ;;
    *)  MIN_DRIVER=0; warn "Unrecognized CUDA major ${CUDA_MAJOR}; skipping driver floor check." ;;
esac
if [[ "${DRIVER_MAJOR:-0}" -lt "$MIN_DRIVER" ]]; then
    die "Driver ${DRIVER_VER} is too old for CUDA ${CUDA_VERSION} (needs >= ${MIN_DRIVER}.x).
     Either upgrade the NVIDIA driver (https://www.nvidia.com/Download/index.aspx)
     or pin a lower CUDA with --cuda-version=<X.Y> (your driver supports up to ${DRIVER_MAX_CUDA:-?})."
fi

# Map CUDA major -> a stable toolkit minor available in the apt repo, unless the
# caller pinned an explicit minor.
APT_CUDA_PKG="cuda-toolkit-${CUDA_MAJOR}-${CUDA_MINOR}"
info "Will install ${APT_CUDA_PKG}"

# --------------------------------------------------------------------------- #
# Add the NVIDIA CUDA apt repo (idempotent). This repo also serves TensorRT and
# cuDNN packages and does NOT require an NVIDIA developer login.
# --------------------------------------------------------------------------- #
banner "Adding NVIDIA apt repo (cuda-keyring)"
KEYRING_DEB="cuda-keyring_1.1-1_all.deb"
REPO_BASE="https://developer.download.nvidia.com/compute/cuda/repos/${NV_DISTRO}/x86_64"
if dpkg -s cuda-keyring >/dev/null 2>&1; then
    info "cuda-keyring already installed; skipping repo add."
else
    info "Fetching ${REPO_BASE}/${KEYRING_DEB}"
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "  + wget ${REPO_BASE}/${KEYRING_DEB} && sudo dpkg -i ${KEYRING_DEB}"
    else
        TMP="$(mktemp -d)"
        wget -q -O "${TMP}/${KEYRING_DEB}" "${REPO_BASE}/${KEYRING_DEB}" || die "Failed to download cuda-keyring."
        priv dpkg -i "${TMP}/${KEYRING_DEB}"
        rm -rf "$TMP"
    fi
fi

banner "apt update"
priv apt-get update

# --------------------------------------------------------------------------- #
# Install packages
# --------------------------------------------------------------------------- #
BUILD_PKGS=(build-essential git pkg-config cmake ninja-build ccache)
NV_PKGS=("$APT_CUDA_PKG"
         libnvinfer-dev libnvinfer-headers-dev libnvinfer-plugin-dev
         libnvinfer-headers-plugin-dev libnvonnxparsers-dev)
LIB_PKGS=(libopencv-dev libspdlog-dev libfmt-dev)
PY_PKGS=()
if [[ "$WITH_PYTHON" -eq 1 ]]; then
    PY_PKGS=(python3-dev python3-pip python3-venv pybind11-dev)
fi

banner "Installing build tooling"
priv apt-get install -y "${BUILD_PKGS[@]}"

banner "Installing CUDA toolkit + TensorRT"
priv apt-get install -y "${NV_PKGS[@]}"

banner "Installing OpenCV / spdlog / fmt"
priv apt-get install -y "${LIB_PKGS[@]}"

if [[ "$WITH_PYTHON" -eq 1 ]]; then
    banner "Installing Python build deps"
    priv apt-get install -y "${PY_PKGS[@]}"
fi

# --------------------------------------------------------------------------- #
# Verify
# --------------------------------------------------------------------------- #
if [[ "$DRY_RUN" -eq 1 || "$NO_SUDO" -eq 1 ]]; then
    banner "Done (no changes were made)"
    info "Re-run without --dry-run/--no-sudo to actually install."
    exit 0
fi

banner "Verifying install"
RC=0
if [[ -x "/usr/local/cuda-${CUDA_MAJOR}.${CUDA_MINOR}/bin/nvcc" ]]; then
    "/usr/local/cuda-${CUDA_MAJOR}.${CUDA_MINOR}/bin/nvcc" --version | tail -1
elif command -v nvcc >/dev/null 2>&1; then
    nvcc --version | tail -1
else
    warn "nvcc not on PATH. Add /usr/local/cuda-${CUDA_MAJOR}.${CUDA_MINOR}/bin to PATH."
fi
dpkg -s libnvinfer-dev >/dev/null 2>&1 && info "libnvinfer-dev: $(dpkg-query -W -f='${Version}' libnvinfer-dev)" || { warn "libnvinfer-dev not installed"; RC=1; }
dpkg -s libspdlog-dev >/dev/null 2>&1 && info "libspdlog-dev OK" || { warn "libspdlog-dev not installed"; RC=1; }
dpkg -s libfmt-dev    >/dev/null 2>&1 && info "libfmt-dev OK"    || { warn "libfmt-dev not installed"; RC=1; }
if [[ "$WITH_PYTHON" -eq 1 ]]; then
    if python3 -c 'import pybind11; print("pybind11", pybind11.__version__)' 2>/dev/null; then :;
    else warn "pybind11 not importable from system python (ok: scikit-build-core provides it in an isolated build env)."; fi
fi

if [[ "$RC" -ne 0 ]]; then
    die "One or more required packages are missing. See warnings above and docs/install/troubleshooting.md."
fi
banner "All dependencies installed. Next: cmake -B build -G Ninja && cmake --build build"
