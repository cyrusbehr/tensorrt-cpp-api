# Phase 0 — Host Environment

Captured 2026-05-29 on the maintainer's GPU box. This records the toolchain the
v7 work is developed and validated against, and the constraints that follow from it.

## Hardware / driver

| Item            | Value                                                  |
| --------------- | ------------------------------------------------------ |
| GPU             | NVIDIA GeForce RTX 3080 Laptop GPU                      |
| Architecture    | Ampere, compute capability **8.6** (`sm_86`)           |
| VRAM            | 16 GB                                                   |
| NVIDIA driver   | **565.57.01**                                          |
| Max CUDA (driver) | **12.7** (per `nvidia-smi`)                          |

## Toolchain present at bootstrap

| Component   | Version / location                                                       |
| ----------- | ------------------------------------------------------------------------ |
| OS          | Ubuntu 24.04.1 LTS (Noble Numbat)                                         |
| CUDA Toolkit| 12.6.3 — `/usr/local/cuda` -> `/usr/local/cuda-12.6`, `nvcc` 12.6.85      |
| cuDNN       | 9.6.0.74 (`libcudnn9-dev-cuda-12`)                                        |
| TensorRT    | tarballs only: `/home/cyrus/work/libs/TensorRT-10.0.0.6` and `...-8.6.1.6`|
| OpenCV      | **4.8.0 with CUDA** at `/usr/local` (`OpenCV_DIR=/usr/local/lib/cmake/opencv4`); apt `libopencv-dev` 4.6.0 also present (no CUDA) |
| GCC / G++   | 13.3.0                                                                    |
| CMake       | 3.28.3                                                                    |
| Ninja       | 1.11.1                                                                    |
| ccache      | `/usr/bin/ccache`                                                         |
| Python      | 3.12.3 (Ubuntu PEP-668 externally-managed; pybind11 absent)              |
| gh CLI      | authenticated as `cyrusbehr` (repo scope) for Phase C                    |

Missing at bootstrap, required to build v6: `libspdlog-dev`, `libfmt-dev`
(the prior `build/CMakeCache.txt` from Dec 2024 predates their removal). These
plus Python/doc/clang tooling are installed via the one Phase 0 sudo gate.

## Constraints that drive the design (and deviations from the plan defaults)

The plan's `§9.4` lists "CUDA Toolkit (latest GA)" and "TensorRT (latest GA)" as
targets, "override only with explicit justification." Justified overrides for
this host:

1. **CUDA stays on the 12.x line; CUDA 13 is not used here.** CUDA 13 requires
   an NVIDIA driver newer than 565 (≈ 580+). Upgrading the driver is host-specific
   and needs `sudo` + a reboot; the plan (`§9.3`) explicitly forbids the script from
   installing the driver. Driver 565 supports up to CUDA 12.7, so **CUDA 12.6**
   (installed) is the development target on this host. The library and
   `install_deps.sh` still *support* CUDA 13 for hosts with a new enough driver
   (auto-detected), but it is not validated here.

2. **TensorRT target is the latest GA 10.x for CUDA 12.x.** The host has TRT
   10.0.0.6 tarball, which is sufficient to prove the v6 build. The exact TRT
   10.x minor to develop v7 against is finalized after Phase B research; the v7
   API version-gates everything that TRT 11 will remove (calibrator family) so it
   keeps compiling across the 10.x→11 boundary.

3. **OpenCV-CUDA is available but treated as optional in v7.** v6 hard-depends on
   `cv::cuda::GpuMat`; v7's engine core will not (decided in Phase D). The 4.8.0
   CUDA build at `/usr/local` is used for the v6 verification build and for the
   optional OpenCV interop module.

`sm_86` is the build/test compute capability; `CMAKE_CUDA_ARCHITECTURES` defaults
to native but the install docs note the broader set a release should target.
