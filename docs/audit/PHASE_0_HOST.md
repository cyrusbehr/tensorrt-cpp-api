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

## Toolchain verification (Phase 0 pass-gate)

The plan's Phase 0 gate is "prove the env can build TensorRT code for v7." Verified
2026-05-29 with a TensorRT + CUDA smoke program (compile + link + run against
TensorRT 10.0.0.6 + CUDA 12.6 + g++ 13.3):

```
cudaGetDeviceCount: no error, devices=1
GPU0: NVIDIA GeForce RTX 3080 Laptop GPU  sm_86
createInferBuilder: 0x...  TensorRT 10.0.0
createNetworkV2(0): 0x...  nbInputs=0
createParser(nvonnxparser): 0x...
TOOLCHAIN OK
```

This exercises exactly what the v7 core needs: TRT headers compile, `nvinfer` +
`nvonnxparser` link, the builder/parser instantiate, CUDA sees the GPU. The only
compiler output is the expected TRT-header deprecation warnings (`IGpuAllocator::
allocate`/`deallocate` -> use the `*Async` variants — already reflected in the E3
stream-ordered allocator design).

### Deviation: the current v6 repo does NOT build on this host as-is

Not because of the TRT/CUDA toolchain (verified above) but because the **OpenCV-CUDA
build at `/usr/local` has drifted from the current CUDA/cuDNN**:

- It was compiled against **CUDA 12.0 + cuDNN 8.9.7** (`OpenCVConfig.cmake:99,104`); the
  host now has CUDA 12.6 + cuDNN 9.6.0. `OpenCVConfig.cmake:110` does a hard
  `VERSION_EQUAL` check and aborts: *"OpenCV ... was compiled with CUDA 12.0 ... rebuild
  with CUDA 12.6."*
- Even bypassing that, only `libcudnn.so.9` exists while `libopencv_dnn.so.4.8.0`
  hard-links the absent `libcudnn.so.8` (cuDNN 8->9 is an ABI break). v6's blanket
  `find_package(OpenCV)` pulls `libopencv_dnn` into `${OpenCV_LIBS}`, so the link/load
  fails.

This is, ironically, a textbook instance of the environment fragility the rewrite
targets (community #45/#46 CUDA-version mismatch, #32/#52/#84 OpenCV-CUDA pain). The
deliberate decision (justified per the plan's "route around the blocker" rule) is to
**not** rebuild OpenCV-CUDA just to reproduce a v6 build, because:

1. It would require a second sudo gate (install to `/usr/local`) plus a long source
   build, for a dependency **v7 removes as a hard requirement** (Phase D / Phase B).
2. The toolchain that v7 actually depends on (CUDA + TensorRT + compiler) is verified
   above; the OpenCV drift is orthogonal to it.
3. v7's engine core compiles against CUDA + TensorRT only and never links OpenCV, so
   this drift cannot block the v7 build — the new `install_deps.sh`/`verify_deps.sh` +
   optional-OpenCV design are precisely the fix.
