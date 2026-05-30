# Installation

`tensorrt_cpp_api` is a Linux / CUDA 12 / TensorRT ≥ 10 / C++20 library. TensorRT and the CUDA
toolkit are **system/externally provided** — they are not fetched by the build.

## Prerequisites

- **CMake ≥ 3.22** and a C++20 compiler (GCC ≥ 11 or Clang ≥ 14).
- **CUDA Toolkit 12.x** (`nvcc` is needed for the preprocessing kernel; the engine core itself
  only needs the CUDA runtime). CUDA 13 requires driver ≥ 580.
- **TensorRT 10.0 – 11.x**, installed one of two ways:
  - **apt** (NVIDIA CUDA network repo): `libnvinfer-dev` + `libnvonnxparsers-dev`. The bundled
    `FindTensorRT` module searches the standard `/usr/include/x86_64-linux-gnu` +
    `/usr/lib/x86_64-linux-gnu` layout automatically.
  - **tarball**: download from NVIDIA and pass `-DTensorRT_DIR=/path/to/TensorRT-10.x` at configure.
- Optional: **spdlog** (`-DTRT_CPP_API_WITH_SPDLOG=ON`), **OpenCV** core (`-DTRT_CPP_API_WITH_OPENCV=ON`),
  **pybind11** + a Python 3.9–3.13 dev environment (for the bindings).

## Build options

| Option | Default | Effect |
|---|---|---|
| `TRT_CPP_API_BUILD_PREPROC` | `ON` | Build the fused preprocessing sublibrary (`::preproc`). |
| `TRT_CPP_API_BUILD_TESTS` | `OFF` | Build the GoogleTest suite (`ctest`; GPU tests are labeled `gpu`). |
| `TRT_CPP_API_BUILD_EXAMPLES` | `OFF` | Build the reference examples in-tree. |
| `TRT_CPP_API_BUILD_PYTHON` | `OFF` | Build the `trtcpp` pybind11 extension. |
| `TRT_CPP_API_WITH_OPENCV` | `OFF` | Build the optional OpenCV interop header/source. |
| `TRT_CPP_API_WITH_SPDLOG` | `OFF` | Build the optional spdlog logger adapter. |
| `CMAKE_CUDA_ARCHITECTURES` | `75;80;86;89;90` | Override for your target GPUs. |

## Build & install (C++)

```sh
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DTRT_CPP_API_BUILD_PREPROC=ON
  # add -DTensorRT_DIR=/opt/TensorRT-10.x for a tarball TensorRT
  # add -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc if nvcc is not on PATH
cmake --build build -j$(nproc)
cmake --install build --prefix /opt/trtcpp
```

The install lays down `include/tensorrt_cpp_api/…`, the static libraries, and a CMake package
(`lib/cmake/tensorrt_cpp_api/`) including the bundled `FindTensorRT` module.

## Consume it downstream

```cmake
find_package(tensorrt_cpp_api REQUIRED)   # set CMAKE_PREFIX_PATH=/opt/trtcpp (and TensorRT_DIR if tarball)
add_executable(myapp main.cpp)
target_link_libraries(myapp PRIVATE
    tensorrt_cpp_api::tensorrt_cpp_api    # core
    tensorrt_cpp_api::preproc)            # optional preprocessing
```

The package config re-resolves CUDA, TensorRT, Threads (and OpenCV/spdlog if the install was built
with them), so the only thing a consumer must provide is the location of a system TensorRT.

## Python bindings

```sh
pip install .                                    # builds the trtcpp wheel via scikit-build-core
# tarball TensorRT: pip install . --config-settings=cmake.define.TensorRT_DIR=/opt/TensorRT-10.x
python -c "import trtcpp; print(trtcpp.version_string())"
```

For zero-copy GPU interop install a matching CuPy (`pip install cupy-cuda12x`); see
[`examples/python`](../examples/python).

## Verifying

```sh
cmake -S . -B build -DTRT_CPP_API_BUILD_TESTS=ON
cmake --build build -j$(nproc)
ctest --test-dir build -LE gpu          # CPU-only tests (no GPU needed)
ctest --test-dir build                  # full suite (needs an NVIDIA GPU)
```

## Troubleshooting

- **`Could NOT find CUDA: ... required is exact version "12.0"`** when building examples/consumers.
  This comes from a CUDA-enabled OpenCV whose CMake config pins an exact CUDA version that differs
  from your toolkit. The library core does **not** use OpenCV; build with the default
  `-DTRT_CPP_API_WITH_OPENCV=OFF`, and prefer the stb-based examples (no OpenCV).
- **`Failed to detect a default CUDA architecture` / `nvcc` not found.** Pass
  `-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc`, or put `nvcc` on `PATH`.
- **`Could NOT find TensorRT`.** Install `libnvinfer-dev`/`libnvonnxparsers-dev`, or pass
  `-DTensorRT_DIR=<tarball-root>`. TensorRT must be in the 10.0–11.x range.
- **Stale-engine rebuilds.** Changing the ONNX, build options, driver/TensorRT version, or GPU
  invalidates a cached engine on purpose; delete `engineCacheDir` to force a clean rebuild.
