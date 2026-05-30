# AGENTS.md

Orientation for AI coding agents (Claude Code, Codex, Cursor) working in this repo. Humans should
read [`README.md`](README.md); if the two ever disagree, README wins for humans and this file wins
for agents — keep them in lockstep.

## What this is

`tensorrt_cpp_api` is a C++ library that wraps NVIDIA TensorRT for CNN-class vision models: build a
TensorRT engine from ONNX, cache it on disk, and run inference. It has optional fused GPU
preprocessing, optional OpenCV interop, and optional zero-copy Python bindings. **Linux + NVIDIA
GPU only.** It is **not** an LLM/transformer serving framework, and Windows is out of scope.

Targets TensorRT >= 10 (written to the TensorRT 11 surface, version-gated), CUDA 12, C++20.

## Build & test

```bash
cmake -S . -B build -DTRT_CPP_API_BUILD_TESTS=ON   # add -DTensorRT_DIR=<root> for a tarball TensorRT
cmake --build build -j
ctest --test-dir build -LE gpu   # CPU-only tests; drop -LE gpu to run the full suite (needs a GPU)
```

`scripts/install_deps.sh` (one-time, needs sudo) and `scripts/verify_deps.sh` help set up a host.
If `nvcc` is not on `PATH`, pass `-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc`.

## Layout

- `include/tensorrt_cpp_api/` — the public headers (one umbrella `all.h`). No `nvinfer1`, OpenCV, or
  spdlog types appear here; the optional `preproc.h` / `opencv_interop.h` are separate.
- `src/` — implementation; `src/detail/` is internal (TensorRT glue, cache, buffers, execution).
- `python/` — the pybind11 `trtcpp` extension. `examples/` — reference programs. `tests/` — GoogleTest.

## Conventions

- **Formatting:** `.clang-format` (LLVM-based, 4-space, 140 col) and `.cmake-format.yaml`. Run
  `pre-commit run --all-files` (or `clang-format -i`) before committing. CI enforces clang-format.
- **Comments:** default to none; comment only the non-obvious *why*, never narrate code.
- **No emojis** anywhere — source, comments, or commit messages.
- **Public API:** no-throw — every fallible call returns `Status` or `Result<T>` (no exceptions);
  name-keyed tensor IO (`unordered_map<string, TensorView>`); caller-provided CUDA streams; no
  third-party types leaked through public headers (PImpl + version-gating).
- **Git:** never force-push, never amend a pushed commit, never skip hooks.

## More

- Usage & concepts: [`docs/quickstart.md`](docs/quickstart.md)
- Install options: [`docs/install.md`](docs/install.md)
- Migrating from v6: [`docs/upgrading_from_v6.md`](docs/upgrading_from_v6.md)

Reference downstream consumers: [`YOLOv8-TensorRT-CPP`](https://github.com/cyrusbehr/YOLOv8-TensorRT-CPP)
and [`YOLOv9-TensorRT-CPP`](https://github.com/cyrusbehr/YOLOv9-TensorRT-CPP).
