# AGENTS.md

Orientation file for LLM coding agents (Claude Code, Codex, Cursor). Humans:
read `README.md` instead — if the two ever disagree, README wins for humans and
this file wins for agents; keep them in lockstep.

## What this repo is

A C++ inference library that wraps NVIDIA TensorRT for CNN-class models (build an
engine from ONNX, cache it, run inference), with optional first-class Python
bindings. Linux + NVIDIA GPU only (no Windows). It is **not** an LLM/transformer
serving framework. Goal: be the canonical open-source C++ reference for
high-performance TensorRT inference on CNNs.

The repo is mid-modernization from **v6** (templated `Engine<T>`, triply-nested
`std::vector` IO, hard OpenCV-CUDA dependency) to **v7** (name-keyed tensor IO,
caller-provided CUDA stream, `Status`/`Result` errors, optional OpenCV interop,
optional Python bindings). Active branch: **`v7-rewrite`** (`main` is kept clean
so the maintainer can revert).

## Master spec

`MODERNIZATION_PLAN.md` at the repo root is the full execution spec (phases A–J,
operating rules, subagent templates, definition of done). Read it before doing
substantive work. This file is just the orientation; it does not duplicate the
plan.

## Build

```bash
scripts/install_deps.sh --with-python   # one-time, fresh host (needs sudo)
scripts/verify_deps.sh                   # non-destructive readiness check
cmake -B build -G Ninja && cmake --build build -j
ctest --test-dir build --output-on-failure
```

If TensorRT lives in a tarball rather than apt, pass its root:
`cmake -B build -G Ninja -DTensorRT_DIR=/path/to/TensorRT-X.Y.Z`.

Host the work is developed on: see `docs/audit/PHASE_0_HOST.md` (RTX 3080 Laptop,
`sm_86`, driver 565, CUDA 12.6, OpenCV-CUDA 4.8 at `/usr/local`, TensorRT 10.0
tarball). Driver 565 caps CUDA at 12.7, so CUDA 13 is not used on this host.

## Style

- Formatting: `.clang-format` (LLVM-based, 4-space indent, 140 col). Run
  `pre-commit run --all-files` or `clang-format -i` before committing.
- **Default to no comments.** Comment only non-obvious WHY, never narrate code.
- No emojis anywhere — source, comments, or commit messages.
- Public API design rules (v7): name-keyed tensor IO (no triply-nested vectors),
  caller-provided CUDA stream, `Status`/`Result` rather than `bool`, no
  third-party types (OpenCV, spdlog, raw nvinfer1) leaked through public headers.
- Commits: one per phase, prefixed with the phase letter, e.g. `[E3] cuda helpers`.
  Never force-push, never amend, never skip hooks.

## When in doubt

- Known v6 issues / bugs: `docs/audit/PHASE_A_FINDINGS.md`
- 2026 ecosystem decisions (versions, deprecations): `docs/audit/PHASE_B_ECOSYSTEM.md`
- What real users need fixed: `docs/audit/PHASE_C_COMMUNITY.md`
- Locked API + architecture: `docs/design/PHASE_D_DESIGN.md`
- v6 -> v7 symbol migration: `docs/upgrading_from_v6.md`

Downstream consumers to keep building (Phase J): `cyrusbehr/YOLOv8-TensorRT-CPP`
and `cyrusbehr/YOLOv9-TensorRT-CPP`.
