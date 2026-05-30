#!/usr/bin/env python3
"""Perf-parity check: Python zero-copy inference vs the C++ baseline.

Times the same zero-copy enqueue loop (preallocated device IO, per-iteration stream sync) from
Python and compares against the C++ `benchmark` example on the identical engine. Asserts the
Python path is within a tolerance (default 20%) of C++ -- i.e. the binding overhead (one GIL
release/acquire + a dict->TensorView build per call) is negligible next to the GPU work.

    python benchmark_parity.py <model.onnx|engine> <path/to/cpp_benchmark> [iters=200] [tol=0.20]
"""
import subprocess
import sys
import time

import cupy as cp

import trtcpp


def main() -> int:
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <model> <cpp_benchmark_bin> [iters=200] [tol=0.20]", file=sys.stderr)
        return 2
    model_path = sys.argv[1]
    cpp_bin = sys.argv[2]
    iters = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    tol = float(sys.argv[4]) if len(sys.argv) > 4 else 0.20

    bo = trtcpp.BuildOptions()
    bo.precision = trtcpp.Precision.Fp16
    bo.engine_cache_dir = "engines"
    engine = trtcpp.EngineBuilder().build_and_load(model_path, bo)

    in_name = engine.input_names()[0]
    out_name = engine.output_names()[0]
    in_dims = engine.tensor_shape(in_name).dims()
    out_dims = engine.tensor_shape(out_name).dims()

    stream = trtcpp.Stream.wrap(cp.cuda.get_current_stream().ptr)
    gpu_in = cp.zeros(in_dims, dtype=cp.float32)
    # Preallocated output Tensor reused across iterations (zero-copy enqueue, mirrors the C++ bench).
    out = trtcpp.Tensor.allocate(trtcpp.DType.Float32, trtcpp.Shape(out_dims), trtcpp.Device.Cuda)
    inputs = {in_name: gpu_in}
    outputs = {out_name: out}

    def once():
        engine.enqueue(inputs, outputs, stream)
        stream.synchronize()

    for _ in range(20):  # warmup
        once()
    t0 = time.perf_counter()
    for _ in range(iters):
        once()
    py_ms = (time.perf_counter() - t0) * 1000.0 / iters

    # C++ baseline.
    res = subprocess.run([cpp_bin, model_path, str(iters)], capture_output=True, text=True)
    cpp_ms = None
    for line in res.stdout.splitlines():
        if line.startswith("cpp_latency_ms"):
            cpp_ms = float(line.split()[1])
    if cpp_ms is None:
        print("could not parse C++ benchmark output:\n" + res.stdout + res.stderr, file=sys.stderr)
        return 1

    overhead = (py_ms - cpp_ms) / cpp_ms
    print(f"C++   : {cpp_ms:.3f} ms/infer  ({1000 / cpp_ms:.0f} inf/s)")
    print(f"Python: {py_ms:.3f} ms/infer  ({1000 / py_ms:.0f} inf/s)")
    print(f"overhead: {overhead * 100:+.1f}%  (tolerance +-{tol * 100:.0f}%)")
    if abs(overhead) <= tol:
        print("PARITY OK")
        return 0
    print("PARITY FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
