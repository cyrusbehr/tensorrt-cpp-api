#!/usr/bin/env python3
"""Zero-copy GPU inference with trtcpp + CuPy.

Demonstrates the binding's zero-copy contract end to end: a CuPy array on the GPU is fed to the
engine with no host<->device copy (via __cuda_array_interface__), inference runs on the caller's
CUDA stream with the GIL released, and the output Tensor is wrapped back into a CuPy array (again
zero-copy) for GPU-side post-processing. The only host transfer is the final tiny top-5 result.

    python classify_zerocopy.py <model.onnx|engine> [image.npy]

If an image .npy ([3,H,W] or [1,3,H,W] float32, ImageNet-normalized) is given it is used;
otherwise a random input is generated. (Decoding/normalizing an image file is exactly what the
preproc sublib + the C++ examples do; this script focuses on the zero-copy Python plumbing.)
"""
import sys

import cupy as cp
import numpy as np

import trtcpp


def main() -> int:
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} <model.onnx|engine> [image.npy]", file=sys.stderr)
        return 2
    model_path = sys.argv[1]

    bo = trtcpp.BuildOptions()
    bo.precision = trtcpp.Precision.Fp16
    bo.engine_cache_dir = "engines"
    engine = trtcpp.EngineBuilder().build_and_load(model_path, bo)

    in_name = engine.input_names()[0]
    _, _, in_h, in_w = engine.tensor_shape(in_name).dims()

    # Build the input as a CuPy array ON THE GPU (no host buffer in the hot path).
    if len(sys.argv) > 2:
        arr = np.load(sys.argv[2]).astype(np.float32).reshape(1, 3, in_h, in_w)
        gpu_input = cp.asarray(arr)
    else:
        gpu_input = cp.random.randn(1, 3, in_h, in_w, dtype=cp.float32)

    # Adopt CuPy's current stream so trtcpp and CuPy order work on the same stream.
    stream = trtcpp.Stream.wrap(cp.cuda.get_current_stream().ptr)

    # Zero-copy in: gpu_input exposes __cuda_array_interface__, so the engine binds its device
    # pointer directly -- no copy. GIL released during inference.
    out = engine.infer_single({in_name: gpu_input}, stream)

    # Zero-copy out: wrap the output Tensor as a CuPy array (shares device memory), then do the
    # softmax/top-k on the GPU. The output Tensor must stay alive while gpu_out aliases it.
    gpu_out = cp.asarray(out).reshape(-1)
    stream.synchronize()  # the engine never implicitly syncs; do it before reading
    probs = cp.exp(gpu_out - gpu_out.max())
    probs /= probs.sum()
    topk = cp.asnumpy(cp.argsort(probs)[::-1][:5])  # only the 5 indices come back to the host

    print(f"input {gpu_input.shape} on {gpu_input.device}, zero-copy in/out")
    print("top-5:")
    for cls in topk:
        print(f"  class {int(cls):4d}  {float(probs[int(cls)]) * 100:6.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
