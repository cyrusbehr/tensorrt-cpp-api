# Python examples

Zero-copy GPU inference with the `trtcpp` bindings and [CuPy](https://cupy.dev/).

```sh
pip install cupy-cuda12x      # match your CUDA major
pip install .                 # build/install the trtcpp wheel (from the repo root)
```

## `classify_zerocopy.py` — zero-copy in and out

```sh
python classify_zerocopy.py ../../models/mobilenetv2-7.onnx
```

A CuPy array on the GPU is fed to the engine with **no host↔device copy** (via
`__cuda_array_interface__`), inference runs on CuPy's current stream with the GIL released, and
the output `Tensor` is wrapped back into a CuPy array (again zero-copy) for a GPU-side softmax.
Only the final five class indices are ever moved to the host.

The contract: the library **never implicitly synchronizes or copies to host**. Keep your input
arrays alive and call `stream.synchronize()` before reading outputs.

## `benchmark_parity.py` — Python vs C++

```sh
python benchmark_parity.py ../../models/mobilenetv2-7.onnx ../../build/examples/benchmark 300
```

Times the same preallocated zero-copy `enqueue` loop from Python and from the C++ `benchmark`
example, and asserts the Python path is within 20% of C++. The Python overhead is just one GIL
release/acquire plus a dict→`TensorView` build per call — negligible next to the GPU work.
Measured on an RTX 3080 (MobileNetV2, FP16): C++ ≈ 0.28 ms/infer, Python ≈ 0.32 ms (≈ +13%).
