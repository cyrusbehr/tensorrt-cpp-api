"""Tests for the trtcpp Python bindings.

CPU/pure tests run anywhere; GPU tests skip when no CUDA device is visible. Run with:
    PYTHONPATH=python pytest python/tests -q
(or, against an installed wheel, just `pytest python/tests`).
"""

import os

import numpy as np
import pytest

import trtcpp
from trtcpp import BuildOptions, DType, Device, EngineBuilder, EnginePool, Precision, Shape, Stream, Tensor, TensorView, TrtcppError

MODELS = os.path.join(os.path.dirname(__file__), "..", "..", "tests", "models")
RELU = os.path.join(MODELS, "relu_1x3x8x8.onnx")

has_gpu = trtcpp.device_count() > 0
gpu = pytest.mark.skipif(not has_gpu, reason="no CUDA device")


# ---- pure / CPU ----

def test_version():
    assert trtcpp.library_version() == (7, 0, 0)
    assert trtcpp.tensorrt_build_version()[0] >= 10
    assert "7.0.0" in trtcpp.version_string()
    assert trtcpp.__version__ == "7.0.0"


def test_shape():
    s = Shape([1, 3, 224, 224])
    assert s.rank() == 4
    assert len(s) == 4
    assert s[2] == 224
    assert s.numel() == 1 * 3 * 224 * 224
    assert not s.is_dynamic()
    assert Shape([-1, 3]).is_dynamic()
    assert s.dims() == [1, 3, 224, 224]
    assert Shape([1, 2]) == Shape([1, 2])


def test_enums():
    assert trtcpp.dtype_to_string(DType.Float16) == "float16"
    assert DType.Float32 != DType.Float16
    assert Precision.Int8Qdq is not None


def test_error_type():
    assert issubclass(TrtcppError, Exception)
    with pytest.raises(TrtcppError) as ei:
        trtcpp.Engine.load_from_file("/nonexistent/does_not_exist.engine")
    assert ei.value.code == trtcpp.StatusCode.NotFound


# ---- GPU ----

@gpu
def test_device_query():
    info = trtcpp.query_device(0)
    assert info.compute_major >= 5
    assert info.total_memory_bytes > 0
    assert info.name


@gpu
def test_tensor_alloc_and_interfaces():
    t = Tensor.allocate(DType.Float32, Shape([2, 3]), Device.Cuda)
    assert t.nbytes == 2 * 3 * 4
    cai = t.__cuda_array_interface__
    assert cai["shape"] == (2, 3)
    assert cai["typestr"] == "<f4"
    assert cai["version"] == 3
    assert cai["data"][0] == t.data_ptr
    assert t.__dlpack_device__() == (2, 0)  # (kDLCUDA, device 0)
    cap = t.__dlpack__()
    assert type(cap).__name__ == "PyCapsule"


@gpu
def test_host_roundtrip_via_numpy_dlpack():
    s = Stream()
    src = np.arange(12, dtype=np.float32).reshape(3, 4)
    dev = Tensor.allocate(DType.Float32, Shape([3, 4]), Device.Cuda)
    dev.copy_from(src, s)  # numpy -> host view (DLPack) -> H2D
    back = np.asarray(dev.to_host(s))  # D2H + sync, then __array_interface__
    assert np.array_equal(back, src)


@gpu
def test_cuda_array_interface_import_is_zero_copy():
    t = Tensor.allocate(DType.Float32, Shape([2, 5]), Device.Cuda)

    class Producer:
        def __init__(self, src):
            self.__cuda_array_interface__ = src.__cuda_array_interface__

    view = TensorView.from_array(Producer(t))
    assert view.data_ptr == t.data_ptr
    assert view.shape.dims() == [2, 5]
    assert view.device == Device.Cuda


@pytest.fixture(scope="module")
def relu_engine_path(tmp_path_factory):
    cache = str(tmp_path_factory.mktemp("engines"))
    bo = BuildOptions()
    bo.precision = Precision.Fp32
    bo.engine_cache_dir = cache
    return EngineBuilder().build_or_load(RELU, bo)


@gpu
def test_engine_infer_relu(relu_engine_path):
    s = Stream()
    eng = trtcpp.Engine.load_from_file(relu_engine_path)
    assert eng.input_names() == ["input"]
    assert eng.output_names() == ["output"]
    x = np.random.randn(1, 3, 8, 8).astype(np.float32)
    dx = Tensor.allocate(DType.Float32, Shape([1, 3, 8, 8]), Device.Cuda)
    dx.copy_from(x, s)
    out = eng.infer_single({"input": dx}, s)
    y = np.asarray(out.to_host(s))
    assert y.shape == (1, 3, 8, 8)
    assert np.allclose(y, np.maximum(x, 0), atol=1e-4)


@gpu
def test_engine_infer_dict_and_output_shapes(relu_engine_path):
    s = Stream()
    eng = trtcpp.Engine.load_from_file(relu_engine_path)
    x = np.zeros((1, 3, 8, 8), dtype=np.float32)
    dx = Tensor.allocate(DType.Float32, Shape([1, 3, 8, 8]), Device.Cuda)
    dx.copy_from(x, s)
    shapes = eng.output_shapes({"input": dx})
    assert shapes["output"].dims() == [1, 3, 8, 8]
    outs = eng.infer({"input": dx}, s)
    assert set(outs) == {"output"}
    assert np.asarray(outs["output"].to_host(s)).shape == (1, 3, 8, 8)


@gpu
def test_engine_pool(relu_engine_path):
    pool = EnginePool.create(relu_engine_path, contexts=2)
    assert pool.size() == 2
    s = Stream()
    x = np.full((1, 3, 8, 8), -1.0, dtype=np.float32)
    dx = Tensor.allocate(DType.Float32, Shape([1, 3, 8, 8]), Device.Cuda)
    dx.copy_from(x, s)
    lease = pool.acquire()
    assert lease.valid
    out = lease.infer_single({"input": dx}, s)
    y = np.asarray(out.to_host(s))
    assert np.allclose(y, 0.0)  # relu of all-negative


@gpu
@pytest.mark.skipif(trtcpp.preproc is None, reason="preproc sublib not built")
def test_preproc_letterbox():
    s = Stream()
    # HWC uint8 image, resize/normalize to NCHW float32.
    img = (np.random.rand(16, 16, 3) * 255).astype(np.uint8)
    src = Tensor.allocate(DType.UInt8, Shape([16, 16, 3]), Device.Cuda)
    src.copy_from(img, s)
    dst = Tensor.allocate(DType.Float32, Shape([1, 3, 8, 8]), Device.Cuda)
    spec = trtcpp.preproc.PreprocSpec()
    spec.scale = [1.0 / 255, 1.0 / 255, 1.0 / 255, 1.0]
    trtcpp.preproc.letterbox_to_tensor(src, dst, spec, s)
    y = np.asarray(dst.to_host(s))
    assert y.shape == (1, 3, 8, 8)
    assert 0.0 <= y.min() and y.max() <= 1.0 + 1e-4
