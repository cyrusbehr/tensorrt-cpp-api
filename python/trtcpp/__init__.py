"""tensorrt_cpp_api - a no-throw C++ TensorRT inference library, exposed to Python.

The whole API lives in the compiled ``_core`` extension; this package re-exports it and adds
the package version. The design mirrors the C++ surface: a no-throw core where every fallible
call raises :class:`TrtcppError` (carrying ``.code`` and the message) instead of returning a
status.

Zero-copy contract:
    * Inputs are dicts ``{name: array}`` where ``array`` is a :class:`Tensor` / :class:`TensorView`
      or any object exposing ``__cuda_array_interface__`` (CuPy, Numba, PyTorch CUDA tensors) or
      ``__dlpack__``. No host<->device copy happens at the boundary.
    * Outputs are :class:`Tensor` objects exposing ``__cuda_array_interface__`` (device) /
      ``__array_interface__`` (host) and ``__dlpack__`` / ``__dlpack_device__`` -- so
      ``torch.from_dlpack(out)`` and ``cupy.asarray(out)`` alias the device memory.

Because inference is asynchronous on the caller's stream and never implicitly synchronizes,
you MUST keep input arrays alive and synchronize the stream before consuming outputs.
"""

from ._core import (  # noqa: F401
    BuildOptions,
    DType,
    Device,
    DeviceInfo,
    Engine,
    EngineBuilder,
    EngineOptions,
    EnginePool,
    Layout,
    Lease,
    OptimizationProfile,
    Precision,
    ProfileShape,
    Shape,
    StatusCode,
    Stream,
    Tensor,
    TensorInfo,
    TensorView,
    TrtcppError,
    device_count,
    dtype_to_string,
    library_version,
    query_device,
    tensorrt_build_version,
    version_string,
)

try:
    from ._core import preproc  # noqa: F401
except ImportError:  # built without the preprocessing sublibrary
    preproc = None

__all__ = [
    "BuildOptions",
    "DType",
    "Device",
    "DeviceInfo",
    "Engine",
    "EngineBuilder",
    "EngineOptions",
    "EnginePool",
    "Layout",
    "Lease",
    "OptimizationProfile",
    "Precision",
    "ProfileShape",
    "Shape",
    "StatusCode",
    "Stream",
    "Tensor",
    "TensorInfo",
    "TensorView",
    "TrtcppError",
    "device_count",
    "dtype_to_string",
    "library_version",
    "preproc",
    "query_device",
    "tensorrt_build_version",
    "version_string",
]

__version__ = ".".join(str(v) for v in library_version())
