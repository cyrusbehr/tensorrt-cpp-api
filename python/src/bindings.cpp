// pybind11 bindings for tensorrt_cpp_api (module trtcpp._core).
//
// Design:
//   * The C++ API is no-throw (Status/Result). Here that maps to Python exceptions: every
//     fallible call is unwrapped, raising trtcpp.TrtcppError (carrying .code and message) on
//     failure and returning the value otherwise.
//   * Zero-copy in: inputs are dicts {name: array} where array is a trtcpp.Tensor/TensorView
//     or any object exposing __cuda_array_interface__ (CuPy, Numba, PyTorch CUDA tensors) or
//     __dlpack__. No host<->device copy happens at the boundary.
//   * Zero-copy out: the owning Tensor exposes __cuda_array_interface__ (device) /
//     __array_interface__ (host) AND __dlpack__/__dlpack_device__, so torch.from_dlpack(out)
//     and cupy.asarray(out) alias the device memory.
//   * Caller stream: trtcpp.Stream wraps a torch/cupy stream handle (Stream.wrap(int)).
//   * No implicit D2H: readback is the explicit Tensor.to_host(stream). The GIL is released
//     around the actual inference/build call (not the Python<->TensorView conversion).
//
// Lifetime contract (documented on the Python side): because inference is asynchronous on the
// caller's stream and never implicitly synchronizes, the caller MUST keep input arrays alive
// and synchronize the stream before consuming outputs.

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "tensorrt_cpp_api/all.h"
#ifdef TRT_CPP_API_PYTHON_WITH_PREPROC
#include "tensorrt_cpp_api/preproc.h"
#endif

#include "dlpack.h"

namespace py = pybind11;
using namespace trtcpp;

namespace {

// ---- error mapping ---------------------------------------------------------------------

py::object &errorClass() {
    static py::object cls;
    return cls;
}

[[noreturn]] void raiseStatus(const Status &s) {
    py::object exc = errorClass()(s.message());
    exc.attr("code") = py::cast(s.code());
    PyErr_SetObject(errorClass().ptr(), exc.ptr());
    throw py::error_already_set();
}

void check(const Status &s) {
    if (!s) {
        raiseStatus(s);
    }
}

template <class T> T unwrap(Result<T> &&r) {
    if (!r) {
        raiseStatus(r.status());
    }
    return std::move(r).value();
}

// ---- dtype <-> numpy typestr / DLPack --------------------------------------------------

std::string typestrOf(DType t) {
    switch (t) {
    case DType::kFloat32:
        return "<f4";
    case DType::kFloat16:
        return "<f2";
    case DType::kInt32:
        return "<i4";
    case DType::kInt64:
        return "<i8";
    case DType::kInt8:
        return "|i1";
    case DType::kUInt8:
        return "|u1";
    case DType::kBool:
        return "|b1";
    default:
        break;
    }
    throw py::value_error("dtype " + std::string(toString(t)) + " has no array-interface typestr");
}

DType dtypeFromTypestr(const std::string &ts) {
    // ts is like "<f4", "|u1"; the leading char is byteorder ('<','>','|','='), then kind, then size.
    if (ts.size() < 3) {
        throw py::value_error("malformed __cuda_array_interface__ typestr: " + ts);
    }
    const char kind = ts[1];
    const int size = std::stoi(ts.substr(2));
    if (kind == 'f' && size == 4)
        return DType::kFloat32;
    if (kind == 'f' && size == 2)
        return DType::kFloat16;
    if (kind == 'i' && size == 4)
        return DType::kInt32;
    if (kind == 'i' && size == 8)
        return DType::kInt64;
    if (kind == 'i' && size == 1)
        return DType::kInt8;
    if (kind == 'u' && size == 1)
        return DType::kUInt8;
    if (kind == 'b' && size == 1)
        return DType::kBool;
    throw py::value_error("unsupported __cuda_array_interface__ typestr for TensorRT IO: " + ts);
}

DLDataType toDLDataType(DType t) {
    switch (t) {
    case DType::kFloat32:
        return {kDLFloat, 32, 1};
    case DType::kFloat16:
        return {kDLFloat, 16, 1};
    case DType::kBFloat16:
        return {kDLBfloat, 16, 1};
    case DType::kInt32:
        return {kDLInt, 32, 1};
    case DType::kInt64:
        return {kDLInt, 64, 1};
    case DType::kInt8:
        return {kDLInt, 8, 1};
    case DType::kUInt8:
        return {kDLUInt, 8, 1};
    case DType::kBool:
        return {kDLBool, 8, 1};
    default:
        break;
    }
    throw py::value_error("dtype " + std::string(toString(t)) + " has no DLPack mapping");
}

DType dtypeFromDL(DLDataType d) {
    if (d.lanes != 1) {
        throw py::value_error("vectorized DLPack dtypes (lanes != 1) are unsupported");
    }
    if (d.code == kDLFloat && d.bits == 32)
        return DType::kFloat32;
    if (d.code == kDLFloat && d.bits == 16)
        return DType::kFloat16;
    if (d.code == kDLBfloat && d.bits == 16)
        return DType::kBFloat16;
    if (d.code == kDLInt && d.bits == 32)
        return DType::kInt32;
    if (d.code == kDLInt && d.bits == 64)
        return DType::kInt64;
    if (d.code == kDLInt && d.bits == 8)
        return DType::kInt8;
    if (d.code == kDLUInt && d.bits == 8)
        return DType::kUInt8;
    if (d.code == kDLBool && d.bits == 8)
        return DType::kBool;
    throw py::value_error("unsupported DLPack dtype (code/bits) for TensorRT IO");
}

Shape makeShape(const std::vector<std::int64_t> &dims) { return Shape(std::span<const std::int64_t>(dims.data(), dims.size())); }

// Resolve the CUDA device a device pointer lives on (avoids assuming the current device).
int deviceOfPointer(const void *ptr) {
    cudaPointerAttributes attr{};
    if (cudaPointerGetAttributes(&attr, ptr) != cudaSuccess) {
        cudaGetLastError(); // clear
        int cur = 0;
        cudaGetDevice(&cur);
        return cur;
    }
    return attr.device;
}

// Require C-contiguous strides. CAI strides are in BYTES; None => contiguous. `strides` is
// untrusted (it comes straight from a foreign object's __cuda_array_interface__), so its
// length is validated against the shape before indexing.
void requireContiguousBytes(const std::vector<std::int64_t> &dims, DType dtype, const std::vector<std::int64_t> &strides) {
    if (strides.size() != dims.size()) {
        throw py::value_error("__cuda_array_interface__ strides length does not match shape rank");
    }
    const auto itemsize = static_cast<std::int64_t>(byteSize(dtype));
    std::int64_t expected = itemsize;
    for (int i = static_cast<int>(dims.size()) - 1; i >= 0; --i) {
        if (strides[static_cast<std::size_t>(i)] != expected) {
            throw py::value_error("TensorRT IO requires a C-contiguous array (non-contiguous strides given)");
        }
        // Overflow-checked: a malformed/huge shape from a foreign __cuda_array_interface__ must not
        // wrap `expected` into a value that spuriously matches a later stride.
        if (__builtin_mul_overflow(expected, dims[static_cast<std::size_t>(i)], &expected)) {
            throw py::value_error("__cuda_array_interface__ shape is too large (stride overflow)");
        }
    }
}

// ---- Python object -> TensorView (zero-copy) -------------------------------------------

TensorView viewFromCAI(const py::dict &cai) {
    auto dataTuple = cai["data"].cast<py::tuple>();
    auto ptr = reinterpret_cast<void *>(dataTuple[0].cast<std::uintptr_t>());
    const DType dtype = dtypeFromTypestr(cai["typestr"].cast<std::string>());

    std::vector<std::int64_t> dims;
    for (auto d : cai["shape"].cast<py::tuple>()) {
        dims.push_back(d.cast<std::int64_t>());
    }
    if (cai.contains("strides") && !cai["strides"].is_none()) {
        std::vector<std::int64_t> strides;
        for (auto s : cai["strides"].cast<py::tuple>()) {
            strides.push_back(s.cast<std::int64_t>());
        }
        requireContiguousBytes(dims, dtype, strides);
    }
    return TensorView(ptr, dtype, makeShape(dims), Device::kCuda, deviceOfPointer(ptr));
}

TensorView viewFromDlpack(py::handle obj) {
    py::object cap = obj.attr("__dlpack__")();
    if (!PyCapsule_IsValid(cap.ptr(), "dltensor")) {
        throw py::type_error("__dlpack__ did not return a valid 'dltensor' capsule");
    }
    auto *mt = static_cast<DLManagedTensor *>(PyCapsule_GetPointer(cap.ptr(), "dltensor"));
    // Standard DLPack consumer protocol: rename to "used_dltensor" so the producer's capsule
    // destructor will NOT also invoke the deleter when `cap` is GC'd -- we are now responsible
    // for the single deleter call below. This is what prevents a double free.
    PyCapsule_SetName(cap.ptr(), "used_dltensor");
    const DLTensor &dt = mt->dl_tensor;

    const Device dev = dt.device.device_type == kDLCUDA ? Device::kCuda : Device::kHost;
    const DType dtype = dtypeFromDL(dt.dtype);
    std::vector<std::int64_t> dims(dt.shape, dt.shape + dt.ndim);
    if (dt.strides) {
        // DLPack strides are in ELEMENTS.
        std::int64_t expected = 1;
        for (int i = dt.ndim - 1; i >= 0; --i) {
            if (dt.strides[i] != expected || __builtin_mul_overflow(expected, dims[static_cast<std::size_t>(i)], &expected)) {
                if (mt->deleter) {
                    mt->deleter(mt);
                }
                throw py::value_error("TensorRT IO requires a C-contiguous DLPack tensor (or shape too large)");
            }
        }
    }
    void *data = static_cast<char *>(dt.data) + dt.byte_offset;
    const int deviceId = dt.device.device_id;
    TensorView v(data, dtype, makeShape(dims), dev, deviceId);
    // Release the borrow now: the address stays valid via the (caller-kept-alive) source object.
    if (mt->deleter) {
        mt->deleter(mt);
    }
    return v;
}

TensorView toTensorView(py::handle obj) {
    if (py::isinstance<Tensor>(obj)) {
        return obj.cast<Tensor &>().view();
    }
    if (py::isinstance<TensorView>(obj)) {
        return obj.cast<TensorView>();
    }
    if (py::hasattr(obj, "__cuda_array_interface__")) {
        return viewFromCAI(obj.attr("__cuda_array_interface__").cast<py::dict>());
    }
    if (py::hasattr(obj, "__dlpack__")) {
        return viewFromDlpack(obj);
    }
    throw py::type_error("expected a trtcpp.Tensor/TensorView or an object exposing "
                         "__cuda_array_interface__ or __dlpack__");
}

std::unordered_map<std::string, TensorView> toViewMap(const py::dict &d) {
    std::unordered_map<std::string, TensorView> m;
    for (auto item : d) {
        m.emplace(item.first.cast<std::string>(), toTensorView(item.second));
    }
    return m;
}

py::dict tensorMapToDict(std::unordered_map<std::string, Tensor> &&m) {
    py::dict out;
    for (auto &kv : m) {
        out[py::str(kv.first)] = py::cast(std::move(kv.second));
    }
    return out;
}

// ---- DLPack export ---------------------------------------------------------------------

struct DlpackCtx {
    std::vector<std::int64_t> shape;
    py::object owner; // keeps the producing trtcpp.Tensor alive while the capsule lives
};

void dlpackDeleter(DLManagedTensor *self) {
    auto *ctx = static_cast<DlpackCtx *>(self->manager_ctx);
    {
        // The consumer may invoke this from a thread that does or does not hold the GIL;
        // gil_scoped_acquire (PyGILState_Ensure) is reentrant-safe for both, and is required
        // because dropping the py::object owner touches the Python refcount.
        py::gil_scoped_acquire gil;
        ctx->owner = py::object();
    }
    delete ctx; // frees ctx->shape, which backed self->dl_tensor.shape for the capsule's life
    delete self;
}

void dlpackCapsuleDestructor(PyObject *capsule) {
    // Only fires if the consumer never renamed/consumed the capsule.
    if (PyCapsule_IsValid(capsule, "dltensor")) {
        auto *mt = static_cast<DLManagedTensor *>(PyCapsule_GetPointer(capsule, "dltensor"));
        if (mt && mt->deleter) {
            mt->deleter(mt);
        }
    }
}

py::capsule tensorDlpack(py::object self) {
    auto &t = self.cast<Tensor &>();
    if (t.empty()) {
        throw py::value_error("cannot export an empty Tensor via DLPack");
    }
    auto *ctx = new DlpackCtx{};
    ctx->shape.reserve(static_cast<std::size_t>(t.shape().rank()));
    for (int i = 0; i < t.shape().rank(); ++i) {
        ctx->shape.push_back(t.shape()[i]);
    }
    ctx->owner = self;

    auto *mt = new DLManagedTensor{};
    mt->dl_tensor.data = t.data();
    mt->dl_tensor.device = DLDevice{t.device() == Device::kCuda ? kDLCUDA : kDLCPU, t.deviceId()};
    mt->dl_tensor.ndim = t.shape().rank();
    mt->dl_tensor.dtype = toDLDataType(t.dtype());
    mt->dl_tensor.shape = ctx->shape.data();
    mt->dl_tensor.strides = nullptr; // C-contiguous
    mt->dl_tensor.byte_offset = 0;
    mt->manager_ctx = ctx;
    mt->deleter = dlpackDeleter;
    return py::capsule(mt, "dltensor", dlpackCapsuleDestructor);
}

py::dict cudaArrayInterface(const Tensor &t) {
    if (t.device() != Device::kCuda) {
        throw py::attribute_error("__cuda_array_interface__ is only available on CUDA tensors; use __array_interface__");
    }
    py::tuple shape(t.shape().rank());
    for (int i = 0; i < t.shape().rank(); ++i) {
        shape[i] = t.shape()[i];
    }
    py::dict d;
    d["shape"] = shape;
    d["typestr"] = typestrOf(t.dtype());
    d["data"] = py::make_tuple(reinterpret_cast<std::uintptr_t>(t.data()), false);
    d["strides"] = py::none(); // C-contiguous
    d["version"] = 3;
    // No "stream" key: the producing stream is the caller's and never implicitly synced.
    // Synchronize that stream before handing this to another library (documented).
    return d;
}

py::dict arrayInterface(const Tensor &t) {
    if (t.device() != Device::kHost) {
        throw py::attribute_error("__array_interface__ is only available on host tensors; use __cuda_array_interface__");
    }
    py::tuple shape(t.shape().rank());
    for (int i = 0; i < t.shape().rank(); ++i) {
        shape[i] = t.shape()[i];
    }
    py::dict d;
    d["shape"] = shape;
    d["typestr"] = typestrOf(t.dtype());
    d["data"] = py::make_tuple(reinterpret_cast<std::uintptr_t>(t.data()), false);
    d["strides"] = py::none();
    d["version"] = 3;
    return d;
}

} // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "tensorrt_cpp_api: a no-throw C++ TensorRT inference library (Python bindings)";

    errorClass() = py::reinterpret_steal<py::object>(PyErr_NewException("trtcpp._core.TrtcppError", PyExc_RuntimeError, nullptr));
    m.attr("TrtcppError") = errorClass();

    // ---- enums ----
    py::enum_<StatusCode>(m, "StatusCode")
        .value("Ok", StatusCode::kOk)
        .value("InvalidArgument", StatusCode::kInvalidArgument)
        .value("NotFound", StatusCode::kNotFound)
        .value("IoError", StatusCode::kIoError)
        .value("CudaError", StatusCode::kCudaError)
        .value("TensorRtError", StatusCode::kTensorRtError)
        .value("ShapeMismatch", StatusCode::kShapeMismatch)
        .value("DtypeMismatch", StatusCode::kDtypeMismatch)
        .value("Unsupported", StatusCode::kUnsupported)
        .value("StaleCache", StatusCode::kStaleCache)
        .value("Internal", StatusCode::kInternal);

    py::enum_<DType>(m, "DType")
        .value("Float32", DType::kFloat32)
        .value("Float16", DType::kFloat16)
        .value("BFloat16", DType::kBFloat16)
        .value("Int32", DType::kInt32)
        .value("Int64", DType::kInt64)
        .value("Int8", DType::kInt8)
        .value("UInt8", DType::kUInt8)
        .value("Bool", DType::kBool)
        .value("Fp8", DType::kFp8)
        .value("Int4", DType::kInt4);
    m.def("dtype_to_string", [](DType t) { return std::string(toString(t)); });

    py::enum_<Device>(m, "Device").value("Cuda", Device::kCuda).value("Host", Device::kHost);
    py::enum_<Layout>(m, "Layout")
        .value("NCHW", Layout::kNCHW)
        .value("NHWC", Layout::kNHWC)
        .value("Linear", Layout::kLinear)
        .value("Unknown", Layout::kUnknown);
    py::enum_<Precision>(m, "Precision")
        .value("Fp32", Precision::kFp32)
        .value("Fp16", Precision::kFp16)
        .value("Int8Qdq", Precision::kInt8Qdq)
        .value("Int8CalibLegacy", Precision::kInt8CalibLegacy)
        .value("Fp8", Precision::kFp8)
        .value("Nvfp4", Precision::kNvfp4);

    // ---- Shape ----
    py::class_<Shape>(m, "Shape")
        .def(py::init<>())
        .def(py::init([](const std::vector<std::int64_t> &dims) { return makeShape(dims); }), py::arg("dims"))
        .def("rank", &Shape::rank)
        .def("numel", &Shape::numel)
        .def("is_dynamic", &Shape::isDynamic)
        .def("dims", [](const Shape &s) { return std::vector<std::int64_t>(s.dims().begin(), s.dims().end()); })
        .def("__len__", &Shape::rank)
        .def("__getitem__", [](const Shape &s, int i) { return s.at(i); })
        .def("__eq__", [](const Shape &a, const Shape &b) { return a == b; })
        .def("__repr__", [](const Shape &s) { return "Shape(" + s.toString() + ")"; });

    // ---- Stream ----
    py::class_<Stream>(m, "Stream")
        .def(py::init([]() { return unwrap(Stream::create()); }), "Create and own a new non-blocking CUDA stream.")
        .def_static(
            "wrap", [](std::uintptr_t handle) { return Stream::wrap(handle); }, py::arg("handle"),
            "Wrap an existing stream handle (e.g. torch.cuda.Stream.cuda_stream); non-owning.")
        .def_property_readonly("handle", &Stream::raw, "The raw stream handle as an integer.")
        .def_property_readonly("owns", &Stream::owns)
        .def("synchronize", [](const Stream &s) { check(s.synchronize()); });

    // ---- device query ----
    py::class_<DeviceInfo>(m, "DeviceInfo")
        .def_readonly("index", &DeviceInfo::index)
        .def_readonly("name", &DeviceInfo::name)
        .def_readonly("uuid", &DeviceInfo::uuid)
        .def_readonly("compute_major", &DeviceInfo::computeMajor)
        .def_readonly("compute_minor", &DeviceInfo::computeMinor)
        .def_readonly("total_memory_bytes", &DeviceInfo::totalMemoryBytes)
        .def("__repr__", [](const DeviceInfo &d) {
            return "DeviceInfo(index=" + std::to_string(d.index) + ", name='" + d.name + "', sm=" + std::to_string(d.computeMajor) + "." +
                   std::to_string(d.computeMinor) + ")";
        });
    m.def("device_count", []() { return unwrap(deviceCount()); });
    m.def("query_device", [](int index) { return unwrap(queryDevice(index)); }, py::arg("index"));

    // ---- Tensor (owning) ----
    py::class_<Tensor>(m, "Tensor")
        .def_static(
            "allocate",
            [](DType dtype, const Shape &shape, Device device, int deviceId) {
                return unwrap(Tensor::allocate(dtype, shape, device, deviceId));
            },
            py::arg("dtype"), py::arg("shape"), py::arg("device"), py::arg("device_id") = 0)
        .def_property_readonly("dtype", &Tensor::dtype)
        .def_property_readonly("shape", &Tensor::shape)
        .def_property_readonly("device", &Tensor::device)
        .def_property_readonly("device_id", &Tensor::deviceId)
        .def_property_readonly("nbytes", &Tensor::nbytes)
        .def_property_readonly("data_ptr", [](const Tensor &t) { return reinterpret_cast<std::uintptr_t>(t.data()); })
        .def(
            "copy_from", [](Tensor &t, py::handle src, const Stream &s) { check(t.copyFrom(toTensorView(src), s)); }, py::arg("src"),
            py::arg("stream"))
        .def(
            "to", [](const Tensor &t, Device d, int deviceId, const Stream &s) { return unwrap(t.to(d, deviceId, s)); }, py::arg("device"),
            py::arg("device_id"), py::arg("stream"))
        .def(
            "to_host", [](const Tensor &t, const Stream &s) { return unwrap(t.toHost(s)); }, py::arg("stream"),
            "Copy to a pinned host Tensor AND synchronize the stream; the result is immediately readable.")
        .def_property_readonly("__cuda_array_interface__", &cudaArrayInterface)
        .def_property_readonly("__array_interface__", &arrayInterface)
        .def(
            "__dlpack__", [](py::object self, py::object /*stream*/) { return tensorDlpack(std::move(self)); },
            py::arg("stream") = py::none())
        .def("__dlpack_device__", [](const Tensor &t) {
            return py::make_tuple(static_cast<int>(t.device() == Device::kCuda ? kDLCUDA : kDLCPU), t.deviceId());
        });

    // ---- TensorView (non-owning) ----
    py::class_<TensorView>(m, "TensorView")
        .def_static(
            "from_array", [](py::handle obj) { return toTensorView(obj); }, py::arg("array"),
            "Build a zero-copy view over an object exposing __cuda_array_interface__ or __dlpack__.")
        .def_property_readonly("dtype", &TensorView::dtype)
        .def_property_readonly("shape", &TensorView::shape)
        .def_property_readonly("device", &TensorView::device)
        .def_property_readonly("device_id", &TensorView::deviceId)
        .def_property_readonly("nbytes", &TensorView::nbytes)
        .def_property_readonly("data_ptr", [](const TensorView &v) { return reinterpret_cast<std::uintptr_t>(v.data()); });

    // ---- TensorInfo ----
    py::class_<TensorInfo>(m, "TensorInfo")
        .def_readonly("name", &TensorInfo::name)
        .def_readonly("is_input", &TensorInfo::isInput)
        .def_readonly("dtype", &TensorInfo::dtype)
        .def_readonly("shape", &TensorInfo::shape)
        .def("__repr__", [](const TensorInfo &i) {
            return "TensorInfo(name='" + i.name + "', input=" + (i.isInput ? "True" : "False") + ", shape=" + i.shape.toString() + ")";
        });

    // ---- build options ----
    py::class_<ProfileShape>(m, "ProfileShape")
        .def(py::init<>())
        .def_readwrite("input_name", &ProfileShape::inputName)
        .def_readwrite("min", &ProfileShape::min)
        .def_readwrite("opt", &ProfileShape::opt)
        .def_readwrite("max", &ProfileShape::max);
    py::class_<OptimizationProfile>(m, "OptimizationProfile").def(py::init<>()).def_readwrite("inputs", &OptimizationProfile::inputs);
    py::class_<BuildOptions>(m, "BuildOptions")
        .def(py::init<>())
        .def_readwrite("precision", &BuildOptions::precision)
        .def_readwrite("profiles", &BuildOptions::profiles)
        .def_readwrite("device_index", &BuildOptions::deviceIndex)
        .def_readwrite("dla_core", &BuildOptions::dlaCore)
        .def_readwrite("workspace_bytes", &BuildOptions::workspaceBytes)
        .def_readwrite("strongly_typed", &BuildOptions::stronglyTyped)
        .def_readwrite("version_compatible", &BuildOptions::versionCompatible)
        .def_readwrite("hardware_compatible", &BuildOptions::hardwareCompatible)
        .def_readwrite("engine_cache_dir", &BuildOptions::engineCacheDir)
        .def_readwrite("timing_cache_path", &BuildOptions::timingCachePath)
        .def_readwrite("plugin_libraries", &BuildOptions::pluginLibraries);
    py::class_<EngineOptions>(m, "EngineOptions")
        .def(py::init<>())
        .def_readwrite("device_index", &EngineOptions::deviceIndex)
        .def_readwrite("plugin_libraries", &EngineOptions::pluginLibraries);

    // ---- EngineBuilder ----
    py::class_<EngineBuilder>(m, "EngineBuilder")
        .def(py::init<>())
        .def(
            "build_from_onnx_file",
            [](EngineBuilder &b, const std::string &path, const BuildOptions &opts) {
                std::optional<Result<std::vector<std::byte>>> r;
                {
                    py::gil_scoped_release rel;
                    r.emplace(b.buildFromOnnxFile(path, opts));
                }
                auto bytes = unwrap(std::move(*r));
                return py::bytes(reinterpret_cast<const char *>(bytes.data()), bytes.size());
            },
            py::arg("onnx_path"), py::arg("options"))
        .def(
            "build_or_load",
            [](EngineBuilder &b, const std::string &path, const BuildOptions &opts) {
                std::optional<Result<std::string>> r;
                {
                    py::gil_scoped_release rel;
                    r.emplace(b.buildOrLoad(path, opts));
                }
                return unwrap(std::move(*r));
            },
            py::arg("onnx_path"), py::arg("options"))
        .def(
            "build_and_load",
            [](EngineBuilder &b, const std::string &path, const BuildOptions &opts, const EngineOptions &eopts) {
                std::optional<Result<Engine>> r;
                {
                    py::gil_scoped_release rel;
                    r.emplace(b.buildAndLoad(path, opts, eopts));
                }
                return unwrap(std::move(*r));
            },
            py::arg("onnx_path"), py::arg("options"), py::arg("engine_options") = EngineOptions{});

    // ---- Engine ----
    py::class_<Engine>(m, "Engine")
        .def_static(
            "load_from_file",
            [](const std::string &path, const EngineOptions &opts) {
                std::optional<Result<Engine>> r;
                {
                    py::gil_scoped_release rel;
                    r.emplace(Engine::loadFromFile(path, opts));
                }
                return unwrap(std::move(*r));
            },
            py::arg("engine_path"), py::arg("options") = EngineOptions{})
        .def_static(
            "load_from_memory",
            [](py::bytes data, const EngineOptions &opts) {
                char *buf = nullptr;
                ssize_t len = 0;
                PYBIND11_BYTES_AS_STRING_AND_SIZE(data.ptr(), &buf, &len);
                std::span<const std::byte> span(reinterpret_cast<const std::byte *>(buf), static_cast<std::size_t>(len));
                std::optional<Result<Engine>> r;
                {
                    py::gil_scoped_release rel;
                    r.emplace(Engine::loadFromMemory(span, opts));
                }
                return unwrap(std::move(*r));
            },
            py::arg("engine_data"), py::arg("options") = EngineOptions{})
        .def("tensors", &Engine::tensors)
        .def("input_names", &Engine::inputNames)
        .def("output_names", &Engine::outputNames)
        .def("nb_optimization_profiles", &Engine::nbOptimizationProfiles)
        .def(
            "tensor_shape", [](const Engine &e, const std::string &n) { return unwrap(e.tensorShape(n)); }, py::arg("name"))
        .def(
            "tensor_dtype", [](const Engine &e, const std::string &n) { return unwrap(e.tensorDType(n)); }, py::arg("name"))
        .def(
            "enqueue",
            [](Engine &e, const py::dict &inputs, const py::dict &outputs, const Stream &s, int profile) {
                auto in = toViewMap(inputs);
                auto out = toViewMap(outputs);
                Status st;
                {
                    py::gil_scoped_release rel;
                    st = e.enqueue(in, out, s, profile);
                }
                check(st);
            },
            py::arg("inputs"), py::arg("outputs"), py::arg("stream"), py::arg("profile_index") = 0)
        .def(
            "infer",
            [](Engine &e, const py::dict &inputs, const Stream &s, int profile) {
                auto in = toViewMap(inputs);
                std::optional<Result<std::unordered_map<std::string, Tensor>>> r;
                {
                    py::gil_scoped_release rel;
                    r.emplace(e.infer(in, s, profile));
                }
                return tensorMapToDict(unwrap(std::move(*r)));
            },
            py::arg("inputs"), py::arg("stream"), py::arg("profile_index") = 0)
        .def(
            "infer_single",
            [](Engine &e, const py::dict &inputs, const Stream &s, int profile) {
                auto in = toViewMap(inputs);
                std::optional<Result<Tensor>> r;
                {
                    py::gil_scoped_release rel;
                    r.emplace(e.inferSingle(in, s, profile));
                }
                return unwrap(std::move(*r));
            },
            py::arg("inputs"), py::arg("stream"), py::arg("profile_index") = 0)
        .def(
            "output_shapes",
            [](Engine &e, const py::dict &inputs, int profile) {
                auto in = toViewMap(inputs);
                return unwrap(e.outputShapes(in, profile));
            },
            py::arg("inputs"), py::arg("profile_index") = 0);

    // ---- EnginePool + Lease ----
    py::class_<EnginePool::Lease>(m, "Lease")
        .def_property_readonly("valid", &EnginePool::Lease::valid)
        .def_property_readonly("profile_index", &EnginePool::Lease::profileIndex)
        .def(
            "enqueue",
            [](EnginePool::Lease &l, const py::dict &inputs, const py::dict &outputs, const Stream &s) {
                auto in = toViewMap(inputs);
                auto out = toViewMap(outputs);
                Status st;
                {
                    py::gil_scoped_release rel;
                    st = l.enqueue(in, out, s);
                }
                check(st);
            },
            py::arg("inputs"), py::arg("outputs"), py::arg("stream"))
        .def(
            "infer",
            [](EnginePool::Lease &l, const py::dict &inputs, const Stream &s) {
                auto in = toViewMap(inputs);
                std::optional<Result<std::unordered_map<std::string, Tensor>>> r;
                {
                    py::gil_scoped_release rel;
                    r.emplace(l.infer(in, s));
                }
                return tensorMapToDict(unwrap(std::move(*r)));
            },
            py::arg("inputs"), py::arg("stream"))
        .def(
            "infer_single",
            [](EnginePool::Lease &l, const py::dict &inputs, const Stream &s) {
                auto in = toViewMap(inputs);
                std::optional<Result<Tensor>> r;
                {
                    py::gil_scoped_release rel;
                    r.emplace(l.inferSingle(in, s));
                }
                return unwrap(std::move(*r));
            },
            py::arg("inputs"), py::arg("stream"));

    py::class_<EnginePool>(m, "EnginePool")
        .def_static(
            "create",
            [](const std::string &path, int contexts, const EngineOptions &opts) {
                std::optional<Result<EnginePool>> r;
                {
                    py::gil_scoped_release rel;
                    r.emplace(EnginePool::create(path, contexts, opts));
                }
                return unwrap(std::move(*r));
            },
            py::arg("engine_path"), py::arg("contexts"), py::arg("options") = EngineOptions{})
        .def(
            "acquire",
            [](EnginePool &p) {
                std::optional<Result<EnginePool::Lease>> r;
                {
                    py::gil_scoped_release rel;
                    r.emplace(p.acquire());
                }
                return unwrap(std::move(*r));
            },
            "Block until a context is free, then return a Lease.")
        .def("try_acquire",
             [](EnginePool &p) -> py::object {
                 auto lease = p.tryAcquire();
                 if (!lease) {
                     return py::none();
                 }
                 return py::cast(std::move(*lease));
             })
        .def("size", &EnginePool::size)
        .def("tensors", &EnginePool::tensors)
        .def("input_names", &EnginePool::inputNames)
        .def("output_names", &EnginePool::outputNames);

    // ---- version ----
    m.def("version_string", &versionString);
    m.def("library_version", []() {
        auto v = libraryVersion();
        return py::make_tuple(v.major, v.minor, v.patch);
    });
    m.def("tensorrt_build_version", []() { return py::make_tuple(tensorrtBuildMajor(), tensorrtBuildMinor()); });

#ifdef TRT_CPP_API_PYTHON_WITH_PREPROC
    auto pre = m.def_submodule("preproc", "Fused GPU preprocessing (letterbox-resize -> normalize -> NCHW).");
    py::class_<preproc::PreprocSpec>(pre, "PreprocSpec")
        .def(py::init<>())
        .def_readwrite("mean", &preproc::PreprocSpec::mean)
        .def_readwrite("scale", &preproc::PreprocSpec::scale)
        .def_readwrite("swap_rb", &preproc::PreprocSpec::swapRB)
        .def_readwrite("keep_aspect_ratio_pad", &preproc::PreprocSpec::keepAspectRatioPad)
        .def_readwrite("pad_value", &preproc::PreprocSpec::padValue);
    pre.def(
        "letterbox_to_tensor",
        [](py::handle src, py::handle dst, const preproc::PreprocSpec &spec, const Stream &s) {
            auto srcView = toTensorView(src);
            auto dstView = toTensorView(dst);
            Status st;
            {
                py::gil_scoped_release rel;
                st = preproc::letterboxToTensor(srcView, dstView, spec, s);
            }
            check(st);
        },
        py::arg("src"), py::arg("dst"), py::arg("spec"), py::arg("stream"));
#endif
}
