# FindTensorRT.cmake -- locate a TensorRT install (NVIDIA apt repo OR tarball) and expose
# the imported target TensorRT::TensorRT (nvinfer + nvonnxparser + headers). Supports
# TensorRT 10.0 through 11.x and errors clearly otherwise. Relocatable: it bakes no
# build-tree paths, so it can be installed alongside the package config (Phase E14/H).
#
# Hints: set -DTensorRT_DIR=<tarball root> (or the env var) to point at a tarball; on a
# host with libnvinfer-dev from the NVIDIA apt repo no hint is needed.

set(_trt_hints)
if(TensorRT_DIR)
    list(APPEND _trt_hints "${TensorRT_DIR}")
endif()
if(DEFINED ENV{TensorRT_DIR})
    list(APPEND _trt_hints "$ENV{TensorRT_DIR}")
endif()

find_path(TensorRT_INCLUDE_DIR
    NAMES NvInfer.h
    HINTS ${_trt_hints}
    PATH_SUFFIXES include
    PATHS /usr/include/x86_64-linux-gnu /usr/include /usr/local/include /usr/local/tensorrt/include)

find_library(TensorRT_nvinfer_LIBRARY
    NAMES nvinfer
    HINTS ${_trt_hints}
    PATH_SUFFIXES lib lib64 targets/x86_64-linux/lib
    PATHS /usr/lib/x86_64-linux-gnu /usr/lib /usr/local/lib)

find_library(TensorRT_nvonnxparser_LIBRARY
    NAMES nvonnxparser
    HINTS ${_trt_hints}
    PATH_SUFFIXES lib lib64 targets/x86_64-linux/lib
    PATHS /usr/lib/x86_64-linux-gnu /usr/lib /usr/local/lib)

if(TensorRT_INCLUDE_DIR AND EXISTS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h")
    file(STRINGS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h" _trt_ver_lines REGEX "#define NV_TENSORRT_(MAJOR|MINOR|PATCH) ")
    string(REGEX REPLACE ".*NV_TENSORRT_MAJOR ([0-9]+).*" "\\1" TensorRT_VERSION_MAJOR "${_trt_ver_lines}")
    string(REGEX REPLACE ".*NV_TENSORRT_MINOR ([0-9]+).*" "\\1" TensorRT_VERSION_MINOR "${_trt_ver_lines}")
    string(REGEX REPLACE ".*NV_TENSORRT_PATCH ([0-9]+).*" "\\1" TensorRT_VERSION_PATCH "${_trt_ver_lines}")
    set(TensorRT_VERSION "${TensorRT_VERSION_MAJOR}.${TensorRT_VERSION_MINOR}.${TensorRT_VERSION_PATCH}")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TensorRT
    REQUIRED_VARS TensorRT_nvinfer_LIBRARY TensorRT_nvonnxparser_LIBRARY TensorRT_INCLUDE_DIR
    VERSION_VAR TensorRT_VERSION)

if(TensorRT_FOUND)
    if(TensorRT_VERSION VERSION_LESS "10.0" OR NOT TensorRT_VERSION VERSION_LESS "12.0")
        message(FATAL_ERROR
            "tensorrt_cpp_api requires TensorRT 10.0 - 11.x, but found ${TensorRT_VERSION} at "
            "${TensorRT_INCLUDE_DIR}.\n"
            "  Point -DTensorRT_DIR=<root> at a supported tarball, or install libnvinfer-dev from "
            "the NVIDIA apt repo (scripts/install_deps.sh).")
    endif()

    if(NOT TARGET TensorRT::nvonnxparser)
        add_library(TensorRT::nvonnxparser UNKNOWN IMPORTED)
        set_target_properties(TensorRT::nvonnxparser PROPERTIES IMPORTED_LOCATION "${TensorRT_nvonnxparser_LIBRARY}")
    endif()
    if(NOT TARGET TensorRT::TensorRT)
        add_library(TensorRT::TensorRT UNKNOWN IMPORTED)
        set_target_properties(TensorRT::TensorRT PROPERTIES
            IMPORTED_LOCATION "${TensorRT_nvinfer_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}"
            INTERFACE_LINK_LIBRARIES TensorRT::nvonnxparser)
    endif()
    set(TensorRT_LIBRARIES ${TensorRT_nvinfer_LIBRARY} ${TensorRT_nvonnxparser_LIBRARY})
    set(TensorRT_INCLUDE_DIRS ${TensorRT_INCLUDE_DIR})
endif()

mark_as_advanced(TensorRT_INCLUDE_DIR TensorRT_nvinfer_LIBRARY TensorRT_nvonnxparser_LIBRARY)
