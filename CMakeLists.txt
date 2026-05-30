cmake_minimum_required(VERSION 3.22)
project(tensorrt_cpp_api VERSION 7.0.0 LANGUAGES CXX)

# The engine core depends only on CUDA + TensorRT. The preprocessing sublibrary, OpenCV interop,
# Python bindings, tests, and examples are independent, opt-in components; all default OFF except
# the preprocessing sublibrary.

option(TRT_CPP_API_BUILD_TESTS "Build unit/integration tests" OFF)
option(TRT_CPP_API_BUILD_EXAMPLES "Build reference examples" OFF)
option(TRT_CPP_API_BUILD_PREPROC "Build the fused preprocessing sublibrary" ON)
option(TRT_CPP_API_BUILD_PYTHON "Build the pybind11 Python bindings" OFF)
option(TRT_CPP_API_WITH_OPENCV "Build the optional OpenCV interop" OFF)
option(TRT_CPP_API_WITH_SPDLOG "Build the optional spdlog logger adapter" OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Enable CTest at the top level so tests registered in subdirectories (tests/, python/) are
# discoverable. Called early -- before add_subdirectory(python) registers the binding test.
if(TRT_CPP_API_BUILD_TESTS)
    enable_testing()
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Modern ccache hook (replaces v6's generator-specific RULE_LAUNCH_COMPILE).
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()

# CUDA runtime: a hard dependency of the engine core (streams, allocation, Tensor cross
# the public API). Plain C++ against the CUDA runtime API -- nvcc is only needed later for
# the preprocessing kernels (.cu), at which point CUDA is enabled as a language.
find_package(CUDAToolkit REQUIRED)

# TensorRT (apt or tarball). Linked PRIVATE: nvinfer1 types never appear in public headers
# (PImpl/forward-decl), so consumers need TRT at runtime but not at compile time. Pass
# -DTensorRT_DIR=<root> for a tarball install.
find_package(TensorRT REQUIRED)

# Threads: the logger and EnginePool use std::mutex/condition_variable. Linked PRIVATE
# (PImpl, so no thread types in public headers); the package config re-resolves it.
find_package(Threads REQUIRED)

# Generate build_config.h recording the library + TensorRT versions, so public headers can
# version-gate (e.g. the legacy calibrator) without including a TensorRT header.
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/build_config.h.in ${CMAKE_CURRENT_BINARY_DIR}/generated/tensorrt_cpp_api/build_config.h
    @ONLY
)

add_library(tensorrt_cpp_api)
add_library(tensorrt_cpp_api::tensorrt_cpp_api ALIAS tensorrt_cpp_api)
target_sources(
    tensorrt_cpp_api
    PRIVATE src/core.cpp
            src/logger.cpp
            src/cuda.cpp
            src/allocator.cpp
            src/device_tensor.cpp
            src/engine_builder.cpp
            src/engine.cpp
            src/engine_pool.cpp
            src/calibrator.cpp
            src/detail/buffers.cpp
            src/detail/calibrator_bridge.cpp
            src/detail/engine_cache.cpp
            src/detail/execution.cpp
            src/detail/sha256.cpp
            src/detail/trt_common.cpp
)
target_include_directories(tensorrt_cpp_api PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(tensorrt_cpp_api PUBLIC CUDA::cudart PRIVATE TensorRT::TensorRT Threads::Threads)
# Build-tree include roots only. The install-tree include dir is supplied once by the
# INCLUDES DESTINATION of install(TARGETS) below (both the public headers and the generated
# build_config.h land under the same <prefix>/include tree), so no $<INSTALL_INTERFACE> here.
target_include_directories(
    tensorrt_cpp_api PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>
)
target_compile_features(tensorrt_cpp_api PUBLIC cxx_std_20)
# POSITION_INDEPENDENT_CODE: the static core is routinely linked into shared objects -- the Python
# extension and downstream shared libraries such as the YOLO sibling repos' libYoloV8_TRT.so -- so
# it must be PIC. (Negligible cost; standard for an embeddable static library.)
set_target_properties(
    tensorrt_cpp_api PROPERTIES CXX_EXTENSIONS OFF POSITION_INDEPENDENT_CODE ON VERSION ${PROJECT_VERSION}
                                SOVERSION ${PROJECT_VERSION_MAJOR}
)
if(NOT MSVC)
    target_compile_options(tensorrt_cpp_api PRIVATE -Wall -Wextra)
endif()

# Optional spdlog logger adapter. PUBLIC so makeSpdlogLogger() is declared and
# spdlog is linked for consumers of a spdlog-enabled install.
if(TRT_CPP_API_WITH_SPDLOG)
    find_package(spdlog REQUIRED)
    target_link_libraries(tensorrt_cpp_api PUBLIC spdlog::spdlog)
    target_compile_definitions(tensorrt_cpp_api PUBLIC TRT_CPP_API_WITH_SPDLOG)
endif()

# Optional OpenCV interop. Requests only the `core` module (which carries cv::Mat
# and cv::cuda::GpuMat) to avoid pulling opencv_dnn and its cuDNN dependency. Strictly
# opt-in; the engine core has no OpenCV in any non-gated header.
if(TRT_CPP_API_WITH_OPENCV)
    find_package(OpenCV REQUIRED COMPONENTS core)
    target_sources(tensorrt_cpp_api PRIVATE src/opencv_interop.cpp)
    target_link_libraries(tensorrt_cpp_api PUBLIC ${OpenCV_LIBS})
    target_include_directories(tensorrt_cpp_api PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_compile_definitions(tensorrt_cpp_api PUBLIC TRT_CPP_API_WITH_OPENCV)
endif()

# Optional preprocessing sublibrary: one fused CUDA (.cu) kernel. Separate target --
# the engine core does NOT depend on it. Enabling it turns on the CUDA language (needs nvcc).
if(TRT_CPP_API_BUILD_PREPROC)
    # Architectures must be set BEFORE enable_language(CUDA): CMake validates the variable
    # during language init, and an unset value on hosts where nvcc can't self-detect a default
    # aborts configuration.
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 75-real 80-real 86-real 89-real 90-virtual)
    endif()
    enable_language(CUDA)
    add_library(tensorrt_cpp_api_preproc src/preproc.cu)
    add_library(tensorrt_cpp_api::preproc ALIAS tensorrt_cpp_api_preproc)
    target_link_libraries(tensorrt_cpp_api_preproc PUBLIC tensorrt_cpp_api::tensorrt_cpp_api CUDA::cudart)
    target_compile_features(tensorrt_cpp_api_preproc PUBLIC cxx_std_20)
    set_target_properties(
        tensorrt_cpp_api_preproc
        PROPERTIES CUDA_STANDARD 20
                   CUDA_STANDARD_REQUIRED ON
                   CXX_EXTENSIONS OFF
                   POSITION_INDEPENDENT_CODE ON # embeddable into shared libs (Python ext, sibling .so)
                   EXPORT_NAME preproc # consumers link tensorrt_cpp_api::preproc (matches the in-tree alias)
                   VERSION ${PROJECT_VERSION}
                   SOVERSION ${PROJECT_VERSION_MAJOR}
    )
endif()

# Python bindings. Added after the core/preproc targets exist so the extension can link
# them. The subdirectory builds trtcpp._core.
if(TRT_CPP_API_BUILD_PYTHON)
    add_subdirectory(python)
endif()

# Reference examples. Normally built standalone against the installed package; this in-tree
# path is a convenience that reuses the just-built targets.
if(TRT_CPP_API_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# ---- install / export (find_package(tensorrt_cpp_api) consumable) ----
# Skipped under scikit-build-core (SKBUILD): a Python wheel must contain only the Python package
# and the compiled extension, not the C++ dev headers / static libs / CMake package config.
if(NOT SKBUILD)
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)
    set(TRTCPP_INSTALL_CMAKEDIR ${CMAKE_INSTALL_LIBDIR}/cmake/tensorrt_cpp_api)

    set(TRTCPP_INSTALL_TARGETS tensorrt_cpp_api)
    if(TARGET tensorrt_cpp_api_preproc)
        list(APPEND TRTCPP_INSTALL_TARGETS tensorrt_cpp_api_preproc)
    endif()

    install(
        TARGETS ${TRTCPP_INSTALL_TARGETS}
        EXPORT tensorrt_cpp_apiTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )

    install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/generated/tensorrt_cpp_api/build_config.h
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/tensorrt_cpp_api
    )
    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindTensorRT.cmake DESTINATION ${TRTCPP_INSTALL_CMAKEDIR})

    install(EXPORT tensorrt_cpp_apiTargets NAMESPACE tensorrt_cpp_api:: DESTINATION ${TRTCPP_INSTALL_CMAKEDIR})

    configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/tensorrt_cpp_api-config.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/tensorrt_cpp_api-config.cmake INSTALL_DESTINATION ${TRTCPP_INSTALL_CMAKEDIR}
    )
    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/tensorrt_cpp_api-config-version.cmake VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
    )
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/tensorrt_cpp_api-config.cmake
                  ${CMAKE_CURRENT_BINARY_DIR}/tensorrt_cpp_api-config-version.cmake DESTINATION ${TRTCPP_INSTALL_CMAKEDIR}
    )
endif() # NOT SKBUILD

if(TRT_CPP_API_BUILD_TESTS)
    add_subdirectory(tests)
endif()
