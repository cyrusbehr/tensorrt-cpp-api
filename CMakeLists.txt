cmake_minimum_required(VERSION 3.22)
project(tensorrt_cpp_api VERSION 7.0.0 LANGUAGES CXX)

# v7 rewrite. The build system grows by phase (see docs/design/PHASE_D_DESIGN.md);
# this is the E1 stage: a dependency-free core-types library + tests. CUDA/TensorRT,
# the preprocessing sublib, OpenCV interop, install/export, and Python bindings are
# wired up in later E phases. All optional components default OFF.

option(TRT_CPP_API_BUILD_TESTS    "Build unit/integration tests"                OFF)
option(TRT_CPP_API_BUILD_EXAMPLES "Build reference examples"                    OFF)
option(TRT_CPP_API_BUILD_PREPROC  "Build the preprocessing sublibrary (E11)"    OFF)
option(TRT_CPP_API_BUILD_PYTHON   "Build the pybind11 Python bindings (E15)"    OFF)
option(TRT_CPP_API_WITH_OPENCV    "Build the optional OpenCV interop (E12/E13)" OFF)
option(TRT_CPP_API_WITH_SPDLOG    "Build the optional spdlog logger adapter"    OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Modern ccache hook (replaces v6's generator-specific RULE_LAUNCH_COMPILE).
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()

add_library(tensorrt_cpp_api)
add_library(tensorrt_cpp_api::tensorrt_cpp_api ALIAS tensorrt_cpp_api)
target_sources(tensorrt_cpp_api PRIVATE
    src/core.cpp
    src/logger.cpp)
target_include_directories(tensorrt_cpp_api PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_compile_features(tensorrt_cpp_api PUBLIC cxx_std_20)
set_target_properties(tensorrt_cpp_api PROPERTIES
    CXX_EXTENSIONS OFF
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR})
if(NOT MSVC)
    target_compile_options(tensorrt_cpp_api PRIVATE -Wall -Wextra)
endif()

# Optional spdlog logger adapter (E2). PUBLIC so makeSpdlogLogger() is declared and
# spdlog is linked for consumers of a spdlog-enabled install.
if(TRT_CPP_API_WITH_SPDLOG)
    find_package(spdlog REQUIRED)
    target_link_libraries(tensorrt_cpp_api PUBLIC spdlog::spdlog)
    target_compile_definitions(tensorrt_cpp_api PUBLIC TRT_CPP_API_WITH_SPDLOG)
endif()

if(TRT_CPP_API_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
