cmake_minimum_required(VERSION 3.18)
# Since CMake3.9, CUDA support can be enabled directly
project(tensorrt_cpp_api LANGUAGES CXX CUDA)

# Use ccache to speed up rebuilds
include(cmake/ccache.cmake)

# Set C++ version and optimization level
set(CMAKE_CXX_STANDARD 11)
if (UNIX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Ofast -DNDEBUG -Wno-deprecated-declarations")
endif()

# When using MSVC, set the encoding to UTF-8.
ADD_COMPILE_OPTIONS("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
ADD_COMPILE_OPTIONS("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")

# For finding FindTensorRT.cmake
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

# TODO: Specify the path to TensorRT root dir
if (NOT TensorRT_DIR)
    set(TensorRT_DIR E:/3rdparty/TensorRT-8.6.1.6)
endif()

# TODO: Specify the path to OpenCV root dir
if (NOT OpenCV_DIR)
    set(OpenCV_DIR E:/3rdparty/OpenCV/opencv470WithCuda/build)
endif()

# We require OpenCV, and TensorRT
find_package(TensorRT REQUIRED)
find_package(OpenCV REQUIRED)

set(ROOT_DIR ${CMAKE_CURRENT_LIST_DIR})

add_subdirectory(TensorRT_CPP_API)
add_subdirectory(Benchmark)