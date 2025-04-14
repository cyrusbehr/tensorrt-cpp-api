cmake_minimum_required(VERSION 3.18)
project(tensorrt_cpp_api)

# Use ccache to speed up rebuilds
include(cmake/ccache.cmake)

# Set C++ version and optimization level
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Ofast -DNDEBUG -Wno-deprecated-declarations")

# For finding FindTensorRT.cmake
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

set(TensorRT_INCLUDE_DIRS /usr/include)
set(TENSORRT_LIB_DIR /usr/lib/x86_64-linux-gnu)

find_library(NVINFER_LIBRARY nvinfer HINTS ${TENSORRT_LIB_DIR})
find_library(NVINFER_PLUGIN_LIBRARY nvinfer_plugin HINTS ${TENSORRT_LIB_DIR})
find_library(NVONNXPARSER_LIBRARY nvonnxparser HINTS ${TENSORRT_LIB_DIR})

if (NOT NVINFER_LIBRARY OR NOT NVINFER_PLUGIN_LIBRARY OR NOT NVONNXPARSER_LIBRARY)
    message(FATAL_ERROR "Could not find TensorRT libraries. Make sure TensorRT is installed.")
endif()

include_directories(${TENSORRT_INCLUDE_DIR})

set(TensorRT_LIBRARIES ${NVINFER_LIBRARY} ${NVINFER_PLUGIN_LIBRARY} ${NVONNXPARSER_LIBRARY})

# Use the correct version of CUDA
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(fmt REQUIRED)

add_library(tensorrt_cpp_api SHARED
        src/engine.cpp)

target_include_directories(tensorrt_cpp_api PUBLIC ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS} include include/interfaces)
target_link_libraries(tensorrt_cpp_api PUBLIC ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${TensorRT_LIBRARIES} fmt::fmt)

add_executable(run_inference_benchmark src/main.cpp)
target_link_libraries(run_inference_benchmark tensorrt_cpp_api fmt::fmt)