cmake_minimum_required(VERSION 3.18)
project(tensorrt_cpp_api)

# Use ccache to speed up rebuilds
include(cmake/ccache.cmake)

# Set C++ version and optimization level
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Ofast -DNDEBUG -Wno-deprecated-declarations")

# Adding experimental/filesystem
set(CXX_FILESYSTEM_LIBRARIES stdc++fs ${CXX_FILESYSTEM_LIBRARIES})

# For finding FindTensorRT.cmake
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# Use the correct version of CUDA
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)

# We require CUDA, OpenCV, and TensorRT
find_package(TensorRT REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

add_library(tensorrt_cpp_api SHARED
        src/engine.cpp)

target_include_directories(tensorrt_cpp_api PUBLIC ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS})
target_link_libraries(tensorrt_cpp_api PUBLIC ${CXX_FILESYSTEM_LIBRARIES} ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${TensorRT_LIBRARIES})

add_executable(run_inference_benchmark src/main.cpp)
target_link_libraries(run_inference_benchmark tensorrt_cpp_api)
