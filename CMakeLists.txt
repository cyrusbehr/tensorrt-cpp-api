cmake_minimum_required(VERSION 3.18)
project(tensorrt_cpp_api)

# Set C++ version and optimization level
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Ofast -DNDEBUG")

# For finding FindTensorRT.cmake
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# TODO: Specify the path to TensorRT root dir
set(TensorRT_DIR /home/cyrus/work/libs/TensorRT-8.0.3.4/)

# Use the correct version of CUDA
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)

# We require CUDA, OpenCV, and TensorRT
find_package(TensorRT REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(tensorrt_cpp_api
        src/main.cpp
        src/engine.cpp)

target_include_directories(tensorrt_cpp_api PUBLIC ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS}
        ${TensorRT_DIR}/samples/common)
target_link_libraries(tensorrt_cpp_api PUBLIC ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${TensorRT_LIBRARIES})
