cmake_minimum_required(VERSION 3.22)
project(trtcpp_consumer LANGUAGES CXX)

# Smoke test that an installed tensorrt_cpp_api is consumable via find_package: configure with
# -DCMAKE_PREFIX_PATH=<install-prefix>.
find_package(tensorrt_cpp_api REQUIRED)

add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE tensorrt_cpp_api::tensorrt_cpp_api)
