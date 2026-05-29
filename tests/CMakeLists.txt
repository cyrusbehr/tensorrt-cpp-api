# Prefer a system GoogleTest; fall back to fetching it (keeps CI/offline-ish hosts
# working without an extra apt dependency).
find_package(GTest QUIET)
if(NOT GTest_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.15.2)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

include(GoogleTest)

# CPU-only unit tests (no GPU required).
add_executable(trtcpp_core_tests
    test_dtype.cpp
    test_shape.cpp
    test_status.cpp
    test_tensorview.cpp
    test_logger.cpp)
target_link_libraries(trtcpp_core_tests PRIVATE tensorrt_cpp_api::tensorrt_cpp_api GTest::gtest_main)
gtest_discover_tests(trtcpp_core_tests)

# GPU-required integration tests, labeled "gpu" so CI can exclude them (ctest -LE gpu).
add_executable(trtcpp_cuda_tests
    test_cuda.cpp
    test_device_tensor.cpp
    test_buffers.cpp)
target_include_directories(trtcpp_cuda_tests PRIVATE ${CMAKE_SOURCE_DIR}/src) # internal detail headers
target_link_libraries(trtcpp_cuda_tests PRIVATE tensorrt_cpp_api::tensorrt_cpp_api GTest::gtest_main)
gtest_discover_tests(trtcpp_cuda_tests PROPERTIES LABELS "gpu")

# White-box tests of the internal TensorRT glue (logger bridge, plugin loader). Need the
# TRT runtime + headers and src/ on the include path; labeled "gpu".
if(TARGET TensorRT::TensorRT)
    add_executable(trtcpp_trt_tests
        test_trt_logger.cpp
        test_trt_plugin.cpp)
    target_include_directories(trtcpp_trt_tests PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(trtcpp_trt_tests PRIVATE tensorrt_cpp_api::tensorrt_cpp_api TensorRT::TensorRT GTest::gtest_main)
    gtest_discover_tests(trtcpp_trt_tests PROPERTIES LABELS "gpu")
endif()
