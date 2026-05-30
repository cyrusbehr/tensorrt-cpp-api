# Prefer a system GoogleTest; fall back to fetching it (keeps CI/offline-ish hosts
# working without an extra apt dependency).
find_package(GTest QUIET)
if(NOT GTest_FOUND)
    include(FetchContent)
    fetchcontent_declare(googletest GIT_REPOSITORY https://github.com/google/googletest.git GIT_TAG v1.15.2)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE) # don't leak gtest into a consumer's install prefix
    fetchcontent_makeavailable(googletest)
endif()

include(GoogleTest)

# CPU-only unit tests (no GPU required).
add_executable(
    trtcpp_core_tests
    test_dtype.cpp
    test_shape.cpp
    test_status.cpp
    test_tensorview.cpp
    test_logger.cpp
    test_sha256.cpp
    test_engine_cache.cpp
    test_version.cpp
)
target_include_directories(trtcpp_core_tests PRIVATE ${CMAKE_SOURCE_DIR}/src) # internal detail headers (CPU-only)
target_link_libraries(trtcpp_core_tests PRIVATE tensorrt_cpp_api::tensorrt_cpp_api GTest::gtest_main)
gtest_discover_tests(trtcpp_core_tests)

# GPU-required integration tests, labeled "gpu" so CI can exclude them (ctest -LE gpu).
add_executable(trtcpp_cuda_tests test_cuda.cpp test_device_tensor.cpp test_buffers.cpp)
target_include_directories(trtcpp_cuda_tests PRIVATE ${CMAKE_SOURCE_DIR}/src) # internal detail headers
target_link_libraries(trtcpp_cuda_tests PRIVATE tensorrt_cpp_api::tensorrt_cpp_api GTest::gtest_main)
gtest_discover_tests(trtcpp_cuda_tests PROPERTIES LABELS "gpu")

# White-box tests of the internal TensorRT glue (logger bridge, plugin loader). Need the
# TRT runtime + headers and src/ on the include path; labeled "gpu".
if(TARGET TensorRT::TensorRT)
    add_executable(
        trtcpp_trt_tests test_trt_logger.cpp test_trt_plugin.cpp test_engine_builder.cpp test_engine.cpp test_engine_pool.cpp
                         test_calibrator.cpp
    )
    target_include_directories(trtcpp_trt_tests PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_compile_definitions(trtcpp_trt_tests PRIVATE TRTCPP_TEST_MODEL_DIR="${CMAKE_SOURCE_DIR}/tests/models")
    target_link_libraries(trtcpp_trt_tests PRIVATE tensorrt_cpp_api::tensorrt_cpp_api TensorRT::TensorRT GTest::gtest_main)
    gtest_discover_tests(trtcpp_trt_tests PROPERTIES LABELS "gpu")
endif()

# Optional OpenCV interop tests (only when WITH_OPENCV; need a working OpenCV-CUDA + GPU).
if(TRT_CPP_API_WITH_OPENCV)
    add_executable(trtcpp_opencv_tests test_opencv_interop.cpp)
    target_link_libraries(trtcpp_opencv_tests PRIVATE tensorrt_cpp_api::tensorrt_cpp_api ${OpenCV_LIBS} GTest::gtest_main)
    gtest_discover_tests(trtcpp_opencv_tests PROPERTIES LABELS "gpu")
endif()

# Preprocessing kernel tests (need a GPU; no TRT).
if(TARGET tensorrt_cpp_api::preproc)
    add_executable(trtcpp_preproc_tests test_preproc.cpp)
    target_link_libraries(trtcpp_preproc_tests PRIVATE tensorrt_cpp_api::preproc GTest::gtest_main)
    gtest_discover_tests(trtcpp_preproc_tests PROPERTIES LABELS "gpu")
endif()
