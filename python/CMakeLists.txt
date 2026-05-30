# Python bindings: the trtcpp._core extension. Built two ways:
#   * in-tree, with -DTRT_CPP_API_BUILD_PYTHON=ON (added by the root CMakeLists), for dev/test;
#   * standalone, by scikit-build-core from the top-level pyproject.toml (the wheel path), which
#     configures the whole project with BUILD_PYTHON=ON.
# Either way the core library targets (tensorrt_cpp_api[, ::preproc]) already exist in this
# build tree, so the extension just links them. The static core is built position-independent
# (set on its target) so it links cleanly into this module .so.

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_core MODULE src/bindings.cpp)
target_compile_features(_core PRIVATE cxx_std_20)
target_link_libraries(_core PRIVATE tensorrt_cpp_api::tensorrt_cpp_api CUDA::cudart)

if(TARGET tensorrt_cpp_api::preproc)
    target_link_libraries(_core PRIVATE tensorrt_cpp_api::preproc)
    target_compile_definitions(_core PRIVATE TRT_CPP_API_PYTHON_WITH_PREPROC)
endif()

if(DEFINED SKBUILD)
    # Wheel build: install the extension into the package dir inside the wheel.
    install(TARGETS _core DESTINATION trtcpp)
else()
    # In-tree dev build: stage the extension inside the source package so `import trtcpp`
    # (with PYTHONPATH=<repo>/python) resolves `from ._core import ...`. The staged .so is
    # gitignored (*.so).
    add_custom_command(
        TARGET _core POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:_core>
                                        ${CMAKE_CURRENT_SOURCE_DIR}/trtcpp/$<TARGET_FILE_NAME:_core>
        COMMENT "Staging trtcpp._core into the source package for in-tree pytest"
    )

    if(TRT_CPP_API_BUILD_TESTS)
        find_program(PYTEST_EXECUTABLE NAMES pytest)
        if(PYTEST_EXECUTABLE)
            add_test(NAME python_bindings COMMAND ${PYTEST_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests -q)
            set_tests_properties(
                python_bindings PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}" LABELS "gpu;python"
            )
        else()
            message(STATUS "pytest not found; skipping the python_bindings test")
        endif()
    endif()
endif()
