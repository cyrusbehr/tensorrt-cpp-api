set(Target TensorRT_CPP_API_Benchmark)

find_package(TensorRT_CPP_API REQUIRED HINTS ${PROJECT_BINARY_DIR}/install/cmake)

include_directories(${TensorRT_CPP_API_INCLUDE_DIRS})
link_directories(${TensorRT_CPP_API_LINK_DIRS})
add_executable(${Target} 
    main.cpp 
)
target_link_libraries(${Target} ${TensorRT_CPP_API_LIBRARIES})