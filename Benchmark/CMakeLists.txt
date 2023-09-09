set(Target TensorRT_CPP_API_Benchmark)

add_executable(${Target} 
    main.cpp 
)
target_link_libraries(${Target} TensorRT_CPP_API)