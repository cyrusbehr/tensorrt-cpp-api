set(Target tensorrt_cpp_api_benchmark)

add_executable(${Target} 
    main.cpp 
)

target_link_libraries(${Target} 
PRIVATE
    tensorrt_cpp_api
)