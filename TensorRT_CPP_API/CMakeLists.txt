set(Target TensorRT_CPP_API)

add_library(${Target} SHARED 
    ${CMAKE_CURRENT_LIST_DIR}/include/engine.h 
    ${CMAKE_CURRENT_LIST_DIR}/src/engine.cpp 
)

target_include_directories(${Target} 
PUBLIC 
    ${OpenCV_INCLUDE_DIRS} 
    ${CUDA_INCLUDE_DIRS} 
    ${TensorRT_INCLUDE_DIRS} 
    ${CMAKE_CURRENT_LIST_DIR}/include 
)

target_link_libraries(${Target} 
PUBLIC 
    ${OpenCV_LIBS} 
    ${CUDA_LIBRARIES} 
    ${CMAKE_THREAD_LIBS_INIT} 
    ${TensorRT_LIBRARIES} 
)

target_compile_definitions(${Target} PRIVATE TENSORRT_CPP_API_EXPORT)

set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/install)

install(TARGETS ${Target}
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
        ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)

install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/include DESTINATION ${CMAKE_INSTALL_PREFIX})