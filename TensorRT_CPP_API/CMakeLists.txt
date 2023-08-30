set(Target tensorrt_cpp_api)

set(Headers
    ${CMAKE_CURRENT_LIST_DIR}/include/engine.h 
)

set(Sources 
    ${CMAKE_CURRENT_LIST_DIR}/src/engine.cpp 
)

add_library(${Target} STATIC "")

target_sources(${Target} 
PRIVATE 
    ${Headers}
    ${Sources}
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

set(CMAKE_INSTALL_PREFIX ${ROOT_DIR}/install)

install(TARGETS ${Target}
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
        ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)

install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/include DESTINATION ${CMAKE_INSTALL_PREFIX})