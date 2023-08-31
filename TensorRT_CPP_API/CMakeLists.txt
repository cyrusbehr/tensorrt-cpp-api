set(Target TensorRT_CPP_API)

set(Headers
    ${CMAKE_CURRENT_LIST_DIR}/include/engine.h 
)

set(Sources 
    ${CMAKE_CURRENT_LIST_DIR}/src/engine.cpp 
)

add_library(${Target} SHARED ${Headers} ${Sources})

target_link_libraries(${Target} 
PUBLIC 
    ${OpenCV_LIBS} 
    ${CUDA_LIBRARIES} 
    ${CMAKE_THREAD_LIBS_INIT} 
    ${TensorRT_LIBRARIES} 
)

target_include_directories(${Target} 
PUBLIC 
    ${OpenCV_INCLUDE_DIRS} 
    ${CUDA_INCLUDE_DIRS} 
    ${TensorRT_INCLUDE_DIRS} 
PRIVATE 
    ${CMAKE_CURRENT_LIST_DIR}/include 
)

target_compile_definitions(${Target} PRIVATE TENSORRT_CPP_API_EXPORT)

set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR}/install)

install(TARGETS ${Target}
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
        ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)

install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/include DESTINATION ${CMAKE_INSTALL_PREFIX})

configure_file(${CMAKE_CURRENT_LIST_DIR}/config/${Target}Config.cmake.in ${PROJECT_BINARY_DIR}/cmake/${Target}Config.cmake)
install(FILES ${PROJECT_BINARY_DIR}/cmake/${Target}Config.cmake DESTINATION ${CMAKE_INSTALL_PREFIX}/cmake)