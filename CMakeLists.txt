cmake_minimum_required(VERSION 3.12)
project(my-img)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT SD_PATH)
    message(FATAL_ERROR "SD_PATH not set. Usage: cmake .. -DSD_PATH=/path/to/stable-diffusion.cpp")
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${SD_PATH}/thirdparty)
include_directories(${SD_PATH}/ggml/include)
include_directories(${SD_PATH}/ggml/src)

add_definitions(-DSTB_IMAGE_IMPLEMENTATION)
add_definitions(-DSTB_IMAGE_WRITE_IMPLEMENTATION)

link_directories(${SD_PATH}/build)

file(GLOB_RECURSE SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

foreach(src ${SRC_FILES})
    get_filename_component(name ${src} NAME_WE)
    get_filename_component(dir ${src} PATH)
    string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}/src/" "" dir ${dir})
    string(REPLACE "/" "-" dir ${dir})
    if(dir STREQUAL "")
        set(final_name ${name})
    else()
        set(final_name ${dir})
    endif()
    add_executable(${final_name} ${src})
    target_link_libraries(${final_name}
        ${SD_PATH}/build/libstable-diffusion.a 
        ${SD_PATH}/build/ggml/src/libggml-cpu.a
        ${SD_PATH}/build/ggml/src/libggml-base.a
        gomp
        pthread
        m
    )
endforeach()
