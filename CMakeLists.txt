cmake_minimum_required(VERSION 3.12)
project(my-img)

# ============================================================
# 重要：默认启用 GPU + Flash Attention
# ============================================================
# 所有新项目必须默认启用 GPU 加速和 Flash Attention
# 除非硬件/环境确实不支持
# ============================================================

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
link_directories(${SD_PATH}/build/ggml/src/ggml-cuda)

# ============================================================
# 编译前置条件
# ============================================================
# stable-diffusion.cpp 必须使用以下选项编译以支持 GPU + Flash Attention:
#   cmake . -DSD_CUDA=ON -DSD_FLASH_ATTN=ON -DCMAKE_BUILD_TYPE=Release
# ============================================================

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
    
    # ============================================================
    # 默认启用 GPU + Flash Attention（链接 CUDA 库）
    # ============================================================
    target_link_libraries(${final_name}
        ${SD_PATH}/build/libstable-diffusion.a 
        ${SD_PATH}/build/ggml/src/ggml-cuda/libggml-cuda.a
        ${SD_PATH}/build/ggml/src/libggml.a
        ${SD_PATH}/build/ggml/src/libggml-cpu.a
        ${SD_PATH}/build/ggml/src/libggml-base.a
        gomp
        pthread
        m
        dl
        cuda
        cudart
        cublas
        rt
    )

    # ============================================================
    # CUDA 编译选项（检测 stable-diffusion.cpp 是否启用了 CUDA）
    # ============================================================
    if(EXISTS ${SD_PATH}/build/CMakeCache.txt)
        file(STRINGS ${SD_PATH}/build/CMakeCache.txt SD_CUDA_CFG)
        if(SD_CUDA_CFG MATCHES "SD_CUDA:BOOL=ON")
            add_definitions(-DSD_CUDA)
            message(STATUS "[my-img] GPU support: ON")
        else()
            message(STATUS "[my-img] GPU support: OFF (using CPU)")
        endif()
        
        file(STRINGS ${SD_PATH}/build/CMakeCache.txt SD_FLASH_ATTN_CFG)
        if(SD_FLASH_ATTN_CFG MATCHES "SD_FLASH_ATTN:UNINITIALIZED=ON" OR SD_FLASH_ATTN_CFG MATCHES "SD_FLASH_ATTN:BOOL=ON")
            add_definitions(-DSD_FLASH_ATTENTION)
            message(STATUS "[my-img] Flash Attention: ON")
        else()
            message(STATUS "[my-img] Flash Attention: OFF")
        endif()
    endif()
endforeach()
