cmake_minimum_required(VERSION 3.15)
project(my-img)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================
# 1. 读取 .env 配置文件
# ============================================================
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/.env)
    file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/.env ENV_FILE)
    foreach(line ${ENV_FILE})
        if(line MATCHES "^SD_PATH=(.*)$")
            set(SD_PATH_FROM_ENV ${CMAKE_MATCH_1})
        endif()
        if(line MATCHES "^CUDA_PATH=(.*)$")
            set(CUDA_PATH_FROM_ENV ${CMAKE_MATCH_1})
        endif()
    endforeach()
    if(SD_PATH_FROM_ENV AND NOT SD_PATH)
        set(SD_PATH ${SD_PATH_FROM_ENV})
    endif()
    if(CUDA_PATH_FROM_ENV AND NOT CUDA_PATH)
        set(CUDA_PATH ${CUDA_PATH_FROM_ENV})
    endif()
endif()

if(SD_PATH)
    string(REPLACE "~" "$ENV{HOME}" SD_PATH ${SD_PATH})
endif()

message(STATUS "CUDA_PATH: ${CUDA_PATH}")

if(NOT SD_PATH OR NOT EXISTS ${SD_PATH}/CMakeLists.txt)
    message(FATAL_ERROR "SD_PATH 路径无效或未设置。当前路径: ${SD_PATH}")
endif()

message(STATUS "[my-img] Using SD_PATH: ${SD_PATH}")

# ============================================================
# 2. 链接已有的 stable-diffusion.cpp 编译产物
# ============================================================
# 直接链接 ~/stable-diffusion.cpp/bin 中的库，不重新编译

# 设置库路径
set(SD_LIB_PATH "${SD_PATH}/build")
set(SD_BIN_PATH "${SD_PATH}/bin")

# 查找 ggml 库 (优先 build 目录静态库，fallback bin 目录动态库)
find_library(GGML_CUDA_LIB NAMES ggml-cuda PATHS ${SD_LIB_PATH}/ggml/src/ggml-cuda ${SD_BIN_PATH} NO_DEFAULT_PATH)
find_library(GGML_CPU_LIB NAMES ggml-cpu PATHS ${SD_LIB_PATH}/ggml/src ${SD_BIN_PATH} NO_DEFAULT_PATH)
find_library(GGML_BASE_LIB NAMES ggml-base PATHS ${SD_LIB_PATH}/ggml/src ${SD_BIN_PATH} NO_DEFAULT_PATH)
find_library(GGML_LIB NAMES ggml PATHS ${SD_LIB_PATH}/ggml/src ${SD_BIN_PATH} NO_DEFAULT_PATH)

# 查找 stable-diffusion 静态库 (在 build 目录)
find_library(SD_LIB NAMES stable-diffusion PATHS ${SD_LIB_PATH} NO_DEFAULT_PATH)

message(STATUS "[my-img] GGML_CUDA_LIB: ${GGML_CUDA_LIB}")
message(STATUS "[my-img] GGML_CPU_LIB: ${GGML_CPU_LIB}")
message(STATUS "[my-img] GGML_BASE_LIB: ${GGML_BASE_LIB}")
message(STATUS "[my-img] SD_LIB: ${SD_LIB}")

if(NOT SD_LIB)
    message(FATAL_ERROR "未找到预编译的库，请先运行 scripts/build_sd_cpp_*.sh 编译 stable-diffusion.cpp")
endif()

# ============================================================
# 3. 编译你的程序
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
    
    # 链接 stable-diffusion 和 ggml（动态库）
    target_link_libraries(${final_name} PRIVATE ${SD_LIB} ${GGML_CUDA_LIB} ${GGML_CPU_LIB} ${GGML_BASE_LIB} gomp pthread m)
    
    # 链接 CUDA 运行时
    if(CUDA_PATH)
        target_link_libraries(${final_name} PRIVATE 
            ${CUDA_PATH}/lib/x86_64-linux-gnu/libcudart.so
            ${CUDA_PATH}/lib/x86_64-linux-gnu/libcublas.so
            ${CUDA_PATH}/lib/x86_64-linux-gnu/libcublasLt.so
            /usr/lib/x86_64-linux-gnu/libcuda.so
        )
    endif()

    target_include_directories(${final_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    
    target_include_directories(${final_name} PRIVATE 
        ${SD_PATH}
        ${SD_PATH}/thirdparty
        ${SD_PATH}/ggml/include
    )

    message(STATUS "[my-img] Created target: ${final_name}")
endforeach()

# 自动复制二进制到 SD_PATH/bin
if(SD_PATH)
    set(OUTPUT_DIR "${SD_PATH}/bin")
    message(STATUS "[my-img] Copying binaries to ${OUTPUT_DIR}")
    foreach(target ${ALL_TARGETS})
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
                $<TARGET_FILE:${target}>
                ${OUTPUT_DIR}/
        )
    endforeach()
endif()
