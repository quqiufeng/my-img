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
# 2. 查找 ONNX Runtime（可选依赖，用于 RemBG 背景抠图）
# ============================================================
set(ONNXRUNTIME_ROOT "")
if(DEFINED ENV{ONNXRUNTIME_ROOT})
    set(ONNXRUNTIME_ROOT $ENV{ONNXRUNTIME_ROOT})
elseif(EXISTS /home/dministrator/onnxruntime-linux-x64-1.20.1)
    set(ONNXRUNTIME_ROOT /home/dministrator/onnxruntime-linux-x64-1.20.1)
endif()

if(ONNXRUNTIME_ROOT)
    find_library(ONNXRUNTIME_LIB NAMES onnxruntime PATHS ${ONNXRUNTIME_ROOT}/lib NO_DEFAULT_PATH)
    find_path(ONNXRUNTIME_INCLUDE_DIR NAMES onnxruntime_cxx_api.h PATHS ${ONNXRUNTIME_ROOT}/include NO_DEFAULT_PATH)
    if(ONNXRUNTIME_LIB AND ONNXRUNTIME_INCLUDE_DIR)
        set(HAS_ONNXRUNTIME ON)
        message(STATUS "[my-img] ONNX Runtime found: ${ONNXRUNTIME_LIB}")
        message(STATUS "[my-img] ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIR}")
    else()
        message(WARNING "[my-img] ONNX Runtime root set but library/headers not found. RemBG will be disabled.")
    endif()
else()
    message(STATUS "[my-img] ONNX Runtime not found. RemBG background removal will be disabled.")
    message(STATUS "[my-img] To enable: download onnxruntime-linux-x64-*.tgz and set ONNXRUNTIME_ROOT")
endif()

# ============================================================
# 3. 链接已有的 stable-diffusion.cpp 编译产物
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
# 第三方库路径
set(THIRDPARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty")

# 特殊处理 sd-engine：编译为静态库
set(SD_ENGINE_SRC 
    "${CMAKE_CURRENT_SOURCE_DIR}/src/sd-engine/core/node.cpp"
)
if(EXISTS ${SD_ENGINE_SRC})
    add_library(sd-engine STATIC 
        "${CMAKE_CURRENT_SOURCE_DIR}/src/sd-engine/core/node.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/sd-engine/core/workflow.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/sd-engine/core/cache.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/sd-engine/core/executor.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/sd-engine/core/init.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/sd-engine/core/workflow_builder.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/sd-engine/nodes/test_nodes.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/sd-engine/nodes/core_nodes.cpp"
    )
    target_include_directories(sd-engine PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/src/sd-engine
        ${THIRDPARTY_DIR}
        ${SD_PATH}
        ${SD_PATH}/include
        ${SD_PATH}/thirdparty
        ${SD_PATH}/ggml/include
        ${SD_PATH}/src
    )
    if(HAS_ONNXRUNTIME)
        target_compile_definitions(sd-engine PRIVATE HAS_ONNXRUNTIME=1)
        target_include_directories(sd-engine PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
        target_link_libraries(sd-engine PRIVATE ${ONNXRUNTIME_LIB})
    endif()
    message(STATUS "[my-img] Created library: sd-engine")
endif()

# 收集所有源文件
file(GLOB_RECURSE SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

# 其他目录编译为可执行文件
foreach(src ${SRC_FILES})
    get_filename_component(name ${src} NAME_WE)
    get_filename_component(dir ${src} PATH)
    string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}/src/" "" rel_dir ${dir})
    
    # 跳过 sd-engine 库中的源文件（避免被 GLOB_RECURSE 当成可执行文件）
    # 但保留 sd-engine/tools 目录下的可执行文件
    if(rel_dir MATCHES "^sd-engine/core" OR rel_dir MATCHES "^sd-engine/nodes")
        continue()
    endif()
    
    string(REPLACE "/" "-" dir ${rel_dir})
    
    if(dir STREQUAL "")
        set(final_name ${name})
    else()
        set(final_name ${dir})
    endif()
    
    # 特殊处理 sd-engine-tools -> sd-workflow
    if(final_name STREQUAL "sd-engine-tools")
        set(final_name "sd-workflow")
    endif()

    add_executable(${final_name} ${src})
    
    # 链接 sd-engine 库（必须在 SD_LIB 之前，因为 sd-engine 依赖 SD_LIB）
    if(TARGET sd-engine)
        target_link_libraries(${final_name} PRIVATE sd-engine)
    endif()
    
    # 链接 stable-diffusion 和 ggml（必须在 sd-engine 之后）
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
        ${SD_PATH}/include
        ${SD_PATH}/thirdparty
        ${SD_PATH}/ggml/include
        ${SD_PATH}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/sd-engine
        ${THIRDPARTY_DIR}
    )
    if(HAS_ONNXRUNTIME)
        target_compile_definitions(${final_name} PRIVATE HAS_ONNXRUNTIME=1)
        target_include_directories(${final_name} PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
        target_link_libraries(${final_name} PRIVATE ${ONNXRUNTIME_LIB})
    endif()

    message(STATUS "[my-img] Created target: ${final_name}")
endforeach()

# ============================================================
# 4. 单元测试
# ============================================================
enable_testing()

# Catch2 amalgamated 源文件
set(CATCH2_SRC "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/catch2/catch_amalgamated.cpp")
set(CATCH2_INC "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/catch2")

# 收集测试源文件
file(GLOB TEST_SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp")

if(EXISTS ${CATCH2_SRC} AND TEST_SRC_FILES)
    add_executable(sd-engine-tests ${TEST_SRC_FILES} ${CATCH2_SRC})
    target_include_directories(sd-engine-tests PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/src/sd-engine
        ${THIRDPARTY_DIR}
        ${SD_PATH}
        ${SD_PATH}/include
        ${SD_PATH}/thirdparty
        ${SD_PATH}/ggml/include
        ${SD_PATH}/src
        ${CATCH2_INC}
    )
    if(HAS_ONNXRUNTIME)
        target_compile_definitions(sd-engine-tests PRIVATE HAS_ONNXRUNTIME=1)
        target_include_directories(sd-engine-tests PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
        target_link_libraries(sd-engine-tests PRIVATE ${ONNXRUNTIME_LIB})
    endif()
    if(TARGET sd-engine)
        target_link_libraries(sd-engine-tests PRIVATE sd-engine)
    endif()
    target_link_libraries(sd-engine-tests PRIVATE ${SD_LIB} ${GGML_CUDA_LIB} ${GGML_CPU_LIB} ${GGML_BASE_LIB} gomp pthread m)
    if(CUDA_PATH)
        target_link_libraries(sd-engine-tests PRIVATE 
            ${CUDA_PATH}/lib/x86_64-linux-gnu/libcudart.so
            ${CUDA_PATH}/lib/x86_64-linux-gnu/libcublas.so
            ${CUDA_PATH}/lib/x86_64-linux-gnu/libcublasLt.so
            /usr/lib/x86_64-linux-gnu/libcuda.so
        )
    endif()
    add_test(NAME sd-engine-tests COMMAND sd-engine-tests)
    message(STATUS "[my-img] Created target: sd-engine-tests")
else()
    message(STATUS "[my-img] Tests disabled: Catch2 or test sources not found")
endif()

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
