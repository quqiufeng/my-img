cmake_minimum_required(VERSION 3.18)
project(my-img VERSION 0.1.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 启用 LTO 以与 sd.cpp 兼容
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)

# ============================================================
# 1. 查找 libtorch
# ============================================================
# 从环境变量、conda 环境或默认路径查找 libtorch
set(LIBTORCH_ROOT "")
if(DEFINED ENV{LIBTORCH_ROOT})
    set(LIBTORCH_ROOT $ENV{LIBTORCH_ROOT})
elseif(DEFINED ENV{CONDA_PREFIX})
    # Conda 环境中的 torch
    set(CONDA_TORCH_CMAKE "$ENV{CONDA_PREFIX}/lib/python3.10/site-packages/torch/share/cmake")
    if(EXISTS ${CONDA_TORCH_CMAKE}/Torch/TorchConfig.cmake)
        set(LIBTORCH_ROOT "$ENV{CONDA_PREFIX}/lib/python3.10/site-packages/torch")
        message(STATUS "[my-img] Found libtorch in conda environment")
    endif()
elseif(EXISTS /opt/libtorch)
    set(LIBTORCH_ROOT /opt/libtorch)
elseif(EXISTS /usr/local/libtorch)
    set(LIBTORCH_ROOT /usr/local/libtorch)
endif()

# 如果还没找到，尝试从 python torch 包获取路径
if(NOT LIBTORCH_ROOT)
    execute_process(
        COMMAND python3 -c "import torch; print(torch.utils.cmake_prefix_path)"
        OUTPUT_VARIABLE PYTHON_TORCH_CMAKE
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(PYTHON_TORCH_CMAKE AND EXISTS "${PYTHON_TORCH_CMAKE}/Torch/TorchConfig.cmake")
        get_filename_component(LIBTORCH_ROOT "${PYTHON_TORCH_CMAKE}" DIRECTORY)
        get_filename_component(LIBTORCH_ROOT "${LIBTORCH_ROOT}" DIRECTORY)
        message(STATUS "[my-img] Found libtorch via python: ${LIBTORCH_ROOT}")
    endif()
endif()

if(NOT LIBTORCH_ROOT)
    message(FATAL_ERROR "未找到 libtorch。请设置 LIBTORCH_ROOT 环境变量或安装 libtorch。\n"
        "conda 安装: conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia\n"
        "pip 安装: pip install torch --index-url https://download.pytorch.org/whl/cu121")
endif()

message(STATUS "[my-img] LIBTORCH_ROOT: ${LIBTORCH_ROOT}")

list(APPEND CMAKE_PREFIX_PATH ${LIBTORCH_ROOT})
find_package(Torch REQUIRED)
message(STATUS "[my-img] Torch version: ${Torch_VERSION}")

# ============================================================
# 2. 查找 CUDA（如果 libtorch 启用了 CUDA）
# ============================================================
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    message(STATUS "[my-img] CUDA compiler found: ${CMAKE_CUDA_COMPILER}")
else()
    message(STATUS "[my-img] CUDA compiler not found, using CPU only")
endif()

# ============================================================
# 3. 第三方依赖（子目录）
# ============================================================
set(THIRDPARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

# nlohmann/json（头文件-only）
add_subdirectory(${THIRDPARTY_DIR}/json)

# GGML（仅用于 GGUF 解析，不参与推理）
# 配置：关闭所有后端，仅保留基础功能用于文件解析
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_CUDA OFF CACHE BOOL "" FORCE)
set(GGML_METAL OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
set(GGML_OPENCL OFF CACHE BOOL "" FORCE)
add_subdirectory(${THIRDPARTY_DIR}/ggml)

# stb_image（头文件-only，不需要 add_subdirectory）
set(STB_IMAGE_INCLUDE ${THIRDPARTY_DIR}/stb)

# ============================================================
# 4. stable-diffusion.cpp 集成
# ============================================================
set(SDCPP_DIR /opt/stable-diffusion.cpp)
set(SDCPP_BUILD_DIR ${SDCPP_DIR}/build)

# 检查 sd.cpp 是否已编译
if(EXISTS "${SDCPP_BUILD_DIR}/libstable-diffusion.a")
    message(STATUS "[my-img] Found stable-diffusion.cpp at ${SDCPP_DIR}")
    set(SDCPP_FOUND TRUE)
else()
    message(FATAL_ERROR "stable-diffusion.cpp not found at ${SDCPP_BUILD_DIR}/libstable-diffusion.a. Please build sd.cpp first.")
endif()

# 手动指定 sd.cpp 的库路径和依赖
# 注意：链接顺序非常重要！
# 使用链接组来解决循环依赖
set(SDCPP_LINK_START "-Wl,--start-group")
set(SDCPP_LINK_END "-Wl,--end-group")

# sd.cpp 的库路径
set(SDCPP_LINK_LIBRARIES
    ${SDCPP_BUILD_DIR}/libstable-diffusion.a
    ${SDCPP_BUILD_DIR}/ggml/src/libggml.a
    ${SDCPP_BUILD_DIR}/ggml/src/libggml-cpu.a
    ${SDCPP_BUILD_DIR}/ggml/src/libggml-base.a
)

# 如果有 CUDA 支持，添加 CUDA 库
if(EXISTS "${SDCPP_BUILD_DIR}/ggml/src/ggml-cuda/libggml-cuda.a")
    message(STATUS "[my-img] Found GGML CUDA backend")
    list(APPEND SDCPP_LINK_LIBRARIES ${SDCPP_BUILD_DIR}/ggml/src/ggml-cuda/libggml-cuda.a)
    find_package(CUDAToolkit)
    if(CUDAToolkit_FOUND)
        list(APPEND SDCPP_LINK_LIBRARIES CUDA::cudart CUDA::cuda_driver CUDA::cublas)
    endif()
endif()

# 添加 OpenMP 支持
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    list(APPEND SDCPP_LINK_LIBRARIES OpenMP::OpenMP_CXX)
endif()

# ============================================================
# 5. 源文件
# ============================================================
set(MYIMG_SOURCES
    src/main.cpp
    src/adapters/sdcpp_adapter.cpp
    src/utils/image_utils.cpp
    src/utils/image_adjust.cpp
    src/utils/png_metadata.cpp
    src/utils/stb_impl.c
)

# ============================================================
# 5. 创建可执行文件
# ============================================================
add_executable(myimg-cli ${MYIMG_SOURCES})

# 包含目录
target_include_directories(myimg-cli PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/engine
    ${CMAKE_CURRENT_SOURCE_DIR}/src/nodes
    ${CMAKE_CURRENT_SOURCE_DIR}/src/backend
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/src/adapters
    ${STB_IMAGE_INCLUDE}
    ${THIRDPARTY_DIR}/ggml/include
    ${SDCPP_DIR}/include
    ${SDCPP_DIR}/ggml/include
)

# 查找 zlib
find_package(ZLIB REQUIRED)

# 链接库
    target_link_libraries(myimg-cli PRIVATE
        ${TORCH_LIBRARIES}
        nlohmann_json::nlohmann_json
        ZLIB::ZLIB
        ${SDCPP_LINK_START}
        ${SDCPP_LINK_LIBRARIES}
        ${SDCPP_LINK_END}
        ${CMAKE_THREAD_LIBS_INIT}
    )
    if(EXISTS "${SDCPP_BUILD_DIR}/ggml/src/ggml-cuda/libggml-cuda.a")
    else()
    endif()

# ============================================================
# 6. 编译选项
# ============================================================
if(MSVC)
    target_compile_options(myimg-cli PRIVATE /W4 /WX-)
else()
    target_compile_options(myimg-cli PRIVATE 
        -Wall 
        -Wextra 
        -Wno-unused-parameter
        -Wno-unused-variable
        -O3
        -march=native
    )
endif()

# 禁用 LTO 以避免与 sd.cpp 的 LTO 版本冲突
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION FALSE)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set_target_properties(myimg-cli PROPERTIES INTERPROCEDURAL_OPTIMIZATION FALSE)
endif()

# 设置 RPATH，使二进制能找到 libtorch 库
if(UNIX)
    set_target_properties(myimg-cli PROPERTIES
        INSTALL_RPATH "${LIBTORCH_ROOT}/lib"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()

# ============================================================
# 7. 安装
# ============================================================
install(TARGETS myimg-cli DESTINATION bin)

# ============================================================
# 8. 信息输出
# ============================================================
message(STATUS "")
message(STATUS "========================================")
message(STATUS "  my-img Configuration")
message(STATUS "========================================")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  libtorch: ${LIBTORCH_ROOT}")
message(STATUS "  CUDA: ${CMAKE_CUDA_COMPILER}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "========================================")
message(STATUS "")

# ============================================================
# 9. 测试
# ============================================================
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_gguf_loader.cpp")
    add_executable(test_gguf_loader 
        tests/test_gguf_loader.cpp
        src/utils/gguf_loader.cpp
    )
    target_include_directories(test_gguf_loader PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/utils
        ${THIRDPARTY_DIR}/ggml/include
    )
    target_link_libraries(test_gguf_loader PRIVATE
        ${TORCH_LIBRARIES}
        ggml
    )
    set_target_properties(test_gguf_loader PROPERTIES
        INSTALL_RPATH "${LIBTORCH_ROOT}/lib"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_vae.cpp")
    add_executable(test_vae 
        tests/test_vae.cpp
        src/backend/vae.cpp
        src/backend/model.cpp
        src/utils/image_utils.cpp
        src/utils/stb_impl.c
        src/utils/gguf_loader.cpp
        src/utils/safetensors.cpp
    )
    target_include_directories(test_vae PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/backend
        ${CMAKE_CURRENT_SOURCE_DIR}/src/utils
        ${STB_IMAGE_INCLUDE}
        ${THIRDPARTY_DIR}/ggml/include
    )
    target_link_libraries(test_vae PRIVATE
        ${TORCH_LIBRARIES}
        ggml
    )
    set_target_properties(test_vae PROPERTIES
        INSTALL_RPATH "${LIBTORCH_ROOT}/lib"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_hires_fix.cpp")
    add_executable(test_hires_fix
        tests/test_hires_fix.cpp
        src/backend/vae.cpp
        src/backend/model.cpp
        src/utils/image_utils.cpp
        src/utils/stb_impl.c
        src/utils/gguf_loader.cpp
        src/utils/safetensors.cpp
    )
    target_include_directories(test_hires_fix PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/backend
        ${CMAKE_CURRENT_SOURCE_DIR}/src/utils
        ${CMAKE_CURRENT_SOURCE_DIR}/src/nodes
        ${STB_IMAGE_INCLUDE}
        ${THIRDPARTY_DIR}/ggml/include
    )
    target_link_libraries(test_hires_fix PRIVATE
        ${TORCH_LIBRARIES}
        ggml
    )
    set_target_properties(test_hires_fix PROPERTIES
        INSTALL_RPATH "${LIBTORCH_ROOT}/lib"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_z_image_model.cpp")
    add_executable(test_z_image_model
        tests/test_z_image_model.cpp
        src/backend/z_image_model.cpp
        src/backend/model.cpp
        src/utils/gguf_loader.cpp
        src/utils/safetensors.cpp
    )
    target_include_directories(test_z_image_model PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/backend
        ${CMAKE_CURRENT_SOURCE_DIR}/src/utils
        ${STB_IMAGE_INCLUDE}
        ${THIRDPARTY_DIR}/ggml/include
    )
    target_link_libraries(test_z_image_model PRIVATE
        ${TORCH_LIBRARIES}
        ggml
    )
    set_target_properties(test_z_image_model PROPERTIES
        INSTALL_RPATH "${LIBTORCH_ROOT}/lib"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_z_image_quick.cpp")
    add_executable(test_z_image_quick
        tests/test_z_image_quick.cpp
        src/backend/z_image_model.cpp
        src/backend/model.cpp
        src/utils/gguf_loader.cpp
        src/utils/safetensors.cpp
    )
    target_include_directories(test_z_image_quick PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/backend
        ${CMAKE_CURRENT_SOURCE_DIR}/src/utils
        ${STB_IMAGE_INCLUDE}
        ${THIRDPARTY_DIR}/ggml/include
    )
    target_link_libraries(test_z_image_quick PRIVATE
        ${TORCH_LIBRARIES}
        ggml
    )
    set_target_properties(test_z_image_quick PROPERTIES
        INSTALL_RPATH "${LIBTORCH_ROOT}/lib"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_sdcpp_adapter.cpp")
    add_executable(test_sdcpp_adapter
        tests/test_sdcpp_adapter.cpp
        src/adapters/sdcpp_adapter.cpp
        src/utils/stb_impl.c
    )
    target_include_directories(test_sdcpp_adapter PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/adapters
        ${STB_IMAGE_INCLUDE}
        ${SDCPP_DIR}/include
        ${SDCPP_DIR}/ggml/include
    )
    target_link_libraries(test_sdcpp_adapter PRIVATE
        ${SDCPP_LINK_START}
        ${SDCPP_LINK_LIBRARIES}
        ${SDCPP_LINK_END}
        ${CMAKE_THREAD_LIBS_INIT}
    )
    if(EXISTS "${SDCPP_BUILD_DIR}/ggml/src/ggml-cuda/libggml-cuda.a")
    else()
    endif()
    set_target_properties(test_sdcpp_adapter PROPERTIES
        INSTALL_RPATH "${LIBTORCH_ROOT}/lib"
        BUILD_WITH_INSTALL_RPATH TRUE
        INTERPROCEDURAL_OPTIMIZATION FALSE
    )
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_txt2img.cpp")
    add_executable(test_txt2img
        tests/test_txt2img.cpp
        src/adapters/sdcpp_adapter.cpp
        src/utils/stb_impl.c
    )
    target_include_directories(test_txt2img PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/adapters
        ${STB_IMAGE_INCLUDE}
        ${SDCPP_DIR}/include
        ${SDCPP_DIR}/ggml/include
    )
    target_link_libraries(test_txt2img PRIVATE
        ${SDCPP_LINK_START}
        ${SDCPP_LINK_LIBRARIES}
        ${SDCPP_LINK_END}
        ${CMAKE_THREAD_LIBS_INIT}
    )
    set_target_properties(test_txt2img PROPERTIES
        INSTALL_RPATH "${LIBTORCH_ROOT}/lib"
        BUILD_WITH_INSTALL_RPATH TRUE
        INTERPROCEDURAL_OPTIMIZATION FALSE
    )
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_hires_fix_real.cpp")
    add_executable(test_hires_fix_real
        tests/test_hires_fix_real.cpp
        src/adapters/sdcpp_adapter.cpp
        src/utils/stb_impl.c
    )
    target_include_directories(test_hires_fix_real PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/adapters
        ${STB_IMAGE_INCLUDE}
        ${SDCPP_DIR}/include
        ${SDCPP_DIR}/ggml/include
    )
    target_link_libraries(test_hires_fix_real PRIVATE
        ${SDCPP_LINK_START}
        ${SDCPP_LINK_LIBRARIES}
        ${SDCPP_LINK_END}
        ${CMAKE_THREAD_LIBS_INIT}
    )
    set_target_properties(test_hires_fix_real PROPERTIES
        INSTALL_RPATH "${LIBTORCH_ROOT}/lib"
        BUILD_WITH_INSTALL_RPATH TRUE
        INTERPROCEDURAL_OPTIMIZATION FALSE
    )
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_image_utils.cpp")
    add_executable(test_image_utils
        tests/test_image_utils.cpp
        src/utils/image_utils.cpp
        src/utils/stb_impl.c
    )
    target_include_directories(test_image_utils PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/utils
        ${STB_IMAGE_INCLUDE}
        ${THIRDPARTY_DIR}/ggml/include
    )
    target_link_libraries(test_image_utils PRIVATE
        ${TORCH_LIBRARIES}
        ggml
    )
    set_target_properties(test_image_utils PROPERTIES
        INSTALL_RPATH "${LIBTORCH_ROOT}/lib"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()
