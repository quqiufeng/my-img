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
    endforeach()
    if(SD_PATH_FROM_ENV AND NOT SD_PATH)
        set(SD_PATH ${SD_PATH_FROM_ENV})
    endif()
endif()

if(SD_PATH)
    string(REPLACE "~" "$ENV{HOME}" SD_PATH ${SD_PATH})
endif()

if(NOT SD_PATH OR NOT EXISTS ${SD_PATH}/CMakeLists.txt)
    message(FATAL_ERROR "SD_PATH 路径无效或未设置。当前路径: ${SD_PATH}")
endif()

message(STATUS "[my-img] Using SD_PATH: ${SD_PATH}")

# ============================================================
# 2. 集成官方项目
# ============================================================
add_subdirectory(${SD_PATH} ${CMAKE_BINARY_DIR}/sd_build)

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
    
    target_link_libraries(${final_name} PRIVATE stable-diffusion)

    target_include_directories(${final_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    
    target_include_directories(${final_name} PRIVATE 
        ${SD_PATH}
        ${SD_PATH}/thirdparty
        ${SD_PATH}/ggml/include
    )

    message(STATUS "[my-img] Created target: ${final_name}")
endforeach()
