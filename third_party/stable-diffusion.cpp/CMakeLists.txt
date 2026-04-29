cmake_minimum_required(VERSION 3.12)
project("stable-diffusion")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

if (MSVC)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
    add_compile_definitions(_SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING)
    add_compile_options(
        $<$<COMPILE_LANGUAGE:C>:/MP>
        $<$<COMPILE_LANGUAGE:CXX>:/MP>
    )
endif()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(SD_STANDALONE ON)
else()
    set(SD_STANDALONE OFF)
endif()

set(SD_SUBMODULE_WEBP FALSE)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/libwebp/CMakeLists.txt")
    set(SD_SUBMODULE_WEBP TRUE)
endif()
if(SD_SUBMODULE_WEBP)
    set(SD_WEBP_DEFAULT ON)
else()
    set(SD_WEBP_DEFAULT ${SD_USE_SYSTEM_WEBP})
endif()

set(SD_SUBMODULE_WEBM FALSE)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/libwebm/CMakeLists.txt")
    set(SD_SUBMODULE_WEBM TRUE)
endif()
if(SD_SUBMODULE_WEBM)
    set(SD_WEBM_DEFAULT ON)
else()
    set(SD_WEBM_DEFAULT ${SD_USE_SYSTEM_WEBM})
endif()

#
# Option list
#

# general
#option(SD_BUILD_TESTS                "sd: build tests"    ${SD_STANDALONE})
option(SD_BUILD_EXAMPLES             "sd: build examples" ${SD_STANDALONE})
option(SD_WEBP                       "sd: enable WebP image I/O support" ${SD_WEBP_DEFAULT})
option(SD_USE_SYSTEM_WEBP            "sd: link against system libwebp" OFF)
option(SD_WEBM                       "sd: enable WebM video output support" ${SD_WEBM_DEFAULT})
option(SD_USE_SYSTEM_WEBM            "sd: link against system libwebm" OFF)
option(SD_CUDA                       "sd: cuda backend" OFF)
option(SD_HIPBLAS                    "sd: rocm backend" OFF)
option(SD_METAL                      "sd: metal backend" OFF)
option(SD_VULKAN                     "sd: vulkan backend" OFF)
option(SD_OPENCL                     "sd: opencl backend" OFF)
option(SD_SYCL                       "sd: sycl backend" OFF)
option(SD_MUSA                       "sd: musa backend" OFF)
option(SD_BUILD_SHARED_LIBS          "sd: build shared libs" OFF)
option(SD_BUILD_SHARED_GGML_LIB      "sd: build ggml as a separate shared lib" OFF)
option(SD_USE_SYSTEM_GGML            "sd: use system-installed GGML library" OFF)
#option(SD_BUILD_SERVER               "sd: build server example"                           ON)

if(SD_CUDA)
    message("-- Use CUDA as backend stable-diffusion")
    set(GGML_CUDA ON)
    add_definitions(-DSD_USE_CUDA)
endif()

if(SD_METAL)
    message("-- Use Metal as backend stable-diffusion")
    set(GGML_METAL ON)
    add_definitions(-DSD_USE_METAL)
endif()

if (SD_VULKAN)
    message("-- Use Vulkan as backend stable-diffusion")
    set(GGML_VULKAN ON)
    add_definitions(-DSD_USE_VULKAN)
endif ()

if (SD_OPENCL)
    message("-- Use OpenCL as backend stable-diffusion")
    set(GGML_OPENCL ON)
    add_definitions(-DSD_USE_OPENCL)
endif ()

if (SD_HIPBLAS)
    message("-- Use HIPBLAS as backend stable-diffusion")
    set(GGML_HIP ON)
    add_definitions(-DSD_USE_CUDA)
endif ()

if(SD_MUSA)
    message("-- Use MUSA as backend stable-diffusion")
    set(GGML_MUSA ON)
    add_definitions(-DSD_USE_CUDA)
endif()

if(SD_WEBP)
    if(NOT SD_SUBMODULE_WEBP AND NOT SD_USE_SYSTEM_WEBP)
        message(FATAL_ERROR "WebP support enabled but no source found.
          Either initialize the submodule:\n  git submodule update --init thirdparty/libwebp\n\n"
          "Or link against system library:\n  cmake (...) -DSD_USE_SYSTEM_WEBP=ON")
    endif()
    if(SD_USE_SYSTEM_WEBP)
        find_package(WebP REQUIRED)
        add_library(webp ALIAS WebP::webp)
        # libwebp CMake target naming is not consistent across versions/distros.
        # Some export WebP::libwebpmux, others export WebP::webpmux.
        if(TARGET WebP::libwebpmux)
            add_library(libwebpmux ALIAS WebP::libwebpmux)
        elseif(TARGET WebP::webpmux)
            add_library(libwebpmux ALIAS WebP::webpmux)
        else()
            message(FATAL_ERROR
                "Could not find a compatible webpmux target in system WebP package. "
                "Expected WebP::libwebpmux or WebP::webpmux."
            )
        endif()
    endif()
endif()

if(SD_WEBM)
    if(NOT SD_WEBP)
        message(FATAL_ERROR "SD_WEBM requires SD_WEBP because WebM output reuses libwebp VP8 encoding.")
    endif()
    if(NOT SD_SUBMODULE_WEBM AND NOT SD_USE_SYSTEM_WEBM)
        message(FATAL_ERROR "WebM support enabled but no source found.
          Either initialize the submodule:\n  git submodule update --init thirdparty/libwebm\n\n"
          "Or link against system library:\n  cmake (...) -DSD_USE_SYSTEM_WEBM=ON")
    endif()
    if(SD_USE_SYSTEM_WEBM)
        find_path(WEBM_INCLUDE_DIR
            NAMES mkvmuxer/mkvmuxer.h mkvparser/mkvparser.h common/webmids.h
            PATH_SUFFIXES webm
            REQUIRED)
        find_library(WEBM_LIBRARY
            NAMES webm libwebm
            REQUIRED)

        add_library(webm UNKNOWN IMPORTED)
        set_target_properties(webm PROPERTIES
            IMPORTED_LOCATION "${WEBM_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${WEBM_INCLUDE_DIR}")
    endif()
endif()

set(SD_LIB stable-diffusion)

file(GLOB SD_LIB_SOURCES CONFIGURE_DEPENDS
    "src/*.h"
    "src/*.cpp"
    "src/*.hpp"
    "src/model_io/*.h"
    "src/model_io/*.cpp"
    "src/tokenizers/*.h"
    "src/tokenizers/*.cpp"
    "src/tokenizers/vocab/*.h"
    "src/tokenizers/vocab/*.cpp"
)

# Exclude stable-diffusion-ext.cpp from direct compilation
# (it is included by stable-diffusion.cpp)
list(REMOVE_ITEM SD_LIB_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/stable-diffusion-ext.cpp")

find_program(GIT_EXE NAMES git git.exe NO_CMAKE_FIND_ROOT_PATH)
if(GIT_EXE)
    execute_process(COMMAND ${GIT_EXE} describe --tags --abbrev=7 --dirty=+
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE SDCPP_BUILD_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    execute_process(COMMAND ${GIT_EXE} rev-parse --short HEAD
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE SDCPP_BUILD_COMMIT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
endif()

if(NOT SDCPP_BUILD_VERSION)
    set(SDCPP_BUILD_VERSION unknown)
endif()
message(STATUS "stable-diffusion.cpp version ${SDCPP_BUILD_VERSION}")

if(NOT SDCPP_BUILD_COMMIT)
    set(SDCPP_BUILD_COMMIT unknown)
endif()
message(STATUS "stable-diffusion.cpp commit ${SDCPP_BUILD_COMMIT}")

set_property(
  SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/src/version.cpp
  APPEND PROPERTY COMPILE_DEFINITIONS
  SDCPP_BUILD_COMMIT=${SDCPP_BUILD_COMMIT} SDCPP_BUILD_VERSION=${SDCPP_BUILD_VERSION}
)

if(SD_BUILD_SHARED_LIBS)
    message("-- Build shared library")
    message(${SD_LIB_SOURCES})
    if(NOT SD_BUILD_SHARED_GGML_LIB)
        set(BUILD_SHARED_LIBS OFF)
    endif()
    add_library(${SD_LIB} SHARED ${SD_LIB_SOURCES})
    add_definitions(-DSD_BUILD_SHARED_LIB)
    target_compile_definitions(${SD_LIB} PRIVATE -DSD_BUILD_DLL)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
else()
    message("-- Build static library")
    if(NOT SD_BUILD_SHARED_GGML_LIB)
        set(BUILD_SHARED_LIBS OFF)
    endif()
    add_library(${SD_LIB} STATIC ${SD_LIB_SOURCES})
endif()

if(SD_SYCL)
    message("-- Use SYCL as backend stable-diffusion")
    set(GGML_SYCL ON)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-narrowing -fsycl")
    add_definitions(-DSD_USE_SYCL)
    # disable fast-math on host, see:
    # https://www.intel.com/content/www/us/en/docs/cpp-compiler/developer-guide-reference/2021-10/fp-model-fp.html
    if (WIN32)
        set(SYCL_COMPILE_OPTIONS /fp:precise)
    else()
        set(SYCL_COMPILE_OPTIONS -fp-model=precise)
    endif()
    message("-- Turn off fast-math for host in SYCL backend")
    target_compile_options(${SD_LIB} PRIVATE ${SYCL_COMPILE_OPTIONS})
endif()

set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

if (NOT SD_USE_SYSTEM_GGML)
    # see https://github.com/ggerganov/ggml/pull/682
    add_definitions(-DGGML_MAX_NAME=128)
endif()

# deps
# Only add ggml if it hasn't been added yet
if (NOT TARGET ggml)
    if (SD_USE_SYSTEM_GGML)
        find_package(ggml REQUIRED)
        if (NOT ggml_FOUND)
            message(FATAL_ERROR "System-installed GGML library not found.")
        endif()
        add_library(ggml ALIAS ggml::ggml)
    else()
        add_subdirectory(ggml)
    endif()
endif()

add_subdirectory(thirdparty)

target_link_libraries(${SD_LIB} PUBLIC ggml zip)
target_include_directories(${SD_LIB} PUBLIC . src include)
target_include_directories(${SD_LIB} PUBLIC . thirdparty)
target_compile_features(${SD_LIB} PUBLIC c_std_11 cxx_std_17)


if (SD_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

set(SD_PUBLIC_HEADERS include/stable-diffusion.h include/stable-diffusion-ext.h)
set_target_properties(${SD_LIB} PROPERTIES PUBLIC_HEADER "${SD_PUBLIC_HEADERS}")

install(TARGETS ${SD_LIB} LIBRARY PUBLIC_HEADER)

# ============================================================================
# sd-engine: C++ Workflow Engine (Command-line ComfyUI)
# ============================================================================

set(SD_ENGINE_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/sd-engine")

# sd-engine core library
set(SD_ENGINE_CORE_SRC
    "${SD_ENGINE_SRC_DIR}/core/node.cpp"
    "${SD_ENGINE_SRC_DIR}/core/workflow.cpp"
    "${SD_ENGINE_SRC_DIR}/core/cache.cpp"
    "${SD_ENGINE_SRC_DIR}/core/executor.cpp"
    "${SD_ENGINE_SRC_DIR}/core/init.cpp"
    "${SD_ENGINE_SRC_DIR}/core/log.cpp"
    "${SD_ENGINE_SRC_DIR}/core/memory_monitor.cpp"
    "${SD_ENGINE_SRC_DIR}/core/job_queue.cpp"
    "${SD_ENGINE_SRC_DIR}/adapter/sd_adapter.cpp"
)

# sd-engine nodes
set(SD_ENGINE_NODE_SRC
    "${SD_ENGINE_SRC_DIR}/nodes/node_utils.cpp"
    "${SD_ENGINE_SRC_DIR}/nodes/test_nodes.cpp"
    "${SD_ENGINE_SRC_DIR}/nodes/loader_nodes.cpp"
    "${SD_ENGINE_SRC_DIR}/nodes/model_loader_nodes.cpp"
    "${SD_ENGINE_SRC_DIR}/nodes/lora_loader_nodes.cpp"
    "${SD_ENGINE_SRC_DIR}/nodes/upscale_loader_nodes.cpp"
    "${SD_ENGINE_SRC_DIR}/nodes/image_nodes.cpp"
    "${SD_ENGINE_SRC_DIR}/nodes/image_io_nodes.cpp"
    "${SD_ENGINE_SRC_DIR}/nodes/generate_nodes.cpp"
    "${SD_ENGINE_SRC_DIR}/nodes/core_nodes.cpp"
    "${SD_ENGINE_SRC_DIR}/nodes/preprocessor_nodes.cpp"
    "${SD_ENGINE_SRC_DIR}/nodes/face_nodes.cpp"
)

# ONNX Runtime (optional, for FaceRestore/FaceSwap)
set(ONNXRUNTIME_DIR "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/onnxruntime")
if(EXISTS "${ONNXRUNTIME_DIR}/include/onnxruntime_cxx_api.h")
    message(STATUS "[sd-engine] ONNX Runtime found: ${ONNXRUNTIME_DIR}")
    set(ONNXRUNTIME_INCLUDE "${ONNXRUNTIME_DIR}/include")
    if(WIN32)
        find_library(ONNXRUNTIME_LIB onnxruntime PATHS "${ONNXRUNTIME_DIR}/lib" NO_DEFAULT_PATH)
    elseif(APPLE)
        find_library(ONNXRUNTIME_LIB onnxruntime PATHS "${ONNXRUNTIME_DIR}/lib" NO_DEFAULT_PATH)
    else()
        find_library(ONNXRUNTIME_LIB onnxruntime PATHS "${ONNXRUNTIME_DIR}/lib" NO_DEFAULT_PATH)
    endif()
    if(NOT ONNXRUNTIME_LIB)
        message(WARNING "[sd-engine] ONNX Runtime headers found but library not found in ${ONNXRUNTIME_DIR}/lib")
        set(ONNXRUNTIME_AVAILABLE FALSE)
    else()
        set(ONNXRUNTIME_AVAILABLE TRUE)
    endif()
else()
    message(STATUS "[sd-engine] ONNX Runtime not found. Face processing nodes disabled.")
    set(ONNXRUNTIME_AVAILABLE FALSE)
endif()

if(EXISTS "${SD_ENGINE_SRC_DIR}/core/node.cpp")
    add_library(sd-engine STATIC ${SD_ENGINE_CORE_SRC} ${SD_ENGINE_NODE_SRC})
    target_include_directories(sd-engine PRIVATE
        "${SD_ENGINE_SRC_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty"
    )
    target_compile_definitions(sd-engine PRIVATE NLOHMANN_JSON_HPP)
    target_link_libraries(sd-engine PRIVATE ${SD_LIB} ggml)
    if(ONNXRUNTIME_AVAILABLE)
        target_include_directories(sd-engine PRIVATE ${ONNXRUNTIME_INCLUDE})
        target_link_libraries(sd-engine PRIVATE ${ONNXRUNTIME_LIB})
        target_compile_definitions(sd-engine PRIVATE ONNXRUNTIME_AVAILABLE)
    endif()
    message(STATUS "[sd-engine] Created library target")
endif()

# Tests
enable_testing()
set(CATCH2_SRC "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/catch_amalgamated.cpp")
set(CATCH2_INC "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty")

file(GLOB TEST_SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp")

if(EXISTS ${CATCH2_SRC} AND TEST_SRC_FILES)
    add_executable(sd-engine-tests ${TEST_SRC_FILES} ${CATCH2_SRC})
    target_include_directories(sd-engine-tests PRIVATE
        "${SD_ENGINE_SRC_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty"
        "${CATCH2_INC}"
    )
    if(TARGET sd-engine)
        target_link_libraries(sd-engine-tests PRIVATE sd-engine)
    endif()
    target_link_libraries(sd-engine-tests PRIVATE ${SD_LIB} ggml)
    add_test(NAME sd-engine-tests COMMAND sd-engine-tests)
    message(STATUS "[sd-engine] Created test target: sd-engine-tests")
else()
    message(STATUS "[sd-engine] Tests disabled: Catch2 or test sources not found")
endif()

# sd-workflow executable
if(EXISTS "${SD_ENGINE_SRC_DIR}/tools/sd-workflow.cpp")
    add_executable(sd-workflow "${SD_ENGINE_SRC_DIR}/tools/sd-workflow.cpp")
    target_include_directories(sd-workflow PRIVATE
        "${SD_ENGINE_SRC_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty"
    )
    if(TARGET sd-engine)
        target_link_libraries(sd-workflow PRIVATE sd-engine)
    endif()
    target_link_libraries(sd-workflow PRIVATE ${SD_LIB} ggml)
    message(STATUS "[sd-engine] Created executable: sd-workflow")
endif()

# sd-engine-server executable
if(EXISTS "${SD_ENGINE_SRC_DIR}/tools/sd-server.cpp")
    add_executable(sd-engine-server "${SD_ENGINE_SRC_DIR}/tools/sd-server.cpp")
    target_include_directories(sd-engine-server PRIVATE
        "${SD_ENGINE_SRC_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty"
    )
    if(TARGET sd-engine)
        target_link_libraries(sd-engine-server PRIVATE sd-engine)
    endif()
    target_link_libraries(sd-engine-server PRIVATE ${SD_LIB} ggml)
    message(STATUS "[sd-engine] Created executable: sd-engine-server")
endif()
