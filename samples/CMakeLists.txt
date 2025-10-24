cmake_minimum_required(VERSION 3.29)

project(ApraPipesSamples CXX CUDA)

# This is a STANDALONE project that finds and links against the already-built aprapipes library
# It does NOT integrate with base/CMakeLists.txt to avoid breaking the main build

message(STATUS "=== ApraPipes Samples - Standalone Build ===")

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add cmake modules directory for FindCUDA.cmake (needed by OpenCV)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../base/cmake")

# VCPKG_INSTALLED_DIR is set via command line in build_samples.ps1
# This tells vcpkg toolchain where to find packages (in main build's vcpkg_installed)

# However, vcpkg toolchain in manifest mode has issues with non-standard locations
# So we also manually add the vcpkg installed directory to CMAKE_PREFIX_PATH
if(DEFINED VCPKG_INSTALLED_DIR)
    list(APPEND CMAKE_PREFIX_PATH "${VCPKG_INSTALLED_DIR}/share")
    message(STATUS "Added to CMAKE_PREFIX_PATH: ${VCPKG_INSTALLED_DIR}/share")
endif()

# ============================================================================
# Find the already-built aprapipes library
# ============================================================================

set(APRAPIPES_BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../_build" CACHE PATH "Path to ApraPipes build directory")
set(APRAPIPES_BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../base")

message(STATUS "Looking for aprapipes library in: ${APRAPIPES_BUILD_DIR}")

# Find the library file (aprapipesd for Debug, aprapipes for others)
# Try current build type directory first, then fallback to others
find_library(APRAPIPES_LIB
    NAMES aprapipes aprapipesd
    PATHS
        ${APRAPIPES_BUILD_DIR}/${CMAKE_BUILD_TYPE}
        ${APRAPIPES_BUILD_DIR}/RelWithDebInfo
        ${APRAPIPES_BUILD_DIR}/Release
        ${APRAPIPES_BUILD_DIR}/Debug
    NO_DEFAULT_PATH
)

if(NOT APRAPIPES_LIB)
    message(FATAL_ERROR
        "aprapipes library not found!\n"
        "Please build the main library first:\n"
        "  cd ${CMAKE_CURRENT_SOURCE_DIR}/..\n"
        "  .\\build_windows_cuda_vs19.ps1\n"
        "\n"
        "Looking in:\n"
        "  ${APRAPIPES_BUILD_DIR}/RelWithDebInfo\n"
        "  ${APRAPIPES_BUILD_DIR}/Debug\n"
        "  ${APRAPIPES_BUILD_DIR}/Release")
endif()

message(STATUS "Found aprapipes library: ${APRAPIPES_LIB}")

# Check include directory exists
if(NOT EXISTS "${APRAPIPES_BASE_DIR}/include")
    message(FATAL_ERROR "aprapipes include directory not found: ${APRAPIPES_BASE_DIR}/include")
endif()

message(STATUS "Using aprapipes headers from: ${APRAPIPES_BASE_DIR}/include")

# ============================================================================
# Find dependencies (same as main library uses)
# ============================================================================

message(STATUS "Finding dependencies via vcpkg...")

# Boost is installed in the main build's vcpkg_installed directory
set(BOOST_ROOT "${APRAPIPES_BUILD_DIR}/vcpkg_installed/x64-windows")
set(Boost_INCLUDE_DIR "${BOOST_ROOT}/include")
set(Boost_LIBRARY_DIR "${BOOST_ROOT}/lib")

# Find Boost (same way as main library)
find_package(Boost REQUIRED COMPONENTS
    system
    thread
    filesystem
    serialization
    log
    chrono
)

message(STATUS "Found Boost: ${Boost_VERSION}")

# Set up Boost library directories for debug/release
set(BOOST_LIB_DIR_RELEASE "${BOOST_ROOT}/lib")
set(BOOST_LIB_DIR_DEBUG "${APRAPIPES_BUILD_DIR}/vcpkg_installed/x64-windows/debug/lib")

# Create a function to get Boost library path with debug/release variants
function(get_boost_libraries OUTPUT_VAR)
    set(BOOST_LIBS "")
    foreach(COMPONENT ${ARGN})
        # Use generator expressions to select debug (-gd) or release libs
        # .lib files use vc140 naming convention
        list(APPEND BOOST_LIBS
            "$<$<CONFIG:Debug>:${BOOST_LIB_DIR_DEBUG}/boost_${COMPONENT}-vc140-mt-gd.lib>"
            "$<$<NOT:$<CONFIG:Debug>>:${BOOST_LIB_DIR_RELEASE}/boost_${COMPONENT}-vc140-mt.lib>"
        )
    endforeach()
    set(${OUTPUT_VAR} ${BOOST_LIBS} PARENT_SCOPE)
endfunction()

# Find CUDA (required since aprapipes uses CUDA)
find_package(CUDAToolkit REQUIRED)
message(STATUS "Found CUDA: ${CUDAToolkit_VERSION}")

# Set hints for Find modules used by OpenCV dependencies
# These modules don't automatically use vcpkg locations in standalone builds
if(DEFINED VCPKG_INSTALLED_DIR)
    # TIFF
    set(TIFF_INCLUDE_DIR "${VCPKG_INSTALLED_DIR}/include" CACHE PATH "TIFF include directory")
    set(TIFF_LIBRARY "${VCPKG_INSTALLED_DIR}/lib/tiff.lib" CACHE FILEPATH "TIFF library")

    # ZLIB
    set(ZLIB_INCLUDE_DIR "${VCPKG_INSTALLED_DIR}/include" CACHE PATH "ZLIB include directory")
    set(ZLIB_LIBRARY "${VCPKG_INSTALLED_DIR}/lib/zlib.lib" CACHE FILEPATH "ZLIB library")

    # PNG
    set(PNG_PNG_INCLUDE_DIR "${VCPKG_INSTALLED_DIR}/include" CACHE PATH "PNG include directory")
    set(PNG_LIBRARY "${VCPKG_INSTALLED_DIR}/lib/libpng16.lib" CACHE FILEPATH "PNG library")

    # JPEG
    set(JPEG_INCLUDE_DIR "${VCPKG_INSTALLED_DIR}/include" CACHE PATH "JPEG include directory")
    set(JPEG_LIBRARY "${VCPKG_INSTALLED_DIR}/lib/jpeg.lib" CACHE FILEPATH "JPEG library")

    # LibArchive
    set(LibArchive_INCLUDE_DIR "${VCPKG_INSTALLED_DIR}/include" CACHE PATH "LibArchive include directory")
    set(LibArchive_LIBRARY "${VCPKG_INSTALLED_DIR}/lib/archive.lib" CACHE FILEPATH "LibArchive library")

    # WebP
    set(WEBP_INCLUDE_DIR "${VCPKG_INSTALLED_DIR}/include" CACHE PATH "WebP include directory")
    set(WEBP_LIBRARY "${VCPKG_INSTALLED_DIR}/lib/webp.lib" CACHE FILEPATH "WebP library")

    # Common include path for all
    set(CMAKE_INCLUDE_PATH "${VCPKG_INSTALLED_DIR}/include" CACHE PATH "Common include path")
    set(CMAKE_LIBRARY_PATH "${VCPKG_INSTALLED_DIR}/lib" CACHE PATH "Common library path")
endif()

# Find OpenCV (required for samples that use vision modules)
# vcpkg toolchain will find it automatically via VCPKG_INSTALLED_DIR
find_package(OpenCV CONFIG REQUIRED)
message(STATUS "Found OpenCV: ${OpenCV_VERSION}")

# Find FFmpeg libraries (required for RTSP and video processing samples)
find_library(FFMPEG_AVCODEC_LIB NAMES avcodec PATHS ${APRAPIPES_BUILD_DIR}/vcpkg_installed/x64-windows/lib NO_DEFAULT_PATH REQUIRED)
find_library(FFMPEG_AVFORMAT_LIB NAMES avformat PATHS ${APRAPIPES_BUILD_DIR}/vcpkg_installed/x64-windows/lib NO_DEFAULT_PATH REQUIRED)
find_library(FFMPEG_AVUTIL_LIB NAMES avutil PATHS ${APRAPIPES_BUILD_DIR}/vcpkg_installed/x64-windows/lib NO_DEFAULT_PATH REQUIRED)
find_library(FFMPEG_SWRESAMPLE_LIB NAMES swresample PATHS ${APRAPIPES_BUILD_DIR}/vcpkg_installed/x64-windows/lib NO_DEFAULT_PATH REQUIRED)
find_library(FFMPEG_SWSCALE_LIB NAMES swscale PATHS ${APRAPIPES_BUILD_DIR}/vcpkg_installed/x64-windows/lib NO_DEFAULT_PATH REQUIRED)
message(STATUS "Found FFmpeg libraries")

# Find MP4 demuxer library (required for MP4 samples)
find_library(MP4LIB_LIB NAMES mp4lib PATHS ${APRAPIPES_BUILD_DIR}/vcpkg_installed/x64-windows/lib NO_DEFAULT_PATH REQUIRED)
message(STATUS "Found MP4 library: ${MP4LIB_LIB}")

# Find NVIDIA CUVID library (required for hardware-accelerated video decoding)
find_library(NVCUVID_LIB NAMES nvcuvid PATHS ${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/Video_Codec_SDK_10.0.26/Lib/x64 NO_DEFAULT_PATH REQUIRED)
message(STATUS "Found NVIDIA CUVID library: ${NVCUVID_LIB}")

# Find NVIDIA NVJPEG library (required for thumbnail_generator sample)
find_library(NVJPEG_LIB NAMES nvjpeg PATHS ${CUDAToolkit_LIBRARY_DIR} NO_DEFAULT_PATH)
if(NVJPEG_LIB)
    message(STATUS "Found NVIDIA NVJPEG library: ${NVJPEG_LIB}")
else()
    message(WARNING "NVJPEG library not found - thumbnail_generator may fail to link")
endif()

# Find OpenH264 library (required for timelapse sample)
find_library(OPENH264_LIB NAMES openh264 PATHS ${APRAPIPES_BUILD_DIR}/vcpkg_installed/x64-windows/lib NO_DEFAULT_PATH)
if(OPENH264_LIB)
    message(STATUS "Found OpenH264 library: ${OPENH264_LIB}")
else()
    message(WARNING "OpenH264 library not found - timelapse may fail to link")
endif()

# Find NVIDIA NvEncode library (required for timelapse H264 encoding)
find_library(NVENCODE_LIB NAMES nvencodeapi PATHS ${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/Video_Codec_SDK_10.0.26/Lib/x64 NO_DEFAULT_PATH)
if(NVENCODE_LIB)
    message(STATUS "Found NVIDIA NvEncode library: ${NVENCODE_LIB}")
else()
    message(WARNING "NvEncode library not found - timelapse may fail to link")
endif()

# ============================================================================
# Helper function to add samples
# ============================================================================

function(add_apra_sample SAMPLE_NAME SOURCE_FILE)
    message(STATUS "  Adding sample: ${SAMPLE_NAME}")

    add_executable(${SAMPLE_NAME} ${SOURCE_FILE})

    # Get Boost libraries with automatic debug/release selection
    get_boost_libraries(BOOST_LIBS_FOR_SAMPLE
        system thread filesystem serialization log chrono
    )

    # Link against aprapipes library and dependencies
    target_link_libraries(${SAMPLE_NAME} PRIVATE
        ${APRAPIPES_LIB}
        ${BOOST_LIBS_FOR_SAMPLE}
        ${OpenCV_LIBRARIES}
        CUDA::cudart
        CUDA::cuda_driver
        ${FFMPEG_AVCODEC_LIB}
        ${FFMPEG_AVFORMAT_LIB}
        ${FFMPEG_AVUTIL_LIB}
        ${FFMPEG_SWRESAMPLE_LIB}
        ${FFMPEG_SWSCALE_LIB}
        ${MP4LIB_LIB}
        ${NVCUVID_LIB}
    )

    # Link optional libraries if found
    if(NVJPEG_LIB)
        target_link_libraries(${SAMPLE_NAME} PRIVATE ${NVJPEG_LIB})
    endif()
    if(OPENH264_LIB)
        target_link_libraries(${SAMPLE_NAME} PRIVATE ${OPENH264_LIB})
    endif()
    if(NVENCODE_LIB)
        target_link_libraries(${SAMPLE_NAME} PRIVATE ${NVENCODE_LIB})
    endif()

    # Include directories
    target_include_directories(${SAMPLE_NAME} PRIVATE
        ${APRAPIPES_BASE_DIR}/include
        ${Boost_INCLUDE_DIRS}
    )

    # Windows-specific settings
    if(WIN32)
        # Use /MP for parallel compilation
        target_compile_options(${SAMPLE_NAME} PRIVATE /MP)

        # Use correct runtime: /MDd for Debug, /MD for Release/RelWithDebInfo
        # This matches how aprapipesd.lib and aprapipes.lib are built
        set_property(TARGET ${SAMPLE_NAME} PROPERTY
            MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL"
        )
    endif()

    # Output to samples subdirectory
    set_target_properties(${SAMPLE_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/$<CONFIG>
    )

    # Copy Boost DLLs to output directory (uses config-aware script)
    add_custom_command(TARGET ${SAMPLE_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND}
            -DCONFIG=$<CONFIG>
            -DAPRAPIPES_BUILD_DIR=${APRAPIPES_BUILD_DIR}
            -DOUTPUT_DIR=$<TARGET_FILE_DIR:${SAMPLE_NAME}>
            -P ${CMAKE_CURRENT_SOURCE_DIR}/copy_dlls.cmake
        COMMENT "Copying Boost DLLs for $<CONFIG> configuration"
        VERBATIM
    )
endfunction()

# ============================================================================
# Add samples
# ============================================================================

message(STATUS "Configuring samples:")

# Basic samples
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/basic/hello_pipeline/main.cpp)
    add_apra_sample(hello_pipeline basic/hello_pipeline/main.cpp)
else()
    message(WARNING "hello_pipeline source not found")
endif()

# Video samples
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/video/face_detection_cpu/main.cpp)
    add_apra_sample(face_detection_cpu video/face_detection_cpu/main.cpp)
else()
    message(WARNING "face_detection_cpu source not found")
endif()

# Network samples
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/network/relay/main.cpp)
    add_apra_sample(relay network/relay/main.cpp)
else()
    message(WARNING "relay source not found")
endif()

# Video samples (continued)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/video/thumbnail_generator/main.cpp)
    add_apra_sample(thumbnail_generator video/thumbnail_generator/main.cpp)
else()
    message(WARNING "thumbnail_generator source not found")
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/video/timelapse/main.cpp)
    add_apra_sample(timelapse video/timelapse/main.cpp)
else()
    message(WARNING "timelapse source not found")
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/video/file_reader/main.cpp)
    add_apra_sample(file_reader video/file_reader/main.cpp)
else()
    message(WARNING "file_reader source not found")
endif()

# Add more samples here as they are created:
# add_apra_sample(webcam_capture video/webcam_capture/main.cpp)
# add_apra_sample(image_resize image/resize/main.cpp)

# ============================================================================
# Meta-target to build all samples
# ============================================================================

add_custom_target(all_samples)
if(TARGET hello_pipeline)
    add_dependencies(all_samples hello_pipeline)
endif()
if(TARGET face_detection_cpu)
    add_dependencies(all_samples face_detection_cpu)
endif()
if(TARGET relay)
    add_dependencies(all_samples relay)
endif()
if(TARGET thumbnail_generator)
    add_dependencies(all_samples thumbnail_generator)
endif()
if(TARGET timelapse)
    add_dependencies(all_samples timelapse)
endif()
if(TARGET file_reader)
    add_dependencies(all_samples file_reader)
endif()

# ============================================================================
# Summary
# ============================================================================

message(STATUS "")
message(STATUS "=== Configuration Summary ===")
message(STATUS "aprapipes library: ${APRAPIPES_LIB}")
message(STATUS "aprapipes headers: ${APRAPIPES_BASE_DIR}/include")
message(STATUS "Boost version: ${Boost_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Output directory: ${CMAKE_BINARY_DIR}/$<CONFIG>")
message(STATUS "============================")
message(STATUS "")
