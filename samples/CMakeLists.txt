cmake_minimum_required(VERSION 3.29)

project(ApraPipesSamples)

# This is a STANDALONE project that finds and links against the already-built aprapipes library
# It does NOT integrate with base/CMakeLists.txt to avoid breaking the main build

message(STATUS "=== ApraPipes Samples - Standalone Build ===")

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================================
# Find the already-built aprapipes library
# ============================================================================

set(APRAPIPES_BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../_build" CACHE PATH "Path to ApraPipes build directory")
set(APRAPIPES_BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../base")

message(STATUS "Looking for aprapipes library in: ${APRAPIPES_BUILD_DIR}")

# Find the library file
find_library(APRAPIPES_LIB
    NAMES aprapipes aprapipesd
    PATHS
        ${APRAPIPES_BUILD_DIR}/RelWithDebInfo
        ${APRAPIPES_BUILD_DIR}/Debug
        ${APRAPIPES_BUILD_DIR}/Release
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

# Find CUDA (required since aprapipes uses CUDA)
find_package(CUDAToolkit REQUIRED)
message(STATUS "Found CUDA: ${CUDAToolkit_VERSION}")

# ============================================================================
# Helper function to add samples
# ============================================================================

function(add_apra_sample SAMPLE_NAME SOURCE_FILE)
    message(STATUS "  Adding sample: ${SAMPLE_NAME}")

    add_executable(${SAMPLE_NAME} ${SOURCE_FILE})

    # Link against aprapipes library and dependencies
    target_link_libraries(${SAMPLE_NAME} PRIVATE
        ${APRAPIPES_LIB}
        ${Boost_LIBRARIES}
        CUDA::cudart
        CUDA::cuda_driver
    )

    # Include directories
    target_include_directories(${SAMPLE_NAME} PRIVATE
        ${APRAPIPES_BASE_DIR}/include
        ${Boost_INCLUDE_DIRS}
    )

    # Windows-specific settings
    if(WIN32)
        # Use /MP for parallel compilation
        target_compile_options(${SAMPLE_NAME} PRIVATE /MP)

        # Set runtime library to match aprapipes
        set_property(TARGET ${SAMPLE_NAME} PROPERTY
            MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL"
        )
    endif()

    # Output to samples subdirectory
    set_target_properties(${SAMPLE_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/$<CONFIG>
    )

    # Copy Boost DLLs to output directory (needed when running from VS)
    add_custom_command(TARGET ${SAMPLE_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${APRAPIPES_BUILD_DIR}/$<CONFIG>/boost_filesystem-vc142-mt-x64-1_84.dll"
            "$<TARGET_FILE_DIR:${SAMPLE_NAME}>/"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${APRAPIPES_BUILD_DIR}/$<CONFIG>/boost_log-vc142-mt-x64-1_84.dll"
            "$<TARGET_FILE_DIR:${SAMPLE_NAME}>/"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${APRAPIPES_BUILD_DIR}/$<CONFIG>/boost_serialization-vc142-mt-x64-1_84.dll"
            "$<TARGET_FILE_DIR:${SAMPLE_NAME}>/"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${APRAPIPES_BUILD_DIR}/$<CONFIG>/boost_thread-vc142-mt-x64-1_84.dll"
            "$<TARGET_FILE_DIR:${SAMPLE_NAME}>/"
        COMMENT "Copying Boost DLLs to ${SAMPLE_NAME} output directory"
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
