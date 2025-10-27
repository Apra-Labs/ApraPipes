# Script to copy required DLLs from vcpkg to sample output directory
# Handles Debug and Release/RelWithDebInfo configurations
# Copies: Boost DLLs, OpenCV DLLs, and OpenCV dependencies

# Determine if this is a debug build
if(CONFIG MATCHES "Debug")
    set(IS_DEBUG TRUE)
    set(BOOST_DLL_SUFFIX "-vc142-mt-gd-x64-1_84.dll")
    set(VCPKG_BIN_DIR "${APRAPIPES_BUILD_DIR}/vcpkg_installed/x64-windows/debug/bin")
else()
    set(IS_DEBUG FALSE)
    set(BOOST_DLL_SUFFIX "-vc142-mt-x64-1_84.dll")
    set(VCPKG_BIN_DIR "${APRAPIPES_BUILD_DIR}/vcpkg_installed/x64-windows/bin")
endif()

message(STATUS "Copying required DLLs for ${CONFIG} configuration from ${VCPKG_BIN_DIR}")

# ============================================================================
# 1. Copy Boost DLLs
# ============================================================================

set(BOOST_COMPONENTS filesystem log serialization thread)

message(STATUS "  [1/2] Copying Boost DLLs...")
set(BOOST_DLL_COUNT 0)

foreach(COMPONENT ${BOOST_COMPONENTS})
    set(DLL_NAME "boost_${COMPONENT}${BOOST_DLL_SUFFIX}")
    set(SOURCE "${VCPKG_BIN_DIR}/${DLL_NAME}")

    if(EXISTS "${SOURCE}")
        file(COPY "${SOURCE}" DESTINATION "${OUTPUT_DIR}")
        message(STATUS "    ✓ ${DLL_NAME}")
        math(EXPR BOOST_DLL_COUNT "${BOOST_DLL_COUNT} + 1")
    else()
        message(WARNING "    ✗ Could not find ${DLL_NAME}")
    endif()
endforeach()

message(STATUS "  Copied ${BOOST_DLL_COUNT} Boost DLLs")

# ============================================================================
# 2. Copy OpenCV DLLs and dependencies
# ============================================================================

message(STATUS "  [2/2] Copying OpenCV DLLs and dependencies...")

# Copy all opencv*.dll files
file(GLOB OPENCV_DLLS "${VCPKG_BIN_DIR}/opencv*.dll")
set(OPENCV_DLL_COUNT 0)

foreach(DLL ${OPENCV_DLLS})
    get_filename_component(DLL_NAME ${DLL} NAME)
    file(COPY "${DLL}" DESTINATION "${OUTPUT_DIR}")
    message(STATUS "    ✓ ${DLL_NAME}")
    math(EXPR OPENCV_DLL_COUNT "${OPENCV_DLL_COUNT} + 1")
endforeach()

# Copy OpenCV dependencies and FFmpeg libraries
# These are required by OpenCV, RTSP, MP4, and video processing modules
set(OPENCV_DEPS
    zlib1.dll
    jpeg62.dll
    libpng16.dll
    tiff.dll
    libwebp.dll
    libwebpdecoder.dll
    libwebpdemux.dll
    libwebpmux.dll
    liblzma.dll
    zstd.dll
    lerc.dll
    libsharpyuv.dll
    libprotobuf.dll
    libprotobuf-lite.dll
    libprotoc.dll
    avcodec-58.dll
    avformat-58.dll
    avutil-56.dll
    swresample-3.dll
    swscale-5.dll
)

set(DEPS_DLL_COUNT 0)

foreach(DLL_NAME ${OPENCV_DEPS})
    set(SOURCE "${VCPKG_BIN_DIR}/${DLL_NAME}")
    if(EXISTS "${SOURCE}")
        file(COPY "${SOURCE}" DESTINATION "${OUTPUT_DIR}")
        message(STATUS "    ✓ ${DLL_NAME}")
        math(EXPR DEPS_DLL_COUNT "${DEPS_DLL_COUNT} + 1")
    endif()
endforeach()

message(STATUS "  Copied ${OPENCV_DLL_COUNT} OpenCV DLLs and ${DEPS_DLL_COUNT} dependency DLLs")
message(STATUS "Total DLLs copied: ${BOOST_DLL_COUNT} Boost + ${OPENCV_DLL_COUNT} OpenCV + ${DEPS_DLL_COUNT} dependencies")
