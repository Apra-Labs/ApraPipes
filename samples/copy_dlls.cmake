# Script to copy Boost DLLs from vcpkg to sample output directory
# Handles Debug (-gd-) and Release/RelWithDebInfo (no -gd-) configurations

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

# List of Boost components needed by samples
set(BOOST_COMPONENTS filesystem log serialization thread)

message(STATUS "Copying Boost DLLs for ${CONFIG} configuration from ${VCPKG_BIN_DIR}")

# Copy each required DLL
foreach(COMPONENT ${BOOST_COMPONENTS})
    set(DLL_NAME "boost_${COMPONENT}${BOOST_DLL_SUFFIX}")
    set(SOURCE "${VCPKG_BIN_DIR}/${DLL_NAME}")

    if(EXISTS "${SOURCE}")
        file(COPY "${SOURCE}" DESTINATION "${OUTPUT_DIR}")
        message(STATUS "  Copied ${DLL_NAME}")
    else()
        message(WARNING "  Could not find ${DLL_NAME} at ${SOURCE}")
    endif()
endforeach()
