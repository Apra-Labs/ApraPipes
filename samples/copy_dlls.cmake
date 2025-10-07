# Script to copy Boost DLLs from main build to sample output directory
# This handles the case where VS builds in Release but main build is RelWithDebInfo

set(DLL_NAMES
    "boost_filesystem-vc142-mt-x64-1_84.dll"
    "boost_log-vc142-mt-x64-1_84.dll"
    "boost_serialization-vc142-mt-x64-1_84.dll"
    "boost_thread-vc142-mt-x64-1_84.dll"
)

# Try multiple source directories in order of preference
set(POSSIBLE_SOURCE_DIRS
    "${APRAPIPES_BUILD_DIR}/${CONFIG}"
    "${APRAPIPES_BUILD_DIR}/RelWithDebInfo"
    "${APRAPIPES_BUILD_DIR}/Release"
    "${APRAPIPES_BUILD_DIR}/Debug"
)

foreach(DLL_NAME ${DLL_NAMES})
    set(FOUND FALSE)
    foreach(SOURCE_DIR ${POSSIBLE_SOURCE_DIRS})
        set(SOURCE_FILE "${SOURCE_DIR}/${DLL_NAME}")
        if(EXISTS "${SOURCE_FILE}")
            message(STATUS "Copying ${DLL_NAME} from ${SOURCE_DIR}")
            file(COPY "${SOURCE_FILE}" DESTINATION "${OUTPUT_DIR}")
            set(FOUND TRUE)
            break()
        endif()
    endforeach()

    if(NOT FOUND)
        message(WARNING "Could not find ${DLL_NAME} in any build directory")
    endif()
endforeach()
