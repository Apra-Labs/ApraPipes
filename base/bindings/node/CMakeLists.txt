# ============================================================
# CMakeLists.txt for Node.js native addon
# Package: @apralabs/aprapipes
# ============================================================

cmake_minimum_required(VERSION 3.15)

# Find node-addon-api from vcpkg
find_package(unofficial-node-addon-api CONFIG REQUIRED)
find_package(unofficial-node-api-headers CONFIG REQUIRED)

# Define the addon target
set(ADDON_NAME "aprapipes")

add_library(${ADDON_NAME} SHARED
    addon.cpp
)

# Set output extension to .node
set_target_properties(${ADDON_NAME} PROPERTIES
    PREFIX ""
    SUFFIX ".node"
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Link dependencies
target_link_libraries(${ADDON_NAME} PRIVATE
    unofficial::node-addon-api::node-addon-api
    aprapipes  # Main ApraPipes library
)

# Include directories
target_include_directories(${ADDON_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/declarative
)

# Platform-specific settings
if(WIN32)
    # Windows: Don't add lib prefix
    set_target_properties(${ADDON_NAME} PROPERTIES
        PREFIX ""
    )
    # Link node.lib
    # Note: User must set NODE_LIB_DIR environment variable or CMake cache variable
    if(DEFINED ENV{NODE_LIB_DIR})
        target_link_directories(${ADDON_NAME} PRIVATE $ENV{NODE_LIB_DIR})
    endif()
elseif(APPLE)
    # macOS: Use undefined dynamic lookup for Node symbols
    set_target_properties(${ADDON_NAME} PROPERTIES
        LINK_FLAGS "-undefined dynamic_lookup"
    )
endif()

# Don't include this target in default ALL builds unless explicitly enabled
# Users must set BUILD_NODE_ADDON=ON to build the Node.js addon
if(NOT BUILD_NODE_ADDON)
    set_target_properties(${ADDON_NAME} PROPERTIES
        EXCLUDE_FROM_ALL TRUE
    )
endif()

# Install target
install(TARGETS ${ADDON_NAME}
    LIBRARY DESTINATION lib/node
    RUNTIME DESTINATION lib/node
)

# Copy to project root for npm usage (if building addon)
add_custom_command(TARGET ${ADDON_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:${ADDON_NAME}>
        ${CMAKE_SOURCE_DIR}/../$<TARGET_FILE_NAME:${ADDON_NAME}>
    COMMENT "Copying Node addon to project root"
)
