vcpkg_check_linkage(ONLY_STATIC_LIBRARY)

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO Apra-Labs/whisper.cpp
    REF 74ad92406e58eee4c0618d8749df48aae91194cc #v1.5.2
    SHA512 897479792902f4cb2f36ef199c699b1cc1472c3b88898027230528b8ad1f27a0de1548fdf8eca44981e558c84cd37c8091375647d208195b1b7443a18a6cfa7e  # This is a temporary value. We will modify this value in the next section.
    HEAD_REF kj/add-Config-for-vcpkg
)


vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    DISABLE_PARALLEL_CONFIGURE
)

vcpkg_cmake_install()
vcpkg_cmake_config_fixup(
    CONFIG_PATH lib/cmake/whisper
    PACKAGE_NAME whisper
    )
vcpkg_copy_pdbs()

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")

file(INSTALL "${SOURCE_PATH}/LICENSE" DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}" RENAME copyright)
configure_file("${CMAKE_CURRENT_LIST_DIR}/usage" "${CURRENT_PACKAGES_DIR}/share/${PORT}/usage" COPYONLY)

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/share")