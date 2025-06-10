vcpkg_check_linkage(ONLY_STATIC_LIBRARY)

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO Apra-Labs/whisper.cpp
    REF 779050a5edfb47f115cd1c1c1e010b1c533bf174 #v1.5.4
    SHA512 fe354506c7377a7a6a783caccd09f1b9333394ada5046fadc77a51b1e2442089b73fcb730cea2ec82d31d205f47426349aa5e6206f294727341a829f90c127bf  # This is a temporary value. We will modify this value in the next section.
    HEAD_REF kj/add-Config-for-vcpkg
    PATCHES "fix-for-arm64.patch"
)

vcpkg_check_features(OUT_FEATURE_OPTIONS FEATURE_OPTIONS
 FEATURES
 "cuda" WHISPER_CUBLAS
)

set(WHISPER_CUBLAS OFF)
if("cuda" IN_LIST FEATURES)
  set(WHISPER_CUBLAS ON)
endif()

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