vcpkg_check_linkage(ONLY_STATIC_LIBRARY)

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO Apra-Labs/whisper.cpp
    REF c3bff0d121e2af823344939643d64a27e4a76ea2 #v1.5.4
    SHA512 d51a32c91340d2b9f18bf5221e134e57a0259bc3a1c803ef427adc6e3de5f54c556232cd4ef070b9c07f93968efd942a61cfe311c2cbca013a928f0eb8055e6f  # This is a temporary value. We will modify this value in the next section.
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
    OPTIONS
        ${FEATURE_OPTIONS}
        -DWHISPER_CUBLAS=${WHISPER_CUBLAS}
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