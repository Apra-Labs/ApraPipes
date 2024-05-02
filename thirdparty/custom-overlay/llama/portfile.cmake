vcpkg_check_linkage(ONLY_STATIC_LIBRARY)

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO Apra-Labs/llama.cpp
    REF e5bd6e1abb146b38649236429c22ed6b4db0f3da
    SHA512 f36a0731e7b5044b1d75297fdd806cf19206a439bc9996bba1ee36b0b2e692e4482d5fac9b7dcd111c7d69bbd900b99ed38b301c572c450a48ad6fd484b3322f
    HEAD_REF kj/vcpkg-port
)

vcpkg_check_features(OUT_FEATURE_OPTIONS FEATURE_OPTIONS
 FEATURES
 "cuda" LLAMA_CUBLAS
)

set(LLAMA_CUBLAS OFF)
if("cuda" IN_LIST FEATURES)
  set(LLAMA_CUBLAS ON)
endif()


vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        ${FEATURE_OPTIONS}
        -DLLAMA_CUBLAS=${LLAMA_CUBLAS}
    DISABLE_PARALLEL_CONFIGURE
)

vcpkg_cmake_install()
vcpkg_cmake_config_fixup(
    CONFIG_PATH lib/cmake/Llama
    PACKAGE_NAME Llama
)
vcpkg_copy_pdbs()

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")

file(INSTALL "${SOURCE_PATH}/LICENSE" DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}" RENAME copyright)
configure_file("${CMAKE_CURRENT_LIST_DIR}/usage" "${CURRENT_PACKAGES_DIR}/share/${PORT}/usage" COPYONLY)

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/share")

if(VCPKG_LIBRARY_LINKAGE STREQUAL "static")
    file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/bin" "${CURRENT_PACKAGES_DIR}/debug/bin")
endif()