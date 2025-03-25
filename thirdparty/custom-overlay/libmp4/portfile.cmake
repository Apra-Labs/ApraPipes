# portfile.cmake for libmp4

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO Apra-Labs/libmp4
    REF db1b88f707e40edf7a77a7dcff58ec023e801caf    
    SHA512 503320d12cc3da5bf5b611d4852ad7d241bcff001ddae6f26b19d17c89746b53863798e71ec80c43853dd4bf9666502a9506f2ae9c3b58fcbd53be2fb11667e2
    HEAD_REF forApraPipes
)
vcpkg_configure_cmake(
    SOURCE_PATH "${SOURCE_PATH}"
    PREFER_NINJA 
)

vcpkg_build_cmake()

vcpkg_install_cmake()

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/share")

file(INSTALL ${SOURCE_PATH}/COPYING DESTINATION ${CURRENT_PACKAGES_DIR}/share/${PORT} RENAME copyright)

