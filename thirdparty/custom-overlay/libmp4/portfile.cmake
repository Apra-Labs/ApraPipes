# portfile.cmake for libmp4

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO Apra-Labs/libmp4
    REF 98f8ae9637093c822f344ec95c8cffbb814dd336    
    SHA512 34c8ced415b5b1e03c0b04148ca5647109a70226af1fdc3c0739c8d88e68294ebe187a59d44008d5bea3fbab7e09b19e311a03712710f62b53444a92e924db4c
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

