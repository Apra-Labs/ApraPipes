vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO dpirch/libfvad
    REF 532ab666c20d3cfda38bca63abbb0f152706c369
    SHA512 926fb7155aae7a4ca6caf8e31a06e96125f8becda45bbb1218b2d2941b4ebf4e90d8552718e497b80a90d21a6813165d5e217cc354919eea4f2297d89226ed86
    HEAD_REF master
)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
)

vcpkg_cmake_install()
vcpkg_copy_pdbs()


file(INSTALL "${SOURCE_PATH}/LICENSE" DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}" RENAME copyright)
