# portfile.cmake for Baresip

vcpkg_from_github(
  OUT_SOURCE_PATH SOURCE_PATH
  REPO Apra-Labs/baresip
  REF ea7840ff25a610e2968fc253aed1d774b7073cf9
  SHA512 12ddd8e44757233a10dca0307d04fd2c6436ba749c2573e11a7257440c2cbec5fb828ea4274f543b36691f5c2f7d9783df53efae0df3635c0208fac64ea4e934
  HEAD_REF forApraPipes
)

vcpkg_cmake_configure(
  SOURCE_PATH "${SOURCE_PATH}"
  OPTIONS
    -DSTATIC=ON
)

vcpkg_cmake_build()

vcpkg_cmake_install()

# Only remove debug directories if they exist (not in release-only builds)
if(EXISTS "${CURRENT_PACKAGES_DIR}/debug")
    file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
    file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/share")
endif()

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")

