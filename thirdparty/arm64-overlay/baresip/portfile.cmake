# portfile.cmake for Baresip

vcpkg_from_github(
  OUT_SOURCE_PATH SOURCE_PATH
  REPO Apra-Labs/baresip
  REF ea7840ff25a610e2968fc253aed1d774b7073cf9
  SHA512 12ddd8e44757233a10dca0307d04fd2c6436ba749c2573e11a7257440c2cbec5fb828ea4274f543b36691f5c2f7d9783df53efae0df3635c0208fac64ea4e934
  HEAD_REF forApraPipes
)

vcpkg_configure_cmake(
  SOURCE_PATH "${SOURCE_PATH}"
  PREFER_NINJA
)

vcpkg_build_cmake()

vcpkg_install_cmake()

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/share")

file(
  INSTALL "${SOURCE_PATH}/LICENSE"
  DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}"
  RENAME license
)

