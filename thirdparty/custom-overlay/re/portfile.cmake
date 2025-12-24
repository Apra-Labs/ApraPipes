# portfile.cmake for lib_re

vcpkg_from_github(
  OUT_SOURCE_PATH SOURCE_PATH
  REPO Apra-Labs/re
  REF 5e516154d4354df8a753849270d235f02e04ac5a
  SHA512 b6875d8b98a06419619c7338ec53cc6c7078f24c3d5cacceac2ad43f201d8f302cdac14ce394c56e3ebf1b0b1692ea7feac4e58bb934e8923dead9608250e757
  HEAD_REF forApraPipes
)

vcpkg_cmake_configure(
  SOURCE_PATH "${SOURCE_PATH}"
  OPTIONS
    -DLIBRE_BUILD_SHARED=OFF
)

vcpkg_cmake_build()

vcpkg_cmake_install()

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
