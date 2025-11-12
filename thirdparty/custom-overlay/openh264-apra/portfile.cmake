vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO Apra-Labs/openh264
    REF 4e3c4edd39c0192b98d10424fefe9c0b6bec1a2e
    SHA512 f33c1e01f1d2ff04dcf16e563eacec0bf43235db8afc241ff63c99e5896a2c97efe47d5eeee225bcfc8c49a214d22c42aea640e4d1626b5cc116cc26d59a201d
    HEAD_REF ForApraPipes
    PATCHES
        0001-respect-default-library-option.patch  # https://github.com/cisco/openh264/pull/3351
)

if((VCPKG_TARGET_ARCHITECTURE STREQUAL "x86" OR VCPKG_TARGET_ARCHITECTURE STREQUAL "x64"))
    vcpkg_find_acquire_program(NASM)
    get_filename_component(NASM_EXE_PATH ${NASM} DIRECTORY)
    vcpkg_add_to_path(${NASM_EXE_PATH})
elseif(VCPKG_TARGET_IS_WINDOWS)
    vcpkg_find_acquire_program(GASPREPROCESSOR)
    foreach(GAS_PATH ${GASPREPROCESSOR})
        get_filename_component(GAS_ITEM_PATH ${GAS_PATH} DIRECTORY)
        vcpkg_add_to_path(${GAS_ITEM_PATH})
    endforeach(GAS_PATH)
endif()

vcpkg_configure_meson(
    SOURCE_PATH ${SOURCE_PATH}
    OPTIONS -Dtests=disabled
)

vcpkg_install_meson()
vcpkg_copy_pdbs()
vcpkg_fixup_pkgconfig()

if(VCPKG_LIBRARY_LINKAGE STREQUAL "static")
    file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/bin" "${CURRENT_PACKAGES_DIR}/debug/bin")
endif()

configure_file("${SOURCE_PATH}/LICENSE" "${CURRENT_PACKAGES_DIR}/share/${PORT}/copyright" COPYONLY)
