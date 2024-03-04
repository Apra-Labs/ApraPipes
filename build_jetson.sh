chmod +x build_scripts/build_dependencies_jetson_cuda.sh
./build_scripts/build_dependencies_jetson_cuda.sh

chmod +x build_documentation.sh
./build_documentation.sh

cd vcpkg
./bootstrap-vcpkg.sh
./vcpkg integrate install
cd ..

CMAKE_THCOUNT=$(sh ./checkProc.sh)
mkdir -p _build
cd _build
export VCPKG_FORCE_SYSTEM_BINARIES=1 && export VCPKG_OVERLAY_PORTS=../thirdparty/custom-overlay && cmake -B . -DENABLE_ARM64=ON -DENABLE_WINDOWS=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo ../base -DCMAKE_TOOLCHAIN_FILE=../vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build . -- -j "$(($(nproc) - 1))"
