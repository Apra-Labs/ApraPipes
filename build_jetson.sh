chmod +x base/build_dependencies_linux_cuda.sh
./base/build_dependencies_jetson_cuda.sh

cd vcpkg
./bootstrap-vcpkg.sh
./vcpkg integrate install
cd ..

CMAKE_THCOUNT=$(sh ./checkProc.sh)
mkdir -p _build
cd _build
cmake -B . -DENABLE_ARM64=ON -DENABLE_WINDOWS=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo ../base -DCMAKE_TOOLCHAIN_FILE=../vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build . -- -j "$(($(nproc) - 1))"
