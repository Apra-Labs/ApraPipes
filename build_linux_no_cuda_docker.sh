chmod +x build_scripts/build_dependencies_linux_no_cuda.sh
sudo ./build_scripts/build_dependencies_linux_no_cuda.sh

chmod +x base/fix-vcpkg-json.sh
./base/fix-vcpkg-json.sh true false false

cd vcpkg
./bootstrap-vcpkg.sh
./vcpkg integrate install
cd ..

CMAKE_THCOUNT=$(sh ./checkProc.sh)
mkdir -p _build
cd _build
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -DENABLE_CUDA=OFF  ../base -DCMAKE_TOOLCHAIN_FILE=../vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build . -- -j "$CMAKE_THCOUNT"

