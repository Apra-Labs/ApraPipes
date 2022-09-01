cp base/vcpkg.nocuda.json base/vcpkg.json
CMAKE_THCOUNT=$(sh ./checkProc.sh)
mkdir -p _build
cd _build
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -DENABLE_CUDA=OFF  ../base -DCMAKE_TOOLCHAIN_FILE=../vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build . -- -j "$CMAKE_THCOUNT"
cd ..

mkdir -p _debugbuild
cd _debugbuild
cmake -DCMAKE_BUILD_TYPE=Debug -DENABLE_CUDA=OFF ../base -DCMAKE_TOOLCHAIN_FILE=../vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build . -- -j "$CMAKE_THCOUNT"
