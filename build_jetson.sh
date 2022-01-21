CMAKE_THCOUNT=$(sh ./checkProc.sh)
mkdir -p _build
cd _build
# cmake -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON -DENABLE_ARM64=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo ../base 

# cmake -DENABLE_ARM64=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo ../base -DCMAKE_TOOLCHAIN_FILE=/mnt/disks/ssd/kashyap/ApraPipes/vcpkg/scripts/buildsystems/vcpkg.cmake
 cmake -B . -DENABLE_ARM64=ON -DENABLE_WINDOWS=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo ../base -DCMAKE_TOOLCHAIN_FILE=../vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build . -- -j "$(($(nproc) - 1))"
