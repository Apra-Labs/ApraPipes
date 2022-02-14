mkdir _build
cd _build
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -DENABLE_CUDA=OFF -DENABLE_WINDOWS=ON -DENABLE_LINUX=OFF -G "Visual Studio 15 2017" -A x64 -DCMAKE_TOOLCHAIN_FILE=../vcpkg/scripts/buildsystems/vcpkg.cmake -A x64 ../base 
cmake --build .
cd ..
pause

mkdir _debugbuild
cd _debugbuild
cmake -DCMAKE_BUILD_TYPE=Debug -DENABLE_CUDA=OFF -DENABLE_WINDOWS=ON -DENABLE_LINUX=OFF -G "Visual Studio 15 2017" -A x64 -DCMAKE_TOOLCHAIN_FILE=../vcpkg/scripts/buildsystems/vcpkg.cmake -A x64 ../base 
cmake --build .
pause